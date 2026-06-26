"""
Fine-tune adversarial SUAVE sobre MAT v7 NOBG (genera v8) — FASE 1 del
schedule adversarial de dos fases.

El objetivo de v8 NO es producir las reconstrucciones finales del corpus,
sino servir de WARM-UP para que despues v9 (peso_adv=0.5) pueda warm-startar
desde un generador que YA ha visto un discriminador. Schedule simetrico al
de LaMa (v8 + v9), para que la comparacion en la memoria sea metodologicamente
equivalente entre arquitecturas.

INPUT:
    - imagenes: ~/tfg/synthetic_dataset_bw_first/images_no_bg/
    - masks:    ~/tfg/synthetic_dataset_bw_first/masks/
    - cache DP: ~/tfg/synthetic_dataset_bw_first/densepose_cache/images/
    - pesos:    ~/tfg/MAT/checkpoints/best_finetuned_mat_v7_nobg.pt

OUTPUT:
    - ~/tfg/MAT/checkpoints/best_finetuned_mat_v8.pt
    - ~/tfg/MAT/checkpoints/last_finetuned_mat_v8.pt
    - ~/tfg/logs/finetune_mat_v8.log
"""

import sys
import random
import logging
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models as tv_models
from PIL import Image
from tqdm import tqdm


BASE = Path("/home/pfc/cescuder/tfg")

DIR_MAT_REPO = BASE / "MAT"
sys.path.insert(0, str(DIR_MAT_REPO))
sys.path.insert(0, str(BASE / "scripts"))

from networks.mat import Generator
from networks.basic_module import Conv2dLayer
from networks.mat import Conv2dLayerPartial

from finetune_mat_v7_densepose import (
    DatasetMATv7,
    sintetizar_uv_en_region,
    construir_mascara_aleatoria,
    PerdidaPerceptual,
    expandir_conv2d_layer,
    expandir_conv2d_partial,
    first_stage_forward_con_dp,
    synthesis_forward_con_dp,
    generator_forward_con_dp,
)


DIR_IMAGENES = BASE / "synthetic_dataset_bw_first" / "images_no_bg"
DIR_MASCARAS_DP15 = BASE / "synthetic_dataset_bw_first" / "masks"
DIR_DP_CACHE = BASE / "synthetic_dataset_bw_first" / "densepose_cache" / "images"

PKL_MAT = DIR_MAT_REPO / "Places_512_FullData_G.pkl"
DIR_CKPT = DIR_MAT_REPO / "checkpoints"
DIR_CKPT.mkdir(parents=True, exist_ok=True)
CKPT_V7_NOBG = DIR_CKPT / "best_finetuned_mat_v7_nobg.pt"
CKPT_BEST = DIR_CKPT / "best_finetuned_mat_v8.pt"
CKPT_LAST = DIR_CKPT / "last_finetuned_mat_v8.pt"


TAMANO_IMG = 512
BATCH_SIZE = 2
EPOCHS = 8                  #simetrico con LaMa v8
LR_G = 1e-5                 #LR mayor que v9: fase suave, mas margen de aprendizaje
LR_D = 4e-5
VAL_SPLIT = 0.10
NUM_WORKERS = 2             #aolin-gpu-4 tiene 7.8GB RAM, mejor con 2 workers
PESO_L1 = 1.0
PESO_PERCEPTUAL = 0.1
PESO_ADV = 0.01             #SUAVE: equivalente a LaMa v8
SEED_SPLIT = 42

EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 1e-4


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(BASE / "logs" / "finetune_mat_v8.log", encoding="utf-8")])
log = logging.getLogger(__name__)


def cargar_mat_v8(device):
    """Construye MAT expandido a 7 canales y carga los pesos v7 NOBG como warm start."""
    if not CKPT_V7_NOBG.exists():
        raise FileNotFoundError(f"v7 NOBG checkpoint not found at {CKPT_V7_NOBG}. Run v7 NOBG MAT first.")

    log.info(f"building MAT generator")
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=TAMANO_IMG, img_channels=3)

    log.info(f"loading MAT base weights from {PKL_MAT}")
    state_base = torch.load(str(PKL_MAT), map_location="cpu", weights_only=False)
    G.load_state_dict(state_base, strict=False)

    log.info("expanding FirstStage.conv_first 4 -> 7 channels")
    G.synthesis.first_stage.conv_first = expandir_conv2d_partial(
        G.synthesis.first_stage.conv_first, in_nuevo=7)
    enc_attr = f"EncConv_Block_{TAMANO_IMG}x{TAMANO_IMG}"
    enc_first = getattr(G.synthesis.enc, enc_attr)
    log.info(f"expanding Encoder.{enc_attr}.conv0 7 -> 10 channels")
    enc_first.conv0 = expandir_conv2d_layer(enc_first.conv0, in_nuevo=10)

    G.synthesis.first_stage.forward = MethodType(first_stage_forward_con_dp, G.synthesis.first_stage)
    G.synthesis.forward = MethodType(synthesis_forward_con_dp, G.synthesis)
    G.forward = MethodType(generator_forward_con_dp, G)

    if CKPT_LAST.exists():
        log.info(f"resuming v8 from {CKPT_LAST}")
        state = torch.load(str(CKPT_LAST), map_location=device, weights_only=False)
        G.load_state_dict(state["generator"], strict=False)
        start_epoch = int(state.get("epoch", 0)) + 1
        best_val = float(state.get("best_val", float("inf")))
        epochs_sin_mejora = int(state.get("epochs_sin_mejora", 0))
    else:
        log.info(f"warm-start desde v7 NOBG: {CKPT_V7_NOBG}")
        state = torch.load(str(CKPT_V7_NOBG), map_location=device, weights_only=False)
        G.load_state_dict(state["generator"], strict=False)
        log.info("pesos v7 NOBG cargados, polishing con adversarial desde epoch 0")
        start_epoch = 0
        best_val = float("inf")
        epochs_sin_mejora = 0

    return G.to(device), start_epoch, best_val, epochs_sin_mejora


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64, n_layers=3):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if device == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}  compute: {torch.cuda.get_device_capability(0)}")
    log.info(f"Hyperparams: BATCH={BATCH_SIZE}, LR_G={LR_G}, LR_D={LR_D}, EPOCHS={EPOCHS}")
    log.info(f"             L1={PESO_L1}, PERC={PESO_PERCEPTUAL}, ADV={PESO_ADV}")

    random.seed(SEED_SPLIT)
    np.random.seed(SEED_SPLIT)
    torch.manual_seed(SEED_SPLIT)

    stems_img = {p.stem for p in DIR_IMAGENES.glob("*.jpg")}
    stems_mask = {p.stem for p in DIR_MASCARAS_DP15.glob("*.png")}
    stems = sorted(stems_img & stems_mask)
    log.info(f"stems disponibles: {len(stems)}")

    random.shuffle(stems)
    n_val = max(1, int(len(stems) * VAL_SPLIT))
    stems_val = stems[:n_val]
    stems_train = stems[n_val:]
    log.info(f"train: {len(stems_train)} | val: {len(stems_val)}")

    ds_train = DatasetMATv7(stems_train, DIR_IMAGENES, DIR_MASCARAS_DP15, DIR_DP_CACHE)
    ds_val = DatasetMATv7(stems_val, DIR_IMAGENES, DIR_MASCARAS_DP15, DIR_DP_CACHE)
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    G, start_epoch, best_val, epochs_sin_mejora = cargar_mat_v8(device)
    G.train()
    D = PatchGANDiscriminator(in_channels=3).to(device).train()

    perceptual_fn = PerdidaPerceptual().to(device).eval()
    params_entrenables_g = [p for p in G.parameters() if p.requires_grad]
    optim_g = torch.optim.Adam(params_entrenables_g, lr=LR_G, betas=(0.5, 0.999))
    optim_d = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(start_epoch, EPOCHS):
        G.train()
        D.train()
        train_g_total = 0.0
        train_d_total = 0.0
        n_batches = 0

        for img_t, mask_t, dp_t in tqdm(dl_train, desc=f"train e{epoch}"):
            img_t = img_t.to(device, non_blocking=True)
            mask_t = mask_t.to(device, non_blocking=True)
            dp_t = dp_t.to(device, non_blocking=True)

            z = torch.randn(img_t.size(0), G.z_dim, device=device)
            c = torch.zeros(img_t.size(0), G.c_dim, device=device)

            pred = G(img_t, mask_t, dp_t, z, c, truncation_psi=1, noise_mode="const")

            # train D
            optim_d.zero_grad(set_to_none=True)
            d_real = D(img_t)
            d_fake = D(pred.detach())
            loss_d_real = bce(d_real, torch.ones_like(d_real))
            loss_d_fake = bce(d_fake, torch.zeros_like(d_fake))
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            loss_d.backward()
            optim_d.step()

            # train G
            optim_g.zero_grad(set_to_none=True)
            d_fake_for_g = D(pred)
            loss_l1 = F.l1_loss(pred, img_t)
            loss_perc = perceptual_fn(pred, img_t)
            loss_adv = bce(d_fake_for_g, torch.ones_like(d_fake_for_g))
            loss_g = PESO_L1 * loss_l1 + PESO_PERCEPTUAL * loss_perc + PESO_ADV * loss_adv
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(params_entrenables_g, max_norm=5.0)
            optim_g.step()

            train_g_total += loss_g.item()
            train_d_total += loss_d.item()
            n_batches += 1

        train_g_avg = train_g_total / max(n_batches, 1)
        train_d_avg = train_d_total / max(n_batches, 1)

        G.eval()
        val_g_total = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for img_t, mask_t, dp_t in tqdm(dl_val, desc=f"val e{epoch}"):
                img_t = img_t.to(device, non_blocking=True)
                mask_t = mask_t.to(device, non_blocking=True)
                dp_t = dp_t.to(device, non_blocking=True)
                z = torch.randn(img_t.size(0), G.z_dim, device=device)
                c = torch.zeros(img_t.size(0), G.c_dim, device=device)
                pred = G(img_t, mask_t, dp_t, z, c, truncation_psi=1, noise_mode="const")
                loss_l1 = F.l1_loss(pred, img_t)
                loss_perc = perceptual_fn(pred, img_t)
                loss_v = PESO_L1 * loss_l1 + PESO_PERCEPTUAL * loss_perc
                val_g_total += loss_v.item()
                n_val_batches += 1
        val_loss = val_g_total / max(n_val_batches, 1)

        log.info(f"epoch {epoch:02d} | train_g={train_g_avg:.4f} | train_d={train_d_avg:.4f} | val={val_loss:.4f}")

        mejora = val_loss < (best_val - EARLY_STOP_MIN_DELTA)
        if mejora:
            best_val = val_loss
            epochs_sin_mejora = 0
        else:
            epochs_sin_mejora += 1

        state_save = {
            "epoch": epoch,
            "best_val": best_val,
            "epochs_sin_mejora": epochs_sin_mejora,
            "generator": G.state_dict(),
        }
        torch.save(state_save, str(CKPT_LAST))

        if mejora:
            torch.save(state_save, str(CKPT_BEST))
            log.info(f"   -> new best val ({val_loss:.4f}), saved")
        else:
            log.info(f"   sin mejora ({epochs_sin_mejora}/{EARLY_STOP_PATIENCE}) best={best_val:.4f}")

        if epochs_sin_mejora >= EARLY_STOP_PATIENCE:
            log.info(f"Early stopping en epoch {epoch}. Best val: {best_val:.4f}")
            break

    log.info(f"FINE-TUNE v8 MAT COMPLETED - best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
