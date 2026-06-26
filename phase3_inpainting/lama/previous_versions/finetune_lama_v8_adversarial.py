"""
Fine-tune adversarial sobre LaMa v7 NOBG (genera lo que llamamos v8).

Punto de partida: el checkpoint v7 NOBG ya entrenado (L1+perceptual, 20 epochs).
Polishing con loss adicional adversarial para romper la "regression to the mean"
del L1. Misma arquitectura 7-canal con conditioning DensePose.

Loss G = L1 + lambda_perc * perceptual + lambda_adv * adversarial
Loss D = BCE(real, 1) + BCE(fake, 0)  con PatchGAN 70x70

INPUT:
    - imagenes:   ~/tfg/synthetic_dataset_bw_first/images_no_bg/
    - mascaras 15 clases: ~/tfg/synthetic_dataset_bw_first/masks/
    - cache UV:   ~/tfg/synthetic_dataset_bw_first/densepose_cache/images/
    - pesos v7 NOBG: ~/tfg/lama_repo/big-lama/models/best_finetuned_v7_nobg.ckpt

OUTPUT:
    - ~/tfg/lama_repo/big-lama/models/best_finetuned_v8.ckpt
    - ~/tfg/lama_repo/big-lama/models/last_finetuned_v8.ckpt
    - ~/tfg/logs/finetune_lama_v8.log
"""

import sys
import random
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models as tv_models
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_dilation

sys.path.insert(0, "/home/pfc/cescuder/tfg/scripts")
from ffc_standalone import FFCResNetGenerator, BIG_LAMA_GENERATOR_KWARGS, cargar_pesos_big_lama


BASE = Path("/home/pfc/cescuder/tfg")

DIR_IMAGENES = BASE / "synthetic_dataset_bw_first" / "images_no_bg"
DIR_MASCARAS_DP15 = BASE / "synthetic_dataset_bw_first" / "masks"
DIR_DP_CACHE = BASE / "synthetic_dataset_bw_first" / "densepose_cache" / "images"

DIR_LAMA_REPO = BASE / "lama_repo"
DIR_BIG_LAMA = DIR_LAMA_REPO / "big-lama"
CKPT_LAMA = DIR_BIG_LAMA / "models" / "best.ckpt"
CKPT_V7_NOBG = DIR_BIG_LAMA / "models" / "best_finetuned_v7_nobg.ckpt"
CKPT_FINETUNED = DIR_BIG_LAMA / "models" / "best_finetuned_v8.ckpt"
CKPT_LAST = DIR_BIG_LAMA / "models" / "last_finetuned_v8.ckpt"


TAMANO_IMG = 256
BATCH_SIZE = 4
EPOCHS = 10
LR_G = 1e-5
LR_D = 4e-5
VAL_SPLIT = 0.10
NUM_WORKERS = 2
PESO_L1 = 1.0
PESO_PERCEPTUAL = 0.1
PESO_ADV = 0.01
SEED_SPLIT = 42

N_PARTES_MIN = 1
N_PARTES_MAX = 3
DILATACION_MASK_PX = 8
NUM_CLASES_DENSEPOSE_15 = 15
MAX_PART_ID = 24.0
PROB_SIMULAR_PROYECCION = 0.5

EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 1e-4


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),logging.FileHandler(BASE / "logs" / "finetune_lama_v8.log", encoding="utf-8")])
log = logging.getLogger(__name__)


def sintetizar_uv_en_region(mascara_region: np.ndarray):
    U = np.zeros(mascara_region.shape, dtype=np.float32)
    V = np.zeros(mascara_region.shape, dtype=np.float32)
    ys, xs = np.where(mascara_region)


    if len(ys) < 2:
        return U, V
    pts = np.column_stack([ys, xs]).astype(np.float64)
    centroide = pts.mean(axis=0)
    pts_c = pts - centroide
    cov = np.cov(pts_c.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx_p = np.argmax(eigvals)
    eje_principal = eigvecs[:, idx_p]
    eje_perp = np.array([-eje_principal[1], eje_principal[0]])
    proj_p = pts_c @ eje_principal
    proj_q = pts_c @ eje_perp
    p_min, p_max = proj_p.min(), proj_p.max()
    q_min, q_max = proj_q.min(), proj_q.max()
    if p_max - p_min > 1e-6:
        proj_p_norm = (proj_p - p_min) / (p_max - p_min)


    else:
        proj_p_norm = np.full_like(proj_p, 0.5)

    if q_max - q_min > 1e-6:
        proj_q_norm = (proj_q - q_min) / (q_max - q_min)
    else:
        proj_q_norm = np.full_like(proj_q, 0.5)
    U[ys, xs] = proj_p_norm.astype(np.float32)
    V[ys, xs] = proj_q_norm.astype(np.float32)

    return U, V


def construir_mascara_aleatoria(mask_dp15: np.ndarray):
    partes_disponibles = [c for c in range(1, NUM_CLASES_DENSEPOSE_15) if (mask_dp15 == c).any()]
    if not partes_disponibles:
        return np.zeros_like(mask_dp15, dtype=np.uint8)
    n_partes = random.randint(N_PARTES_MIN, min(N_PARTES_MAX, len(partes_disponibles)))
    elegidas = random.sample(partes_disponibles, n_partes)
    mascara = np.zeros_like(mask_dp15, dtype=bool)
    for c in elegidas:
        mascara |= (mask_dp15 == c)
    if DILATACION_MASK_PX > 0:
        mascara = binary_dilation(mascara, iterations=DILATACION_MASK_PX)
    return mascara.astype(np.uint8)


class SyntheticInpaintCondDataset(Dataset):
    def __init__(self, tripletas, tamano):
        self.tripletas = tripletas
        self.tamano = tamano

    def __len__(self):
        return len(self.tripletas)

    def __getitem__(self, idx):
        img_path, mask_path, dp_path = self.tripletas[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except (OSError, Image.UnidentifiedImageError):
            return self.__getitem__((idx + 1) % len(self))

        mask_dp15 = np.array(Image.open(mask_path))
        if mask_dp15.ndim == 3:
            mask_dp15 = mask_dp15[..., 0]

        img = img.resize((self.tamano, self.tamano), Image.BILINEAR)
        mask_dp15 = np.array(Image.fromarray(mask_dp15).resize((self.tamano, self.tamano), Image.NEAREST))

        mascara_bin = construir_mascara_aleatoria(mask_dp15)

        d = np.load(dp_path)
        I_full = d["I"].astype(np.float32)
        U_full = d["U"].astype(np.float32)
        V_full = d["V"].astype(np.float32)
        if I_full.shape != (self.tamano, self.tamano):
            I_full = np.array(Image.fromarray(I_full.astype(np.uint8)).resize((self.tamano, self.tamano), Image.NEAREST)).astype(np.float32)
            U_full = np.array(Image.fromarray(U_full).resize((self.tamano, self.tamano), Image.BILINEAR)).astype(np.float32)
            V_full = np.array(Image.fromarray(V_full).resize((self.tamano, self.tamano), Image.BILINEAR)).astype(np.float32)

        if random.random() < PROB_SIMULAR_PROYECCION and mascara_bin.any():
            region_bool = mascara_bin.astype(bool)
            u_grad, v_grad = sintetizar_uv_en_region(region_bool)
            U_full[region_bool] = u_grad[region_bool]
            V_full[region_bool] = v_grad[region_bool]
            partes_visibles = I_full[(mask_dp15 > 0) & (mask_dp15 < NUM_CLASES_DENSEPOSE_15)]
            partes_visibles = partes_visibles[partes_visibles > 0]
            if partes_visibles.size > 0:
                I_full[region_bool] = float(np.median(partes_visibles))

        img_np = np.array(img).astype(np.float32) / 127.5 - 1.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float()

        mask_t = torch.from_numpy(mascara_bin.astype(np.float32)).unsqueeze(0)

        I_norm = (I_full / MAX_PART_ID).clip(0, 1)
        U_norm = U_full.clip(0, 1)
        V_norm = V_full.clip(0, 1)
        cond_t = torch.from_numpy(np.stack([I_norm, U_norm, V_norm], axis=0)).float()

        return img_t, mask_t, cond_t


def listar_tripletas():
    tripletas = []
    for img_path in sorted(DIR_IMAGENES.iterdir()):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        mask_path = DIR_MASCARAS_DP15 / (img_path.stem + ".png")
        dp_path = DIR_DP_CACHE / (img_path.stem + ".npz")
        if mask_path.exists() and dp_path.exists():
            tripletas.append((img_path, mask_path, dp_path))
    return tripletas


def expandir_primer_conv(generator, in_orig=4, in_nuevo=7):
    modificadas = 0
    def reemplazar_en(parent: nn.Module):
        nonlocal modificadas
        for nombre, hijo in list(parent.named_children()):
            if isinstance(hijo, nn.Conv2d) and hijo.in_channels == in_orig:
                conv_nuevo = nn.Conv2d(
                    in_channels=in_nuevo,
                    out_channels=hijo.out_channels,
                    kernel_size=hijo.kernel_size,
                    stride=hijo.stride,
                    padding=hijo.padding,
                    dilation=hijo.dilation,
                    groups=hijo.groups,
                    bias=hijo.bias is not None,
                    padding_mode=hijo.padding_mode,
                )
                with torch.no_grad():
                    conv_nuevo.weight.zero_()
                    conv_nuevo.weight[:, :in_orig].copy_(hijo.weight)
                    nn.init.normal_(conv_nuevo.weight[:, in_orig:], mean=0.0, std=0.01)
                    if hijo.bias is not None:
                        conv_nuevo.bias.copy_(hijo.bias)
                setattr(parent, nombre, conv_nuevo)
                modificadas += 1
            else:
                reemplazar_en(hijo)
    reemplazar_en(generator)
    return modificadas


def cargar_generator_v8(device):
    """Construye el generator 7-canal y carga los pesos v7 NOBG como warm start."""
    if not CKPT_LAMA.exists():
        raise FileNotFoundError(f"big-lama checkpoint not found at {CKPT_LAMA}")
    if not CKPT_V7_NOBG.exists():
        raise FileNotFoundError(f"v7 NOBG checkpoint not found at {CKPT_V7_NOBG}. Run v7 NOBG first.")

    log.info("building FFCResNetGenerator with big-lama params (input_nc=4)")
    generator = FFCResNetGenerator(**BIG_LAMA_GENERATOR_KWARGS)

    log.info(f"loading big-lama base weights (4ch) from {CKPT_LAMA}")
    cargar_pesos_big_lama(generator, str(CKPT_LAMA), device=device)

    log.info("expanding first conv 4 -> 7 channels (RGB + mask + I + U + V)")
    expandir_primer_conv(generator, in_orig=4, in_nuevo=7)

    if CKPT_LAST.exists():
        log.info(f"resuming v8 from {CKPT_LAST}")
        state = torch.load(str(CKPT_LAST), map_location=device, weights_only=False)
        gen_sd = {k[len("generator."):]: v for k, v in state["state_dict"].items() if k.startswith("generator.")}
        generator.load_state_dict(gen_sd, strict=False)
        start_epoch = int(state.get("epoch", 0)) + 1
        best_val = float(state.get("best_val", float("inf")))
        log.info(f"  resuming at epoch {start_epoch} (previous best={best_val:.4f})")
    else:
        log.info(f"warm-start desde v7 NOBG: {CKPT_V7_NOBG}")
        state = torch.load(str(CKPT_V7_NOBG), map_location=device, weights_only=False)
        gen_sd = {k[len("generator."):]: v for k, v in state["state_dict"].items() if k.startswith("generator.")}
        generator.load_state_dict(gen_sd, strict=False)
        log.info("  pesos v7 NOBG cargados, polishing con adversarial desde epoch 1")
        start_epoch = 1
        best_val = float("inf")

    generator = generator.to(device).train()
    return generator, start_epoch, best_val


class PatchGANDiscriminator(nn.Module):
    """Discriminador PatchGAN 70x70. Recibe imagen RGB (3 canales) y devuelve mapa
    de patches con valores 0-1 (sigmoid).
    """
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


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(4)])
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(4, 9)])
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(9, 16)])
        self.slice4 = nn.Sequential(*[vgg[i] for i in range(16, 23)])
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _norm(self, x):
        x_01 = (x + 1.0) / 2.0
        return (x_01 - self.mean) / self.std

    def forward(self, pred, target):
        p, t = self._norm(pred), self._norm(target)
        loss = 0.0
        for slc in (self.slice1, self.slice2, self.slice3, self.slice4):
            p, t = slc(p), slc(t)
            loss = loss + F.l1_loss(p, t)
        return loss


def guardar_checkpoint(generator, path, epoch, best_val):
    state_full = {"state_dict": {}, "epoch": int(epoch), "best_val": float(best_val)}
    for k, v in generator.state_dict().items():
        state_full["state_dict"][f"generator.{k}"] = v.detach().cpu()
    torch.save(state_full, str(path))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    log.info(f"Hyperparams: BATCH={BATCH_SIZE}, LR_G={LR_G}, LR_D={LR_D}, EPOCHS={EPOCHS}, IMG={TAMANO_IMG}")
    log.info(f"             L1={PESO_L1}, PERC={PESO_PERCEPTUAL}, ADV={PESO_ADV}")

    random.seed(SEED_SPLIT)
    np.random.seed(SEED_SPLIT)
    torch.manual_seed(SEED_SPLIT)

    tripletas = listar_tripletas()
    log.info(f"tripletas: {len(tripletas)}")
    if not tripletas:
        log.error("No hay tripletas. Saliendo.")
        return

    random.shuffle(tripletas)
    n_val = max(1, int(len(tripletas) * VAL_SPLIT))
    val_tripletas = tripletas[:n_val]
    train_tripletas = tripletas[n_val:]
    log.info(f"train: {len(train_tripletas)} | val: {len(val_tripletas)}")

    ds_train = SyntheticInpaintCondDataset(train_tripletas, TAMANO_IMG)
    ds_val = SyntheticInpaintCondDataset(val_tripletas, TAMANO_IMG)
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    generator, start_epoch, best_val = cargar_generator_v8(device)
    discriminator = PatchGANDiscriminator(in_channels=3).to(device).train()

    perceptual = VGGPerceptualLoss().to(device).eval()
    optim_g = torch.optim.Adam(generator.parameters(), lr=LR_G, betas=(0.5, 0.999))
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=LR_D, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    epochs_sin_mejora = 0

    for epoch in range(start_epoch, EPOCHS + 1):
        generator.train()
        discriminator.train()
        train_g_total = 0.0
        train_d_total = 0.0
        n_batches = 0

        for img_t, mask_t, cond_t in tqdm(dl_train, desc=f"epoch {epoch} train"):
            img_t = img_t.to(device, non_blocking=True)
            mask_t = mask_t.to(device, non_blocking=True)
            cond_t = cond_t.to(device, non_blocking=True)

            masked_img = img_t * (1 - mask_t)
            entrada = torch.cat([masked_img, mask_t, cond_t], dim=1)

            salida = generator(entrada)
            pred = masked_img + salida * mask_t

            # train D
            optim_d.zero_grad(set_to_none=True)
            d_real = discriminator(img_t)
            d_fake = discriminator(pred.detach())
            loss_d_real = bce(d_real, torch.ones_like(d_real))
            loss_d_fake = bce(d_fake, torch.zeros_like(d_fake))
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            loss_d.backward()
            optim_d.step()

            # train G
            optim_g.zero_grad(set_to_none=True)
            d_fake_for_g = discriminator(pred)
            loss_l1 = F.l1_loss(pred * mask_t, img_t * mask_t)
            loss_perc = perceptual(pred, img_t)
            loss_adv = bce(d_fake_for_g, torch.ones_like(d_fake_for_g))
            loss_g = PESO_L1 * loss_l1 + PESO_PERCEPTUAL * loss_perc + PESO_ADV * loss_adv
            loss_g.backward()
            optim_g.step()

            train_g_total += loss_g.item()
            train_d_total += loss_d.item()
            n_batches += 1

        train_g_avg = train_g_total / max(n_batches, 1)
        train_d_avg = train_d_total / max(n_batches, 1)

        generator.eval()
        val_g_total = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for img_t, mask_t, cond_t in tqdm(dl_val, desc=f"epoch {epoch} val"):
                img_t = img_t.to(device, non_blocking=True)
                mask_t = mask_t.to(device, non_blocking=True)
                cond_t = cond_t.to(device, non_blocking=True)
                masked_img = img_t * (1 - mask_t)
                entrada = torch.cat([masked_img, mask_t, cond_t], dim=1)
                salida = generator(entrada)
                pred = masked_img + salida * mask_t
                loss_l1 = F.l1_loss(pred * mask_t, img_t * mask_t)
                loss_perc = perceptual(pred, img_t)
                loss_v = PESO_L1 * loss_l1 + PESO_PERCEPTUAL * loss_perc
                val_g_total += loss_v.item()
                n_val_batches += 1
        val_loss = val_g_total / max(n_val_batches, 1)

        log.info(f"epoch {epoch:02d} | train_g={train_g_avg:.4f} | train_d={train_d_avg:.4f} | val={val_loss:.4f}")

        guardar_checkpoint(generator, CKPT_LAST, epoch, best_val)

        mejora = val_loss < (best_val - EARLY_STOP_MIN_DELTA)
        if mejora:
            best_val = val_loss
            epochs_sin_mejora = 0
            guardar_checkpoint(generator, CKPT_FINETUNED, epoch, best_val)
            log.info(f"   -> new best val ({val_loss:.4f}), saved")
        else:
            epochs_sin_mejora += 1
            log.info(f"   sin mejora ({epochs_sin_mejora}/{EARLY_STOP_PATIENCE}) best={best_val:.4f}")

        if epochs_sin_mejora >= EARLY_STOP_PATIENCE:
            log.info(f"Early stopping en epoch {epoch}. Best val: {best_val:.4f}")
            break

    log.info(f"FINE-TUNE v8 COMPLETED - best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
