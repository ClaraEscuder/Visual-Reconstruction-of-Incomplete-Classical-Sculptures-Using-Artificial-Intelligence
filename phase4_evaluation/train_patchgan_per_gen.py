"""
Entrenamiento de un PatchGAN evaluador ESPECIFICO para UN generador concreto.

Uso (CLI):
    python train_patchgan_per_gen.py --generador lama
    python train_patchgan_per_gen.py --generador mat
    python train_patchgan_per_gen.py --generador sd

Por cada generador entrena un D separado:
  - Positivos: whole_body (escultura clasica real)
  - Negativos: SOLO outputs del generador especificado

Asi cada D aprende los artefactos ESPECIFICOS de su generador, y el val_acc
final indica cuanto se distingue ese paradigma de la escultura real.

OUTPUT:
  - checkpoints: ~/tfg/MAT/checkpoints/patchgan_eval_{generador}.pt
  - log:         ~/tfg/logs/patchgan_eval_{generador}.log

INTERPRETACION en la memoria:
  - val_acc alto (~0.85) -> D distingue facilmente este paradigma de real
                           -> el paradigma tiene artefactos detectables
  - val_acc bajo (~0.55) -> D apenas distingue
                           -> el paradigma es perceptualmente cercano a real
"""

import sys
import argparse
import random
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

BASE = Path("/home/pfc/cescuder/tfg")
#positivos CON FONDO original para coherencia metodologica con los negativos
#composited (ambos clases CON fondo -> D solo discrimina calidad del cuerpo)
DIR_POSITIVOS = BASE / "dataset_classificado" / "whole_body"
DIR_CKPT_OUT = BASE / "MAT" / "checkpoints"
DIR_CKPT_OUT.mkdir(parents=True, exist_ok=True)
DIR_LOGS = BASE / "logs"
DIR_LOGS.mkdir(parents=True, exist_ok=True)

#mapping de generador -> carpeta de outputs COMPOSITED (con fondo)
GENERADORES = {
    "lama": BASE / "inpainting_results" / "lama_v9_adversarial_v8masks_composited",
    "mat":  BASE / "inpainting_results" / "mat_v9_adversarial_v8masks_composited",
    "sd":   BASE / "inpainting_results" / "sd_controlnet_v8masks_composited",}

#hyperparams (mismos que el train general optimizado, para coherencia metodologica)
TAMANO_IMG = 256
TAMANO_AUG = 286
BATCH_SIZE = 8
EPOCHS = 15
LR = 2e-4
WEIGHT_DECAY = 5e-4
DROPOUT = 0.4
NUM_WORKERS = 2
VAL_SPLIT = 0.10
SEED_SPLIT = 42
NDF = 32
N_LAYERS = 2
LABEL_SMOOTH_POS = 0.9
LABEL_SMOOTH_NEG = 0.1

class PatchGAN(nn.Module):
    def __init__(self, in_channels=3, ndf=32, n_layers=2, dropout=0.4):
        super().__init__()
        kw, padw = 4, 1
        seq = [
            nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(dropout),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            seq += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, True),
                nn.Dropout2d(dropout),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        seq += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.InstanceNorm2d(ndf * nf_mult, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(dropout),
        ]
        seq += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*seq)

    def forward(self, x):
        return self.model(x)


class SculptureDataset(Dataset):
    def __init__(self, archivos, etiqueta, tamano, augment=False, tamano_aug=286):
        self.archivos = archivos
        self.etiqueta = etiqueta
        self.tamano = tamano
        self.augment = augment
        self.tamano_aug = tamano_aug

    def __len__(self):
        return len(self.archivos)

    def __getitem__(self, idx):
        p = self.archivos[idx]
        try:
            img = Image.open(p).convert("RGB")
        except (OSError, Image.UnidentifiedImageError):
            return self.__getitem__((idx + 1) % len(self))

        if self.augment:
            img = img.resize((self.tamano_aug, self.tamano_aug), Image.BILINEAR)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            angulo = random.uniform(-10.0, 10.0)
            img = img.rotate(angulo, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
            arr = np.array(img).astype(np.float32)
            factor_brillo = random.uniform(0.8, 1.2)
            factor_contraste = random.uniform(0.8, 1.2)
            arr = arr * factor_brillo
            media = arr.mean(axis=(0, 1), keepdims=True)
            arr = (arr - media) * factor_contraste + media
            arr = arr.clip(0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            offset_x = random.randint(0, self.tamano_aug - self.tamano)
            offset_y = random.randint(0, self.tamano_aug - self.tamano)
            img = img.crop((offset_x, offset_y, offset_x + self.tamano, offset_y + self.tamano))
        else:
            img = img.resize((self.tamano, self.tamano), Image.BILINEAR)

        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1), self.etiqueta


def listar_archivos(carpeta, exts={".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}):
    return [f for f in carpeta.iterdir() if f.suffix in exts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generador", required=True, choices=list(GENERADORES.keys()),
                        help="Nombre del generador: lama, mat o sd")
    args = parser.parse_args()
    generador = args.generador

    #configurar logging por generador
    log_file = DIR_LOGS / f"patchgan_eval_{generador}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_file, encoding="utf-8")])
    log = logging.getLogger(__name__)

    dir_neg = GENERADORES[generador]
    ckpt_d = DIR_CKPT_OUT / f"patchgan_eval_{generador}.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"========== ENTRENANDO D ESPECIFICO PARA: {generador.upper()} ==========")
    log.info(f"Device: {device}")
    log.info(f"Positivos en: {DIR_POSITIVOS}")
    log.info(f"Negativos en: {dir_neg}")
    log.info(f"Checkpoint output: {ckpt_d}")

    if not DIR_POSITIVOS.exists() or not dir_neg.exists():
        log.error(f"missing input dirs: pos={DIR_POSITIVOS.exists()}  neg={dir_neg.exists()}")
        return

    random.seed(SEED_SPLIT)
    np.random.seed(SEED_SPLIT)
    torch.manual_seed(SEED_SPLIT)

    archivos_pos = listar_archivos(DIR_POSITIVOS)
    archivos_neg = listar_archivos(dir_neg)
    log.info(f"positivos: {len(archivos_pos)}  negativos: {len(archivos_neg)}")

    random.shuffle(archivos_pos)
    random.shuffle(archivos_neg)

    n_val_p = max(1, int(len(archivos_pos) * VAL_SPLIT))
    n_val_n = max(1, int(len(archivos_neg) * VAL_SPLIT))

    pos_train = archivos_pos[n_val_p:]
    pos_val = archivos_pos[:n_val_p]
    neg_train = archivos_neg[n_val_n:]
    neg_val = archivos_neg[:n_val_n]

    ds_pos_train = SculptureDataset(pos_train, 1.0, TAMANO_IMG, augment=True, tamano_aug=TAMANO_AUG)
    ds_neg_train = SculptureDataset(neg_train, 0.0, TAMANO_IMG, augment=True, tamano_aug=TAMANO_AUG)
    ds_pos_val = SculptureDataset(pos_val, 1.0, TAMANO_IMG, augment=False)
    ds_neg_val = SculptureDataset(neg_val, 0.0, TAMANO_IMG, augment=False)

    dl_pos_train = DataLoader(ds_pos_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    dl_neg_train = DataLoader(ds_neg_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    dl_pos_val = DataLoader(ds_pos_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    dl_neg_val = DataLoader(ds_neg_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    D = PatchGAN(in_channels=3, ndf=NDF, n_layers=N_LAYERS, dropout=DROPOUT).to(device).train()
    n_params = sum(p.numel() for p in D.parameters())
    log.info(f"PatchGAN: ndf={NDF}, n_layers={N_LAYERS}, dropout={DROPOUT}, params={n_params:,}")
    optim = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999), weight_decay=WEIGHT_DECAY)
    bce = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        D.train()
        train_loss, train_acc, n = 0.0, 0.0, 0
        iter_neg = iter(dl_neg_train)
        for img_pos, lbl_pos in tqdm(dl_pos_train, desc=f"e{epoch} [{generador}]"):
            try:
                img_neg, lbl_neg = next(iter_neg)
            except StopIteration:
                iter_neg = iter(dl_neg_train)
                img_neg, lbl_neg = next(iter_neg)
            img_pos = img_pos.to(device, non_blocking=True)
            img_neg = img_neg.to(device, non_blocking=True)
            out_pos = D(img_pos)
            out_neg = D(img_neg)
            target_pos = torch.full_like(out_pos, LABEL_SMOOTH_POS)
            target_neg = torch.full_like(out_neg, LABEL_SMOOTH_NEG)
            loss = bce(out_pos, target_pos) + bce(out_neg, target_neg)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            train_loss += loss.item()
            acc = ((out_pos.mean(dim=[1, 2, 3]) > 0).float().mean().item() +
                   (out_neg.mean(dim=[1, 2, 3]) < 0).float().mean().item()) / 2.0
            train_acc += acc
            n += 1
        train_loss /= max(n, 1)
        train_acc /= max(n, 1)

        D.eval()
        val_acc, n_val = 0.0, 0
        with torch.no_grad():
            for img, _ in dl_pos_val:
                img = img.to(device)
                out = D(img)
                val_acc += (out.mean(dim=[1, 2, 3]) > 0).float().mean().item()
                n_val += 1
            for img, _ in dl_neg_val:
                img = img.to(device)
                out = D(img)
                val_acc += (out.mean(dim=[1, 2, 3]) < 0).float().mean().item()
                n_val += 1
        val_acc /= max(n_val, 1)

        log.info(f"e{epoch:02d}  train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(D.state_dict(), str(ckpt_d))
            log.info(f"   -> new best val_acc ({val_acc:.3f}), saved to {ckpt_d.name}")

    log.info(f"========== {generador.upper()} TRAINING COMPLETED - best val_acc: {best_val_acc:.3f} ==========")


if __name__ == "__main__":
    main()
