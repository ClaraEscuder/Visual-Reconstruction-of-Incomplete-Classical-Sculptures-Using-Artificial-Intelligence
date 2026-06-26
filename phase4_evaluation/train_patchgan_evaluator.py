"""
Entrenamiento del discriminador PatchGAN evaluador del TFG (Phase 4).

Este discriminador NO es el que se usa dentro del training adversarial de v9 (ese
es interno y se descarta). Aqui entrenamos uno NUEVO y SEPARADO, cuyo unico
proposito es servir como metrica de calidad domain-specific sobre las
reconstrucciones finales.

Estructura del entrenamiento:
  - Positivos: imagenes de whole_body (escultura clasica REAL e intacta)
  - Negativos: outputs del mejor generador inpainting (LaMa v9, MAT v9, o SD)

El discriminador aprende a distinguir "escultura clasica real" de
"reconstruccion artificial". Una vez entrenado, sirve como puntuador: cada
reconstruccion del corpus broken_body se pasa por D y se obtiene un score de
"domain consistency".

INPUT:
  - positivos: ~/tfg/background_removed/whole_body/   (1755 esculturas)
  - negativos: ~/tfg/inpainting_results/{generador_elegido}/  (configurable)

OUTPUT:
  - ~/tfg/MAT/checkpoints/patchgan_evaluator.pt  (los pesos del D entrenado)
  - ~/tfg/logs/patchgan_evaluator.log
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
from PIL import Image
from tqdm import tqdm


BASE = Path("/home/pfc/cescuder/tfg")

#positivos CON FONDO original (no rembg+SAM) para evitar bias de presencia/ausencia
#de fondo en el discriminador. El D debe discriminar por calidad de la
#reconstruccion del cuerpo, no por contexto de fondo.
DIR_POSITIVOS = BASE / "dataset_classificado" / "whole_body"

#pool MEZCLADO de negativos: las reconstrucciones recompuestas sobre su fondo
#original (carpetas _composited). Ambas clases (positivos y negativos) tienen
#fondo de fotografia real -> D solo puede aprender de la calidad del cuerpo,
#no del fondo.
DIRS_NEGATIVOS_POOL = [
    BASE / "inpainting_results" / "lama_v9_adversarial_v8masks_composited",
    BASE / "inpainting_results" / "mat_v9_adversarial_v8masks_composited",
    BASE / "inpainting_results" / "sd_controlnet_v8masks_composited",
]

DIR_CKPT_OUT = BASE / "MAT" / "checkpoints"
DIR_CKPT_OUT.mkdir(parents=True, exist_ok=True)
CKPT_D = DIR_CKPT_OUT / "patchgan_evaluator.pt"

TAMANO_IMG = 256
TAMANO_AUG = 286 #carga a 286, recorta aleatoriamente a 256 (augmentation)
BATCH_SIZE = 8
EPOCHS = 15 #reducido: con augmentation no hacen falta 30
LR = 2e-4
WEIGHT_DECAY = 5e-4 #regularizacion L2 en Adam
DROPOUT = 0.4  #regularizacion estructural en el D
NUM_WORKERS = 2
VAL_SPLIT = 0.10
SEED_SPLIT = 42
NDF = 32 #reducido desde 64: menos capacidad del D, menos overfitting
N_LAYERS = 2 #reducido desde 3: arquitectura mas pequena
LABEL_SMOOTH_POS = 0.9 #label smoothing: targets soft en vez de duros 1/0
LABEL_SMOOTH_NEG = 0.1


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(BASE / "logs" / "patchgan_evaluator.log", encoding="utf-8")])
log = logging.getLogger(__name__)


class PatchGAN(nn.Module):
    #PatchGAN modificado para uso como EVALUADOR (no como D en training GAN):
    #  - InstanceNorm en vez de BatchNorm (estable con batches pequenos)
    #  - Dropout entre capas conv (regularizacion estructural)
    #  - Capacidad reducida (ndf=32, n_layers=2) para evitar memorizar el dataset
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
    #Dataset con augmentation fuerte para entrenamiento, sin augmentation para val.
    #Augmentations: flip horizontal, rotacion +-10 deg, color jitter, random crop.
    #Multiplica efectivamente el dataset por ~20x en diversidad visual sin coste
    #de almacenamiento, lo que combate el overfitting agresivo del D.
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
            #cargamos a tamano mayor para luego hacer random crop a tamano final
            img = img.resize((self.tamano_aug, self.tamano_aug), Image.BILINEAR)

            #flip horizontal con probabilidad 0.5
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            #rotacion aleatoria +-10 grados
            angulo = random.uniform(-10.0, 10.0)
            img = img.rotate(angulo, resample=Image.BILINEAR, fillcolor=(0, 0, 0))

            #color jitter: brillo, contraste y saturacion +-20%
            arr = np.array(img).astype(np.float32)
            factor_brillo = random.uniform(0.8, 1.2)
            factor_contraste = random.uniform(0.8, 1.2)
            arr = arr * factor_brillo
            media = arr.mean(axis=(0, 1), keepdims=True)
            arr = (arr - media) * factor_contraste + media
            arr = arr.clip(0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

            #random crop de tamano_aug a tamano final
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    log.info(f"Positivos en: {DIR_POSITIVOS}")
    log.info(f"Pool de negativos: {[str(d) for d in DIRS_NEGATIVOS_POOL]}")

    if not DIR_POSITIVOS.exists():
        log.error("DIR_POSITIVOS no existe")
        return

    random.seed(SEED_SPLIT)
    np.random.seed(SEED_SPLIT)
    torch.manual_seed(SEED_SPLIT)

    archivos_pos = listar_archivos(DIR_POSITIVOS)

    #pool mezclado: cogemos todos los outputs de cada generador disponible y
    #los concatenamos. el D queda balanceado contra los 3 a la vez
    archivos_neg = []
    for d in DIRS_NEGATIVOS_POOL:
        if d.exists():
            f_list = listar_archivos(d)
            archivos_neg.extend(f_list)
            log.info(f"  negativos de {d.name}: {len(f_list)}")
        else:
            log.warning(f"  {d.name} no existe en disco, saltando")
    if not archivos_neg:
        log.error("pool de negativos vacio, abortando")
        return
    log.info(f"positivos: {len(archivos_pos)}  negativos (pool total): {len(archivos_neg)}")

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
    log.info(f"PatchGAN evaluador: ndf={NDF}, n_layers={N_LAYERS}, dropout={DROPOUT}")
    log.info(f"Numero de parametros: {n_params:,}  ({n_params/1e6:.2f}M)")
    log.info(f"Augmentation activada: flip + rot+-10 + color jitter + random crop {TAMANO_AUG}->{TAMANO_IMG}")
    log.info(f"Label smoothing: pos={LABEL_SMOOTH_POS}, neg={LABEL_SMOOTH_NEG}")
    log.info(f"Weight decay: {WEIGHT_DECAY}")
    optim = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999), weight_decay=WEIGHT_DECAY)
    bce = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        D.train()
        train_loss, train_acc, n = 0.0, 0.0, 0
        iter_neg = iter(dl_neg_train)
        for img_pos, lbl_pos in tqdm(dl_pos_train, desc=f"epoch {epoch}"):
            try:
                img_neg, lbl_neg = next(iter_neg)
            except StopIteration:
                iter_neg = iter(dl_neg_train)
                img_neg, lbl_neg = next(iter_neg)
            img_pos = img_pos.to(device, non_blocking=True)
            img_neg = img_neg.to(device, non_blocking=True)
            out_pos = D(img_pos)
            out_neg = D(img_neg)
            #label smoothing: en vez de 1.0/0.0 duros, usamos 0.9/0.1 -> el D
            #no puede llegar a confianza extrema, regulariza y mejora generalizacion
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

        log.info(f"epoch {epoch:02d}  train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(D.state_dict(), str(CKPT_D))
            log.info(f"   -> new best val_acc ({val_acc:.3f}), saved to {CKPT_D.name}")

    log.info(f"PATCHGAN EVALUATOR TRAINING COMPLETED - best val_acc: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
