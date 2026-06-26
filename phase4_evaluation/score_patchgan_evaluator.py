"""
Aplica el discriminador PatchGAN evaluador entrenado a TODAS las variantes de
inpainting y produce un score promedio por variante.

El score representa la "domain consistency" media de las reconstrucciones de
cada variante respecto al dominio de escultura clasica intacta aprendido por
el discriminador. Valores altos = D opina que se parece a una escultura
clasica real. Valores bajos = D opina que parece artificial.

INPUT:
  - D entrenado: ~/tfg/MAT/checkpoints/patchgan_evaluator.pt
  - outputs de cada variante en ~/tfg/inpainting_results/*/

OUTPUT:
  - ~/tfg/inpainting_results/metrics/patchgan_scores.csv
"""

import sys
import csv
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

BASE = Path("/home/pfc/cescuder/tfg")
CKPT_D = BASE / "MAT" / "checkpoints" / "patchgan_evaluator.pt"
DIR_RESULTADOS = BASE / "inpainting_results"
DIR_METRICS = DIR_RESULTADOS / "metrics"
DIR_METRICS.mkdir(parents=True, exist_ok=True)
OUT_CSV = DIR_METRICS / "patchgan_scores.csv"

#variantes que evaluamos (cualquiera que exista en disco se procesa)
VARIANTES = [
    #variantes BASELINE (SIN pipeline) --> todas CON fondo original:
    "lama", #LaMa v1 baseline (con fondo)
    "mat_v1_real", #MAT v1 baseline (con fondo)
    "sd_baseline", #SD raw (con fondo)

    #variantes CON PIPELINE COMPLETO recomendadas (composited, con fondo):
    "lama_v9_adversarial_v8masks_composited", #LaMa v9 recompuesto sobre fondo
    "mat_v9_adversarial_v8masks_composited", #MAT v9 recompuesto sobre fondo
    "sd_controlnet_v8masks_composited", #SD+ControlNet recompuesto sobre fondo

    #variantes intermedias (ablation interna):
    "lama_v7_densepose_cond",
    "lama_v7_densepose_cond_nobg",
    "mat_v7_densepose_cond",
    "mat_v7_densepose_cond_nobg",
    "lama_v8_adversarial",
    "lama_v8_adversarial_v8masks",

    #variantes SIN composited (para comparar el efecto del fondo):
    "lama_v9_adversarial_v8masks",
    "mat_v9_adversarial_v8masks",
    "sd_v8masks",
    "sd_controlnet_v8masks",
    "sd_v8masks_composited",

    #otras variantes legacy (apendice):
    "lama_v2_mask_prefill",
    "lama_v3_tight_crop",
    "lama_v4_marble_bg",
    "lama_v5_iterativo",
    "lama_v6_finetuned",
    "mat_v2_mask_prefill",
    "mat_v3_tight_crop",
    "mat_v4_marble_bg",
    "mat_v5_iterativo",
    "mat_v6_finetuned",
]

TAMANO_IMG = 256

#hyperparams arquitectura: DEBEN coincidir EXACTAMENTE con los del training
#(train_patchgan_evaluator.py). Si cambias alli, cambia aqui tambien.
NDF = 32
N_LAYERS = 2
DROPOUT = 0.4


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(BASE / "logs" / "patchgan_scores.log", encoding="utf-8")])
log = logging.getLogger(__name__)


class PatchGAN(nn.Module):
    #DEBE ser identica a la del training (train_patchgan_evaluator.py):
    #InstanceNorm + Dropout + ndf=32 + n_layers=2
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


def cargar_discriminador(device):
    D = PatchGAN(in_channels=3, ndf=NDF, n_layers=N_LAYERS, dropout=DROPOUT).to(device).eval()
    state = torch.load(str(CKPT_D), map_location=device, weights_only=True)
    D.load_state_dict(state)
    return D


def score_imagen(img_path, D, device):
    img = Image.open(img_path).convert("RGB").resize((TAMANO_IMG, TAMANO_IMG), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        out = D(t)
    #score por imagen = sigmoid(media del mapa). cerca de 1 = "parece escultura clasica real"
    return float(torch.sigmoid(out.mean()).item())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    if not CKPT_D.exists():
        log.error(f"PatchGAN evaluator checkpoint not found: {CKPT_D}. Train it first.")
        return

    D = cargar_discriminador(device)
    log.info(f"PatchGAN evaluator cargado desde {CKPT_D}")

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    resultados = []
    for v in VARIANTES:
        carpeta = DIR_RESULTADOS / v
        if not carpeta.exists():
            log.warning(f"variante {v} no existe en disco, saltando")
            continue
        archivos = [f for f in carpeta.iterdir() if f.suffix in exts]
        if not archivos:
            log.warning(f"variante {v} vacia, saltando")
            continue

        scores = []
        for f in tqdm(archivos, desc=f"score {v}"):
            try:
                scores.append(score_imagen(f, D, device))
            except Exception as e:
                log.error(f"error en {f.name}: {e}")
        if not scores:
            continue

        mean_score = float(np.mean(scores))
        median_score = float(np.median(scores))
        std_score = float(np.std(scores))
        log.info(f"  {v}: n={len(scores)}  mean={mean_score:.4f}  median={median_score:.4f}  std={std_score:.4f}")
        resultados.append({
            "variante": v, "n": len(scores),
            "mean_score": mean_score, "median_score": median_score, "std_score": std_score,
        })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variante", "n", "mean_score", "median_score", "std_score"])
        writer.writeheader()
        for r in resultados:
            writer.writerow(r)

    log.info(f"PATCHGAN SCORES SAVED to {OUT_CSV}")


if __name__ == "__main__":
    main()
