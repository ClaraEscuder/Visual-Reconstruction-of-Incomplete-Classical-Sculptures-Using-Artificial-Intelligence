"""
Scoring per-generador: aplica cada D especifico a SU generador correspondiente.

Carga los 3 PatchGAN evaluadores entrenados (D_lama, D_mat, D_sd) y los aplica
a sus respectivas carpetas de outputs. Produce un CSV con:
  - val_acc del D entrenado (de los logs de training, lo extraemos)
  - score promedio del D sobre TODOS los outputs del generador
  - score promedio del D sobre los whole_body (positivos) para comparar

NOTA importante en la memoria:
  Los SCORES no son directamente comparables entre generadores (cada D tiene
  decision boundary distinta). Lo que SI es comparable es:
    - val_acc por generador: "cuanto distingue el D ese paradigma de real"
    - separacion entre score(reconstrucciones) y score(reales) por cada D

OUTPUT:
  - ~/tfg/inpainting_results/metrics/patchgan_per_gen_scores.csv
"""

import sys
import re
import csv
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

BASE = Path("/home/pfc/cescuder/tfg")
#positivos CON FONDO original (coherente con el training de cada D_xx)
DIR_POSITIVOS = BASE / "dataset_classificado" / "whole_body"
DIR_CKPT_OUT = BASE / "MAT" / "checkpoints"
DIR_LOGS = BASE / "logs"
DIR_METRICS = BASE / "inpainting_results" / "metrics"
DIR_METRICS.mkdir(parents=True, exist_ok=True)
OUT_CSV = DIR_METRICS / "patchgan_per_gen_scores.csv"

#cada D_xx fue entrenado con la variante composited correspondiente como negativos
#por tanto se evalua sobre esa misma carpeta (consistencia metodologica)
GENERADORES = {
    "lama": BASE / "inpainting_results" / "lama_v9_adversarial_v8masks_composited",
    "mat":  BASE / "inpainting_results" / "mat_v9_adversarial_v8masks_composited",
    "sd":   BASE / "inpainting_results" / "sd_controlnet_v8masks_composited",}

TAMANO_IMG = 256
NDF = 32
N_LAYERS = 2
DROPOUT = 0.4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(BASE / "logs" / "patchgan_per_gen_scores.log", encoding="utf-8")])
log = logging.getLogger(__name__)


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

def cargar_D(ckpt_path, device):
    D = PatchGAN(in_channels=3, ndf=NDF, n_layers=N_LAYERS, dropout=DROPOUT).to(device).eval()
    state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    D.load_state_dict(state)
    return D


def score_imagen(img_path, D, device):
    img = Image.open(img_path).convert("RGB").resize((TAMANO_IMG, TAMANO_IMG), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        out = D(t)
    return float(torch.sigmoid(out.mean()).item())


def extraer_val_acc_del_log(log_path):
    """Extrae el best val_acc del log de training."""
    if not log_path.exists():
        return float("nan")
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            contenido = f.read()
        #buscamos lineas tipo "best val_acc: 0.823"
        matches = re.findall(r"best val_acc:\s*([\d.]+)", contenido)
        if matches:
            return float(matches[-1])  #el ultimo (final) best
        #fallback: extraer todos los val_acc y devolver el max
        matches = re.findall(r"val_acc=([\d.]+)", contenido)
        if matches:
            return max(float(m) for m in matches)
    except Exception:
        pass
    return float("nan")

#----------------------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    archivos_pos = [f for f in DIR_POSITIVOS.iterdir() if f.suffix in exts]
    log.info(f"positivos (whole_body): {len(archivos_pos)}")

    resultados = []
    for generador, dir_gen in GENERADORES.items():
        log.info(f"=== procesando D_{generador} ===")

        ckpt = DIR_CKPT_OUT / f"patchgan_eval_{generador}.pt"
        if not ckpt.exists():
            log.error(f"  checkpoint no existe: {ckpt}. salta.")
            continue

        #extraer val_acc del log de training (lo guardamos como info)
        log_train = DIR_LOGS / f"patchgan_eval_{generador}.log"
        val_acc = extraer_val_acc_del_log(log_train)
        log.info(f"  val_acc del training: {val_acc:.3f}")

        D = cargar_D(ckpt, device)

        archivos_gen = [f for f in dir_gen.iterdir() if f.suffix in exts]
        log.info(f"  outputs en {dir_gen.name}: {len(archivos_gen)}")

        #score sobre el generador (esperamos < 0.5: D dice "es fake")
        scores_gen = []
        for f in tqdm(archivos_gen, desc=f"D_{generador} sobre {generador}"):
            try:
                scores_gen.append(score_imagen(f, D, device))
            except Exception as e:
                log.error(f"  error en {f.name}: {e}")
        mean_gen = float(np.mean(scores_gen)) if scores_gen else float("nan")
        median_gen = float(np.median(scores_gen)) if scores_gen else float("nan")
        std_gen = float(np.std(scores_gen)) if scores_gen else float("nan")

        #score sobre los reales (esperamos > 0.5: D dice "es real")
        scores_real = []
        for f in tqdm(archivos_pos, desc=f"D_{generador} sobre whole_body"):
            try:
                scores_real.append(score_imagen(f, D, device))
            except Exception as e:
                log.error(f"  error en {f.name}: {e}")
        mean_real = float(np.mean(scores_real)) if scores_real else float("nan")
        median_real = float(np.median(scores_real)) if scores_real else float("nan")

        #separacion: diferencia entre score(real) y score(generador)
        #cuanto mayor, mas distingue el D ese generador de la realidad
        separacion = mean_real - mean_gen

        log.info(f"  score sobre {generador}: mean={mean_gen:.4f} median={median_gen:.4f} std={std_gen:.4f}")
        log.info(f"  score sobre real (whole_body): mean={mean_real:.4f} median={median_real:.4f}")
        log.info(f"  SEPARACION (real - gen): {separacion:.4f}")

        resultados.append({
            "generador": generador,
            "val_acc_training": val_acc,
            "n_outputs": len(scores_gen),
            "mean_score_gen": mean_gen,
            "median_score_gen": median_gen,
            "std_score_gen": std_gen,
            "n_reales": len(scores_real),
            "mean_score_real": mean_real,
            "median_score_real": median_real,
            "separacion": separacion,
        })

    if resultados:
        fieldnames = list(resultados[0].keys())
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in resultados:
                writer.writerow(r)
        log.info(f"PATCHGAN PER-GEN SCORES SAVED to {OUT_CSV}")

        log.info("")
        log.info("===== RANKING POR SEPARACION (menor = mas realista) =====")
        ordenados = sorted(resultados, key=lambda r: r["separacion"])
        for r in ordenados:
            log.info(f"  {r['generador']}: separacion={r['separacion']:.4f}  val_acc={r['val_acc_training']:.3f}")
    else:
        log.warning("no se calcularon resultados")


if __name__ == "__main__":
    main()
