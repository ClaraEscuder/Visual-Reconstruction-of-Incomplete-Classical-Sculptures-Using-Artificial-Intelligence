"""
Metricas finales del TFG sobre todas las variantes de inpainting.

Calcula dos bloques:

(A) Metricas SIN ground truth (todas las variantes, corpus broken_body):
    - FID respecto al dominio whole_body (escultura clasica intacta)
    - cromaticidad media en LAB (a, b) y desviacion estandar
    - delta-L (brillo) respecto a la pieza original

(B) Metricas CON ground truth (corpus sintetico whole_body_holed_brokenized,
    donde tenemos pareja original/rota/restaurada):
    - PSNR
    - SSIM
    - LPIPS

INPUT:
  - originales (positivos para FID):    ~/tfg/background_removed/whole_body/
  - variantes a evaluar:                ~/tfg/inpainting_results/{variante}/
  - sinteticos GT:                      ~/tfg/synthetic_eval/whole/ (intactas)
                                        ~/tfg/synthetic_eval/restored_{variante}/

OUTPUT:
  - ~/tfg/inpainting_results/metrics/metricas_finales.csv
"""

import sys
import csv
import logging
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    TORCHMETRICS_OK = True
except ImportError:
    TORCHMETRICS_OK = False

try:
    from skimage.color import rgb2lab
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False


BASE = Path("/home/pfc/cescuder/tfg")
DIR_REAL_WHOLE = BASE / "background_removed" / "whole_body"
DIR_BROKEN_ORIG = BASE / "background_removed" / "broken_body"
DIR_RESULTADOS = BASE / "inpainting_results"
DIR_METRICS = DIR_RESULTADOS / "metrics"
DIR_METRICS.mkdir(parents=True, exist_ok=True)
OUT_CSV = DIR_METRICS / "metricas_finales.csv"

DIR_SYNTH_WHOLE = BASE / "synthetic_eval" / "whole"

VARIANTES_A_EVALUAR = [
    #tabla 2: pipeline ablation LaMa:
    "lama", #baseline out-of-the-box (CON fondo)
    "lama_v7_densepose_cond_nobg", #+DensePose conditioning + nobg
    "lama_v9_adversarial_v8masks_composited", #v9 recompuesto sobre fondo

    #tabla 2: pipeline ablation MAT:
    "mat_v1_real", #baseline out-of-the-box (CON fondo)
    "mat_v7_densepose_cond_nobg", #+DensePose conditioning + nobg
    "mat_v9_adversarial_v8masks_composited", #v9 recompuesto sobre fondo

    #tabla 3: ablation SD:
    "sd_baseline", #SD raw (con fondo) + mask v8
    "sd_v8masks_composited", #sd_v8masks recompuesto sobre fondo
    "sd_controlnet_v8masks", #SD + Multi-ControlNet
    "sd_controlnet_v8masks_composited", #SD+CN recompuesto sobre fondo

    #tabla 1: comparacion central (versiones sin fondo, comparable a metricas previas):
    "lama_v8_adversarial_v8masks",
    "lama_v9_adversarial_v8masks",
    "mat_v9_adversarial_v8masks",
    "sd_v8masks",
    "sd_v8masks",
]

TAMANO_FID = 299

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(BASE / "logs" / "metricas_finales.log", encoding="utf-8")])
log = logging.getLogger(__name__)

def listar_archivos(carpeta):
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    return [f for f in carpeta.iterdir() if f.suffix in exts]


def cargar_uint8(p, size):
    img = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.array(img).astype(np.uint8)


def fid_variante(archivos_var, archivos_real, device):
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    fid.set_dtype(torch.float64)
    for f in tqdm(archivos_real, desc="FID real"):
        arr = cargar_uint8(f, TAMANO_FID)
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        fid.update(t, real=True)
    for f in tqdm(archivos_var, desc="FID fake"):
        arr = cargar_uint8(f, TAMANO_FID)
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        fid.update(t, real=False)
    return float(fid.compute().item())


def chroma_y_brillo(archivos_var, archivos_orig_lookup):
    #cromaticidad LAB media y std + delta-L respecto al original asociado
    a_vals, b_vals = [], []
    dL_vals = []
    for f in tqdm(archivos_var, desc="chroma+dL"):
        try:
            img_v = np.array(Image.open(f).convert("RGB")).astype(np.float32) / 255.0
        except Exception:
            continue
        lab_v = rgb2lab(img_v)
        a_vals.append(float(np.mean(lab_v[..., 1])))
        b_vals.append(float(np.mean(lab_v[..., 2])))
        L_v = float(np.mean(lab_v[..., 0]))

        stem = f.stem
        #sufijos ORDENADOS por longitud descendente (longest match wins) para
        #evitar que "_sd" trunque erroneamente "_sdv8" como "_v8"
        sufijos_por_longitud = [
            "_composited", "_sdbaseline", "_lamav8", "_lamav9", "_matv7", "_matv9",
            "_sdv8", "_sdcn", "_lama", "_mat", "_sd",
        ]
        for suf in sufijos_por_longitud:
            if stem.endswith(suf):
                stem = stem[:-len(suf)]
                break
        orig = archivos_orig_lookup.get(stem)
        if orig is None:
            continue
        try:
            img_o = np.array(Image.open(orig).convert("RGB").resize(
                (img_v.shape[1], img_v.shape[0]), Image.BILINEAR)).astype(np.float32) / 255.0
            L_o = float(np.mean(rgb2lab(img_o)[..., 0]))
            dL_vals.append(L_v - L_o)
        except Exception:
            continue

    return {
        "chroma_a_mean": float(np.mean(a_vals)) if a_vals else float("nan"),
        "chroma_a_std": float(np.std(a_vals)) if a_vals else float("nan"),
        "chroma_b_mean": float(np.mean(b_vals)) if b_vals else float("nan"),
        "chroma_b_std": float(np.std(b_vals)) if b_vals else float("nan"),
        "dL_mean": float(np.mean(dL_vals)) if dL_vals else float("nan"),
        "dL_abs_mean": float(np.mean(np.abs(dL_vals))) if dL_vals else float("nan"),
    }


def psnr_ssim_lpips(carpeta_restored, carpeta_gt, device):
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

    archivos = listar_archivos(carpeta_restored)
    gt_lookup = {p.stem: p for p in listar_archivos(carpeta_gt)}

    p_vals, s_vals, l_vals = [], [], []
    sufijos_por_longitud = [
        "_composited", "_sdbaseline", "_restored", "_lamav8", "_lamav9", "_matv7", "_matv9",
        "_sdv8", "_sdcn", "_lama", "_mat", "_sd",
    ]
    for f in tqdm(archivos, desc=f"PSNR/SSIM/LPIPS {carpeta_restored.name}"):
        stem = f.stem
        for suf in sufijos_por_longitud:
            if stem.endswith(suf):
                stem = stem[:-len(suf)]
                break
        gt = gt_lookup.get(stem)
        if gt is None:
            continue
        try:
            a = np.array(Image.open(f).convert("RGB").resize((256, 256), Image.BILINEAR)).astype(np.float32) / 255.0
            b = np.array(Image.open(gt).convert("RGB").resize((256, 256), Image.BILINEAR)).astype(np.float32) / 255.0
        except Exception:
            continue
        ta = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)
        tb = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            p_vals.append(float(psnr(ta, tb).item()))
            s_vals.append(float(ssim(ta, tb).item()))
            l_vals.append(float(lpips_m(ta, tb).item()))

    if not p_vals:
        return {"psnr": float("nan"), "ssim": float("nan"), "lpips": float("nan"), "n_gt": 0}
    return {
        "psnr": float(np.mean(p_vals)),
        "ssim": float(np.mean(s_vals)),
        "lpips": float(np.mean(l_vals)),
        "n_gt": len(p_vals),
    }


def main():
    if not TORCHMETRICS_OK:
        log.error("torchmetrics no esta instalado. pip install torchmetrics[image]")
        return
    if not SKIMAGE_OK:
        log.error("skimage no esta instalado. pip install scikit-image")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    archivos_real = listar_archivos(DIR_REAL_WHOLE)
    archivos_orig_lookup = {p.stem: p for p in listar_archivos(DIR_BROKEN_ORIG)}
    log.info(f"reales whole_body: {len(archivos_real)}  broken_body originales: {len(archivos_orig_lookup)}")

    resultados = []
    for v in VARIANTES_A_EVALUAR:
        carpeta = DIR_RESULTADOS / v
        if not carpeta.exists():
            log.warning(f"variante {v} no existe, saltando")
            continue
        archivos_var = listar_archivos(carpeta)
        if not archivos_var:
            log.warning(f"variante {v} vacia, saltando")
            continue

        log.info(f"=== variante {v} (n={len(archivos_var)}) ===")
        fid = fid_variante(archivos_var, archivos_real, device)
        log.info(f"  FID: {fid:.3f}")

        cb = chroma_y_brillo(archivos_var, archivos_orig_lookup)
        log.info(f"  chroma a: {cb['chroma_a_mean']:.2f}+-{cb['chroma_a_std']:.2f}  "
                 f"b: {cb['chroma_b_mean']:.2f}+-{cb['chroma_b_std']:.2f}  "
                 f"|dL|: {cb['dL_abs_mean']:.2f}")

        #PSNR/SSIM/LPIPS si existe un corpus sintetico con GT
        carpeta_synth = BASE / "synthetic_eval" / f"restored_{v}"
        if carpeta_synth.exists() and DIR_SYNTH_WHOLE.exists():
            psl = psnr_ssim_lpips(carpeta_synth, DIR_SYNTH_WHOLE, device)
            log.info(f"  PSNR: {psl['psnr']:.2f}  SSIM: {psl['ssim']:.3f}  LPIPS: {psl['lpips']:.3f}  (n={psl['n_gt']})")
        else:
            psl = {"psnr": float("nan"), "ssim": float("nan"), "lpips": float("nan"), "n_gt": 0}

        resultados.append({
            "variante": v, "n": len(archivos_var),
            "FID": fid,
            **cb,
            **psl,
        })

    if resultados:
        fieldnames = list(resultados[0].keys())
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in resultados:
                writer.writerow(r)
        log.info(f"METRICAS FINALES SAVED to {OUT_CSV}")
    else:
        log.warning("no se calcularon metricas (sin variantes en disco)")


if __name__ == "__main__":
    main()
