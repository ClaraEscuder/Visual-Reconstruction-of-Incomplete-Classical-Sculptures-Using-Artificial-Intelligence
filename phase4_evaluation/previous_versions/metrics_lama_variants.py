"""
Metricas para comparar las 6 variantes de LaMa sobre broken_body.

Variantes evaluadas (cualquiera que exista en disco se evalua; las que falten se saltan):
    LaMa v1 baseline         -> inpainting_results/lama/
    LaMa v2 mask prefill     -> inpainting_results/lama_v2_mask_prefill/
    LaMa v3 tight crop       -> inpainting_results/lama_v3_tight_crop/
    LaMa v4 marble bg        -> inpainting_results/lama_v4_marble_bg/
    LaMa v5 iterativo        -> inpainting_results/lama_v5_iterativo/
    LaMa v6 fine-tuned       -> inpainting_results/lama_v6_finetuned/
    LaMa v7 densepose cond.  -> inpainting_results/lama_v7_densepose_cond/

    MAT v1 fondo blanco      -> inpainting_results/mat_v1_fondoblanco/  (historico)
    MAT v1 real              -> inpainting_results/mat_v1_real/
    MAT v2 mask prefill      -> inpainting_results/mat_v2_mask_prefill/
    MAT v3 tight crop        -> inpainting_results/mat_v3_tight_crop/
    MAT v4 marble bg         -> inpainting_results/mat_v4_marble_bg/
    MAT v5 iterativo         -> inpainting_results/mat_v5_iterativo/

Metricas calculadas:

    Por imagen (CSV detallado):
        - chroma_mask: saturacion media (HSV) dentro de la mascara. Bajo = acromatico/marmol, alto = color saturado de fondo (mal).
        - std_mask: desviacion estandar de luminosidad en la mascara. Muy bajo = propagacion plana (mal), moderado = textura.
        - dl_mask: diferencia L (CIELAB) media entre la mascara y el cuerpo visible. Cuando esta cerca de 0 la luminosidad
                   generada se parece a la del cuerpo (probablemente marmol); muy grande = "blanco" o "negro" propagado.

    Agregadas por variante (CSV resumen):
        - FID contra la distribucion whole_body (con pytorch-fid)
        - media y mediana de las anteriores

    Visual:
        - grid_lama_variants.png: collage de N esculturas representativas con la original,
          la mascara, y la reconstruccion de cada variante presente.

OUTPUT:
    - ~/tfg/inpainting_results/metrics/per_image.csv
    - ~/tfg/inpainting_results/metrics/per_variant.csv
    - ~/tfg/inpainting_results/metrics/grid_lama_variants.png
"""

import csv
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


BASE = Path("/home/pfc/cescuder/tfg")
DIR_IMG_ORIGINAL = BASE / "dataset_classificado" / "broken_body"
DIR_MASCARAS = BASE / "masks" / "broken_body"
DIR_RESULTADOS = BASE / "inpainting_results"
DIR_WHOLE = BASE / "dataset_classificado" / "whole_body"
DIR_METRICS = DIR_RESULTADOS / "metrics"

#mapeo nombre_variante -> (carpeta, sufijo del fichero output, etiqueta legible)
VARIANTES = [
    #LaMa
    ("lama_v1_baseline",DIR_RESULTADOS / "lama", "_lama.png", "LaMa v1 baseline"),
    ("lama_v2_mask_prefill",DIR_RESULTADOS / "lama_v2_mask_prefill", "_lamav2.png",  "LaMa v2 mask prefill"),
    ("lama_v3_tight_crop", DIR_RESULTADOS / "lama_v3_tight_crop", "_lamav3.png",  "LaMa v3 tight crop"),
    ("lama_v4_marble_bg", DIR_RESULTADOS / "lama_v4_marble_bg", "_lamav4.png",  "LaMa v4 marble bg"),
    ("lama_v5_iterativo", DIR_RESULTADOS / "lama_v5_iterativo", "_lamav5.png",  "LaMa v5 iterativo"),
    ("lama_v6_finetuned", DIR_RESULTADOS / "lama_v6_finetuned", "_lamav6.png",  "LaMa v6 finetuned"),
    ("lama_v7_densepose_cond", DIR_RESULTADOS / "lama_v7_densepose_cond", "_lamav7.png",  "LaMa v7 densepose cond"),
    #MAT
    ("mat_v1_fondoblanco", DIR_RESULTADOS / "mat_v1_fondoblanco", "_mat.png", "MAT v1 fondo blanco (historico)"),
    ("mat_v1_real", DIR_RESULTADOS / "mat_v1_real", "_matv1.png", "MAT v1 real"),
    ("mat_v2_mask_prefill", DIR_RESULTADOS / "mat_v2_mask_prefill","_matv2.png", "MAT v2 mask prefill"),
    ("mat_v3_tight_crop", DIR_RESULTADOS / "mat_v3_tight_crop", "_matv3.png", "MAT v3 tight crop"),
    ("mat_v4_marble_bg", DIR_RESULTADOS / "mat_v4_marble_bg", "_matv4.png", "MAT v4 marble bg"),
    ("mat_v5_iterativo", DIR_RESULTADOS / "mat_v5_iterativo", "_matv5.png", "MAT v5 iterativo"),
    ("mat_v6_finetuned", DIR_RESULTADOS / "mat_v6_finetuned", "_matv6.png", "MAT v6 finetuned"),
    ("mat_v7_densepose_cond", DIR_RESULTADOS / "mat_v7_densepose_cond", "_matv7.png", "MAT v7 densepose cond"),
]

#numero de esculturas representativas para el grid visual
N_GRID = 6

#umbral para detectar pixeles de fondo blanco al calcular las stats de cuerpo
UMBRAL_BLANCO = 245


#LOGGING:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(BASE / "logs" / "metrics_lama_variants.log", encoding="utf-8"),])
log = logging.getLogger(__name__)


#UTILIDADES DE COLOR:
def rgb_a_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convierte un array (..., 3) en RGB [0,255] a HSV [0,255]. Devuelve mismo shape."""
    from colorsys import rgb_to_hsv
    flat = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    hsv = np.array([rgb_to_hsv(*p) for p in flat], dtype=np.float32) * 255.0
    return hsv.reshape(rgb.shape)


def rgb_a_lab(rgb: np.ndarray) -> np.ndarray:
    """Conversion RGB->LAB rapida sin depender de skimage.
       Aproximacion: pasamos por sRGB->XYZ->LAB con la matriz D65."""
    arr = rgb.astype(np.float32) / 255.0
    #sRGB linearizado
    mask = arr <= 0.04045
    arr_lin = np.where(mask, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]], dtype=np.float32)
    xyz = arr_lin @ M.T
    #referencia D65
    xn, yn, zn = 0.95047, 1.0, 1.08883
    xyz_norm = xyz / np.array([xn, yn, zn], dtype=np.float32)
    delta = 6/29
    f = np.where(xyz_norm > delta**3,
                 np.cbrt(np.clip(xyz_norm, 1e-12, None)),
                 xyz_norm / (3 * delta**2) + 4/29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


#METRICAS POR IMAGEN:
def metricas_por_imagen(img_original_np: np.ndarray, img_pred_np: np.ndarray, mask_np: np.ndarray) -> dict:
    """
    Calcula chroma_mask, std_mask, dl_mask para una unica imagen reconstruida.
    Si la mascara esta vacia devuelve None (no se puede medir nada).
    """
    if mask_np.sum() == 0:
        return None

    #pixeles regenerados dentro de la mascara
    pred_mask = img_pred_np[mask_np]

    #chroma = saturacion HSV media
    hsv_mask = rgb_a_hsv(pred_mask.reshape(-1, 1, 3)).reshape(-1, 3)
    chroma_mask = float(hsv_mask[:, 1].mean())

    #std de luminosidad en mascara: aprox simple usando media de canales
    lum = pred_mask.astype(np.float32).mean(axis=1)
    std_mask = float(lum.std())

    #referencia del cuerpo visible: pixeles de la imagen ORIGINAL que no son blanco y no estan en la mascara
    no_blanco = (img_original_np.mean(axis=2) < UMBRAL_BLANCO)
    cuerpo = no_blanco & (~mask_np)
    if cuerpo.sum() == 0:
        dl_mask = float("nan")
    else:
        lab_pred = rgb_a_lab(pred_mask)
        lab_cuerpo = rgb_a_lab(img_original_np[cuerpo])
        L_pred = lab_pred[..., 0].mean()
        L_cuerpo = lab_cuerpo[..., 0].mean()
        dl_mask = float(abs(L_pred - L_cuerpo))

    return {"chroma_mask": chroma_mask, "std_mask": std_mask, "dl_mask": dl_mask}


#FID:
def calcular_fid(dir_pred: Path, dir_ref: Path) -> float:
    """
    Calcula FID entre dos carpetas usando pytorch-fid via subprocess.
    Devuelve NaN si falla por la razon que sea.
    """
    try:
        salida = subprocess.run(
            [sys.executable, "-m", "pytorch_fid", str(dir_pred), str(dir_ref), "--device", "cuda"],
            capture_output=True, text=True, timeout=3600,)
        #pytorch-fid imprime "FID:  XX.YY" al stdout
        for linea in salida.stdout.splitlines():
            if "FID" in linea:
                partes = linea.replace(":", " ").split()
                for tok in partes[::-1]:
                    try:
                        return float(tok)
                    except ValueError:
                        continue
        log.warning(f"FID: could not parse output for {dir_pred.name}")
        log.warning(f"   returncode: {salida.returncode}")
        log.warning(f"   stdout (full): {salida.stdout}")
        log.warning(f"   stderr (full): {salida.stderr}")
        return float("nan")
    except Exception as e:
        log.warning(f"FID computation failed for {dir_pred.name}: {e}")
        return float("nan")


#GRID VISUAL:
def construir_grid(seleccion: list, salida: Path):
    """
    seleccion = lista de dicts {nombre_img, img_orig (PIL), mask (PIL), preds: {variante: PIL}}
    Crea un grid: una fila por escultura, columnas = original, mask, v1, v2, ...
    """
    #orden de columnas: original, mask, luego todas las variantes en orden de definicion
    columnas = ["original", "mask"] + [v[0] for v in VARIANTES]
    n_cols = len(columnas)
    n_filas = len(seleccion)

    #cada celda a 256x256 para que ocupe poco
    celda = 256
    grid = Image.new("RGB", (n_cols * celda, n_filas * celda), color=(20, 20, 20))

    for i, item in enumerate(seleccion):
        for j, col in enumerate(columnas):
            if col == "original":
                img = item["img_orig"]
            elif col == "mask":
                img = item["mask"].convert("RGB")
            else:
                img = item["preds"].get(col)
                if img is None:
                    continue
            img_c = img.resize((celda, celda), Image.BILINEAR)
            grid.paste(img_c, (j * celda, i * celda))

    grid.save(salida)
    log.info(f"grid saved: {salida}")


#MAIN:
def main():
    DIR_METRICS.mkdir(parents=True, exist_ok=True)

    if not DIR_IMG_ORIGINAL.exists() or not DIR_MASCARAS.exists():
        log.error("Input dirs not found")
        return

    #detectar variantes presentes en disco
    variantes_presentes = []
    for nombre, carpeta, sufijo, etiqueta in VARIANTES:
        if carpeta.exists():
            n = len(list(carpeta.glob("*" + sufijo)))
            log.info(f"   {nombre}: {n} images at {carpeta}")
            if n > 0:
                variantes_presentes.append((nombre, carpeta, sufijo, etiqueta))
        else:
            log.info(f"   {nombre}: NOT PRESENT (skipped)")
    if not variantes_presentes:
        log.error("No variants found in disk. Run the LaMa inference scripts first.")
        return

    extensiones = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    imagenes = sorted([f for f in DIR_IMG_ORIGINAL.iterdir() if f.suffix in extensiones])
    log.info(f"broken_body images: {len(imagenes)}")

    #--- METRICAS POR IMAGEN ---
    per_image_path = DIR_METRICS / "per_image.csv"
    with open(per_image_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["image", "variant", "chroma_mask", "std_mask", "dl_mask"]
        writer.writerow(header)

        per_variant_stats = {v[0]: {"chroma": [], "std": [], "dl": []} for v in variantes_presentes}

        for img_path in tqdm(imagenes, desc="per-image metrics"):
            mask_path = DIR_MASCARAS / (img_path.stem + "_mask.png")
            if not mask_path.exists():
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                mask = Image.open(mask_path).convert("L")
                if mask.size != img.size:
                    mask = mask.resize(img.size, Image.NEAREST)
                img_np = np.array(img)
                mask_np = (np.array(mask) > 127)
            except Exception as e:
                log.warning(f"cannot read {img_path.name}: {e}")
                continue

            for nombre, carpeta, sufijo, etiqueta in variantes_presentes:
                pred_path = carpeta / (img_path.stem + sufijo)
                if not pred_path.exists():
                    continue
                try:
                    pred = Image.open(pred_path).convert("RGB")
                    if pred.size != img.size:
                        pred = pred.resize(img.size, Image.BILINEAR)
                    pred_np = np.array(pred)
                    m = metricas_por_imagen(img_np, pred_np, mask_np)
                    if m is None:
                        continue
                    writer.writerow([img_path.name, nombre, m["chroma_mask"], m["std_mask"], m["dl_mask"]])
                    per_variant_stats[nombre]["chroma"].append(m["chroma_mask"])
                    per_variant_stats[nombre]["std"].append(m["std_mask"])
                    if not np.isnan(m["dl_mask"]):
                        per_variant_stats[nombre]["dl"].append(m["dl_mask"])
                except Exception as e:
                    log.warning(f"error in {pred_path.name}: {e}")

    log.info(f"per-image metrics saved: {per_image_path}")

    #--- METRICAS AGREGADAS POR VARIANTE + FID ---
    per_variant_path = DIR_METRICS / "per_variant.csv"
    with open(per_variant_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "n", "chroma_mean", "chroma_median",
                         "std_mean", "std_median", "dl_mean", "dl_median", "fid_vs_wholebody"])

        for nombre, carpeta, sufijo, etiqueta in variantes_presentes:
            st = per_variant_stats[nombre]
            n = len(st["chroma"])
            if n == 0:
                continue

            chroma_mean = float(np.mean(st["chroma"]))
            chroma_med  = float(np.median(st["chroma"]))
            std_mean = float(np.mean(st["std"]))
            std_med  = float(np.median(st["std"]))
            dl_mean = float(np.mean(st["dl"])) if st["dl"] else float("nan")
            dl_med  = float(np.median(st["dl"])) if st["dl"] else float("nan")

            log.info(f"computing FID for {nombre}...")
            fid = calcular_fid(carpeta, DIR_WHOLE)
            log.info(f"   FID({nombre}) = {fid:.2f}")

            writer.writerow([nombre, n, chroma_mean, chroma_med,
                             std_mean, std_med, dl_mean, dl_med, fid])

    log.info(f"per-variant metrics saved: {per_variant_path}")

    #--- GRID VISUAL ---
    #escogemos N_GRID esculturas que tengan reconstruccion en TODAS las variantes presentes,
    #para que el grid quede balanceado y comparable.
    candidatas = []
    for img_path in imagenes:
        mask_path = DIR_MASCARAS / (img_path.stem + "_mask.png")
        if not mask_path.exists():
            continue
        preds = {}
        completo = True
        for nombre, carpeta, sufijo, _ in variantes_presentes:
            p = carpeta / (img_path.stem + sufijo)
            if p.exists():
                preds[nombre] = p
            else:
                completo = False
                break
        if completo:
            candidatas.append((img_path, mask_path, preds))

    log.info(f"candidates for grid (have all variants): {len(candidatas)}")
    if candidatas:
        #muestreo equiespaciado para tener variedad y no las primeras N alfabeticas
        if len(candidatas) <= N_GRID:
            sel = candidatas
        else:
            paso = len(candidatas) // N_GRID
            sel = [candidatas[i * paso] for i in range(N_GRID)]

        seleccion = []
        for img_path, mask_path, preds in sel:
            seleccion.append({
                "nombre_img": img_path.name,
                "img_orig": Image.open(img_path).convert("RGB"),
                "mask": Image.open(mask_path).convert("L"),
                "preds": {k: Image.open(v).convert("RGB") for k, v in preds.items()},})

        construir_grid(seleccion, DIR_METRICS / "grid_lama_variants.png")

    log.info("ALL METRICS COMPLETED")


if __name__ == "__main__":
    main()

