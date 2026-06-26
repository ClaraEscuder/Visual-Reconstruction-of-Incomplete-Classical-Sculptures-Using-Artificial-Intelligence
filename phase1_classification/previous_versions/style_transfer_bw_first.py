"""
Genera una version revisada del dataset sintetico para fine-tunear LaMa.

Diferencia clave con style_transfer.py (el que se uso para DeepLabv3+):
    - En la version original las imagenes COCO pasaban primero a gris, se coloreaban con el
      tono del marmol y se hacia histogram matching, pero al final se mezclaba con la imagen
      ORIGINAL via alpha (alpha=0.8). Ese ultimo paso filtraba un 20% del color del fondo
      original, por eso las imagenes finales conservaban un poco de tinte de fondo (verde
      vegetacion, azul cielo, etc).

    - Aqui forzamos la conversion a B/N COMPLETA al inicio y eliminamos la mezcla alpha al
      final (alpha=1.0 efectivo). El resultado es marmol limpio sin trazas del color
      original.

Salida:
  - imagenes en  ~/tfg/synthetic_dataset_bw_first/images/
  - mascaras en  ~/tfg/synthetic_dataset_bw_first/masks/  (copias de las DensePose originales)
"""

import random
import logging
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


BASE = Path("/home/pfc/cescuder/tfg")

DIR_CONTENIDO = BASE / "densepose_dataset" / "images"
DIR_MASCARAS = BASE / "densepose_dataset" / "masks"
DIR_ESTILOS = BASE / "dataset_esculturas" / "archive_2" / "images" / "zeus"
DIR_SALIDA_IMG = BASE / "synthetic_dataset_bw_first" / "images"
DIR_SALIDA_MASK = BASE / "synthetic_dataset_bw_first" / "masks"

#numero maximo de imagenes de estilo (marmol) que cargamos en memoria para muestrear
MAX_ESTILOS = 200


#LOGGING:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(str(BASE / "logs" / "style_transfer_bw_first.log"), encoding="utf-8"),])
log = logging.getLogger(__name__)


#STYLE TRANSFER B/N PRIMERO:
def match_histograma_canal(canal_src, canal_ref):
    src_flat = canal_src.flatten()
    ref_flat = canal_ref.flatten()

    hist_src, _ = np.histogram(src_flat, bins=256, range=(0, 256))
    cdf_src = hist_src.cumsum().astype(float)
    cdf_src /= cdf_src[-1]

    hist_ref, _ = np.histogram(ref_flat, bins=256, range=(0, 256))
    cdf_ref = hist_ref.cumsum().astype(float)
    cdf_ref /= cdf_ref[-1]

    lookup = np.zeros(256, dtype=np.uint8)
    ref_idx = 0
    for src_val in range(256):
        while ref_idx < 255 and cdf_ref[ref_idx] < cdf_src[src_val]:
            ref_idx += 1
        lookup[src_val] = ref_idx

    return lookup[canal_src]


def aplicar_estilo_marmol_bw_first(img_contenido: np.ndarray, img_estilo: np.ndarray) -> np.ndarray:
    """
    Convierte img_contenido a marmol partiendo de B/N puro y sin alpha-mix con la original.

    Pasos:
      1) Conversion total a escala de grises (3 canales replicados).
      2) Tinte con el tono medio del marmol de referencia.
      3) Histogram matching canal a canal contra el marmol.
      4) (no hay paso 4: ya no mezclamos con la original)
    """
    #1.gris puro
    gris = np.mean(img_contenido, axis=2, keepdims=True)
    img_gris = np.repeat(gris, 3, axis=2).astype(np.uint8)

    #2.tono medio del marmol
    tono = np.mean(img_estilo, axis=(0, 1))

    #3.colorear el gris con el tono del marmol
    coloreado = np.zeros_like(img_gris, dtype=np.float32)
    for c in range(3):
        factor = tono[c] / 128.0
        coloreado[:, :, c] = np.clip(img_gris[:, :, c] * factor, 0, 255)
    coloreado = coloreado.astype(np.uint8)

    #4.histogram matching para textura (mismo que el original)
    texturizado = np.zeros_like(coloreado)
    for c in range(3):
        texturizado[:, :, c] = match_histograma_canal(coloreado[:, :, c], img_estilo[:, :, c])

    return texturizado


#MAIN:
def main():
    DIR_SALIDA_IMG.mkdir(parents=True, exist_ok=True)
    DIR_SALIDA_MASK.mkdir(parents=True, exist_ok=True)

    log.info("STYLE TRANSFER (B/N FIRST, NO ALPHA-MIX)")

    extensiones = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    paths_estilos = [p for p in DIR_ESTILOS.rglob("*") if p.suffix in extensiones]
    if not paths_estilos:
        log.error(f"There are no style images at: {DIR_ESTILOS}")
        return

    log.info(f"sampling up to {MAX_ESTILOS} style images")
    if len(paths_estilos) > MAX_ESTILOS:
        paths_estilos = random.sample(paths_estilos, MAX_ESTILOS)

    estilos = []
    for p in tqdm(paths_estilos, desc="loading styles"):
        try:
            estilos.append(np.array(Image.open(p).convert("RGB")))
        except Exception:
            continue
    log.info(f"styles loaded: {len(estilos)}")

    contenidos = [p for p in DIR_CONTENIDO.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not contenidos:
        log.error(f"There are no content images at: {DIR_CONTENIDO}")
        return
    log.info(f"content images to process: {len(contenidos)}")

    procesadas = 0
    saltadas = 0
    errores = 0

    for img_path in tqdm(contenidos, desc="bw-first style transfer"):
        try:
            salida_path = DIR_SALIDA_IMG / img_path.name
            mask_src = DIR_MASCARAS / (img_path.stem + ".png")
            mask_dst = DIR_SALIDA_MASK / (img_path.stem + ".png")

            #checkpoint implicito: si la imagen ya esta, garantizar tambien que la mask este copiada
            if salida_path.exists():
                if mask_src.exists() and not mask_dst.exists():
                    shutil.copy2(mask_src, mask_dst)
                saltadas += 1
                continue

            if not mask_src.exists():
                #COCO image sin su mascara DensePose: la descartamos del dataset de fine-tune
                continue

            img = np.array(Image.open(img_path).convert("RGB"))
            estilo = random.choice(estilos)
            resultado = aplicar_estilo_marmol_bw_first(img, estilo)

            Image.fromarray(resultado).save(salida_path, quality=95)
            shutil.copy2(mask_src, mask_dst)

            procesadas += 1
            if procesadas % 500 == 0:
                log.info(f"processed: {procesadas}/{len(contenidos)}")

        except Exception as e:
            log.warning(f"Error in {img_path.name}: {e}")
            errores += 1

    log.info(f"COMPLETED - Processed: {procesadas}, Skipped (already done): {saltadas}, Errors: {errores}")
    log.info(f"Output images: {DIR_SALIDA_IMG}")
    log.info(f"Output masks:  {DIR_SALIDA_MASK}")


if __name__ == "__main__":
    main()
