"""
Aplica style transfer estadístico a imágenes COCO usando esculturas
de mármol como referencia de estilo.

METHOD: Histogram Matching + Color Transfer
-no requiere pesos preentrenados ni decoder
-funciona directamente sin entrenamiento
-transfiere el tono, contraste y textura visual del mármol
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
DIR_SALIDA_IMG = BASE / "synthetic_dataset" / "images"
DIR_SALIDA_MASK = BASE / "synthetic_dataset" / "masks"

ALPHA = 0.8


#LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),logging.FileHandler(str(Path("/home/pfc/cescuder/tfg/logs") / "style_transfer.log"), encoding="utf-8"),])
log = logging.getLogger(__name__)



#STYLE TRANSFER ESTADÍSTICO:

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


def aplicar_estilo_marmol(img_contenido, img_estilo, alpha):
    #1.convertir contenido a escala de grises
    gris = np.mean(img_contenido, axis=2, keepdims=True)
    img_gris = np.repeat(gris, 3, axis=2).astype(np.uint8)

    #2.tono medio del mármol
    tono = np.mean(img_estilo, axis=(0, 1))

    #3.colorear el gris con el tono del mármol
    coloreado = np.zeros_like(img_gris, dtype=np.float32)
    for c in range(3):
        factor = tono[c] / 128.0
        coloreado[:, :, c] = np.clip(img_gris[:, :, c] * factor, 0, 255)
    coloreado = coloreado.astype(np.uint8)

    #4.histogram matching para textura
    texturizado = np.zeros_like(coloreado)
    for c in range(3):
        texturizado[:, :, c] = match_histograma_canal(coloreado[:, :, c], img_estilo[:, :, c])

    #5.mezcla final
    resultado = (alpha*texturizado + (1-alpha) * img_contenido).astype(np.uint8)
    return resultado


#MAIN:---------------------------

def main():
    DIR_SALIDA_IMG.mkdir(parents=True, exist_ok=True)
    DIR_SALIDA_MASK.mkdir(parents=True, exist_ok=True)

    log.info("STYLE TRANSFER")

    #cargar estilos
    extensiones = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    paths_estilos = [p for p in DIR_ESTILOS.rglob("*") if p.suffix in extensiones]
    if not paths_estilos:
        log.error(f"There are no images: {DIR_ESTILOS}")
        return

    log.info(f"getting {min(len(paths_estilos), 200)} images of style!!!")
    if len(paths_estilos) > 200:
        paths_estilos = random.sample(paths_estilos, 200)

    estilos = []
    for p in tqdm(paths_estilos, desc="getting styles"):
        try:
            estilos.append(np.array(Image.open(p).convert("RGB")))
        except Exception:
            continue

    log.info(f"styles downloaded: {len(estilos)}")

    # Cargar contenidos
    contenidos = [p for p in DIR_CONTENIDO.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not contenidos:
        log.error(f"There are no images: {DIR_CONTENIDO}")
        return

    log.info(f"images to process: {len(contenidos)}")

    procesadas = 0
    errores = 0

    for img_path in tqdm(contenidos, desc="Style transfer"):
        try:
            salida_path = DIR_SALIDA_IMG / img_path.name
            mask_src = DIR_MASCARAS / (img_path.stem + ".png")
            mask_dst = DIR_SALIDA_MASK / (img_path.stem + ".png")

            if salida_path.exists():
                if mask_src.exists() and not mask_dst.exists():
                    shutil.copy2(mask_src, mask_dst)
                continue

            if not mask_src.exists():
                continue

            img = np.array(Image.open(img_path).convert("RGB"))
            estilo = random.choice(estilos)
            resultado = aplicar_estilo_marmol(img, estilo, ALPHA)

            Image.fromarray(resultado).save(salida_path, quality=95)
            shutil.copy2(mask_src, mask_dst)

            procesadas += 1
            if procesadas % 200 == 0:
                log.info(f"processed: {procesadas}/{len(contenidos)}")

        except Exception as e:
            log.warning(f"Error in {img_path.name}: {e}")
            errores += 1


    log.info(f"COMPLETED - Processed: {procesadas}, ERRORS: {errores}")
    log.info(f"Output: {DIR_SALIDA_IMG}")



if __name__ == "__main__":
    main()
