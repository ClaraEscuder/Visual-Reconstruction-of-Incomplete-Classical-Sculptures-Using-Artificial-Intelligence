#Variante v3 de LaMa sobre broken_body: recorta apretado alrededor del cuerpo+mascara antes
#de pasar la imagen a LaMa. La hipotesis es que LaMa propaga fondo porque al ver toda la imagen
#"entiende" que hay un fondo (pared museo, blanco, pedestal) y que la mascara forma parte de el.
#Si le damos un recorte donde la mascara esta rodeada mayoritariamente por el cuerpo, la decision
#mas coherente con su entrenamiento (rellenar con el entorno) sera marmol y no escena.
#
#Pipeline:
#  1) bbox del cuerpo (pixeles no-blancos) U bbox de la mascara
#  2) anyadir padding PORCENTUAL alrededor (no fijo en px porque las imagenes tienen tamaños muy distintos)
#  3) recortar, correr LaMa sobre el recorte
#  4) pegar el recorte regenerado de vuelta en la imagen original
#
#INPUT:
#  - imagenes: ~/tfg/dataset_classificado/broken_body/
#  - mascaras: ~/tfg/masks/broken_body/
#OUTPUT:
#  - imagenes reconstruidas: ~/tfg/inpainting_results/lama_v3_tight_crop/

import logging
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from simple_lama_inpainting import SimpleLama


DIR_IMAGENES = Path("/home/pfc/cescuder/tfg/dataset_classificado/broken_body")
DIR_MASCARAS = Path("/home/pfc/cescuder/tfg/masks/broken_body")
DIR_SALIDA = Path("/home/pfc/cescuder/tfg/inpainting_results/lama_v3_tight_crop")

#padding alrededor del bbox combinado cuerpo+mascara, como fraccion del lado mayor del bbox
#valores tipicos 0.05-0.15. demasiado bajo = LaMa no tiene contexto suficiente; demasiado alto = volvemos a meter mucho fondo
PAD_FRAC = 0.10

#tamaño minimo del crop (para que LaMa no reciba imagenes ridiculas)
TAM_MINIMO_CROP = 256

#umbral para detectar pixeles no-cuerpo (fondo blanco de las imagenes background_removed)
UMBRAL_BLANCO = 245


#LOGGING:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("/home/pfc/cescuder/tfg/logs/lama_v3_tight_crop.log", encoding="utf-8"),])
log = logging.getLogger(__name__)


#CALCULO DEL BBOX DE CROP:
def bbox_combinado(img_np: np.ndarray, mask_np: np.ndarray):
    """
    Devuelve (x0, y0, x1, y1) del bbox que contiene cuerpo (no-blanco) U mascara,
    con padding PAD_FRAC del lado mayor del bbox y asegurando un tamaño minimo.
    """
    h, w = mask_np.shape

    no_blanco = (img_np.mean(axis=2) < UMBRAL_BLANCO)
    region = no_blanco | mask_np

    if region.sum() == 0:
        #imagen completamente blanca y sin mascara: devolvemos toda la imagen como fallback
        return 0, 0, w, h

    ys, xs = np.where(region)
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1

    #padding proporcional al lado mas grande del bbox
    lado = max(x1 - x0, y1 - y0)
    pad = int(round(lado * PAD_FRAC))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)

    #garantizar tamaño minimo expandiendo simetricamente si hace falta
    if (x1 - x0) < TAM_MINIMO_CROP:
        extra = TAM_MINIMO_CROP - (x1 - x0)
        x0 = max(0, x0 - extra // 2)
        x1 = min(w, x0 + TAM_MINIMO_CROP)
        x0 = max(0, x1 - TAM_MINIMO_CROP)
    if (y1 - y0) < TAM_MINIMO_CROP:
        extra = TAM_MINIMO_CROP - (y1 - y0)
        y0 = max(0, y0 - extra // 2)
        y1 = min(h, y0 + TAM_MINIMO_CROP)
        y0 = max(0, y1 - TAM_MINIMO_CROP)

    return x0, y0, x1, y1


#MAIN:
def main():
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    log.info(f"Crop params: PAD_FRAC={PAD_FRAC}, TAM_MINIMO_CROP={TAM_MINIMO_CROP}")

    if not DIR_IMAGENES.exists():
        log.error(f"Input images directory not found: {DIR_IMAGENES}")
        return
    if not DIR_MASCARAS.exists():
        log.error(f"Input masks directory not found: {DIR_MASCARAS}")
        return

    extensiones = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    imagenes = [f for f in DIR_IMAGENES.iterdir() if f.suffix in extensiones]
    log.info(f"Images to process: {len(imagenes)}")

    log.info("Loading LaMa model (frozen weights)...")
    lama = SimpleLama()
    log.info("LaMa loaded. Starting v3 (tight crop) inpainting...")

    procesadas = 0
    sin_mascara = 0
    errores = 0

    for img_path in tqdm(imagenes, desc="LaMa v3 crop"):
        try:
            salida_path = DIR_SALIDA / (img_path.stem + "_lamav3.png")
            if salida_path.exists():
                procesadas += 1
                continue

            mask_path = DIR_MASCARAS / (img_path.stem + "_mask.png")
            if not mask_path.exists():
                log.warning(f"Mask not found for: {img_path.name} - skipping")
                sin_mascara += 1
                continue

            img  = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            if mask.size != img.size:
                mask = mask.resize(img.size, Image.NEAREST)

            img_np  = np.array(img)
            mask_np = (np.array(mask) > 127)

            #calcular el crop y aplicarlo a imagen y mascara
            x0, y0, x1, y1 = bbox_combinado(img_np, mask_np)
            img_crop  = Image.fromarray(img_np[y0:y1, x0:x1])
            mask_crop = Image.fromarray((mask_np[y0:y1, x0:x1].astype(np.uint8)) * 255, mode="L")

            #LaMa sobre el recorte; el wrapper interno puede redondear H/W a multiplos
            #de 8 asi que el shape del resultado puede no coincidir exactamente con el
            #del crop. forzamos el resize para que encaje siempre en el slot del bbox
            resultado_crop = lama(img_crop, mask_crop)
            altura_crop = y1 - y0
            ancho_crop  = x1 - x0
            if resultado_crop.size != (ancho_crop, altura_crop):
                resultado_crop = resultado_crop.resize((ancho_crop, altura_crop), Image.BILINEAR)
            resultado_crop_np = np.array(resultado_crop)

            #pegar el crop regenerado en su sitio dentro de la imagen original
            img_final = img_np.copy()
            img_final[y0:y1, x0:x1] = resultado_crop_np
            Image.fromarray(img_final).save(salida_path)

            procesadas += 1

        except Exception as e:
            log.warning(f"Error in {img_path.name}: {e}")
            errores += 1

    log.info("LAMA v3 (TIGHT CROP) COMPLETED")
    log.info(f"Processed: {procesadas}")
    log.info(f"Skipped (no mask): {sin_mascara}")
    log.info(f"Errors: {errores}")
    log.info(f"Output at: {DIR_SALIDA}")


if __name__ == "__main__":
    main()
