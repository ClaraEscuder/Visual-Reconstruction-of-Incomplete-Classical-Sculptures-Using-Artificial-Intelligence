#Aplica LaMa (Large Mask Inpainting with Fourier Convolutions) sobre las esculturas broken_body
#con sus mascaras de inpainting generadas por create_patches.py --> reconstruye las partes faltantes
#LaMa se usa con pesos CONGELADOS (frozen weights), sin reentrenamiento
#INPUT:
#  - imagenes: ~/tfg/dataset_classificado/broken_body/ (PNG con fondo blanco, sin fondo original)
#  - mascaras: ~/tfg/masks/broken_body/  (PNG binario: blanco=reconstruir, negro=conservar)
#OUTPUT:
#  - imagenes reconstruidas: ~/tfg/inpainting_results/lama/

import logging
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

#simple-lama-inpainting es un wrapper de LaMa que permite usarlo con pip sin clonar el repo completo
#pip install simple-lama-inpainting
from simple_lama_inpainting import SimpleLama

DIR_IMAGENES = Path("/home/pfc/cescuder/tfg/dataset_classificado/broken_body")
DIR_MASCARAS = Path("/home/pfc/cescuder/tfg/masks/broken_body")
DIR_SALIDA = Path("/home/pfc/cescuder/tfg/inpainting_results/lama")


#LOGGING:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("/home/pfc/cescuder/tfg/logs/lama_inpainting.log", encoding="utf-8"),])
log = logging.getLogger(__name__)


#MAIN:
def main():
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    if not DIR_IMAGENES.exists():
        log.error(f"Input images directory not found: {DIR_IMAGENES}")
        return
    if not DIR_MASCARAS.exists():
        log.error(f"Input masks directory not found: {DIR_MASCARAS}")
        return

    extensiones = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    imagenes = [f for f in DIR_IMAGENES.iterdir() if f.suffix in extensiones]

    if not imagenes:
        log.error("No images found in input directory.")
        return

    log.info(f"Images to process: {len(imagenes)}")
    log.info("Loading LaMa model (frozen weights)...")

    #cargar LaMa con pesos preentrenados congelados --> no se reentrena
    lama = SimpleLama()
    log.info("LaMa loaded. Starting inpainting...")

    procesadas = 0
    sin_mascara = 0
    errores = 0

    for img_path in tqdm(imagenes, desc="LaMa inpainting"):
        try:
            #checkpoint implicito: saltar si el resultado ya existe
            salida_path = DIR_SALIDA / (img_path.stem + "_lama.png")
            if salida_path.exists():
                procesadas += 1
                continue

            #buscar la mascara correspondiente (misma raiz, sufijo _mask.png)
            mask_path = DIR_MASCARAS / (img_path.stem + "_mask.png")
            if not mask_path.exists():
                log.warning(f"Mask not found for: {img_path.name} — skipping")
                sin_mascara += 1
                continue

            img  = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")  #escala de grises: blanco=reconstruir

            #LaMa espera imagen RGB y mascara L (0=conservar, 255=reconstruir)
            resultado = lama(img, mask)

            resultado.save(salida_path)
            procesadas += 1

        except Exception as e:
            log.warning(f"Error in {img_path.name}: {e}")
            errores += 1

    log.info("LAMA INPAINTING COMPLETED")
    log.info(f"Processed: {procesadas}")
    log.info(f"Skipped (no mask): {sin_mascara}")
    log.info(f"Errors: {errores}")
    log.info(f"Output at: {DIR_SALIDA}")


if __name__ == "__main__":
    main()
