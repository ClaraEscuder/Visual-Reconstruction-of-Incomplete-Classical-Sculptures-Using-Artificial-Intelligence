#Variante v4 de LaMa sobre broken_body: pintar el fondo de marmol antes de pasar la imagen a
#LaMa y luego recomponer el resultado sobre el fondo BLANCO original. La idea es la opuesta a
#v3: en vez de "esconder" el fondo recortandolo, se lo cambiamos por marmol. LaMa, que tiende a
#propagar el entorno hacia la mascara, ahora deberia propagar marmol en lugar de blanco.
#
#Pipeline:
#  1) Detectar fondo: pixeles casi-blancos (las imagenes broken_body ya tienen el fondo
#     sustituido por blanco puro por delete_background.py).
#  2) Sustituir los pixeles de fondo (fuera de la mascara) por un color/textura de marmol
#     muestreada del cuerpo visible.
#  3) Correr LaMa sobre esta imagen "marmolizada".
#  4) Componer el resultado final: dentro de la mascara → pixeles de LaMa; fuera → imagen
#     original (con su fondo blanco), porque solo nos interesa regenerar la zona enmascarada.
#
#INPUT:
#  - imagenes: ~/tfg/dataset_classificado/broken_body/   (fondo ya blanco)
#  - mascaras: ~/tfg/masks/broken_body/
#OUTPUT:
#  - imagenes reconstruidas: ~/tfg/inpainting_results/lama_v4_marble_bg/

import logging
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from simple_lama_inpainting import SimpleLama


DIR_IMAGENES = Path("/home/pfc/cescuder/tfg/dataset_classificado/broken_body")
DIR_MASCARAS = Path("/home/pfc/cescuder/tfg/masks/broken_body")
DIR_SALIDA = Path("/home/pfc/cescuder/tfg/inpainting_results/lama_v4_marble_bg")

#cuantos pixeles del cuerpo muestreamos para estimar color medio + std
N_MUESTRAS_CUERPO = 5000
#sigma del ruido gaussiano aplicado al fondo marmolizado (en escala 0-255)
SIGMA_RUIDO_BG = 8.0
#umbral para detectar pixeles de fondo blanco
UMBRAL_BLANCO = 245


#LOGGING:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("/home/pfc/cescuder/tfg/logs/lama_v4_marble_bg.log", encoding="utf-8"),])
log = logging.getLogger(__name__)


#MARMOLIZAR EL FONDO:
def marmolizar_fondo(img_np: np.ndarray, mask_np: np.ndarray):
    """
    Sustituye los pixeles de fondo blanco por marmol estimado del cuerpo.
    Devuelve (img_marmolizada, fondo_bool):
        - img_marmolizada: copia de img_np con el fondo reemplazado
        - fondo_bool: mascara (H,W) bool de donde se considero fondo (para recompostar al final)
    """
    no_blanco = (img_np.mean(axis=2) < UMBRAL_BLANCO)
    cuerpo = no_blanco & (~mask_np)
    fondo  = ~no_blanco & (~mask_np)  #fondo blanco que esta FUERA de la mascara

    if cuerpo.sum() == 0:
        color_medio = np.array([180, 175, 170], dtype=np.float32)
    else:
        pixeles_cuerpo = img_np[cuerpo]
        if len(pixeles_cuerpo) > N_MUESTRAS_CUERPO:
            idx = np.random.choice(len(pixeles_cuerpo), N_MUESTRAS_CUERPO, replace=False)
            pixeles_cuerpo = pixeles_cuerpo[idx]
        color_medio = pixeles_cuerpo.mean(axis=0)

    #construir un fondo marmol = color medio + ruido gaussiano (textura suave)
    h, w = mask_np.shape
    parche = np.tile(color_medio, (h, w, 1)).astype(np.float32)
    ruido = np.random.normal(0, SIGMA_RUIDO_BG, (h, w, 3))
    parche = np.clip(parche + ruido, 0, 255).astype(np.uint8)

    img_marmolizada = img_np.copy()
    img_marmolizada[fondo] = parche[fondo]
    return img_marmolizada, fondo


#MAIN:
def main():
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    log.info(f"Marble-bg params: N_MUESTRAS_CUERPO={N_MUESTRAS_CUERPO}, SIGMA_RUIDO_BG={SIGMA_RUIDO_BG}")

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
    log.info("LaMa loaded. Starting v4 (marble background) inpainting...")

    procesadas = 0
    sin_mascara = 0
    errores = 0

    for img_path in tqdm(imagenes, desc="LaMa v4 marble-bg"):
        try:
            salida_path = DIR_SALIDA / (img_path.stem + "_lamav4.png")
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

            #marmolizar el fondo y correr LaMa
            img_marmolizada, _ = marmolizar_fondo(img_np, mask_np)
            img_pil = Image.fromarray(img_marmolizada)
            salida_lama = lama(img_pil, mask)
            #SimpleLama puede redondear H/W a multiplos de 8 internamente; forzamos el
            #shape original para que el pegado de mas abajo no peta por broadcast mismatch
            if salida_lama.size != img.size:
                salida_lama = salida_lama.resize(img.size, Image.BILINEAR)
            salida_lama_np = np.array(salida_lama)

            #recomponer: dentro de la mascara nos quedamos con lo que ha generado LaMa;
            #fuera, restauramos la imagen ORIGINAL (con su fondo blanco) para no contaminar el
            #resto de la escultura ni el fondo con el marmol sintetico que metimos como condicion
            img_final = img_np.copy()
            img_final[mask_np] = salida_lama_np[mask_np]
            Image.fromarray(img_final).save(salida_path)

            procesadas += 1

        except Exception as e:
            log.warning(f"Error in {img_path.name}: {e}")
            errores += 1

    log.info("LAMA v4 (MARBLE BACKGROUND) COMPLETED")
    log.info(f"Processed: {procesadas}")
    log.info(f"Skipped (no mask): {sin_mascara}")
    log.info(f"Errors: {errores}")
    log.info(f"Output at: {DIR_SALIDA}")


if __name__ == "__main__":
    main()
