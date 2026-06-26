#Variante v2 de LaMa sobre broken_body: pre-rellena la mascara con color de marmol muestreado
#del propio cuerpo ANTES de pasar la imagen a LaMa. La hipotesis es que LaMa, al tener pixeles
#bajo la mascara que ya se parecen al marmol del cuerpo, propaga marmol y no fondo.
#
#Diferencia con v1 (baseline): en v1 los pixeles bajo la mascara son los originales (fondo museo,
#blanco, etc). Aqui los sobreescribimos con muestras del cuerpo + un poco de ruido gaussiano para
#dar textura. LaMa sigue recibiendo la misma mascara binaria (region a regenerar = la misma).
#
#INPUT:
#  - imagenes: ~/tfg/dataset_classificado/broken_body/
#  - mascaras: ~/tfg/masks/broken_body/  (PNG binario: blanco=reconstruir, negro=conservar)
#OUTPUT:
#  - imagenes reconstruidas: ~/tfg/inpainting_results/lama_v2_mask_prefill/

import logging
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from simple_lama_inpainting import SimpleLama


DIR_IMAGENES = Path("/home/pfc/cescuder/tfg/dataset_classificado/broken_body")
DIR_MASCARAS = Path("/home/pfc/cescuder/tfg/masks/broken_body")
DIR_SALIDA = Path("/home/pfc/cescuder/tfg/inpainting_results/lama_v2_mask_prefill")

#parametros del pre-rellenado
#cuantos pixeles del cuerpo muestreamos para estimar el color medio
N_MUESTRAS_CUERPO = 5000
#desviacion del ruido gaussiano que aplicamos al pre-rellenado (en escala 0-255)
#valores muy bajos = parche plano que LaMa ignora; muy altos = textura ruidosa rara
SIGMA_RUIDO = 12.0
#umbral para considerar un pixel como "fondo blanco" (no-cuerpo) al muestrear
#sobre las imagenes broken_body el fondo ya esta sustituido por blanco puro
UMBRAL_BLANCO = 245


#LOGGING:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("/home/pfc/cescuder/tfg/logs/lama_v2_mask_prefill.log", encoding="utf-8"),])
log = logging.getLogger(__name__)


#PRE-RELLENADO DE LA MASCARA:
def prefill_mascara_con_marmol(img_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
    """
    Sobreescribe los pixeles de img_np que estan dentro de la mascara con muestras
    del color medio del cuerpo visible, mas un poco de ruido gaussiano para que
    LaMa no vea un parche plano que pueda tratar como "objeto extra".

    img_np: (H, W, 3) uint8
    mask_np: (H, W) bool   True = region a reconstruir
    Devuelve: copia de img_np con la region enmascarada sustituida por marmol estimado.
    """
    #identificar pixeles de cuerpo: no estan dentro de la mascara y no son fondo blanco
    no_blanco = (img_np.mean(axis=2) < UMBRAL_BLANCO)
    cuerpo = no_blanco & (~mask_np)

    if cuerpo.sum() == 0:
        #caso degenerado: no hay cuerpo visible (escultura entera o casi entera enmascarada)
        #fallback a gris medio neutro que al menos no es ningun color de fondo
        color_medio = np.array([180, 175, 170], dtype=np.float32)
    else:
        pixeles_cuerpo = img_np[cuerpo]
        if len(pixeles_cuerpo) > N_MUESTRAS_CUERPO:
            idx = np.random.choice(len(pixeles_cuerpo), N_MUESTRAS_CUERPO, replace=False)
            pixeles_cuerpo = pixeles_cuerpo[idx]
        color_medio = pixeles_cuerpo.mean(axis=0)

    #construir el parche: color medio + ruido gaussiano por canal
    h, w = mask_np.shape
    parche = np.tile(color_medio, (h, w, 1)).astype(np.float32)
    ruido = np.random.normal(0, SIGMA_RUIDO, (h, w, 3))
    parche = np.clip(parche + ruido, 0, 255).astype(np.uint8)

    img_out = img_np.copy()
    img_out[mask_np] = parche[mask_np]
    return img_out


#MAIN:
def main():
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    log.info(f"Pre-fill params: N_MUESTRAS_CUERPO={N_MUESTRAS_CUERPO}, SIGMA_RUIDO={SIGMA_RUIDO}")

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
    log.info("LaMa loaded. Starting v2 (mask prefill) inpainting...")

    procesadas = 0
    sin_mascara = 0
    errores = 0

    for img_path in tqdm(imagenes, desc="LaMa v2 prefill"):
        try:
            #checkpoint implicito: saltar si ya esta hecho
            salida_path = DIR_SALIDA / (img_path.stem + "_lamav2.png")
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

            #alinear tamaños por si la mascara fue generada a otra resolucion
            if mask.size != img.size:
                mask = mask.resize(img.size, Image.NEAREST)

            img_np  = np.array(img)
            mask_np = (np.array(mask) > 127)   #bool

            #aplicar el pre-rellenado de marmol y volver a PIL para entregarsela a LaMa
            img_prefilled = prefill_mascara_con_marmol(img_np, mask_np)
            img_pil = Image.fromarray(img_prefilled)

            resultado = lama(img_pil, mask)
            resultado.save(salida_path)
            procesadas += 1

        except Exception as e:
            log.warning(f"Error in {img_path.name}: {e}")
            errores += 1

    log.info("LAMA v2 (MASK PREFILL) COMPLETED")
    log.info(f"Processed: {procesadas}")
    log.info(f"Skipped (no mask): {sin_mascara}")
    log.info(f"Errors: {errores}")
    log.info(f"Output at: {DIR_SALIDA}")


if __name__ == "__main__":
    main()
