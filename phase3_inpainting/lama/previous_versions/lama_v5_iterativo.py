#Variante v5 de LaMa sobre broken_body: inpainting ITERATIVO en anillos crecientes desde elborde del cuerpo
#HIPOTESIS:
#LaMa funciona bien cuando la mascara esta rodeada de contenido similar a lo que tiene que generar. Si la mascara entera esta lejos del cuerpo y
#rodeada de fondo, propaga fondo. Pero si rompemos la mascara grande en una secuencia de
#anillos delgados que se van pegando al cuerpo, en cada paso la zona a rellenar esta junto a
#marmol y LaMa extiende marmol. El anillo siguiente ya tiene como vecino el anillo anterior
#(ahora marmol) y propaga marmol tambien. Y asi sucesivamente hasta cubrir la mascara entera.
#
#Pipeline para cada imagen:
#  1) Calcular distancia (euclidiana) de cada pixel de la mascara al cuerpo mas cercano.
#  2) Dividir la mascara en N anillos por cuantiles de distancia (el 1/N mas cercano = anillo 1,
#     siguiente 1/N = anillo 2, etc).
#  3) Para anillo en 1..N: mascara_paso = solo anillo. Run LaMa sobre la imagen actual.
#     Actualizar la imagen sustituyendo el anillo por la salida de LaMa.
#  4) Resultado final: imagen tras N pasadas.
#
#Coste: N veces mas lento que v1 (N inferencias de LaMa por imagen). Compensa si funciona.
#
#INPUT:
#  - imagenes: ~/tfg/dataset_classificado/broken_body/
#  - mascaras: ~/tfg/masks/broken_body/
#OUTPUT:
#  - imagenes reconstruidas: ~/tfg/inpainting_results/lama_v5_iterativo/

import logging
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

from simple_lama_inpainting import SimpleLama


DIR_IMAGENES = Path("/home/pfc/cescuder/tfg/dataset_classificado/broken_body")
DIR_MASCARAS = Path("/home/pfc/cescuder/tfg/masks/broken_body")
DIR_SALIDA = Path("/home/pfc/cescuder/tfg/inpainting_results/lama_v5_iterativo")

#numero de anillos en los que partimos la mascara
#mas anillos = mas suave la propagacion pero mas tiempo de calculo
N_ANILLOS = 4

#umbral para detectar pixeles no-cuerpo (fondo blanco)
UMBRAL_BLANCO = 245

#tamaño minimo de pixeles en un anillo para procesarlo (anillos demasiado pequeños se saltan)
TAM_MIN_ANILLO = 50


#LOGGING:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("/home/pfc/cescuder/tfg/logs/lama_v5_iterativo.log", encoding="utf-8"),])
log = logging.getLogger(__name__)


#PARTIR LA MASCARA EN ANILLOS POR DISTANCIA AL CUERPO:
def anillos_por_distancia(img_np: np.ndarray, mask_np: np.ndarray, n_anillos: int) -> list:
    """
    Devuelve una lista de mascaras booleanas (una por anillo), ordenadas de mas cercano a mas
    lejano del cuerpo. Cada mascara cubre 1/n_anillos de los pixeles de la mascara original
    (aproximadamente, segun cuantiles de distancia).
    """
    no_blanco = (img_np.mean(axis=2) < UMBRAL_BLANCO)
    cuerpo = no_blanco & (~mask_np)

    if cuerpo.sum() == 0:
        #sin cuerpo visible no podemos calcular distancias, devolvemos la mascara entera como
        #un solo "anillo" (degenerado pero seguro)
        return [mask_np.copy()]

    #distancia euclidiana de cada pixel al pixel de cuerpo mas cercano
    dist = distance_transform_edt(~cuerpo)

    #distancias unicamente dentro de la mascara original
    dist_in_mask = dist[mask_np]
    if dist_in_mask.size == 0:
        return []

    #cuantiles para partir en n_anillos
    cuantiles = np.linspace(0, 1, n_anillos + 1)[1:]  #por ejemplo n=4 -> [0.25, 0.5, 0.75, 1.0]
    umbrales = np.quantile(dist_in_mask, cuantiles)

    anillos = []
    umbral_prev = -np.inf
    for u in umbrales:
        anillo = mask_np & (dist > umbral_prev) & (dist <= u)
        if anillo.sum() >= TAM_MIN_ANILLO:
            anillos.append(anillo)
        umbral_prev = u

    return anillos


#MAIN:
def main():
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    log.info(f"Iterative params: N_ANILLOS={N_ANILLOS}, TAM_MIN_ANILLO={TAM_MIN_ANILLO}")

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
    log.info("LaMa loaded. Starting v5 (iterative rings) inpainting...")

    procesadas = 0
    sin_mascara = 0
    sin_anillos = 0
    errores = 0

    for img_path in tqdm(imagenes, desc="LaMa v5 iterativo"):
        try:
            salida_path = DIR_SALIDA / (img_path.stem + "_lamav5.png")
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

            #partir mascara en anillos
            anillos = anillos_por_distancia(img_np, mask_np, N_ANILLOS)
            if len(anillos) == 0:
                log.warning(f"No usable rings for: {img_path.name} - skipping")
                sin_anillos += 1
                continue

            #iterar: en cada paso solo enmascaramos el anillo en curso, LaMa lo regenera,
            #integramos el resultado y pasamos al siguiente anillo
            img_actual_np = img_np.copy()
            for anillo in anillos:
                mask_paso = (anillo.astype(np.uint8) * 255)
                img_actual_pil = Image.fromarray(img_actual_np)
                mask_paso_pil  = Image.fromarray(mask_paso, mode="L")

                resultado_paso = lama(img_actual_pil, mask_paso_pil)
                #SimpleLama puede redondear H/W a multiplos de 8 internamente; forzamos
                #el shape original para que la indexacion booleana de mas abajo no peta
                if resultado_paso.size != img_actual_pil.size:
                    resultado_paso = resultado_paso.resize(img_actual_pil.size, Image.BILINEAR)
                resultado_paso_np = np.array(resultado_paso)

                img_actual_np[anillo] = resultado_paso_np[anillo]

            Image.fromarray(img_actual_np).save(salida_path)
            procesadas += 1

        except Exception as e:
            log.warning(f"Error in {img_path.name}: {e}")
            errores += 1

    log.info("LAMA v5 (ITERATIVE RINGS) COMPLETED")
    log.info(f"Processed: {procesadas}")
    log.info(f"Skipped (no mask): {sin_mascara}")
    log.info(f"Skipped (no usable rings): {sin_anillos}")
    log.info(f"Errors: {errores}")
    log.info(f"Output at: {DIR_SALIDA}")


if __name__ == "__main__":
    main()
