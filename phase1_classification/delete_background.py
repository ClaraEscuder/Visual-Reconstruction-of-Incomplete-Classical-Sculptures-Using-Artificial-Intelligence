#elimina el fondo de las esculturas combinando dos modelos:
#1. rembg (U2-Net u2net_human_seg) con cascada de contraste --> mascara A
#2. SAM (Segment Anything Model con SamPredictor y center-point) --> mascara B
#3. union de A y B (OR logico) --> mascara final mas completa, donde rembg recorta mal un brazo fino SAM lo recupera y viceversa
#se aplica sobre las carpetas whole_body, broken_body y head_only de dataset_classificado. head_only entra al pipeline porque DeepLabv3+ confunde frecuentemente bustos con torsos rotos y viceversa, y necesitamos que DensePose lo redistribuya en la fase de clasificacion posterior. no_human queda fuera porque ya esta confirmado que no contiene figuras humanas.
#salida: imagenes con fondo eliminado y reemplazado por blanco puro (255,255,255), guardadas como PNG manteniendo la estructura whole_body / broken_body / head_only bajo background_removed.

import logging
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageEnhance
from tqdm import tqdm

from rembg import remove, new_session
from segment_anything import sam_model_registry, SamPredictor


DIR_BASE_ENTRADA = Path("/home/pfc/cescuder/tfg/dataset_classificado")
DIR_BASE_SALIDA = Path("/home/pfc/cescuder/tfg/background_removed")
CARPETAS_PROCESAR = ["whole_body", "broken_body", "head_only"]

#color de fondo tras la eliminacion (blanco neutro para no interferir con el tono del marmol durante el inpainting)
FONDO_RGB = (255, 255, 255)

MODELO_REMBG = "u2net_human_seg"

#umbral para detectar si rembg fallo: si mas del 95% del resultado es blanco U2-Net no distinguio la figura del fondo
UMBRAL_FALLO_BLANCO = 0.95

#factores de contraste en cascada para rembg, se aplican secuencialmente si los anteriores fallan
FACTORES_CONTRASTE = [2.0, 3.0, 4.0]

SAM_PESOS = Path("/home/pfc/cescuder/tfg/scripts/sam_vit_h.pth")
SAM_TIPO = "vit_h"

#si SAM selecciona mas del 80% de la imagen probablemente cogio el fondo, se descarta su mascara
SAM_AREA_MAXIMA = 0.80


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("/home/pfc/cescuder/tfg/logs/delete_background.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)


def cargar_sam(device):
    log.info(f"loading SAM ({SAM_TIPO}) from {SAM_PESOS}")
    sam = sam_model_registry[SAM_TIPO](checkpoint=str(SAM_PESOS))
    sam = sam.to(device)
    predictor = SamPredictor(sam)
    log.info("SAM loaded")
    return predictor


def obtener_mascara_rembg(img: Image.Image, session) -> np.ndarray:
    #intenta primero la imagen original y reintenta con factores de contraste crecientes si rembg devuelve mascara casi vacia
    img_sin_fondo = remove(img, session=session)
    alpha = np.array(img_sin_fondo.convert("RGBA"))[:, :, 3]
    mascara = (alpha > 127).astype(np.uint8)

    if mascara.sum() / mascara.size >= (1 - UMBRAL_FALLO_BLANCO):
        return mascara

    for factor in FACTORES_CONTRASTE:
        enhancer = ImageEnhance.Contrast(img)
        img_contraste = enhancer.enhance(factor)
        img_sin_fondo_c = remove(img_contraste, session=session)
        alpha_c = np.array(img_sin_fondo_c.convert("RGBA"))[:, :, 3]
        mascara_c = (alpha_c > 127).astype(np.uint8)
        if mascara_c.sum() / mascara_c.size >= (1 - UMBRAL_FALLO_BLANCO):
            return mascara_c

    return mascara


def obtener_mascara_sam(img: Image.Image, predictor) -> np.ndarray:
    #usa una cuadricula de 9 puntos en la zona central de la imagen como prompt positivo. devuelve None si SAM seleccionara mas del 80% (probablemente cogio el fondo)
    img_np = np.array(img.convert("RGB"))
    img_h, img_w = img_np.shape[:2]

    puntos = []
    for fx in [0.35, 0.50, 0.65]:
        for fy in [0.35, 0.50, 0.65]:
            puntos.append([int(img_w * fx), int(img_h * fy)])

    puntos_np = np.array(puntos)
    labels_np = np.ones(len(puntos), dtype=int)

    predictor.set_image(img_np)
    mascaras, scores, _ = predictor.predict(
        point_coords=puntos_np,
        point_labels=labels_np,
        multimask_output=True,
    )

    if mascaras is None or len(mascaras) == 0:
        return None

    mejor_idx = np.argmax(scores)
    mascara = mascaras[mejor_idx].astype(np.uint8)

    if mascara.sum() / mascara.size > SAM_AREA_MAXIMA:
        return None

    return mascara


def componer_con_mascara(img: Image.Image, mascara: np.ndarray, fondo: tuple) -> Image.Image:
    fondo_img = Image.new("RGB", img.size, fondo)
    mascara_pil = Image.fromarray((mascara * 255).astype(np.uint8))
    fondo_img.paste(img.convert("RGB"), mask=mascara_pil)
    return fondo_img


def procesar_carpeta(nombre_carpeta, session, predictor):
    dir_entrada = DIR_BASE_ENTRADA / nombre_carpeta
    dir_salida = DIR_BASE_SALIDA / nombre_carpeta
    dir_salida.mkdir(parents=True, exist_ok=True)

    if not dir_entrada.exists():
        log.warning(f"carpeta no encontrada, se omite: {dir_entrada}")
        return 0, 0

    extensiones = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".webp"}
    imagenes = [f for f in dir_entrada.iterdir() if f.suffix in extensiones]
    log.info(f"{nombre_carpeta}: {len(imagenes)} imagenes")

    procesadas = 0
    errores = 0

    for img_path in tqdm(imagenes, desc=f"removing bg {nombre_carpeta}"):
        try:
            ruta_salida = dir_salida / (img_path.stem + ".png")
            if ruta_salida.exists():
                procesadas += 1
                continue

            img = Image.open(img_path).convert("RGB")
            mascara_rembg = obtener_mascara_rembg(img, session)
            mascara_sam = obtener_mascara_sam(img, predictor)

            if mascara_sam is not None:
                mascara_final = np.where((mascara_rembg == 1) | (mascara_sam == 1), 1, 0).astype(np.uint8)
            else:
                mascara_final = mascara_rembg

            img_limpia = componer_con_mascara(img, mascara_final, FONDO_RGB)
            img_limpia.save(ruta_salida)
            procesadas += 1
        except Exception as e:
            log.warning(f"error en {img_path.name}: {e}")
            errores += 1

    return procesadas, errores


def main():
    DIR_BASE_SALIDA.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device: {device}")
    log.info(f"models: rembg ({MODELO_REMBG}) + SAM ({SAM_TIPO}) -> union of masks")
    log.info(f"output background: RGB{FONDO_RGB}")
    log.info(f"contrast cascade factors: {FACTORES_CONTRASTE}")
    log.info(f"folders to process: {CARPETAS_PROCESAR}")

    if not SAM_PESOS.exists():
        log.error(f"SAM weights not found: {SAM_PESOS}")
        return

    session = new_session(MODELO_REMBG)
    predictor = cargar_sam(device)
    log.info("models loaded, processing")

    total_procesadas = 0
    total_errores = 0

    for carpeta in CARPETAS_PROCESAR:
        proc, err = procesar_carpeta(carpeta, session, predictor)
        total_procesadas += proc
        total_errores += err

    log.info("background removal completed")
    log.info(f"total processed: {total_procesadas}")
    log.info(f"total errors: {total_errores}")
    log.info(f"output at: {DIR_BASE_SALIDA}")


if __name__ == "__main__":
    main()