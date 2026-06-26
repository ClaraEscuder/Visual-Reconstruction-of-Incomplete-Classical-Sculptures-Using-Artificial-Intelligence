"""
Descarga las imagenes de COCO con anotaciones DensePose y genera mascaras de segmentación por 14 partes del cuerpo

Las anotaciones DensePose mapean cada píxel de persona a una de las 14 parts del cuerpo humano:
    1 Torso
    2 Right Hand
    3 Left Hand
    4 Left Foot
    5 Right Foot
    6 Upper Leg Right
    7 Upper Leg Left
    8 Lower Leg Right
    9 Lower Leg Left
    10 Upper Arm Left
    11 Upper Arm Right
    12 Lower Arm Left
    13 Lower Arm Right
    14 Head
"""

import os
import json
import zipfile
import shutil
import logging
import numpy as np
import requests
from pathlib import Path
from tqdm import tqdm
from PIL import Image


OUTPUT_DIR = Path("/home/pfc/cescuder/tfg/densepose_dataset")

#max de imagenes a procesar (None = sería coger todas ASÍ QUE LO CAMBIARE SI ESTO ES INSUFICIENTE PARA EL FINETUNING!!!!!)
MAX_IMAGENES = None

#URLs DEL DATASET:
COCO_IMAGES_URL= "http://images.cocodataset.org/zips/train2017.zip"
DENSEPOSE_ANN_URL = (
    "https://dl.fbaipublicfiles.com/densepose/"
    "densepose_coco_2014_train.json")

# Clases DensePose (indice 1-14)
DENSEPOSE_CLASES = {
    1: "Torso",
    2: "Right_Hand",
    3: "Left_Hand",
    4: "Left_Foot",
    5: "Right_Foot",
    6: "Upper_Leg_Right",
    7: "Upper_Leg_Left",
    8: "Lower_Leg_Right",
    9: "Lower_Leg_Left",
    10: "Upper_Arm_Left",
    11: "Upper_Arm_Right",
    12: "Lower_Arm_Left",
    13: "Lower_Arm_Right",
    14: "Head",
}


#LOGGING:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(OUTPUT_DIR.parent / "descargar_densepose.log", encoding="utf-8"),])
log = logging.getLogger(__name__)


#DESCARGA CON PROGRESO:
def descargar_archivo(url: str, destino: Path, descripcion: str) -> Path:
    #descarga un archivo mostrando barra de progreso (omite si ya existe)
    if destino.exists():
        log.info(f"It already exists so omiting: {destino.name}")
        return destino

    log.info(f"downloading {descripcion}!!!!")
    destino.parent.mkdir(parents=True, exist_ok=True)

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(destino, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=descripcion
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    log.info(f"downloaded: {destino}")
    return destino


def extraer_zip(zip_path: Path, destino: Path):
    #extrae un zip si el directorio destino no existe ya
    if destino.exists() and any(destino.iterdir()):
        log.info(f"Ya extraído: {destino}")
        return

    log.info(f"Extrayendo {zip_path.name}...")
    destino.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(destino)
    log.info("Extracción completada.")


#GENERACIÓN DE MÁSCARAS DENSEPOSE:
def generar_mascara_densepose(ann: dict, alto: int, ancho: int) -> np.ndarray:
    """

    convierte una anotación DensePose a una mask 2D de partes corporales.

    DensePose almacena las partes en formato UV comprimido dentro del bounding
    box de la persona. Cada píxel dentro del bbox tiene un valor de parte (1-14)
    o 0 si no pertenece a ninguna parte.

    RETURNS: array con valores 0-14.
    """
    mascara = np.zeros((alto, ancho), dtype=np.uint8)

    dp = ann.get("dp_masks")
    if dp is None:
        return mascara

    # Bounding box de la persona [x, y, w, h]
    x, y, w, h = [int(v) for v in ann["bbox"]]
    x = max(0, x)
    y = max(0, y)
    w = min(w, ancho - x)
    h = min(h, alto - y)

    if w <= 0 or h <= 0:
        return mascara

    #dp_masks es una lista de 14 elementos (uno por parte corporal)
    #cada elemento es una mask RLE comprimida de 256x256 que cubre el bbox
    for parte_idx, rle in enumerate(dp, start=1):
        if rle is None:
            continue

        #decodificar RLE: dp_masks usa RLE binario de 256x256
        m = _decodificar_rle_densepose(rle)  # (256, 256) binaria

        #redimensionar al tamaño real del bbox
        m_resized = np.array(
            Image.fromarray(m).resize((w, h), Image.NEAREST)
        )

        #escribir en la mask global solo donde hay píxeles de esta parte
        region = mascara[y:y+h, x:x+w]
        region[m_resized > 0] = parte_idx
        mascara[y:y+h, x:x+w] = region

    return mascara

def _decodificar_rle_densepose(rle_comprimido) -> np.ndarray:
    
    #decodifica el formato RLE de dp_masks de DensePose
    #RLE = lista de enteros que alternan entre runs de 0s y 1s, aplanado en orden column-major para una máscara de 256x256
    
    total = 256*256
    mascara_flat = np.zeros(total, dtype=np.uint8)

    if isinstance(rle_comprimido, list):
        pos = 0
        valor = 0
        for count in rle_comprimido:
            if valor == 1:
                mascara_flat[pos:pos+count] = 1
            pos += count
            valor = 1 - valor
    elif isinstance(rle_comprimido, dict) and "counts" in rle_comprimido:
        #formato COCO RLE estándar
        try:
            from pycocotools import mask as coco_mask
            m = coco_mask.decode(rle_comprimido)
            return m.astype(np.uint8)
        except ImportError:
            pass

    return mascara_flat.reshape(256, 256, order="F")


#MAIN:
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dir_imagenes = OUTPUT_DIR / "images"
    dir_mascaras = OUTPUT_DIR / "masks"
    dir_imagenes.mkdir(exist_ok=True)
    dir_mascaras.mkdir(exist_ok=True)

    tmp_dir = OUTPUT_DIR / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

   
    # 1. Descargar anotaciones DensePose--------------------------------------------------------
    ann_path = tmp_dir / "densepose_coco_2014_train.json"
    descargar_archivo(DENSEPOSE_ANN_URL, ann_path, "anotaciones DensePose")

    log.info("Cargando anotaciones DensePose...")
    with open(ann_path, encoding="utf-8") as f:
        datos = json.load(f)

    #construir indice imagen_id → info imagen
    id_a_imagen = {img["id"]: img for img in datos["images"]}

    #filtrar anotaciones que tengan dp_masks
    anotaciones_dp = [
        ann for ann in datos["annotations"]
        if ann.get("dp_masks") is not None]
    log.info(f"Anotaciones con DensePose: {len(anotaciones_dp)}")

    #agrupar por imagen
    imagen_a_anns: dict = {}
    for ann in anotaciones_dp:
        iid = ann["image_id"]
        imagen_a_anns.setdefault(iid, []).append(ann)

    ids_con_dp = list(imagen_a_anns.keys())
    if MAX_IMAGENES:
        ids_con_dp = ids_con_dp[:MAX_IMAGENES]

    log.info(f"Imágenes a procesar: {len(ids_con_dp)}")

    # 2. Descargar imágenes COCO (solo las necesarias)----------------------------------------------------------
    log.info("Descargando imágenes COCO necesarias...")
    session = requests.Session()
    #DensePose usa anotaciones de COCO 2014
    #los archivos tienen formato COCO_train2014_XXXXXXXX.jpg
    #URL correcta es train2014, no train2017
    base_url_2014 = "http://images.cocodataset.org/train2014/"
    base_url_2017 = "http://images.cocodataset.org/train2017/"

    descargadas = 0
    errores = 0

    for img_id in tqdm(ids_con_dp, desc="Descargando imágenes"):
        info = id_a_imagen.get(img_id)
        if info is None:
            continue

        nombre = info["file_name"]
        destino_img = dir_imagenes / nombre

        if not destino_img.exists():
            try:
                # Determinar URL correcta según el nombre del archivo
                if "train2014" in nombre:
                    url = base_url_2014 + nombre
                else:
                    url = base_url_2017 + nombre
                r = session.get(url, timeout=30)
                r.raise_for_status()
                with open(destino_img, "wb") as f:
                    f.write(r.content)
                descargadas += 1
            except Exception as e:
                log.warning(f"Error descargando {nombre}: {e}")
                errores += 1
                continue

    log.info(f"Imágenes descargadas: {descargadas}  |  Errores: {errores}")


    # 3. Generar máscaras de partes corporales-----------------------------------------------------------------------
    log.info("Generando máscaras de partes corporales...")
    generadas = 0
    omitidas = 0

    for img_id in tqdm(ids_con_dp, desc="Generando máscaras"):
        info = id_a_imagen.get(img_id)
        if info is None:
            continue

        nombre = info["file_name"]
        img_path = dir_imagenes / nombre
        if not img_path.exists():
            omitidas += 1
            continue

        nombre_mask = Path(nombre).stem + ".png"
        mask_path = dir_mascaras / nombre_mask
        if mask_path.exists():
            omitidas += 1
            continue

        try:
            alto = info["height"]
            ancho = info["width"]

            # Combinar todas las personas de la imagen en una sola máscara
            mascara_final = np.zeros((alto, ancho), dtype=np.uint8)
            for ann in imagen_a_anns[img_id]:
                m = generar_mascara_densepose(ann, alto, ancho)
                # Si hay solapamiento entre personas, la parte más reciente gana
                mascara_final[m > 0] = m[m > 0]

            Image.fromarray(mascara_final).save(mask_path)
            generadas += 1

        except Exception as e:
            log.warning(f"Error procesando {nombre}: {e}")

    log.info(f"Máscaras generadas: {generadas}  |  Omitidas: {omitidas}")


    # 4. Guardar metadatos del dataset------------------------------------------------------------
    meta = {
        "descripcion": "Dataset DensePose para fine-tuning DeepLabv3+ en esculturas",
        "num_clases": 15,
        "clases": {
            "0": "Background",
            **{str(k): v for k, v in DENSEPOSE_CLASES.items()}},
        "num_imagenes": generadas,
        "directorio_imagenes": str(dir_imagenes),
        "directorio_mascaras": str(dir_mascaras),
    }
    meta_path = OUTPUT_DIR / "dataset_info.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info(f"Metadatos guardados: {meta_path}")

    # Limpiar temporales
    shutil.rmtree(tmp_dir, ignore_errors=True)


    log.info("COMPLETED")
    log.info(f"Images:{dir_imagenes}")
    log.info(f"Masks:{dir_mascaras}")


if __name__ == "__main__":
    main()

