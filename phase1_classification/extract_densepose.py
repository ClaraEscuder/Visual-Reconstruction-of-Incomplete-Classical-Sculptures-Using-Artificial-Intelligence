#extrae los campos densos de DensePose para las esculturas y guarda los resultados en .npz como cache para fases posteriores (clasificacion con DensePose y generacion de mascaras de inpainting).
#se procesan las carpetas whole_body, broken_body y head_only de background_removed. head_only entra al pipeline de DensePose porque DeepLabv3+ confunde con frecuencia bustos con torsos rotos. la fase posterior (classify_with_densepose) leera las predicciones de DensePose y redistribuira las imagenes entre las tres categorias segun lo que DensePose detecte realmente. no_human queda fuera porque ya esta confirmado que no contiene figuras humanas.
#por que esto es un script aparte: DensePose es caro (~1-3 segundos por imagen en GPU) y los .npz se reutilizan en multiples pasos posteriores. iterar algoritmos sobre arrays leidos de disco es mucho mas rapido que recomputar el modelo entero cada vez.
#salida por imagen: ~/tfg/densepose_cache/<carpeta>/<id>.npz con:
#  I: array (H, W) con el indice de parte corporal (0=fondo, 1..24=partes)
#  U, V: arrays (H, W) con las coordenadas UV sobre la superficie canonica del cuerpo
#  bbox: array (4,) con [x, y, w, h] de la caja de la persona detectada
#  score: float con la confianza de la deteccion (~0.95 en buenas detecciones)
#  img_shape: array (2,) con [H, W] de la imagen completa

import sys
import logging
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, "/home/pfc/cescuder/tfg/detectron2_repo/projects/DensePose")

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from densepose import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor


DIR_BASE_ENTRADA = Path("/home/pfc/cescuder/tfg/background_removed")
DIR_BASE_CACHE = Path("/home/pfc/cescuder/tfg/densepose_cache")
CARPETAS_PROCESAR = ["whole_body", "broken_body", "head_only"]

CONFIG_FILE = "/home/pfc/cescuder/tfg/detectron2_repo/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"
WEIGHTS_FILE = "/home/pfc/cescuder/tfg/densepose_weights/model_final_162be9.pkl"

#umbral de confianza para aceptar una deteccion. en escultura clasica los valores caen tipicamente en 0.85-0.99 cuando el modelo esta seguro
SCORE_THRESHOLD = 0.5


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/home/pfc/cescuder/tfg/logs/extract_densepose.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def cargar_predictor():
    log.info("loading densepose")
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(CONFIG_FILE)
    cfg.MODEL.WEIGHTS = WEIGHTS_FILE
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"  device: {cfg.MODEL.DEVICE}")
    predictor = DefaultPredictor(cfg)
    log.info("  densepose loaded")
    return predictor


def procesar_imagen(predictor, img_path):
    #procesa una imagen y devuelve los campos densos I, U, V, el bbox en formato (x, y, w, h) y el score. si DensePose no detecta a nadie devuelve None. si detecta varias personas se queda con la de mayor score (suele ser la figura principal en una escultura)
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    H, W = img.shape[:2]

    with torch.no_grad():
        outputs = predictor(img)["instances"]

    if len(outputs) == 0:
        return None

    scores = outputs.scores.cpu().numpy()
    idx_mejor = int(np.argmax(scores))
    score = float(scores[idx_mejor])

    #convertimos el bbox que devuelve detectron2 (x1, y1, x2, y2) a formato (x, y, w, h) para tener un solo formato consistente en todo el pipeline
    x1, y1, x2, y2 = outputs.pred_boxes.tensor.cpu().numpy()[idx_mejor].astype(np.int32)
    bbox = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.int32)

    extractor = DensePoseResultExtractor()
    densepose_results, _ = extractor(outputs)

    if densepose_results is None or len(densepose_results) <= idx_mejor:
        return None

    dp_result = densepose_results[idx_mejor]

    labels_box = dp_result.labels.cpu().numpy().astype(np.uint8)
    uv_box = dp_result.uv.cpu().numpy()
    H_box, W_box = labels_box.shape
    U_box = uv_box[0]
    V_box = uv_box[1]

    I = np.zeros((H, W), dtype=np.uint8)
    U = np.zeros((H, W), dtype=np.float32)
    V = np.zeros((H, W), dtype=np.float32)

    x, y, w, h = bbox
    h_target = min(H_box, h, H - y)
    w_target = min(W_box, w, W - x)
    if h_target <= 0 or w_target <= 0:
        return None

    I[y:y + h_target, x:x + w_target] = labels_box[:h_target, :w_target]
    U[y:y + h_target, x:x + w_target] = U_box[:h_target, :w_target]
    V[y:y + h_target, x:x + w_target] = V_box[:h_target, :w_target]

    return {
        "I": I,
        "U": U,
        "V": V,
        "bbox": bbox,
        "score": score,
        "img_shape": np.array([H, W], dtype=np.int32),
    }


def procesar_carpeta(predictor, nombre_carpeta):
    dir_entrada = DIR_BASE_ENTRADA / nombre_carpeta
    dir_cache = DIR_BASE_CACHE / nombre_carpeta
    dir_cache.mkdir(parents=True, exist_ok=True)

    if not dir_entrada.exists():
        log.warning(f"carpeta no encontrada, se omite: {dir_entrada}")
        return 0, 0, 0, 0

    extensiones = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    imagenes = sorted(f for f in dir_entrada.iterdir() if f.suffix in extensiones)
    log.info(f"{nombre_carpeta}: {len(imagenes)} imagenes")

    procesadas = 0
    saltadas = 0
    sin_deteccion = 0
    errores = 0

    for img_path in tqdm(imagenes, desc=f"densepose {nombre_carpeta}"):
        try:
            cache_path = dir_cache / f"{img_path.stem}.npz"
            if cache_path.exists():
                saltadas += 1
                continue

            resultado = procesar_imagen(predictor, img_path)
            if resultado is None:
                sin_deteccion += 1
                continue

            np.savez_compressed(
                cache_path,
                I=resultado["I"],
                U=resultado["U"],
                V=resultado["V"],
                bbox=resultado["bbox"],
                score=np.array([resultado["score"]], dtype=np.float32),
                img_shape=resultado["img_shape"],
            )
            procesadas += 1
        except Exception as e:
            log.warning(f"error en {img_path.name}: {e}")
            errores += 1

    return procesadas, saltadas, sin_deteccion, errores


def main():
    DIR_BASE_CACHE.mkdir(parents=True, exist_ok=True)

    if not Path(WEIGHTS_FILE).exists():
        log.error(f"weights not found: {WEIGHTS_FILE}")
        return

    predictor = cargar_predictor()
    log.info(f"folders to process: {CARPETAS_PROCESAR}")

    total_proc = 0
    total_skip = 0
    total_no_det = 0
    total_err = 0
    for carpeta in CARPETAS_PROCESAR:
        proc, skip, no_det, err = procesar_carpeta(predictor, carpeta)
        total_proc += proc
        total_skip += skip
        total_no_det += no_det
        total_err += err

    log.info("densepose extraction completed")
    log.info(f"  processed:        {total_proc}")
    log.info(f"  skipped (cached): {total_skip}")
    log.info(f"  no detection:     {total_no_det}")
    log.info(f"  errors:           {total_err}")
    log.info(f"  cache:            {DIR_BASE_CACHE}")


if __name__ == "__main__":
    main()
