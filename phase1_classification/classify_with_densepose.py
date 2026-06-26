#segunda pasada de clasificacion sobre las esculturas usando los .npz de DensePose ya extraidos.
#proposito: refinar la clasificacion inicial de DeepLabv3+ entre las tres categorias (whole_body, broken_body, head_only) usando la firma anatomica que devuelve DensePose. DensePose tiene mas vocabulario (24 caras lateralizadas frente a 14 clases sin lateralizar), trabaja sobre la cobertura observada por region (% del cuerpo detectado) y captura mejor las poses dinamicas y los casos donde una extremidad sale parcialmente segmentada.
#por que entran las tres carpetas: en pruebas con DeepLabv3+ aproximadamente un 40% de head_only contenia falsos positivos (torsos rotos confundidos con bustos, no-humanos confundidos con cabezas) y de manera analoga whole_body y broken_body tambien contienen errores. DensePose redistribuye todo basandose en lo que realmente ve.
#logica de reclasificacion (V2 con regla anatomica de continuidad distal):
#  1. si DensePose no genero npz para una imagen --> es no_human encubierto, se mueve a no_human
#  2. si genero npz, cuento cobertura por region y aplico la regla de continuidad distal:
#     una extremidad solo cuenta como REALMENTE PRESENTE si hay continuidad anatomica:
#       - brazo presente = upper_arm presente Y (lower_arm presente O hand presente)
#       - pierna presente = upper_leg presente Y (lower_leg presente O foot presente)
#     esto descarta los hombros y muslos sueltos que en un busto SIEMPRE aparecen pero no son extremidades de verdad
#  3. con esa definicion, la decision es:
#     a. todas las regiones obligatorias presentes (con continuidad distal) --> whole_body
#     b. ninguna extremidad realmente presente --> head_only (busto: hombros y/o trozos de torso pero sin brazos ni piernas reales)
#     c. caso intermedio --> broken_body
#cuando una imagen se reclasifica se mueven los tres archivos asociados (original con fondo, sin fondo y npz) a la carpeta destino para mantener la coherencia entre directorios a lo largo del pipeline.

import json
import logging
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm


DIR_DATASET = Path("/home/pfc/cescuder/tfg/dataset_classificado")
DIR_BG_REMOVED = Path("/home/pfc/cescuder/tfg/background_removed")
DIR_DENSEPOSE_CACHE = Path("/home/pfc/cescuder/tfg/densepose_cache")

CARPETA_WHOLE = "whole_body"
CARPETA_BROKEN = "broken_body"
CARPETA_HEAD = "head_only"
CARPETA_NO_HUMAN = "no_human"
CARPETAS_FUENTE = [CARPETA_WHOLE, CARPETA_BROKEN, CARPETA_HEAD]


#mapeo de las 24 caras de la malla SMPL a 14 regiones anatomicas (ver paper de DensePose, Guler et al. 2018, figura 2)
REGIONES = {
    "torso": [1, 2],
    "right_hand": [3],
    "left_hand": [4],
    "left_foot": [5],
    "right_foot": [6],
    "upper_leg_R": [7, 9],
    "upper_leg_L": [8, 10],
    "lower_leg_R": [11, 13],
    "lower_leg_L": [12, 14],
    "upper_arm_L": [15, 17],
    "upper_arm_R": [16, 18],
    "lower_arm_L": [19, 21],
    "lower_arm_R": [20, 22],
    "head": [23, 24],
}


COBERTURA_ESPERADA = {
    "torso": 0.180,
    "right_hand": 0.018,
    "left_hand": 0.018,
    "left_foot": 0.020,
    "right_foot": 0.020,
    "upper_leg_R": 0.060,
    "upper_leg_L": 0.060,
    "lower_leg_R": 0.050,
    "lower_leg_L": 0.050,
    "upper_arm_L": 0.040,
    "upper_arm_R": 0.040,
    "lower_arm_L": 0.040,
    "lower_arm_R": 0.040,
    "head": 0.060,
}


PADRE = {
    "left_foot": "lower_leg_L",
    "right_foot": "lower_leg_R",
    "lower_leg_L": "upper_leg_L",
    "lower_leg_R": "upper_leg_R",
    "upper_leg_L": "torso",
    "upper_leg_R": "torso",
    "left_hand": "lower_arm_L",
    "right_hand": "lower_arm_R",
    "lower_arm_L": "upper_arm_L",
    "lower_arm_R": "upper_arm_R",
    "upper_arm_L": "torso",
    "upper_arm_R": "torso",
    "head": "torso",
}


HIJOS = {}
for _h, _p in PADRE.items():
    HIJOS.setdefault(_p, []).append(_h)


#regiones que tienen que estar TODAS presentes para considerar el cuerpo entero. la cabeza queda fuera porque la definicion del TFG acepta whole_body sin cabeza
REGIONES_OBLIGATORIAS_WHOLE = [
    "torso",
    "right_hand", "left_hand",
    "left_foot", "right_foot",
    "upper_leg_R", "upper_leg_L",
    "lower_leg_R", "lower_leg_L",
    "upper_arm_L", "upper_arm_R",
    "lower_arm_L", "lower_arm_R",]


UMBRAL_PRESENCIA = 0.30


#configuracion de la regla de continuidad distal: cada brazo o pierna esta organizada en una cadena anatomica desde el torso hacia el extremo distal. para que una extremidad cuente como "realmente presente" tiene que haber al menos un eslabon mas alla del proximal. los hombros sueltos (upper_arm sin antebrazo ni mano) que aparecen en un busto NO cuentan como brazo presente. analogamente para piernas
CADENAS_EXTREMIDADES = {
    "brazo_L": {
        "proximal": "upper_arm_L",
        "distales": ["lower_arm_L", "left_hand"],
    },
    "brazo_R": {
        "proximal": "upper_arm_R",
        "distales": ["lower_arm_R", "right_hand"],
    },
    "pierna_L": {
        "proximal": "upper_leg_L",
        "distales": ["lower_leg_L", "left_foot"],
    },
    "pierna_R": {
        "proximal": "upper_leg_R",
        "distales": ["lower_leg_R", "right_foot"],
    },
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/home/pfc/cescuder/tfg/logs/classify_with_densepose.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def cargar_body_parts(npz_path):
    data = np.load(str(npz_path), allow_pickle=True)
    keys = list(data.keys())
    for k in ["I", "body_parts", "parts", "labels"]:
        if k in keys:
            return data[k].astype(np.uint8)
    raise ValueError(f"no se encontro array de partes en {npz_path}, claves: {keys}")


def calcular_coberturas(body_parts):
    pix_total = 0
    counts = {}
    for region, caras in REGIONES.items():
        m = np.zeros(body_parts.shape, dtype=bool)
        for c in caras:
            m |= (body_parts == c)
        n = int(m.sum())
        counts[region] = n
        pix_total += n
    if pix_total == 0:
        return {r: 0.0 for r in REGIONES}, 0
    coberturas = {r: counts[r] / pix_total for r in REGIONES}
    return coberturas, pix_total


def region_presente_directo(region, coberturas):
    #version simple sin recursion: una region cuenta como presente solo si su cobertura observada >= 30% de la esperada
    #se usa para evaluar la presencia anatomica real, sin propagar a traves de descendientes
    esp = COBERTURA_ESPERADA.get(region, 0.0)
    if esp == 0.0:
        return True
    return coberturas.get(region, 0.0) >= UMBRAL_PRESENCIA * esp


def region_presente(region, coberturas):
    #version con recursion por descendientes: util para decidir si una region esta presente DE ALGUNA FORMA en la cadena anatomica (cuenta tanto si la propia region tiene cobertura como si algun descendiente esta detectado)
    #se usa para el chequeo whole_body donde queremos ser permisivos: si DensePose detecta la mano pero pierde el antebrazo, anatomicamente no se puede tener mano sin antebrazo, asi que el antebrazo cuenta como presente
    if region_presente_directo(region, coberturas):
        return True
    for hijo in HIJOS.get(region, []):
        if region_presente(hijo, coberturas):
            return True
    return False


def extremidad_realmente_presente(cadena, coberturas):
    #regla anatomica de continuidad distal: una extremidad (brazo o pierna) cuenta como REALMENTE PRESENTE solo si:
    #  - el segmento proximal (hombro o muslo) esta presente Y hay al menos un segmento distal presente (antebrazo/mano para brazos, pantorrilla/pie para piernas), O
    #  - hay un segmento distal presente sin proximal (caso de manos o pies sueltos detectados sin que se vea el resto, raro pero posible en esculturas reconstruidas parcialmente)
    #lo que NO cuenta como extremidad: solo proximal (hombro o muslo) sin nada distal. esto es lo que descarta los hombros de busto que aparecian incorrectamente clasificados como broken_body en V1
    proximal = cadena["proximal"]
    distales = cadena["distales"]

    proximal_presente = region_presente_directo(proximal, coberturas)
    algun_distal_presente = any(region_presente_directo(d, coberturas) for d in distales)

    if proximal_presente and algun_distal_presente:
        return True
    if algun_distal_presente:
        return True
    return False


def clasificar(coberturas, pix_total):
    if pix_total == 0:
        return "broken_body", "sin_cuerpo_detectado"

    #verificamos si todas las regiones obligatorias estan presentes (criterio permisivo con descendientes)
    faltantes = [r for r in REGIONES_OBLIGATORIAS_WHOLE if not region_presente(r, coberturas)]

    if not faltantes:
        return "whole_body", "ok"

    #no es whole_body. ahora aplicamos la regla de continuidad distal para decidir entre head_only y broken_body
    #una extremidad solo cuenta si tiene al menos proximal+distal o solo distal. los hombros sueltos no cuentan
    extremidades_reales = [
        nombre for nombre, cadena in CADENAS_EXTREMIDADES.items()
        if extremidad_realmente_presente(cadena, coberturas)
    ]

    if len(extremidades_reales) == 0:
        #ninguna extremidad real detectada, es un busto independientemente de los muñones de hombro
        return "head_only", "sin_extremidades_reales"

    return "broken_body", f"faltan: {','.join(faltantes)}"


def encontrar_imagen(stem, carpeta):
    #busca el archivo de imagen probando varias extensiones (puede estar como .jpg o .png segun la fuente original)
    for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".webp"]:
        candidato = carpeta / f"{stem}{ext}"
        if candidato.exists():
            return candidato
    return None


def mover_archivos(stem, carpeta_origen, carpeta_destino):
    #mueve los tres archivos asociados a una imagen (original con fondo, sin fondo y .npz) si existen. devuelve cuantos archivos se movieron
    movidos = 0

    img_orig = encontrar_imagen(stem, DIR_DATASET / carpeta_origen)
    if img_orig is not None:
        destino = DIR_DATASET / carpeta_destino / img_orig.name
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(img_orig), str(destino))
        movidos += 1

    img_bg = encontrar_imagen(stem, DIR_BG_REMOVED / carpeta_origen)
    if img_bg is not None:
        destino = DIR_BG_REMOVED / carpeta_destino / img_bg.name
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(img_bg), str(destino))
        movidos += 1

    npz_origen = DIR_DENSEPOSE_CACHE / carpeta_origen / f"{stem}.npz"
    if npz_origen.exists():
        destino = DIR_DENSEPOSE_CACHE / carpeta_destino / npz_origen.name
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(npz_origen), str(destino))
        movidos += 1

    return movidos


def detectar_no_human_en_carpeta(carpeta_actual):
    #recorre las imagenes de la carpeta que NO tienen npz (DensePose no detecto nada). estas se mueven a no_human porque no son figuras humanas reconocibles. devuelve la lista de stems movidos
    dir_imgs = DIR_DATASET / carpeta_actual
    dir_npz = DIR_DENSEPOSE_CACHE / carpeta_actual

    if not dir_imgs.exists():
        return []

    extensiones = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".webp"}
    imgs = [f for f in dir_imgs.iterdir() if f.suffix in extensiones]

    movidos = []
    for img in imgs:
        npz = dir_npz / f"{img.stem}.npz"
        if not npz.exists():
            n = mover_archivos(img.stem, carpeta_actual, CARPETA_NO_HUMAN)
            movidos.append({
                "stem": img.stem,
                "from": carpeta_actual,
                "to": CARPETA_NO_HUMAN,
                "motivo": "densepose_sin_deteccion",
                "moved": True,
                "files_moved": n,
            })
            log.info(f"moved (no detection): {img.stem}  {carpeta_actual} -> {CARPETA_NO_HUMAN}")
    return movidos


def procesar_carpeta_con_npz(carpeta_actual):
    #procesa los .npz de DensePose disponibles para esta carpeta y los redistribuye segun la firma anatomica
    dir_npz = DIR_DENSEPOSE_CACHE / carpeta_actual
    if not dir_npz.exists():
        log.warning(f"directorio npz no encontrado, se omite: {dir_npz}")
        return [], 0, 0, 0

    npz_files = sorted(dir_npz.glob("*.npz"))
    log.info(f"{carpeta_actual}: {len(npz_files)} npz a evaluar")

    decisiones = []
    confirmadas = 0
    movidas = 0
    errores = 0

    for npz_path in tqdm(npz_files, desc=f"classifying {carpeta_actual}"):
        stem = npz_path.stem
        try:
            body_parts = cargar_body_parts(npz_path)
            coberturas, pix_total = calcular_coberturas(body_parts)
            categoria_nueva, motivo = clasificar(coberturas, pix_total)

            if categoria_nueva == carpeta_actual:
                confirmadas += 1
                decisiones.append({
                    "stem": stem,
                    "from": carpeta_actual,
                    "to": categoria_nueva,
                    "motivo": motivo,
                    "moved": False,
                })
                continue

            n = mover_archivos(stem, carpeta_actual, categoria_nueva)
            movidas += 1
            decisiones.append({
                "stem": stem,
                "from": carpeta_actual,
                "to": categoria_nueva,
                "motivo": motivo,
                "moved": True,
                "files_moved": n,
            })
            log.info(f"moved: {stem}  {carpeta_actual} -> {categoria_nueva}  ({motivo})")
        except Exception as e:
            log.warning(f"error en {stem}: {e}")
            errores += 1

    return decisiones, confirmadas, movidas, errores


def main():
    log.info("classify with densepose started (V2 con regla de continuidad distal)")
    log.info(f"folders to process: {CARPETAS_FUENTE}")

    todas_decisiones = []
    total_conf = 0
    total_mov = 0
    total_err = 0
    total_no_human = 0

    #fase 1: las imagenes sin npz son no_human encubiertos --> se mueven antes de procesar los npz para no afectar al recorrido
    log.info("phase 1: detecting hidden no_human (images without densepose detection)")
    for carpeta in CARPETAS_FUENTE:
        movidos_no_human = detectar_no_human_en_carpeta(carpeta)
        todas_decisiones.extend(movidos_no_human)
        total_no_human += len(movidos_no_human)
    log.info(f"  moved to no_human: {total_no_human}")

    #fase 2: clasificacion segun firma anatomica de DensePose
    log.info("phase 2: anatomical signature classification")
    for carpeta in CARPETAS_FUENTE:
        decisiones, conf, mov, err = procesar_carpeta_con_npz(carpeta)
        todas_decisiones.extend(decisiones)
        total_conf += conf
        total_mov += mov
        total_err += err

    json_path = Path("/home/pfc/cescuder/tfg/logs/classify_with_densepose.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(todas_decisiones, f, indent=2, ensure_ascii=False)

    log.info("classify with densepose completed")
    log.info(f"  moved to no_human:    {total_no_human}")
    log.info(f"  confirmed in place:   {total_conf}")
    log.info(f"  moved between cats:   {total_mov}")
    log.info(f"  errors:               {total_err}")
    log.info(f"  decisions log:        {json_path}")

    #conteos finales
    log.info("final counts:")
    for carpeta in [CARPETA_WHOLE, CARPETA_BROKEN, CARPETA_HEAD, CARPETA_NO_HUMAN]:
        d = DIR_DATASET / carpeta
        n = sum(1 for _ in d.iterdir() if _.is_file()) if d.exists() else 0
        log.info(f"  {carpeta}: {n}")


if __name__ == "__main__":
    main()
