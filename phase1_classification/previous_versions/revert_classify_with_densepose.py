#revert de los movimientos hechos por classify_with_densepose.py V1.
#lee el log JSON con todas las decisiones y devuelve cada archivo a su carpeta de origen.
#esto deja el dataset_classificado/ y el densepose_cache/ tal y como estaban justo
#despues de extract_densepose.py, listos para que se ejecute la version V2 corregida.
#
#funcionamiento:
#  - lee /home/pfc/cescuder/tfg/logs/classify_with_densepose.json
#  - para cada decision con "moved": true, deshace el movimiento (carpeta destino -> carpeta origen)
#  - mueve los tres archivos asociados (original, sin fondo y .npz)
#  - salta las decisiones con "moved": false (eran confirmaciones, no movimientos)
#
#es idempotente: si una imagen ya esta en su carpeta de origen no pasa nada (silencioso),
#asi que se puede relanzar sin problema.

import json
import logging
import shutil
from pathlib import Path

DIR_DATASET = Path("/home/pfc/cescuder/tfg/dataset_classificado")
DIR_BG_REMOVED = Path("/home/pfc/cescuder/tfg/background_removed")
DIR_DENSEPOSE_CACHE = Path("/home/pfc/cescuder/tfg/densepose_cache")
JSON_DECISIONES = Path("/home/pfc/cescuder/tfg/logs/classify_with_densepose.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/home/pfc/cescuder/tfg/logs/revert_classify_densepose.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def encontrar_imagen(stem, carpeta):
    #busca el archivo de imagen probando varias extensiones
    for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".webp"]:
        candidato = carpeta / f"{stem}{ext}"
        if candidato.exists():
            return candidato
    return None


def revertir_movimiento(stem, carpeta_origen_original, carpeta_destino_movido):
    #revierte el movimiento: lo que esta en carpeta_destino_movido vuelve a carpeta_origen_original
    #devuelve cuantos archivos se movieron
    movidos = 0

    #imagen original con fondo
    img_orig = encontrar_imagen(stem, DIR_DATASET / carpeta_destino_movido)
    if img_orig is not None:
        destino = DIR_DATASET / carpeta_origen_original / img_orig.name
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(img_orig), str(destino))
        movidos += 1

    #imagen sin fondo
    img_bg = encontrar_imagen(stem, DIR_BG_REMOVED / carpeta_destino_movido)
    if img_bg is not None:
        destino = DIR_BG_REMOVED / carpeta_origen_original / img_bg.name
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(img_bg), str(destino))
        movidos += 1

    #npz de densepose
    npz_origen = DIR_DENSEPOSE_CACHE / carpeta_destino_movido / f"{stem}.npz"
    if npz_origen.exists():
        destino = DIR_DENSEPOSE_CACHE / carpeta_origen_original / npz_origen.name
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(npz_origen), str(destino))
        movidos += 1

    return movidos


def main():
    log.info("revert classify with densepose started")

    if not JSON_DECISIONES.exists():
        log.error(f"json de decisiones no encontrado: {JSON_DECISIONES}")
        return

    with open(JSON_DECISIONES, "r", encoding="utf-8") as f:
        decisiones = json.load(f)

    log.info(f"total decisiones en el json: {len(decisiones)}")

    revertidos = 0
    sin_movimiento = 0
    no_encontrados = 0
    errores = 0

    for d in decisiones:
        try:
            #saltamos las confirmaciones (no se movieron)
            if not d.get("moved", False):
                sin_movimiento += 1
                continue

            stem = d["stem"]
            carpeta_origen = d["from"]
            carpeta_destino = d["to"]

            n = revertir_movimiento(stem, carpeta_origen, carpeta_destino)
            if n == 0:
                no_encontrados += 1
                log.warning(f"no se encontraron archivos para {stem} en {carpeta_destino}")
            else:
                revertidos += 1
                log.info(f"reverted: {stem}  {carpeta_destino} -> {carpeta_origen}  ({n} files)")
        except Exception as e:
            log.warning(f"error en decision {d.get('stem', '?')}: {e}")
            errores += 1

    log.info("revert completed")
    log.info(f"  revertidos:           {revertidos}")
    log.info(f"  sin movimiento (skip): {sin_movimiento}")
    log.info(f"  no encontrados:       {no_encontrados}")
    log.info(f"  errores:              {errores}")

    #conteos finales del dataset_classificado
    log.info("conteos finales del dataset_classificado tras revert:")
    for carpeta in ["whole_body", "broken_body", "head_only", "no_human"]:
        d = DIR_DATASET / carpeta
        n = sum(1 for _ in d.iterdir() if _.is_file()) if d.exists() else 0
        log.info(f"  {carpeta}: {n}")

    #conteos finales del densepose_cache
    log.info("conteos finales del densepose_cache tras revert:")
    for carpeta in ["whole_body", "broken_body", "head_only"]:
        d = DIR_DENSEPOSE_CACHE / carpeta
        n = sum(1 for _ in d.iterdir() if _.is_file()) if d.exists() else 0
        log.info(f"  {carpeta}: {n}")


if __name__ == "__main__":
    main()
