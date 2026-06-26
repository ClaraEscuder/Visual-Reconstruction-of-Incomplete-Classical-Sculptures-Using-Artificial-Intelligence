#wrapper que reusa la logica de extract_densepose.py pero apuntando al dataset
#sintetico bw-first (las imagenes que van a usarse para entrenar LaMa-v7).
#
#unico cambio: reescribimos las 3 constantes de paths antes de llamar al main()
#de extract_densepose. la implementacion del modelo, el predictor, el threshold,
#todo lo demas se hereda intacto del script original.

import sys
from pathlib import Path

sys.path.insert(0, "/home/pfc/cescuder/tfg/scripts")

import extract_densepose as ed

ed.DIR_BASE_ENTRADA = Path("/home/pfc/cescuder/tfg/synthetic_dataset_bw_first")
ed.DIR_BASE_CACHE = Path("/home/pfc/cescuder/tfg/synthetic_dataset_bw_first/densepose_cache")
ed.CARPETAS_PROCESAR = ["images"]


if __name__ == "__main__":
    ed.main()
