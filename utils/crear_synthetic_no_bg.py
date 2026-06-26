"""
Genera la version sin fondo (fondo blanco puro) del synthetic_dataset_bw_first.

Para cada imagen, lee la mask DensePose de 15 clases (0=fondo, 1-14=partes del
cuerpo) y reemplaza los pixeles donde mask==0 con (255, 255, 255).

No reentrena nada ni recomputa DensePose: las masks y el densepose_cache que ya
existen siguen siendo validos para las imagenes resultantes porque las
coordenadas espaciales no cambian.

INPUT:
    - imagenes: ~/tfg/synthetic_dataset_bw_first/images/
    - masks:    ~/tfg/synthetic_dataset_bw_first/masks/

OUTPUT:
    - imagenes sin fondo: ~/tfg/synthetic_dataset_bw_first/images_no_bg/
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


BASE = Path("/home/pfc/cescuder/tfg/synthetic_dataset_bw_first")
DIR_IMG_SRC = BASE / "images"
DIR_MSK = BASE / "masks"
DIR_IMG_DST = BASE / "images_no_bg"

DIR_IMG_DST.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("/home/pfc/cescuder/tfg/logs/crear_synthetic_no_bg.log", encoding="utf-8")])
log = logging.getLogger(__name__)


def main():
    archivos = sorted(DIR_IMG_SRC.glob("*.jpg"))
    log.info(f"Total imagenes a procesar: {len(archivos)}")
    log.info(f"Destino: {DIR_IMG_DST}")

    procesadas, ya_existian, sin_mask, errores = 0, 0, 0, 0

    for img_path in tqdm(archivos, desc="creando NOBG"):
        out_path = DIR_IMG_DST / img_path.name

        if out_path.exists():
            ya_existian += 1
            continue

        mask_path = DIR_MSK / (img_path.stem + ".png")
        if not mask_path.exists():
            sin_mask += 1
            continue

        try:
            img = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path))
            if mask.ndim == 3:
                mask = mask[..., 0]

            if img.shape[:2] != mask.shape:
                log.warning(f"shapes no coinciden en {img_path.name}: img={img.shape[:2]} vs mask={mask.shape}")
                errores += 1
                continue

            img[mask == 0] = 255

            Image.fromarray(img).save(out_path, quality=95)
            procesadas += 1
        except Exception as e:
            errores += 1
            log.error(f"error en {img_path.name}: {e}")

    log.info(f"DONE")
    log.info(f"  procesadas:    {procesadas}")
    log.info(f"  ya existian:   {ya_existian}")
    log.info(f"  sin mask:      {sin_mask}")
    log.info(f"  errores:       {errores}")
    log.info(f"  output:        {DIR_IMG_DST}")


if __name__ == "__main__":
    main()
