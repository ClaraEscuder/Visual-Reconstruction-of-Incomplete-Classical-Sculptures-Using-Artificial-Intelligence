"""
Aplica data augmentation sobre las imagenes clasificadas en DATASET_CLASSIFICADO
y guarda los resultados en FINAL_DATASET_AUG, manteniendo la misma estructura
de subcarpetas (whole_body, broken_body, head_only).

Por cada imagen original se generan N_VERSIONES versiones aumentadas usando
exactamente las mismas augmentaciones que en finetune_deeplabv3.py:

Augmentaciones de COLOR (solo imagen):
-conversion a escala de grises / desaturacion parcial
-ajuste de brillo
-ajuste de contraste
-ruido gaussiano (simula textura granular de piedra)
-blur suave (condiciones de fotografia de museo)

Augmentaciones GEOMETRICAS (imagen):
-flip horizontal
-rotacion aleatoria (-25, +25 grados)
-zoom con recorte aleatorio

En cada carpeta de salida quedan:
-la imagen original (copiada tal cual)
-N_VERSIONES imagenes aumentadas con sufijo _augN

"""

import random
import logging
import shutil
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm


DIR_ENTRADA = Path("/home/pfc/cescuder/tfg/dataset_classificado")
DIR_SALIDA = Path("/home/pfc/cescuder/tfg/final_dataset_aug")

#subcarpetas a procesar (no_human se descarta, no necesita augmentacion)
SUBCARPETAS = ["whole_body", "broken_body", "head_only"]

#versiones aumentadas por imagen original
N_VERSIONES = 5


#LOGGING:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("/home/pfc/cescuder/tfg/logs/data_augmentation.log",encoding="utf-8"),])
log = logging.getLogger(__name__)


#AUGMENTACIONES — identicas a las de finetune_deeplabv3.py:
def augmentar_imagen(img: Image.Image) -> Image.Image:

    #COLOR:

    # Blanco y negro / desaturacion parcial
    if random.random() < 0.3:
        img = img.convert("L").convert("RGB")
    elif random.random() < 0.4:
        factor = random.uniform(0.1, 0.6)
        img = ImageEnhance.Color(img).enhance(factor)

    # Brillo
    if random.random() < 0.4:
        factor = random.uniform(0.5, 1.6)
        img = ImageEnhance.Brightness(img).enhance(factor)

    # Contraste
    if random.random() < 0.4:
        factor = random.uniform(0.5, 1.8)
        img = ImageEnhance.Contrast(img).enhance(factor)

    # Ruido gaussiano (simula textura granular de piedra/marmol)
    if random.random() < 0.3:
        intensidad = random.uniform(10, 35)
        arr = np.array(img, dtype=np.float32)
        arr = np.clip(arr + np.random.normal(0, intensidad, arr.shape), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # Blur suave (fotos de museo a veces ligeramente desenfocadas)
    if random.random() < 0.25:
        radio = random.uniform(0.5, 2.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=radio))

    #GEOMETRICAS:

    # Flip horizontal
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Rotacion aleatoria
    if random.random() < 0.5:
        angulo = random.uniform(-25, 25)
        img = img.rotate(angulo, resample=Image.BILINEAR, fillcolor=(128, 128, 128))

    # Zoom con recorte aleatorio
    if random.random() < 0.35:
        w, h= img.size
        factor = random.uniform(0.78, 0.95)
        cw, ch = int(w * factor), int(h * factor)
        l = random.randint(0, w - cw)
        t = random.randint(0, h - ch)
        img = img.crop((l, t, l + cw, t + ch)).resize((w, h), Image.LANCZOS)

    return img

#MAIN:
def main():
    log.info(f"Versions per image: {N_VERSIONES} aumentadas + 1 original")

    extensiones = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".webp"}
    total_originales = 0
    total_generadas = 0

    for subcarpeta in SUBCARPETAS:
        dir_in  = DIR_ENTRADA / subcarpeta
        dir_out = DIR_SALIDA / subcarpeta
        dir_out.mkdir(parents=True, exist_ok=True)

        if not dir_in.exists():
            log.warning(f"No exists: {dir_in} — omiting.")
            continue

        imagenes = [f for f in dir_in.iterdir() if f.suffix in extensiones]
        log.info(f"\n[{subcarpeta}] {len(imagenes)} found images")

        for img_path in tqdm(imagenes, desc=f"{subcarpeta}"):
            try:
                # 1. Copiar original si no existe ya
                dest_original = dir_out / img_path.name
                if not dest_original.exists():
                    shutil.copy2(img_path, dest_original)
                total_originales += 1

                # 2. Generar N_VERSIONES augmentadas (saltar las que ya existen)
                img = Image.open(img_path).convert("RGB")
                for i in range(1, N_VERSIONES + 1):
                    nombre=f"{img_path.stem}_aug{i}{img_path.suffix}"
                    dest_aug = dir_out / nombre
                    if dest_aug.exists():
                        total_generadas += 1
                        continue
                    img_aug  = augmentar_imagen(img)
                    img_aug.save(dest_aug, quality=95)
                    total_generadas += 1

            except Exception as e:
                log.warning(f"Error in {img_path.name}: {e}")

    log.info("AUGMENTATION COMPLETED")
    log.info(f"Copied originals: {total_originales}")
    log.info(f"Generated versions: {total_generadas}")
    log.info(f"Total images: {total_originales + total_generadas}")


if __name__=="__main__":
    main()