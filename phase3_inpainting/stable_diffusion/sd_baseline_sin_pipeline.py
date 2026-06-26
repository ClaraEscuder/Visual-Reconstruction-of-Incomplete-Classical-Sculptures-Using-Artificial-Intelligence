"""
Stable Diffusion Inpainting BASELINE - sin pipeline.

Esta es la version "sin tu trabajo": aplica SD directamente sobre las imagenes
RAW (con fondo intacto) usando las masks v8 anatomicas como condicion espacial.

Sirve como punto de comparacion en la Tabla 3 de ablation de SD:
  - SD baseline (este script): raw + mask v8                  [sin nobg, sin DP cond]
  - SD intermedio (sd_v8masks ya hecho): nobg + mask v8       [con nobg, sin DP cond]
  - SD completo (sd_controlnet_densepose.py): nobg + mask v8 + DP cond via CN

Aisla el efecto del BACKGROUND REMOVAL al mantener la mask v8 fija y solo
cambiar la imagen de entrada.

INPUT:
  - imagenes RAW (con fondo):  ~/tfg/dataset_clasificado/broken_body/
  - masks v8:                  ~/tfg/masks/broken_body_v8/{stem}_mask.png

OUTPUT:
  - reconstrucciones: ~/tfg/inpainting_results/sd_baseline/
"""

import logging
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionInpaintPipeline

BASE = Path("/home/pfc/cescuder/tfg")
#imagenes RAW (con fondo): probamos dos rutas posibles segun donde esten
DIR_IMAGENES_OPC1 = BASE / "dataset_clasificado" / "broken_body"
DIR_IMAGENES_OPC2 = BASE / "dataset_classificado" / "broken_body"
DIR_IMAGENES_OPC3 = BASE / "dataset" / "broken_body"

#elegir la primera que exista
DIR_IMAGENES = None
for p in [DIR_IMAGENES_OPC1, DIR_IMAGENES_OPC2, DIR_IMAGENES_OPC3]:
    if p.exists():
        DIR_IMAGENES = p
        break

DIR_MASCARAS = BASE / "masks" / "broken_body_v8"
DIR_SALIDA   = BASE / "inpainting_results" / "sd_baseline"

MODELO_SD = "runwayml/stable-diffusion-inpainting"

#mismos hyperparams que sd_v8masks: comparacion justa
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 8.5
TAMANO_LARGO = 512

PROMPT = (
    "a classical Greek marble statue, anatomically correct, full body, "
    "white marble texture, museum lighting, hellenistic style, smooth carving, "
    "sculptural anatomy, photographic, sharp details"
)

NEGATIVE_PROMPT = (
    "broken, deformed, modern, ugly, low quality, blurry, painting, drawing, "
    "cartoon, sketch, abstract, distorted, asymmetric, missing limb, "
    "extra limb, mutated, plastic, glossy"
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(BASE / "logs" / "sd_baseline.log", encoding="utf-8")])
log = logging.getLogger(__name__)


def cargar_pipeline(device):
    log.info(f"Cargando pipeline {MODELO_SD} (fp16 para fit en VRAM)")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODELO_SD,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    return pipe


def redimensionar_para_sd(img, mask, tamano_largo):
    w, h = img.size
    if w >= h:
        new_w = tamano_largo
        new_h = int(h * tamano_largo / w)
    else:
        new_h = tamano_largo
        new_w = int(w * tamano_largo / h)
    new_w = max(8, (new_w // 8) * 8)
    new_h = max(8, (new_h // 8) * 8)
    img_r = img.resize((new_w, new_h), Image.BILINEAR)
    mask_r = mask.resize((new_w, new_h), Image.NEAREST)
    return img_r, mask_r

#---------------------------------------------------------------------------
def main():
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if device == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}  compute: {torch.cuda.get_device_capability(0)}")

    if DIR_IMAGENES is None:
        log.error("No se encontro directorio de imagenes RAW. Probadas: "
                  f"{DIR_IMAGENES_OPC1}, {DIR_IMAGENES_OPC2}, {DIR_IMAGENES_OPC3}")
        log.error("Edita el script con la ruta correcta")
        return

    log.info(f"Imagenes RAW (con fondo): {DIR_IMAGENES}")
    log.info(f"Mascaras v8: {DIR_MASCARAS}")
    log.info(f"Salida: {DIR_SALIDA}")

    if not DIR_MASCARAS.exists():
        log.error(f"Mask dir not found: {DIR_MASCARAS}")
        return

    extensiones = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    imagenes = sorted([f for f in DIR_IMAGENES.iterdir() if f.suffix in extensiones])
    log.info(f"Images to process: {len(imagenes)}")

    pipe = cargar_pipeline(device)

    procesadas, sin_mask, errores = 0, 0, 0

    for img_path in tqdm(imagenes, desc="SD baseline"):
        try:
            #stem cleaning para emparejar masks v8 (que tienen stem limpio)
            stem_limpio = img_path.stem
            for _ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                if stem_limpio.endswith(_ext):
                    stem_limpio = stem_limpio[:-len(_ext)]
                    break

            salida_path = DIR_SALIDA / (stem_limpio + "_sdbaseline.png")
            if salida_path.exists():
                procesadas += 1
                continue

            mask_path = DIR_MASCARAS / (stem_limpio + "_mask.png")
            if not mask_path.exists():
                sin_mask += 1
                continue

            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            tamano_original = img.size

            img_r, mask_r = redimensionar_para_sd(img, mask, TAMANO_LARGO)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = pipe(
                    prompt=PROMPT,
                    negative_prompt=NEGATIVE_PROMPT,
                    image=img_r,
                    mask_image=mask_r,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                ).images[0]

            output_resized = output.resize(tamano_original, Image.BILINEAR)
            output_resized.save(salida_path)
            procesadas += 1

        except Exception as e:
            errores += 1
            log.error(f"Error en {img_path.name}: {e}")

    log.info("SD BASELINE (sin pipeline) COMPLETED")
    log.info(f"Processed: {procesadas}")
    log.info(f"Skipped (no mask): {sin_mask}")
    log.info(f"Errors: {errores}")
    log.info(f"Output at: {DIR_SALIDA}")


if __name__ == "__main__":
    main()
