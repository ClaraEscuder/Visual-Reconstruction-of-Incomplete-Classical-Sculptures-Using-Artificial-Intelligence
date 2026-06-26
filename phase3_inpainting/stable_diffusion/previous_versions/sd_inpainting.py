"""
Inferencia con Stable Diffusion Inpainting sobre broken_body usando los
componentes de la pipeline general. Esto es el "tercer paradigma" del
estudio comparativo, aplicado sobre la misma infraestructura que LaMa/MAT.

Reutiliza, sin cambios, las partes agnosticas al generador:
  - dataset_classificado/broken_body  (clasificacion DeepLabv3+ + DensePose +
    revision manual)
  - background_removed/broken_body    (eliminacion de fondo con rembg+SAM)
  - masks/broken_body_v8              (mask generator anatomico con per-segment
    + dilatacion minima + extremidades ahusadas + manos/pies perpendiculares)

Lo unico que cambia respecto a LaMa/MAT es el generador: SD trabaja sobre el
espacio latente de un VAE y aplica denoising guiado por un prompt textual con
clasifier-free guidance. SD no necesita finetune en este dominio porque sus
priors (LAION-5B, ~5.8B pares image-text) ya contienen un volumen relevante de
escultura clasica y representacion anatomica.

INPUT:
  - imagenes:  ~/tfg/background_removed/broken_body/
  - masks:     ~/tfg/masks/broken_body_v8/{stem}_mask.png

OUTPUT:
  - reconstrucciones: ~/tfg/inpainting_results/sd_v8masks/
"""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionInpaintPipeline


BASE = Path("/home/pfc/cescuder/tfg")

DIR_IMAGENES = BASE / "background_removed" / "broken_body"
DIR_MASCARAS = BASE / "masks" / "broken_body_v8"
DIR_SALIDA   = BASE / "inpainting_results" / "sd_v8masks"

MODELO_SD = "runwayml/stable-diffusion-inpainting"

#parametros documentados en la memoria del TFG: 50 denoising steps es estandar
#para calidad de produccion, guidance_scale 8.5 sesga fuerte al prompt
#manteniendo coherencia con el contexto visible
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
              logging.FileHandler(BASE / "logs" / "sd_v8masks.log", encoding="utf-8")])
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
    #desactivamos progress bar interna del pipeline porque ya tenemos tqdm
    pipe.set_progress_bar_config(disable=True)
    #attention slicing para reducir picos de memoria en imagenes grandes
    pipe.enable_attention_slicing()
    return pipe


def redimensionar_para_sd(img: Image.Image, mask: Image.Image, tamano_largo: int):
    #SD requiere dimensiones multiplos de 8. preservamos aspect ratio y
    #escalamos para que el lado mas largo coincida con tamano_largo
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


def main():
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if device == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}  compute: {torch.cuda.get_device_capability(0)}")

    if not DIR_IMAGENES.exists():
        log.error(f"Input dir not found: {DIR_IMAGENES}")
        return
    if not DIR_MASCARAS.exists():
        log.error(f"Mask dir not found: {DIR_MASCARAS}")
        return

    extensiones = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    imagenes = sorted([f for f in DIR_IMAGENES.iterdir() if f.suffix in extensiones])
    log.info(f"Images to process: {len(imagenes)}")

    pipe = cargar_pipeline(device)

    procesadas, sin_mask, errores = 0, 0, 0

    for img_path in tqdm(imagenes, desc="SD v8masks"):
        try:
            #algunas imagenes de background_removed conservan la extension
            #original embebida (ej "Calf-Bearer.jpg.png"), mientras que las masks
            #v8 fueron escritas con el stem limpio. limpiamos el stem antes de
            #buscar la mask para que ambos esquemas funcionen
            stem_limpio = img_path.stem
            for _ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                if stem_limpio.endswith(_ext):
                    stem_limpio = stem_limpio[:-len(_ext)]
                    break

            salida_path = DIR_SALIDA / (stem_limpio + "_sdv8.png")
            #si ya existe (resume), saltamos para permitir reanudar tras una caida
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

    log.info("SD INPAINTING COMPLETED")
    log.info(f"Processed: {procesadas}")
    log.info(f"Skipped (no mask): {sin_mask}")
    log.info(f"Errors: {errores}")
    log.info(f"Output at: {DIR_SALIDA}")


if __name__ == "__main__":
    main()
