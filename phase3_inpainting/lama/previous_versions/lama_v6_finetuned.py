#Variante v6 de LaMa sobre broken_body: carga los pesos fine-tuneados (sobre el dataset
#sintetico bw-first) y aplica el inpainting sobre las 662 esculturas broken_body. La idea
#es que tras el fine-tuning LaMa ha visto miles de "miembros amputados" sobre figuras de
#marmol y ha aprendido a generar marmol dentro de las mascaras en vez de propagar fondo.
#
#Diferencia con v1 (baseline): mismo modelo arquitectonicamente, pero con los pesos
#actualizados por finetune_lama.py. Todo lo demas igual.
#
#INPUT:
#  - imagenes: ~/tfg/dataset_classificado/broken_body/
#  - mascaras: ~/tfg/masks/broken_body/
#  - pesos fine-tuneados: ~/tfg/lama_repo/big-lama/models/best_finetuned.ckpt
#OUTPUT:
#  - imagenes reconstruidas: ~/tfg/inpainting_results/lama_v6_finetuned/

import sys
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

#standalone FFCResNetGenerator para no depender de saicinpainting (rota en cluster)
sys.path.insert(0, "/home/pfc/cescuder/tfg/scripts")
from ffc_standalone import FFCResNetGenerator, BIG_LAMA_GENERATOR_KWARGS, cargar_pesos_big_lama


BASE = Path("/home/pfc/cescuder/tfg")

DIR_IMAGENES = BASE / "dataset_classificado" / "broken_body"
DIR_MASCARAS = BASE / "masks" / "broken_body"
DIR_SALIDA = BASE / "inpainting_results" / "lama_v6_finetuned"

DIR_LAMA_REPO = BASE / "lama_repo"
DIR_BIG_LAMA = DIR_LAMA_REPO / "big-lama"
CONFIG_LAMA = DIR_BIG_LAMA / "config.yaml"
CKPT_FINETUNED = DIR_BIG_LAMA / "models" / "best_finetuned.ckpt"

#resolucion a la que pasamos la imagen al modelo (LaMa generaliza bien entre resoluciones)
TAMANO_IMG = 512


#LOGGING:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(BASE / "logs" / "lama_v6_finetuned.log", encoding="utf-8"),])
log = logging.getLogger(__name__)


#CARGAR GENERATOR FINE-TUNEADO:
def cargar_generator_finetuneado(device):
    if not CKPT_FINETUNED.exists():
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found at {CKPT_FINETUNED}. Run finetune_lama.py first.")

    log.info(f"building FFCResNetGenerator with big-lama params")
    generator = FFCResNetGenerator(**BIG_LAMA_GENERATOR_KWARGS)

    log.info(f"loading fine-tuned checkpoint: {CKPT_FINETUNED}")
    missing, unexpected = cargar_pesos_big_lama(generator, str(CKPT_FINETUNED), device=device)
    log.info(f"  state_dict loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")
    generator = generator.to(device).eval()
    log.info("LaMa fine-tuned generator loaded")
    return generator


#PREPROCESAR / POSTPROCESAR:
def preprocesar(img: Image.Image, mask: Image.Image, tamano: int):
    img_r = img.resize((tamano, tamano), Image.BILINEAR)
    mask_r = mask.resize((tamano, tamano), Image.NEAREST)

    img_np = np.array(img_r).astype(np.float32) / 255.0
    #normalizar a [-1, 1] (la misma convencion usada en finetune_lama.py)
    img_np = (img_np - 0.5) / 0.5
    mask_np = (np.array(mask_r) > 127).astype(np.float32)

    img_t  = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
    return img_t, mask_t


def postprocesar(img_orig_np: np.ndarray, pred_t: torch.Tensor, mask_orig_np: np.ndarray, tamano_orig: tuple):
    """
    Devuelve una PIL Image al tamaño original:
        - fuera de la mascara: pixeles ORIGINALES (sin tocar)
        - dentro de la mascara: pixeles generados por el modelo
    """
    pred_np = pred_t[0].permute(1, 2, 0).cpu().numpy()
    pred_np = ((pred_np * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

    w_orig, h_orig = tamano_orig
    pred_pil = Image.fromarray(pred_np).resize((w_orig, h_orig), Image.BILINEAR)
    pred_resized = np.array(pred_pil)

    final = img_orig_np.copy()
    final[mask_orig_np] = pred_resized[mask_orig_np]
    return Image.fromarray(final)


#MAIN:
def main():
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    if not DIR_IMAGENES.exists():
        log.error(f"Input images directory not found: {DIR_IMAGENES}")
        return
    if not DIR_MASCARAS.exists():
        log.error(f"Input masks directory not found: {DIR_MASCARAS}")
        return

    extensiones = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    imagenes = [f for f in DIR_IMAGENES.iterdir() if f.suffix in extensiones]
    log.info(f"Images to process: {len(imagenes)}")

    generator = cargar_generator_finetuneado(device)

    procesadas = 0
    sin_mascara = 0
    errores = 0

    for img_path in tqdm(imagenes, desc="LaMa v6 finetuned"):
        try:
            salida_path = DIR_SALIDA / (img_path.stem + "_lamav6.png")
            if salida_path.exists():
                procesadas += 1
                continue

            mask_path = DIR_MASCARAS / (img_path.stem + "_mask.png")
            if not mask_path.exists():
                log.warning(f"Mask not found for: {img_path.name} - skipping")
                sin_mascara += 1
                continue

            img  = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            if mask.size != img.size:
                mask = mask.resize(img.size, Image.NEAREST)

            img_np_orig  = np.array(img)
            mask_np_orig = (np.array(mask) > 127)

            img_t, mask_t = preprocesar(img, mask, TAMANO_IMG)
            img_t  = img_t.to(device)
            mask_t = mask_t.to(device)

            with torch.no_grad():
                masked_img = img_t * (1 - mask_t)
                entrada = torch.cat([masked_img, mask_t], dim=1)
                salida = generator(entrada)
                #componer en espacio normalizado igual que durante el training
                pred = masked_img + salida * mask_t

            resultado = postprocesar(img_np_orig, pred, mask_np_orig, img.size)
            resultado.save(salida_path)
            procesadas += 1

        except Exception as e:
            log.warning(f"Error in {img_path.name}: {e}")
            errores += 1

    log.info("LAMA v6 (FINE-TUNED) COMPLETED")
    log.info(f"Processed: {procesadas}")
    log.info(f"Skipped (no mask): {sin_mascara}")
    log.info(f"Errors: {errores}")
    log.info(f"Output at: {DIR_SALIDA}")


if __name__ == "__main__":
    main()
