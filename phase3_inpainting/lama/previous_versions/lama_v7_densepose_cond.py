#Inferencia LaMa-v7 con conditioning DensePose sobre broken_body.
#
#carga los pesos finetuneados v7 (que tienen primer conv expandido a 7 canales),
#lee imagen + mask + cond.npz generado por compute_mask_from_densepose_v6_1.py y
#construye el tensor de 7 canales que espera el modelo.
#
#cond.npz aporta:
#   I_pred  : (H, W) uint8   - parte SMPL en cada pixel (0=fondo, 1-24=partes)
#   U_pred  : (H, W) float16 - coordenada U sobre el mesh SMPL
#   V_pred  : (H, W) float16 - coordenada V sobre el mesh SMPL
#estos canales contienen tanto el DensePose REAL del cuerpo visible como la
#PROYECCION que v6.1 calculo para el miembro faltante. el modelo decide que
#textura/anatomia generar usando estos canales como conditioning estructural.
#
#INPUT:
#  - imagenes: ~/tfg/dataset_classificado/broken_body/
#  - mascaras + cond: ~/tfg/masks/broken_body/{stem}_mask.png + {stem}_cond.npz
#  - pesos: ~/tfg/lama_repo/big-lama/models/best_finetuned_v7.ckpt
#OUTPUT:
#  - imagenes reconstruidas: ~/tfg/inpainting_results/lama_v7_densepose_cond/

import sys
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

#standalone FFCResNetGenerator
sys.path.insert(0, "/home/pfc/cescuder/tfg/scripts")
from ffc_standalone import FFCResNetGenerator, BIG_LAMA_GENERATOR_KWARGS


BASE = Path("/home/pfc/cescuder/tfg")

DIR_IMAGENES = BASE / "background_removed" / "broken_body"
DIR_MASCARAS = BASE / "masks" / "broken_body"
DIR_SALIDA   = BASE / "inpainting_results" / "lama_v7_densepose_cond_nobg"

DIR_LAMA_REPO = BASE / "lama_repo"
DIR_BIG_LAMA = DIR_LAMA_REPO / "big-lama"
CONFIG_LAMA = DIR_BIG_LAMA / "config.yaml"
CKPT_FINETUNED = DIR_BIG_LAMA / "models" / "best_finetuned_v7_nobg.ckpt"

TAMANO_IMG = 512
MAX_PART_ID = 24.0


#LOGGING:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(BASE / "logs" / "lama_v7_densepose_cond_nobg.log", encoding="utf-8"),])
log = logging.getLogger(__name__)


#EXPANSION DEL PRIMER CONV (debe coincidir con la de finetune_lama_v7.py):
def expandir_primer_conv(generator, in_orig=4, in_nuevo=7):
    modificadas = 0
    def reemplazar_en(parent: nn.Module):
        nonlocal modificadas
        for nombre, hijo in list(parent.named_children()):
            if isinstance(hijo, nn.Conv2d) and hijo.in_channels == in_orig:
                conv_nuevo = nn.Conv2d(
                    in_channels=in_nuevo,
                    out_channels=hijo.out_channels,
                    kernel_size=hijo.kernel_size,
                    stride=hijo.stride,
                    padding=hijo.padding,
                    dilation=hijo.dilation,
                    groups=hijo.groups,
                    bias=hijo.bias is not None,
                    padding_mode=hijo.padding_mode,
                )
                with torch.no_grad():
                    conv_nuevo.weight.zero_()
                    conv_nuevo.weight[:, :in_orig].copy_(hijo.weight)
                    nn.init.normal_(conv_nuevo.weight[:, in_orig:], mean=0.0, std=0.01)
                    if hijo.bias is not None:
                        conv_nuevo.bias.copy_(hijo.bias)
                setattr(parent, nombre, conv_nuevo)
                modificadas += 1
            else:
                reemplazar_en(hijo)
    reemplazar_en(generator)
    return modificadas


#CARGAR GENERATOR v7 FINE-TUNEADO:
def cargar_generator_v7(device):
    if not CKPT_FINETUNED.exists():
        raise FileNotFoundError(f"v7 finetuned checkpoint not found at {CKPT_FINETUNED}. Run finetune_lama_v7.py first.")

    #construimos vanilla (4 canales), expandimos a 7, y luego cargamos los pesos
    #finetuneados que ya estan en formato 7-canal
    log.info("building vanilla FFCResNetGenerator (input_nc=4)")
    generator = FFCResNetGenerator(**BIG_LAMA_GENERATOR_KWARGS)

    log.info("expanding first conv 4 -> 7 channels")
    n = expandir_primer_conv(generator, in_orig=4, in_nuevo=7)
    log.info(f"  convs modified: {n}")

    log.info(f"loading fine-tuned weights from: {CKPT_FINETUNED}")
    state = torch.load(str(CKPT_FINETUNED), map_location=device, weights_only=False)
    state_dict = state.get("state_dict", state)
    state_gen = {}
    for k, v in state_dict.items():
        if k.startswith("generator."):
            state_gen[k[len("generator."):]] = v
    missing, unexpected = generator.load_state_dict(state_gen, strict=False)
    log.info(f"  state_dict loaded (missing={len(missing)}, unexpected={len(unexpected)})")

    generator = generator.to(device).eval()
    return generator


#CONSTRUCCION DEL TENSOR DE 7 CANALES:
def construir_entrada(img: Image.Image, mask: Image.Image, cond_path: Path, tamano: int):
    """
    Devuelve:
        entrada_t (1, 7, H, W) float32 = RGB normalizado + mask + part_id_norm + U + V
        img_orig_np (H_orig, W_orig, 3) uint8
        mask_orig_np (H_orig, W_orig) bool
        tamano_orig (W_orig, H_orig)
    """
    w_orig, h_orig = img.size
    img_r = img.resize((tamano, tamano), Image.BILINEAR)
    mask_r = mask.resize((tamano, tamano), Image.NEAREST)

    img_np = np.array(img_r).astype(np.float32) / 255.0
    img_np = (img_np - 0.5) / 0.5
    mask_np = (np.array(mask_r) > 127).astype(np.float32)

    cond = np.load(cond_path)
    I_pred = cond["I_pred"]
    U_pred = cond["U_pred"].astype(np.float32)
    V_pred = cond["V_pred"].astype(np.float32)
    I_pil = Image.fromarray(I_pred).resize((tamano, tamano), Image.NEAREST)
    U_pil = Image.fromarray(U_pred).resize((tamano, tamano), Image.BILINEAR)
    V_pil = Image.fromarray(V_pred).resize((tamano, tamano), Image.BILINEAR)
    I_arr = np.array(I_pil).astype(np.float32) / MAX_PART_ID
    U_arr = np.array(U_pil).astype(np.float32)
    V_arr = np.array(V_pil).astype(np.float32)

    img_t  = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
    cond_t = torch.from_numpy(np.stack([I_arr, U_arr, V_arr], axis=0)).unsqueeze(0)

    masked_img = img_t * (1 - mask_t)
    entrada = torch.cat([masked_img, mask_t, cond_t], dim=1)

    return entrada, mask_t, np.array(img), (np.array(mask) > 127), (w_orig, h_orig)


def postprocesar(img_orig_np: np.ndarray, pred_t: torch.Tensor, mask_t_full: torch.Tensor,
                 entrada_t: torch.Tensor, mask_orig_np: np.ndarray, tamano_orig: tuple):
    """
    Compone el resultado final:
      - fuera de la mascara: pixeles originales (sin tocar)
      - dentro de la mascara: pixeles regenerados por LaMa-v7
    """
    masked_img_t = entrada_t[:, :3]
    pred_compuesto = masked_img_t + pred_t * mask_t_full
    pred_np = pred_compuesto[0].permute(1, 2, 0).cpu().numpy()
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

    generator = cargar_generator_v7(device)

    procesadas = 0
    sin_mascara = 0
    sin_cond = 0
    errores = 0

    for img_path in tqdm(imagenes, desc="LaMa v7 densepose-cond"):
        try:
            salida_path = DIR_SALIDA / (img_path.stem + "_lamav7.png")
            if salida_path.exists():
                procesadas += 1
                continue

            mask_path = DIR_MASCARAS / (img_path.stem + "_mask.png")
            cond_path = DIR_MASCARAS / (img_path.stem + "_cond.npz")
            if not mask_path.exists():
                log.warning(f"Mask not found for: {img_path.name} - skipping")
                sin_mascara += 1
                continue
            if not cond_path.exists():
                log.warning(f"Cond.npz not found for: {img_path.name} - skipping (run compute_mask_from_densepose_v6_1.py first)")
                sin_cond += 1
                continue

            img  = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            if mask.size != img.size:
                mask = mask.resize(img.size, Image.NEAREST)

            entrada, mask_full, img_orig_np, mask_orig_np, tamano_orig = construir_entrada(
                img, mask, cond_path, TAMANO_IMG)
            entrada = entrada.to(device)
            mask_full = mask_full.to(device)

            with torch.no_grad():
                pred = generator(entrada)

            resultado = postprocesar(img_orig_np, pred, mask_full, entrada, mask_orig_np, tamano_orig)
            resultado.save(salida_path)
            procesadas += 1

        except Exception as e:
            log.warning(f"Error in {img_path.name}: {e}")
            errores += 1

    log.info("LAMA v7 (DENSEPOSE COND) COMPLETED")
    log.info(f"Processed: {procesadas}")
    log.info(f"Skipped (no mask): {sin_mascara}")
    log.info(f"Skipped (no cond): {sin_cond}")
    log.info(f"Errors: {errores}")
    log.info(f"Output at: {DIR_SALIDA}")


if __name__ == "__main__":
    main()
