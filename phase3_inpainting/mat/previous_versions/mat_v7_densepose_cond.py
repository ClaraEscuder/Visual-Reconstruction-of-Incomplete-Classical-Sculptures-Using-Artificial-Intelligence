"""
Inferencia de MAT finetuneado con conditioning DensePose sobre broken_body.

Lee imagen + mascara binaria + cond.npz generado por compute_mask_from_densepose_v6_1.py.
El cond.npz contiene I_pred, U_pred, V_pred ya proyectados sobre la region enmascarada
(usando simetria/PCA del cuerpo visible). Construye el tensor extra de 3 canales y lo
pasa al MAT expandido que se finetuneo en finetune_mat_v7_densepose.py.

INPUT:
    - imagenes:        ~/tfg/dataset_classificado/broken_body/
    - mascaras y cond: ~/tfg/masks/broken_body/{stem}_mask.png + {stem}_cond.npz
    - pesos:           ~/tfg/MAT/checkpoints/best_finetuned_mat_v7.pt

OUTPUT:
    - imagenes reconstruidas: ~/tfg/inpainting_results/mat_v7_densepose_cond/
"""

import sys
import logging
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


BASE = Path("/home/pfc/cescuder/tfg")

DIR_MAT_REPO = BASE / "MAT"
sys.path.insert(0, str(DIR_MAT_REPO))
sys.path.insert(0, str(BASE / "scripts"))

from networks.mat import Generator
from networks.basic_module import Conv2dLayer
from networks.mat import Conv2dLayerPartial

from finetune_mat_v7_densepose import (
    expandir_conv2d_layer,
    expandir_conv2d_partial,
    first_stage_forward_con_dp,
    synthesis_forward_con_dp,
    generator_forward_con_dp,)


DIR_IMAGENES = BASE / "background_removed" / "broken_body"
DIR_MASCARAS = BASE / "masks" / "broken_body"
DIR_SALIDA = BASE / "inpainting_results" / "mat_v7_densepose_cond_nobg"

PKL_MAT = DIR_MAT_REPO / "Places_512_FullData_G.pkl"
CKPT_FINETUNED = DIR_MAT_REPO / "checkpoints" / "best_finetuned_mat_v7_nobg.pt"

TAMANO_IMG = 512
MAX_PART_ID = 24.0


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(BASE / "logs" / "mat_v7_densepose_cond_nobg.log", encoding="utf-8")])
log = logging.getLogger(__name__)


def cargar_generator_finetuneado(device):
    if not CKPT_FINETUNED.exists():
        raise FileNotFoundError(f"No existe checkpoint finetuneado: {CKPT_FINETUNED}. "
                                "Ejecuta primero finetune_mat_v7_densepose.py.")

    log.info(f"Construyendo MAT base y expandiendo entradas (4->7, 7->10)")
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=TAMANO_IMG, img_channels=3)
    state_base = torch.load(str(PKL_MAT), map_location="cpu", weights_only=False)
    G.load_state_dict(state_base, strict=False)

    G.synthesis.first_stage.conv_first = expandir_conv2d_partial(
        G.synthesis.first_stage.conv_first, in_nuevo=7)
    enc_attr = f"EncConv_Block_{TAMANO_IMG}x{TAMANO_IMG}"
    enc_first = getattr(G.synthesis.enc, enc_attr)
    enc_first.conv0 = expandir_conv2d_layer(enc_first.conv0, in_nuevo=10)

    G.synthesis.first_stage.forward = MethodType(first_stage_forward_con_dp, G.synthesis.first_stage)
    G.synthesis.forward = MethodType(synthesis_forward_con_dp, G.synthesis)
    G.forward = MethodType(generator_forward_con_dp, G)

    log.info(f"Cargando pesos finetuneados desde {CKPT_FINETUNED}")
    state_ft = torch.load(str(CKPT_FINETUNED), map_location="cpu", weights_only=False)
    G.load_state_dict(state_ft["generator"], strict=False)

    G = G.to(device).eval().requires_grad_(False)
    return G


def cargar_cond_npz(stem: str):
    """Devuelve (I, U, V) en float32 al tamano original de la imagen, o None si
    falta el cond.
    """
    cond_path = DIR_MASCARAS / f"{stem}_cond.npz"
    if not cond_path.exists():
        return None
    d = np.load(cond_path)
    I = d["I_pred"].astype(np.float32) if "I_pred" in d.files else None
    U = d["U_pred"].astype(np.float32) if "U_pred" in d.files else None
    V = d["V_pred"].astype(np.float32) if "V_pred" in d.files else None
    if I is None or U is None or V is None:
        return None
    return I, U, V


def preprocesar(img: Image.Image, mask: Image.Image, dp_iuv):
    img_r = img.resize((TAMANO_IMG, TAMANO_IMG), Image.BILINEAR)
    mask_r = mask.resize((TAMANO_IMG, TAMANO_IMG), Image.NEAREST)
    img_np = np.array(img_r).astype(np.float32) / 127.5 - 1.0
    mask_np = (np.array(mask_r) > 127).astype(np.float32)
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    mask_bin = torch.from_numpy(1.0 - mask_np).unsqueeze(0).unsqueeze(0)

    if dp_iuv is None:
        dp_t = torch.zeros(1, 3, TAMANO_IMG, TAMANO_IMG)
    else:
        I, U, V = dp_iuv
        I = np.array(Image.fromarray(I.astype(np.uint8)).resize((TAMANO_IMG, TAMANO_IMG), Image.NEAREST)).astype(np.float32)
        U = np.array(Image.fromarray(U).resize((TAMANO_IMG, TAMANO_IMG), Image.BILINEAR)).astype(np.float32)
        V = np.array(Image.fromarray(V).resize((TAMANO_IMG, TAMANO_IMG), Image.BILINEAR)).astype(np.float32)
        I_norm = (I / MAX_PART_ID).clip(0, 1) * 2.0 - 1.0
        U_norm = U.clip(0, 1) * 2.0 - 1.0
        V_norm = V.clip(0, 1) * 2.0 - 1.0
        dp_np = np.stack([I_norm, U_norm, V_norm], axis=0)
        dp_t = torch.from_numpy(dp_np).unsqueeze(0).float()
    return img_t, mask_bin, dp_t


def postprocesar(output_t: torch.Tensor, tamano_original: tuple) -> Image.Image:
    output_np = output_t[0].permute(1, 2, 0).cpu().numpy()
    output_np = ((output_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    img_out = Image.fromarray(output_np)
    return img_out.resize(tamano_original, Image.BILINEAR)


def main():
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    extensiones = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    imagenes = [f for f in DIR_IMAGENES.iterdir() if f.suffix in extensiones]
    log.info(f"Images to process: {len(imagenes)}")

    G = cargar_generator_finetuneado(device)

    procesadas, sin_mask, sin_cond, errores = 0, 0, 0, 0
    for img_path in tqdm(imagenes, desc="MAT v7 densepose cond"):
        try:
            salida_path = DIR_SALIDA / (img_path.stem + "_matv7.png")
            if salida_path.exists():
                procesadas += 1
                continue

            mask_path = DIR_MASCARAS / (img_path.stem + "_mask.png")
            if not mask_path.exists():
                sin_mask += 1
                continue

            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            tamano_original = img.size

            dp_iuv = cargar_cond_npz(img_path.stem)
            if dp_iuv is None:
                sin_cond += 1

            img_t, mask_t, dp_t = preprocesar(img, mask, dp_iuv)
            img_t = img_t.to(device)
            mask_t = mask_t.to(device)
            dp_t = dp_t.to(device)

            with torch.no_grad():
                z = torch.randn(1, G.z_dim, device=device)
                c = torch.zeros(1, G.c_dim, device=device)
                output = G(img_t, mask_t, dp_t, z, c, truncation_psi=1, noise_mode="const")

            img_out = postprocesar(output, tamano_original)
            img_out.save(salida_path)
            procesadas += 1
        except Exception as e:
            errores += 1
            log.error(f"Error en {img_path.name}: {e}")

    log.info(f"MAT v7 densepose cond COMPLETED")
    log.info(f"Processed: {procesadas}")
    log.info(f"Skipped (no mask): {sin_mask}")
    log.info(f"Without cond (zeros): {sin_cond}")
    log.info(f"Errors: {errores}")
    log.info(f"Output at: {DIR_SALIDA}")


if __name__ == "__main__":
    main()
