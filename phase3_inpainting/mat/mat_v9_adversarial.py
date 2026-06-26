"""
Inferencia de MAT v9 adversarial sobre broken_body bg-removed con masks v8.

Carga los pesos v9 (finetune adversarial con peso 0.5 desde MAT v7 NOBG) y
procesa las imagenes de broken_body sin fondo usando las masks anatomicas v8
y el conditioning DensePose.

INPUT:
    - imagenes:        ~/tfg/background_removed/broken_body/
    - mascaras y cond: ~/tfg/masks/broken_body_v8/{stem}_mask.png + {stem}_cond.npz
    - pesos:           ~/tfg/MAT/checkpoints/best_finetuned_mat_v9.pt

OUTPUT:
    - reconstrucciones: ~/tfg/inpainting_results/mat_v9_adversarial_v8masks/
"""

import sys
import logging
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


BASE = Path("/home/pfc/cescuder/tfg")

DIR_MAT_REPO = BASE / "MAT"
sys.path.insert(0, str(DIR_MAT_REPO))
sys.path.insert(0, str(BASE / "scripts"))

from networks.mat import Generator

from finetune_mat_v7_densepose import (
    expandir_conv2d_layer,
    expandir_conv2d_partial,
    first_stage_forward_con_dp,
    synthesis_forward_con_dp,
    generator_forward_con_dp,
)

DIR_IMAGENES = BASE / "background_removed" / "broken_body"
DIR_MASCARAS = BASE / "masks" / "broken_body_v8"
DIR_SALIDA = BASE / "inpainting_results" / "mat_v9_adversarial_v8masks"

PKL_MAT = DIR_MAT_REPO / "Places_512_FullData_G.pkl"
CKPT_V9 = DIR_MAT_REPO / "checkpoints" / "best_finetuned_mat_v9.pt"

TAMANO_IMG = 512
MAX_PART_ID = 24.0


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(BASE / "logs" / "mat_v9_adversarial.log", encoding="utf-8")])
log = logging.getLogger(__name__)


def cargar_generator_v9(device):
    if not CKPT_V9.exists():
        raise FileNotFoundError(f"v8 checkpoint not found: {CKPT_V9}")

    log.info("building MAT base + expanding to 7-channel")
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

    log.info(f"loading v8 weights from {CKPT_V9}")
    state = torch.load(str(CKPT_V9), map_location="cpu", weights_only=False)
    G.load_state_dict(state["generator"], strict=False)
    return G.to(device).eval().requires_grad_(False)


def cargar_cond_npz(stem: str):
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

def postprocesar(output_t, tamano_original):
    out_np = output_t[0].permute(1, 2, 0).cpu().numpy()
    out_np = ((out_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    img_out = Image.fromarray(out_np)
    return img_out.resize(tamano_original, Image.BILINEAR)

def main():
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    extensiones = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    imagenes = [f for f in DIR_IMAGENES.iterdir() if f.suffix in extensiones]
    log.info(f"Images to process: {len(imagenes)}")

    G = cargar_generator_v9(device)

    procesadas, sin_mask, sin_cond, errores = 0, 0, 0, 0
    for img_path in tqdm(imagenes, desc="MAT v9 adversarial"):
        try:
            #limpiamos extensiones embebidas del stem para que el lookup de
            #mask y cond.npz coincida con el esquema de naming de v8 masks
            stem_limpio = img_path.stem
            for _ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                if stem_limpio.endswith(_ext):
                    stem_limpio = stem_limpio[:-len(_ext)]
                    break

            salida_path = DIR_SALIDA / (stem_limpio + "_matv9.png")
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

            dp_iuv = cargar_cond_npz(stem_limpio)
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

    log.info("MAT v9 adversarial COMPLETED")
    log.info(f"Processed: {procesadas}")
    log.info(f"Skipped (no mask): {sin_mask}")
    log.info(f"Without cond: {sin_cond}")
    log.info(f"Errors: {errores}")
    log.info(f"Output at: {DIR_SALIDA}")


if __name__ == "__main__":
    main()
