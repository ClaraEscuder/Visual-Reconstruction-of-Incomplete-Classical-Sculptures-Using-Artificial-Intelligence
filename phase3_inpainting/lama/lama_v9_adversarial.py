"""
Inferencia de LaMa v9 adversarial intensificado sobre broken_body bg-removed.

Carga los pesos v9 (resultado del finetune adversarial con peso 0.5 desde v8)
y procesa las imagenes de broken_body sin fondo usando masks anatomicas v8 y
conditioning DensePose.

INPUT:
    - imagenes:        ~/tfg/background_removed/broken_body/
    - mascaras y cond: ~/tfg/masks/broken_body_v8/{stem}_mask.png + {stem}_cond.npz
    - pesos:           ~/tfg/lama_repo/big-lama/models/best_finetuned_v9.ckpt

OUTPUT:
    - reconstrucciones: ~/tfg/inpainting_results/lama_v9_adversarial_v8masks/
"""

import sys
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/pfc/cescuder/tfg/scripts")
from ffc_standalone import FFCResNetGenerator, BIG_LAMA_GENERATOR_KWARGS

BASE = Path("/home/pfc/cescuder/tfg")

DIR_IMAGENES = BASE / "background_removed" / "broken_body"
DIR_MASCARAS = BASE / "masks" / "broken_body_v8"
DIR_SALIDA = BASE / "inpainting_results" / "lama_v9_adversarial_v8masks"

DIR_LAMA_REPO = BASE / "lama_repo"
DIR_BIG_LAMA = DIR_LAMA_REPO / "big-lama"
CKPT_V9 = DIR_BIG_LAMA / "models" / "best_finetuned_v9.ckpt"

TAMANO_IMG = 512
MAX_PART_ID = 24.0


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(BASE / "logs" / "lama_v9_adversarial_v8masks.log", encoding="utf-8")])
log = logging.getLogger(__name__)


def expandir_primer_conv(generator, in_orig=4, in_nuevo=7):
    def reemplazar_en(parent: nn.Module):
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
                    if hijo.bias is not None:
                        conv_nuevo.bias.copy_(hijo.bias)
                setattr(parent, nombre, conv_nuevo)
            else:
                reemplazar_en(hijo)
    reemplazar_en(generator)


def cargar_generator_v9(device):
    if not CKPT_V9.exists():
        raise FileNotFoundError(f"v8 checkpoint not found: {CKPT_V9}")
    log.info("building FFCResNetGenerator 4ch -> expanding to 7ch")
    generator = FFCResNetGenerator(**BIG_LAMA_GENERATOR_KWARGS)
    expandir_primer_conv(generator, in_orig=4, in_nuevo=7)
    log.info(f"loading v8 weights from {CKPT_V9}")
    state = torch.load(str(CKPT_V9), map_location=device, weights_only=False)
    gen_sd = {k[len("generator."):]: v for k, v in state["state_dict"].items() if k.startswith("generator.")}
    missing, unexpected = generator.load_state_dict(gen_sd, strict=False)
    log.info(f"  loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")
    return generator.to(device).eval().requires_grad_(False)


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
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)

    if dp_iuv is None:
        cond_t = torch.zeros(1, 3, TAMANO_IMG, TAMANO_IMG)
    else:
        I, U, V = dp_iuv
        I = np.array(Image.fromarray(I.astype(np.uint8)).resize((TAMANO_IMG, TAMANO_IMG), Image.NEAREST)).astype(np.float32)
        U = np.array(Image.fromarray(U).resize((TAMANO_IMG, TAMANO_IMG), Image.BILINEAR)).astype(np.float32)
        V = np.array(Image.fromarray(V).resize((TAMANO_IMG, TAMANO_IMG), Image.BILINEAR)).astype(np.float32)
        I_norm = (I / MAX_PART_ID).clip(0, 1)
        U_norm = U.clip(0, 1)
        V_norm = V.clip(0, 1)
        cond_t = torch.from_numpy(np.stack([I_norm, U_norm, V_norm], axis=0)).unsqueeze(0).float()
    return img_t, mask_t, cond_t

def postprocesar(output_t: torch.Tensor, tamano_original: tuple, masked_img_t: torch.Tensor, mask_t: torch.Tensor) -> Image.Image:
    pred_compuesto = masked_img_t + output_t * mask_t
    out_np = pred_compuesto[0].permute(1, 2, 0).cpu().numpy()
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
    for img_path in tqdm(imagenes, desc="LaMa v9 adversarial"):
        try:
            #algunas imagenes de background_removed conservan la extension
            #original embebida (ej "Calf-Bearer.jpg.png"), mientras que las masks
            #v8 fueron escritas con el stem limpio. limpiamos el stem antes de
            #buscar la mask y el cond.npz para que ambos esquemas funcionen
            stem_limpio = img_path.stem
            for _ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                if stem_limpio.endswith(_ext):
                    stem_limpio = stem_limpio[:-len(_ext)]
                    break

            salida_path = DIR_SALIDA / (stem_limpio + "_lamav9.png")
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

            img_t, mask_t, cond_t = preprocesar(img, mask, dp_iuv)
            img_t = img_t.to(device)
            mask_t = mask_t.to(device)
            cond_t = cond_t.to(device)

            masked_img = img_t * (1 - mask_t)
            entrada = torch.cat([masked_img, mask_t, cond_t], dim=1)

            with torch.no_grad():
                output = G(entrada)

            img_out = postprocesar(output, tamano_original, masked_img, mask_t)
            img_out.save(salida_path)
            procesadas += 1
        except Exception as e:
            errores += 1
            log.error(f"Error en {img_path.name}: {e}")

    log.info("LaMa v9 adversarial COMPLETED")
    log.info(f"Processed: {procesadas}")
    log.info(f"Skipped (no mask): {sin_mask}")
    log.info(f"Without cond: {sin_cond}")
    log.info(f"Errors: {errores}")
    log.info(f"Output at: {DIR_SALIDA}")


if __name__ == "__main__":
    main()
