#Variante v4 de MAT sobre broken_body: pinta el fondo (zonas no-cuerpo) de marmol antes
#de pasar la imagen a MAT, luego recompone el resultado SOBRE EL FONDO ORIGINAL (solo conserva pixels de MAT dentro de la mascara)

import logging
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


DIR_IMAGENES = Path("/home/pfc/cescuder/tfg/dataset_classificado/broken_body")
DIR_MASCARAS = Path("/home/pfc/cescuder/tfg/masks/broken_body")
DIR_SALIDA = Path("/home/pfc/cescuder/tfg/inpainting_results/mat_v4_marble_bg")
DIR_MAT_REPO = Path("/home/pfc/cescuder/tfg/MAT")
PESOS_MAT = DIR_MAT_REPO / "Places_512_FullData_G.pkl"
TAMANO_MAT = 512

N_MUESTRAS_CUERPO = 5000
SIGMA_RUIDO_BG = 8.0
UMBRAL_BLANCO = 245


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("/home/pfc/cescuder/tfg/logs/mat_v4_marble_bg.log", encoding="utf-8"),])
log = logging.getLogger(__name__)


def marmolizar_fondo(img_np, mask_np):
    no_blanco = (img_np.mean(axis=2) < UMBRAL_BLANCO)
    cuerpo = no_blanco & (~mask_np)
    fondo = ~no_blanco & (~mask_np)

    if cuerpo.sum() == 0:
        color_medio = np.array([180, 175, 170], dtype=np.float32)
    else:
        pixeles = img_np[cuerpo]
        if len(pixeles) > N_MUESTRAS_CUERPO:
            idx = np.random.choice(len(pixeles), N_MUESTRAS_CUERPO, replace=False)
            pixeles = pixeles[idx]
        color_medio = pixeles.mean(axis=0)

    h, w = mask_np.shape
    parche = np.tile(color_medio, (h, w, 1)).astype(np.float32)
    ruido = np.random.normal(0, SIGMA_RUIDO_BG, (h, w, 3))
    parche = np.clip(parche + ruido, 0, 255).astype(np.uint8)

    img_out = img_np.copy()
    img_out[fondo] = parche[fondo]
    return img_out


def cargar_mat(device):
    sys.path.insert(0, str(DIR_MAT_REPO))
    from networks.mat import Generator
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=TAMANO_MAT, img_channels=3)
    G.load_state_dict(torch.load(str(PESOS_MAT), map_location=device, weights_only=False), strict=False)
    return G.to(device).eval().requires_grad_(False)


def correr_mat_full(img_pil, mask_pil, G, device, tamano_orig):
    img_r = img_pil.resize((TAMANO_MAT, TAMANO_MAT), Image.BILINEAR)
    mask_r = mask_pil.resize((TAMANO_MAT, TAMANO_MAT), Image.NEAREST)
    img_np = np.array(img_r).astype(np.float32) / 127.5 - 1.0
    mask_np = (np.array(mask_r) > 127).astype(np.float32)
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        z = torch.randn(1, G.z_dim, device=device)
        c = torch.zeros(1, G.c_dim, device=device)
        output = G(img_t, mask_t, z, c, truncation_psi=1, noise_mode="const")
    out_np = output[0].permute(1, 2, 0).cpu().numpy()
    out_np = ((out_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    out_pil = Image.fromarray(out_np).resize(tamano_orig, Image.BILINEAR)
    return out_pil


def main():
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    G = cargar_mat(device)

    extensiones = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    imagenes = [f for f in DIR_IMAGENES.iterdir() if f.suffix in extensiones]
    log.info(f"Images to process: {len(imagenes)}")

    procesadas, sin_mascara, errores = 0, 0, 0
    for img_path in tqdm(imagenes, desc="MAT v4 marble-bg"):
        try:
            salida_path = DIR_SALIDA / (img_path.stem + "_matv4.png")
            if salida_path.exists():
                procesadas += 1
                continue

            mask_path = DIR_MASCARAS / (img_path.stem + "_mask.png")
            if not mask_path.exists():
                sin_mascara += 1
                continue

            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            if mask.size != img.size:
                mask = mask.resize(img.size, Image.NEAREST)

            img_np = np.array(img)
            mask_np = (np.array(mask) > 127)

            #marmolizar el fondo y correr MAT:
            img_marmolizada = marmolizar_fondo(img_np, mask_np)
            img_pil = Image.fromarray(img_marmolizada)
            salida_mat = correr_mat_full(img_pil, mask, G, device, img.size)
            salida_mat_np = np.array(salida_mat)

            #recomponer: dentro mascara -> MAT, fuera -> original:
            img_final = img_np.copy()
            img_final[mask_np] = salida_mat_np[mask_np]
            Image.fromarray(img_final).save(salida_path)
            procesadas += 1

        except Exception as e:
            log.warning(f"Error in {img_path.name}: {e}")
            errores += 1

    log.info(f"MAT v4 COMPLETED — Processed:{procesadas}, NoMask:{sin_mascara}, Errors:{errores}")
    log.info(f"Output: {DIR_SALIDA}")


if __name__ == "__main__":
    main()
