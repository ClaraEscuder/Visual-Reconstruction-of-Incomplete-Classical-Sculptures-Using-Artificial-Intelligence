#Variante v3 de MAT sobre broken_body: recorta apretado alrededor del cuerpo+mascara
#antes de pasar la imagen a MAT, corre MAT (que redimensiona internamente a 512), pega el
#crop regenerado de vuelta en la imagen original.

import logging
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


DIR_IMAGENES = Path("/home/pfc/cescuder/tfg/dataset_classificado/broken_body")
DIR_MASCARAS = Path("/home/pfc/cescuder/tfg/masks/broken_body")
DIR_SALIDA = Path("/home/pfc/cescuder/tfg/inpainting_results/mat_v3_tight_crop")
DIR_MAT_REPO = Path("/home/pfc/cescuder/tfg/MAT")
PESOS_MAT = DIR_MAT_REPO / "Places_512_FullData_G.pkl"
TAMANO_MAT = 512

PAD_FRAC = 0.10
TAM_MINIMO_CROP = 256
UMBRAL_BLANCO = 245


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("/home/pfc/cescuder/tfg/logs/mat_v3_tight_crop.log", encoding="utf-8"),])
log = logging.getLogger(__name__)


def bbox_combinado(img_np, mask_np):
    h, w = mask_np.shape
    no_blanco = (img_np.mean(axis=2) < UMBRAL_BLANCO)
    region = no_blanco | mask_np
    if region.sum() == 0:
        return 0, 0, w, h
    ys, xs = np.where(region)
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    lado = max(x1 - x0, y1 - y0)
    pad = int(round(lado * PAD_FRAC))
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad); y1 = min(h, y1 + pad)
    if (x1 - x0) < TAM_MINIMO_CROP:
        extra = TAM_MINIMO_CROP - (x1 - x0)
        x0 = max(0, x0 - extra // 2)
        x1 = min(w, x0 + TAM_MINIMO_CROP)
        x0 = max(0, x1 - TAM_MINIMO_CROP)
    if (y1 - y0) < TAM_MINIMO_CROP:
        extra = TAM_MINIMO_CROP - (y1 - y0)
        y0 = max(0, y0 - extra // 2)
        y1 = min(h, y0 + TAM_MINIMO_CROP)
        y0 = max(0, y1 - TAM_MINIMO_CROP)
    return x0, y0, x1, y1


def cargar_mat(device):
    sys.path.insert(0, str(DIR_MAT_REPO))
    from networks.mat import Generator
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=TAMANO_MAT, img_channels=3)
    G.load_state_dict(torch.load(str(PESOS_MAT), map_location=device, weights_only=False), strict=False)
    return G.to(device).eval().requires_grad_(False)


def correr_mat(img_pil, mask_pil, G, device):
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
    return Image.fromarray(out_np)


def main():
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    G = cargar_mat(device)

    extensiones = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    imagenes = [f for f in DIR_IMAGENES.iterdir() if f.suffix in extensiones]
    log.info(f"Images to process: {len(imagenes)}")

    procesadas, sin_mascara, errores = 0, 0, 0
    for img_path in tqdm(imagenes, desc="MAT v3 crop"):
        try:
            salida_path = DIR_SALIDA / (img_path.stem + "_matv3.png")
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

            #recorte apretado al cuerpo + mascara
            x0, y0, x1, y1 = bbox_combinado(img_np, mask_np)
            img_crop = Image.fromarray(img_np[y0:y1, x0:x1])
            mask_crop = Image.fromarray((mask_np[y0:y1, x0:x1].astype(np.uint8)) * 255, mode="L")

            #MAT sobre el recorte (siempre devuelve 512x512, hay que volver al tamano del crop)
            resultado_crop = correr_mat(img_crop, mask_crop, G, device)
            altura_crop = y1 - y0
            ancho_crop = x1 - x0
            if resultado_crop.size != (ancho_crop, altura_crop):
                resultado_crop = resultado_crop.resize((ancho_crop, altura_crop), Image.BILINEAR)

            img_final = img_np.copy()
            img_final[y0:y1, x0:x1] = np.array(resultado_crop)
            Image.fromarray(img_final).save(salida_path)
            procesadas += 1

        except Exception as e:
            log.warning(f"Error in {img_path.name}: {e}")
            errores += 1

    log.info(f"MAT v3 COMPLETED — Processed:{procesadas}, NoMask:{sin_mascara}, Errors:{errores}")
    log.info(f"Output: {DIR_SALIDA}")


if __name__ == "__main__":
    main()
