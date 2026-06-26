#Variante v5 de MAT sobre broken_body: inpainting iterativo en anillos crecientes desde el borde del cuerpo
#en cada paso solo se enmascara un anillo (parte de la mascaraoriginal)
#MAT extiende marmol, en el siguiente paso el anillo crece. usa z FIJO para que los pasos sucesivos no varien por ruido aleatorio

import logging
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt


DIR_IMAGENES = Path("/home/pfc/cescuder/tfg/dataset_classificado/broken_body")
DIR_MASCARAS = Path("/home/pfc/cescuder/tfg/masks/broken_body")
DIR_SALIDA = Path("/home/pfc/cescuder/tfg/inpainting_results/mat_v5_iterativo")
DIR_MAT_REPO = Path("/home/pfc/cescuder/tfg/MAT")
PESOS_MAT = DIR_MAT_REPO / "Places_512_FullData_G.pkl"
TAMANO_MAT = 512

N_ANILLOS = 4
UMBRAL_BLANCO = 245
TAM_MIN_ANILLO = 50
SEED_Z = 42


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("/home/pfc/cescuder/tfg/logs/mat_v5_iterativo.log", encoding="utf-8"),])
log = logging.getLogger(__name__)


def anillos_por_distancia(img_np, mask_np, n_anillos):
    no_blanco = (img_np.mean(axis=2) < UMBRAL_BLANCO)
    cuerpo = no_blanco & (~mask_np)
    if cuerpo.sum() == 0:
        return [mask_np.copy()]
    dist = distance_transform_edt(~cuerpo)
    dist_in_mask = dist[mask_np]
    if dist_in_mask.size == 0:
        return []
    cuantiles = np.linspace(0, 1, n_anillos + 1)[1:]
    umbrales = np.quantile(dist_in_mask, cuantiles)
    anillos = []
    prev = -np.inf
    for u in umbrales:
        a = mask_np & (dist > prev) & (dist <= u)
        if a.sum() >= TAM_MIN_ANILLO:
            anillos.append(a)
        prev = u
    return anillos


def cargar_mat(device):
    sys.path.insert(0, str(DIR_MAT_REPO))
    from networks.mat import Generator
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=TAMANO_MAT, img_channels=3)
    G.load_state_dict(torch.load(str(PESOS_MAT), map_location=device, weights_only=False), strict=False)
    return G.to(device).eval().requires_grad_(False)


def correr_mat(img_pil, mask_pil, G, device, z_fijo):
    tamano_orig = img_pil.size
    img_r = img_pil.resize((TAMANO_MAT, TAMANO_MAT), Image.BILINEAR)
    mask_r = mask_pil.resize((TAMANO_MAT, TAMANO_MAT), Image.NEAREST)
    img_np = np.array(img_r).astype(np.float32) / 127.5 - 1.0
    mask_np = (np.array(mask_r) > 127).astype(np.float32)
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
    c = torch.zeros(1, G.c_dim, device=device)
    with torch.no_grad():
        output = G(img_t, mask_t, z_fijo, c, truncation_psi=1, noise_mode="const")
    out_np = output[0].permute(1, 2, 0).cpu().numpy()
    out_np = ((out_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out_np).resize(tamano_orig, Image.BILINEAR)


def main():
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    G = cargar_mat(device)

    #z fijo: reproducible y los pasos del iterativo comparten estilo StyleGAN
    torch.manual_seed(SEED_Z)
    z_fijo = torch.randn(1, G.z_dim, device=device)

    extensiones = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    imagenes = [f for f in DIR_IMAGENES.iterdir() if f.suffix in extensiones]
    log.info(f"Images to process: {len(imagenes)}")

    procesadas, sin_mascara, sin_anillos, errores = 0, 0, 0, 0
    for img_path in tqdm(imagenes, desc="MAT v5 iterativo"):
        try:
            salida_path = DIR_SALIDA / (img_path.stem + "_matv5.png")
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

            anillos = anillos_por_distancia(img_np, mask_np, N_ANILLOS)
            if len(anillos) == 0:
                sin_anillos += 1
                continue

            #iterativo: en cada paso, enmascarar solo el anillo en curso
            img_actual_np = img_np.copy()
            for anillo in anillos:
                mask_paso_pil = Image.fromarray((anillo.astype(np.uint8) * 255), mode="L")
                img_actual_pil = Image.fromarray(img_actual_np)
                resultado_paso = correr_mat(img_actual_pil, mask_paso_pil, G, device, z_fijo)
                if resultado_paso.size != img_actual_pil.size:
                    resultado_paso = resultado_paso.resize(img_actual_pil.size, Image.BILINEAR)
                resultado_paso_np = np.array(resultado_paso)
                img_actual_np[anillo] = resultado_paso_np[anillo]

            Image.fromarray(img_actual_np).save(salida_path)
            procesadas += 1

        except Exception as e:
            log.warning(f"Error in {img_path.name}: {e}")
            errores += 1

    log.info(f"MAT v5 COMPLETED - Processed:{procesadas}, NoMask:{sin_mascara}, NoRings:{sin_anillos}, Errors:{errores}")
    log.info(f"Output: {DIR_SALIDA}")


if __name__ == "__main__":
    main()
