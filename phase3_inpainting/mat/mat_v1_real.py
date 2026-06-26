#Variante v1 de MAT sobre broken_body (equivalente a v1 de LaMa = baseline puro).
#es esencialmente el mat_inpainting.py de marzo, pero con output dir diferente para no
#machacar el run de "mat_v1_fondoblanco" anterior, que correspondia al workaround de
#cuando gcc 12 no se cargaba bien.
#
#destrabado: el slurm ahora hace `module load gcc/12.2.1` antes de python, asi MAT compila
#sus kernels CUDA al vuelo y deja de petar por GCC 8.
#
#INPUT:
#  - imagenes: ~/tfg/dataset_classificado/broken_body/  (con fondo blanco neutro)
#  - mascaras: ~/tfg/masks/broken_body/  (PNG binario)
#OUTPUT:
#  - imagenes reconstruidas: ~/tfg/inpainting_results/mat_v1_real/

import logging
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


DIR_IMAGENES = Path("/home/pfc/cescuder/tfg/dataset_classificado/broken_body")
DIR_MASCARAS = Path("/home/pfc/cescuder/tfg/masks/broken_body")
DIR_SALIDA = Path("/home/pfc/cescuder/tfg/inpainting_results/mat_v1_real")
DIR_MAT_REPO = Path("/home/pfc/cescuder/tfg/MAT")

PESOS_MAT = DIR_MAT_REPO / "Places_512_FullData_G.pkl"
TAMANO_MAT = 512


#LOGGING:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("/home/pfc/cescuder/tfg/logs/mat_v1_real.log", encoding="utf-8"),])
log = logging.getLogger(__name__)


#CARGAR MAT:
def cargar_mat(device):
    sys.path.insert(0, str(DIR_MAT_REPO))
    from networks.mat import Generator

    log.info(f"Loading MAT weights from: {PESOS_MAT}")
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=TAMANO_MAT, img_channels=3)
    state_dict = torch.load(str(PESOS_MAT), map_location=device, weights_only=False)
    missing, unexpected = G.load_state_dict(state_dict, strict=False)
    log.info(f"Generator created (missing: {len(missing)}, unexpected: {len(unexpected)})")
    G = G.to(device).eval().requires_grad_(False)
    return G


#PREPROCESAR / POSTPROCESAR:
def preprocesar(img: Image.Image, mask: Image.Image):
    img_r = img.resize((TAMANO_MAT, TAMANO_MAT), Image.BILINEAR)
    mask_r = mask.resize((TAMANO_MAT, TAMANO_MAT), Image.NEAREST)
    img_np = np.array(img_r).astype(np.float32) / 127.5 - 1.0
    mask_np = (np.array(mask_r) > 127).astype(np.float32)
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
    return img_t, mask_t


def postprocesar(output_t: torch.Tensor, tamano_original: tuple) -> Image.Image:
    output_np = output_t[0].permute(1, 2, 0).cpu().numpy()
    output_np = ((output_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    img_out = Image.fromarray(output_np)
    return img_out.resize(tamano_original, Image.BILINEAR)


#MAIN:
def main():
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    if not DIR_MAT_REPO.exists() or not PESOS_MAT.exists():
        log.error("MAT repo or weights missing")
        return
    if not DIR_IMAGENES.exists() or not DIR_MASCARAS.exists():
        log.error("inputs missing")
        return

    extensiones = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    imagenes = [f for f in DIR_IMAGENES.iterdir() if f.suffix in extensiones]
    log.info(f"Images to process: {len(imagenes)}")

    G = cargar_mat(device)

    procesadas = 0
    sin_mascara = 0
    errores = 0

    for img_path in tqdm(imagenes, desc="MAT v1 real"):
        try:
            salida_path = DIR_SALIDA / (img_path.stem + "_matv1.png")
            if salida_path.exists():
                procesadas += 1
                continue

            mask_path = DIR_MASCARAS / (img_path.stem + "_mask.png")
            if not mask_path.exists():
                sin_mascara += 1
                continue

            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            tamano_original = img.size

            img_t, mask_t = preprocesar(img, mask)
            img_t = img_t.to(device)
            mask_t = mask_t.to(device)

            with torch.no_grad():
                z = torch.randn(1, G.z_dim, device=device)
                c = torch.zeros(1, G.c_dim, device=device)
                output = G(img_t, mask_t, z, c, truncation_psi=1, noise_mode="const")

            resultado = postprocesar(output, tamano_original)
            resultado.save(salida_path)
            procesadas += 1

        except Exception as e:
            log.warning(f"Error in {img_path.name}: {e}")
            errores += 1

    log.info("MAT v1 REAL COMPLETED")
    log.info(f"Processed: {procesadas}")
    log.info(f"Skipped (no mask): {sin_mascara}")
    log.info(f"Errors: {errores}")
    log.info(f"Output at: {DIR_SALIDA}")


if __name__ == "__main__":
    main()
