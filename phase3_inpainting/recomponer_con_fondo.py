"""
Recomposicion de reconstrucciones sobre el fondo original.

Las reconstrucciones de v9 (LaMa, MAT) y SD operan sobre imagenes sin fondo
(background_removed via rembg+SAM), asi que sus outputs tienen fondo negro.
Esto introduce un bias en evaluadores visuales (PatchGAN, FID) que correlacionan
"presencia de fondo" con "realismo" (problema observado en patchgan_scores.csv:
cluster A=0.506 con fondo, cluster B=0.499 sin fondo).

Este script recompone cada reconstruccion sobre el fondo de su imagen original,
eliminando el bias del fondo negro.

ESTRATEGIA:
  Para cada pixel del output composited:
    - Si pixel pertenece al CUERPO (segun alpha_original OR mask_v8):
        -> usar la reconstruccion
        (esto cubre: parte intacta del cuerpo + parte reconstruida nueva)
    - Si pixel es FONDO (fuera del cuerpo y mask):
        -> usar la imagen original (museo, pared, etc.)

La mascara de compositing = alpha_original OR mask_v8 garantiza que NO se pega
nada en regiones que no son cuerpo.

INPUT:
  - originales:           ~/tfg/dataset_classificado/broken_body/{stem}.jpg
  - bg-removed (alpha):   ~/tfg/background_removed/broken_body/{stem}.jpg.png
  - mask v8:              ~/tfg/masks/broken_body_v8/{stem}_mask.png
  - reconstrucciones:     ~/tfg/inpainting_results/{variante}/{stem}_{suf}.png

OUTPUT:
  - ~/tfg/inpainting_results/{variante}_composited/{stem}_composited.png
"""

import sys
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from scipy.ndimage import binary_dilation, gaussian_filter
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


BASE = Path("/home/pfc/cescuder/tfg")
DIR_ORIG = BASE / "dataset_classificado" / "broken_body"
DIR_BG_REMOVED = BASE / "background_removed" / "broken_body"
DIR_MASKS_V8 = BASE / "masks" / "broken_body_v8"

#variantes a recomponer y su sufijo en el filename
VARIANTES = {
    "lama_v9_adversarial_v8masks":    "_lamav9",
    "mat_v9_adversarial_v8masks":     "_matv9",
    "sd_v8masks":                     "_sdv8",
    "sd_controlnet_v8masks":          "_sdcn",
    #sd_baseline NO necesita recomposicion: ya tiene fondo original
}


#parametros de blending para suavizar bordes
FEATHER_SIGMA = 1.0 #sigma para gaussian blur del alpha (suaviza bordes)
DILATE_PX = 1 #dilatar mask cuerpo 1px para evitar pixeles huerfanos en bordes


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(BASE / "logs" / "recomponer_con_fondo.log", encoding="utf-8")])
log = logging.getLogger(__name__)

def stem_limpio_fn(p):
    """Strip embedded extensions from stem (e.g. 'Apollo.jpg' -> 'Apollo')."""
    s = p.stem
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        if s.endswith(ext):
            s = s[:-len(ext)]
            break
    return s


def stem_limpio_fn_str(s):
    """Como stem_limpio_fn pero opera sobre strings (no Paths). Util para
    limpiar nombres que ya son strings (ej. p.stem o resultados intermedios)"""
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        if s.endswith(ext):
            return s[:-len(ext)]
    return s


def cargar_alpha(p):
    """Carga la mascara alfa del cuerpo desde la imagen background_removed
    Devuelve (H, W) uint8 donde 255 = cuerpo, 0 = fondo
    """

    img = Image.open(p)
    if img.mode in ("RGBA", "LA"):
        alpha = np.array(img.split()[-1])
    elif img.mode == "P" and "transparency" in img.info:
        alpha = np.array(img.convert("RGBA").split()[-1])
    else:
        #JPG sin alfa: threshold sobre suma de canales (negro <30 = fondo)
        arr = np.array(img.convert("RGB"))
        alpha = ((arr.sum(axis=2) > 30).astype(np.uint8)) * 255
    return alpha


def cargar_mask_v8(stem):
    """Carga la mask v8 anatomica. Devuelve (H, W) uint8 o None si no existe"""
    p = DIR_MASKS_V8 / f"{stem}_mask.png"
    if not p.exists():
        return None
    return np.array(Image.open(p).convert("L"))


def construir_mascara_compositing(alpha_orig, mask_v8, ancho, alto):
    """Construye la mascara de compositing: union de alpha_cuerpo y mask_v8

    pasos:
      1. Resize ambas al tamano del original
      2. Threshold a binario (>127)
      3. Union (OR logico)
      4. Dilatacion ligera para evitar pixeles huerfanos en bordes
      5. Feathering gaussiano para suavizar las transiciones

    devuelve (H, W) float32 en [0, 1]
    """
    #resize al tamano del original
    alpha_img = Image.fromarray(alpha_orig).resize((ancho, alto), Image.NEAREST)
    alpha_arr = np.array(alpha_img)

    if mask_v8 is not None:
        mask_img = Image.fromarray(mask_v8).resize((ancho, alto), Image.NEAREST)
        mask_arr = np.array(mask_img)
        #union: pixel pertenece al cuerpo si esta en alpha original O en mask v8
        union = np.maximum(alpha_arr, mask_arr)
    else:
        union = alpha_arr

    #threshold y dilatacion opcional
    if SCIPY_OK and DILATE_PX > 0:
        binary = union > 127
        dilated = binary_dilation(binary, iterations=DILATE_PX)
        union = (dilated.astype(np.uint8)) * 255

    #feathering: suaviza la mascara con un blur gaussiano para que la transicion
    #cuerpo<->fondo no tenga una linea dura
    if SCIPY_OK and FEATHER_SIGMA > 0:
        union = gaussian_filter(union.astype(np.float32), sigma=FEATHER_SIGMA)

    #normalizar a [0, 1]
    return (union.astype(np.float32) / 255.0).clip(0.0, 1.0)


def recomponer(orig_rgb, recon_rgb, alpha_compositing):
    """Compone la reconstruccion sobre el fondo original con alpha blending

    orig_rgb: (H, W, 3) uint8
    recon_rgb: (H, W, 3) uint8 (mismo tamano que orig)
    alpha_compositing: (H, W) float32 en [0, 1]

    return: (H, W, 3) uint8
    """
    alpha = alpha_compositing[..., None]   #(H, W, 1) para broadcasting
    composite = orig_rgb.astype(np.float32) * (1.0 - alpha) + recon_rgb.astype(np.float32) * alpha
    return np.clip(composite, 0, 255).astype(np.uint8)


def main():
    if not SCIPY_OK:
        log.warning("scipy no instalado: feathering y dilatacion desactivados")

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    #lookup originales por stem LIMPIO (algunos archivos en dataset_classificado
    #tienen doble extension como "Apollo.jpg.jpg" -> p.stem = "Apollo.jpg" pero
    #queremos como key "Apollo" para que matchee con stems limpios de las
    #reconstrucciones. Usamos stem_limpio_fn que strip extension embebida.
    if not DIR_ORIG.exists():
        log.error(f"No existe DIR_ORIG: {DIR_ORIG}")
        return
    orig_lookup = {}
    for p in DIR_ORIG.iterdir():
        if p.suffix in exts:
            #aplicar stem_limpio_fn iterativamente hasta que no cambie
            #(maneja casos con .jpg.jpg, .jpg.png, etc.)
            stem = p.stem
            while True:
                nuevo = stem_limpio_fn_str(stem)
                if nuevo == stem:
                    break
                stem = nuevo
            orig_lookup[stem] = p

    #lookup bg_removed por stem limpio (igual que orig_lookup: maneja doble ext)
    if not DIR_BG_REMOVED.exists():
        log.error(f"No existe DIR_BG_REMOVED: {DIR_BG_REMOVED}")
        return
    bg_lookup = {}
    for p in DIR_BG_REMOVED.iterdir():
        if p.suffix in exts:
            stem = p.stem
            while True:
                nuevo = stem_limpio_fn_str(stem)
                if nuevo == stem:
                    break
                stem = nuevo
            bg_lookup[stem] = p

    log.info(f"Lookup originales: {len(orig_lookup)}  bg_removed: {len(bg_lookup)}")

    for variante, sufijo_recon in VARIANTES.items():
        dir_recon = BASE / "inpainting_results" / variante
        if not dir_recon.exists():
            log.warning(f"variante {variante} no existe en disco, salta")
            continue

        dir_out = BASE / "inpainting_results" / f"{variante}_composited"
        dir_out.mkdir(parents=True, exist_ok=True)

        archivos = sorted([f for f in dir_recon.iterdir() if f.suffix in exts])
        log.info(f"=== variante {variante}: {len(archivos)} reconstrucciones ===")

        procesadas, sin_orig, sin_alpha, errores, ya_existian = 0, 0, 0, 0, 0
        for f in tqdm(archivos, desc=f"compositing {variante}"):
            try:
                #extraer stem limpio del filename de la reconstruccion
                stem = f.stem
                if stem.endswith(sufijo_recon):
                    stem = stem[:-len(sufijo_recon)]

                #FIX naming bug: muchas reconstrucciones tienen extension(es)
                #embebidas (ej: "Apollo.jpg_lamav9.png" o incluso "Apollo.jpg.jpg_lamav9").
                #Limpiamos ITERATIVAMENTE hasta que no quede ninguna extension al final,
                #para matchear con orig_lookup que tiene "Apollo" como key.
                while True:
                    nuevo = stem_limpio_fn_str(stem)
                    if nuevo == stem:
                        break
                    stem = nuevo

                #RESUME: si la composited ya existe, saltamos. Esto permite lanzar
                #el script multiples veces (ej: una vez con outputs parciales de un
                #inpainter que aun esta corriendo, y otra vez cuando termina) sin
                #reprocesar imagenes ya hechas.
                out_path = dir_out / f"{stem}_composited.png"
                if out_path.exists():
                    ya_existian += 1
                    continue

                if stem not in orig_lookup:
                    sin_orig += 1
                    continue
                if stem not in bg_lookup:
                    sin_alpha += 1
                    continue

                #cargar imagen original (con fondo)
                orig = Image.open(orig_lookup[stem]).convert("RGB")
                w, h = orig.size
                orig_arr = np.array(orig)

                #cargar alpha del bg-removed
                alpha_orig = cargar_alpha(bg_lookup[stem])

                #cargar mask v8 (puede no existir para algunas imagenes)
                mask_v8 = cargar_mask_v8(stem)

                #construir mascara de compositing
                alpha_comp = construir_mascara_compositing(alpha_orig, mask_v8, w, h)

                #cargar reconstruccion y resize al tamano del original
                recon = Image.open(f).convert("RGB").resize((w, h), Image.BILINEAR)
                recon_arr = np.array(recon)

                #recomponer
                composite = recomponer(orig_arr, recon_arr, alpha_comp)

                #out_path ya esta definido al inicio del try (para el resume check)
                Image.fromarray(composite).save(out_path)
                procesadas += 1

            except Exception as e:
                errores += 1
                log.error(f"  error en {f.name}: {e}")

        log.info(f"  procesadas: {procesadas}  ya_existian: {ya_existian}  "
                 f"sin_orig: {sin_orig}  sin_alpha: {sin_alpha}  errores: {errores}")
        log.info(f"  output en: {dir_out}")

    log.info("RECOMPOSICION CON FONDO COMPLETED")


if __name__ == "__main__":
    main()
