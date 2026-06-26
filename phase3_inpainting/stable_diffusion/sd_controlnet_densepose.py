"""
Stable Diffusion Inpainting + Multi-ControlNet (Seg + Pose) condicionado por DensePose.

Esta es la version COMPLETA del pipeline para SD: maximiza la simetria
metodologica respecto a LaMa v9 y MAT v9 (que reciben DensePose IUV como
input de 3 canales).

Estrategia Multi-ControlNet:
  - ControlNet-Seg: recibe DensePose I (24 segmentos) coloreado como
    segmentation map denso -> equivale al canal I de LaMa/MAT
  - ControlNet-OpenPose: recibe esqueleto OpenPose derivado de los centroides
    de cada segmento DP -> aporta info topologica estructural

Las coordenadas U,V de superficie SMPL no se transmiten directamente (requeriria
un ControlNet custom-trained sobre DensePose IUV, que queda como future work).
Su efecto estructural queda capturado por la combinacion Seg + Pose.

INPUT:
  - imagenes: ~/tfg/background_removed/broken_body/
  - masks:    ~/tfg/masks/broken_body_v8/{stem}_mask.png
  - dp cond:  ~/tfg/masks/broken_body_v8/{stem}_cond.npz

OUTPUT:
  - reconstrucciones: ~/tfg/inpainting_results/sd_controlnet_v8masks/
"""

import logging
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel

BASE = Path("/home/pfc/cescuder/tfg")
DIR_IMAGENES = BASE / "background_removed" / "broken_body"
DIR_MASCARAS = BASE / "masks" / "broken_body_v8"
DIR_SALIDA = BASE / "inpainting_results" / "sd_controlnet_v8masks"

MODELO_SD = "runwayml/stable-diffusion-inpainting"
MODELO_CN_SEG = "lllyasviel/sd-controlnet-seg"
MODELO_CN_POSE = "lllyasviel/sd-controlnet-openpose"

NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 8.5
#pesos de cada controlnet: seg lleva mas peso porque transmite mas info densa
CONTROLNET_SCALE_SEG = 0.8
CONTROLNET_SCALE_POSE = 0.5
TAMANO_LARGO = 512
MAX_PART_ID = 24

PROMPT = (
    "a classical Greek marble statue, anatomically correct, full body, "
    "white marble texture, museum lighting, hellenistic style, smooth carving, "
    "sculptural anatomy, photographic, sharp details"
)

NEGATIVE_PROMPT = (
    "broken, deformed, modern, ugly, low quality, blurry, painting, drawing, "
    "cartoon, sketch, abstract, distorted, asymmetric, missing limb, "
    "extra limb, mutated, plastic, glossy"
)


#paleta para colorear los 25 valores de DensePose I (0=background, 1-24=partes corporales)
#colores distintos y separados para que ControlNet-Seg pueda discriminar regiones
PALETA_DP = np.array([
    [0, 0, 0], #0 = background
    [255, 0, 0], #1 = Torso back
    [255, 85, 0], #2 = Torso front
    [255, 170, 0], #3 = Rhand
    [255, 255, 0], #4 = Lhand
    [170, 255, 0], #5 = Rfoot
    [85, 255, 0], #6 = Lfoot
    [0, 255, 0], #7 = Rupper leg back
    [0, 255, 85], #8 = Lupper leg back
    [0, 255, 170], #9 = Rupper leg front
    [0, 255, 255], #10 = Lupper leg front
    [0, 170, 255], #11 = Rlower leg back
    [0, 85, 255], #12 = Llower leg back
    [0, 0, 255], #13 = Rlower leg front
    [85, 0, 255], #14 = Llower leg front
    [170, 0, 255], #15 = Rupper arm back
    [255, 0, 255], #16 = Lupper arm back
    [255, 0, 170], #17 = Rupper arm front
    [255, 0, 85], #18 = Lupper arm front
    [128, 128, 0], #19 = Rlower arm back
    [0, 128, 128], #20 = Llower arm back
    [128, 0, 128], #21 = Rlower arm front
    [200, 100, 50], #22 = Llower arm front
    [50, 200, 100], #23 = Head back
    [100, 50, 200], #24 = Head front
], dtype=np.uint8)


#mapping DensePose part IDs -> OpenPose body joints (COCO-18)
DP_TO_OPENPOSE = {
    24: 0, 23: 1,
    17: 2, 15: 2, 21: 3, 19: 3, 3: 4,
    18: 5, 16: 5, 22: 6, 20: 6, 4: 7,
    9: 8, 7: 8, 13: 9, 11: 9, 5: 10,
    10: 11, 8: 11, 14: 12, 12: 12, 6: 13,}

OPENPOSE_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
]

OPENPOSE_JOINT_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
]

OPENPOSE_BONE_COLORS = [
    (153, 0, 0), (153, 51, 0), (153, 102, 0), (153, 153, 0), (102, 153, 0),
    (51, 153, 0), (0, 153, 0), (0, 153, 51), (0, 153, 102), (0, 153, 153),
    (0, 102, 153), (0, 51, 153), (0, 0, 153),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(BASE / "logs" / "sd_controlnet_densepose.log", encoding="utf-8")])
log = logging.getLogger(__name__)

def densepose_a_seg_image(I_pred, ancho, alto):
    """
    Convierte DensePose I (HxW, valores 0-24) en una imagen RGB donde cada
    segmento corporal tiene un color distinto y constante (paleta PALETA_DP).

    Esto se pasa a ControlNet-Seg como control image, equivaliendo
    estructuralmente al canal I que reciben LaMa/MAT.
    """
    h_src, w_src = I_pred.shape
    I_clipped = np.clip(I_pred, 0, MAX_PART_ID).astype(np.int32)
    #lookup table: mapear cada valor de I a su color RGB
    rgb = PALETA_DP[I_clipped]   #(H, W, 3)
    img = Image.fromarray(rgb, mode="RGB")
    img = img.resize((ancho, alto), Image.NEAREST)
    return img


def densepose_a_openpose_image(I_pred, ancho, alto):
    """
    Convierte DensePose I en una imagen tipo OpenPose: joints (circulos)
    + bones (lineas). Usa centroides de segmentos DP como aproximacion de
    joints anatomicos.
    """
    canvas = Image.new("RGB", (ancho, alto), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    h_src, w_src = I_pred.shape
    centroides_op = {}

    for dp_id in range(1, MAX_PART_ID + 1):
        mask_part = (I_pred == dp_id)
        if not mask_part.any():
            continue
        if dp_id not in DP_TO_OPENPOSE:
            continue
        joint_op = DP_TO_OPENPOSE[dp_id]
        ys, xs = np.where(mask_part)
        cy = float(ys.mean()) / h_src * alto
        cx = float(xs.mean()) / w_src * ancho
        centroides_op.setdefault(joint_op, []).append((cx, cy))

    joints_xy = {}
    for joint_op, lista in centroides_op.items():
        xs = [c[0] for c in lista]
        ys = [c[1] for c in lista]
        joints_xy[joint_op] = (sum(xs) / len(xs), sum(ys) / len(ys))

    for idx, (j1, j2) in enumerate(OPENPOSE_BONES):
        if j1 in joints_xy and j2 in joints_xy:
            color = OPENPOSE_BONE_COLORS[idx % len(OPENPOSE_BONE_COLORS)]
            x1, y1 = joints_xy[j1]
            x2, y2 = joints_xy[j2]
            draw.line([(x1, y1), (x2, y2)], fill=color, width=4)

    for joint_op, (cx, cy) in joints_xy.items():
        color = OPENPOSE_JOINT_COLORS[joint_op % len(OPENPOSE_JOINT_COLORS)]
        r = 4
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=color)

    return canvas


def cargar_pipeline_multi(device):
    log.info(f"Cargando ControlNet-Seg: {MODELO_CN_SEG}")
    controlnet_seg = ControlNetModel.from_pretrained(MODELO_CN_SEG, torch_dtype=torch.float16)

    log.info(f"Cargando ControlNet-OpenPose: {MODELO_CN_POSE}")
    controlnet_pose = ControlNetModel.from_pretrained(MODELO_CN_POSE, torch_dtype=torch.float16)

    log.info(f"Cargando SD inpainting con Multi-ControlNet (Seg + Pose): {MODELO_SD}")
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        MODELO_SD,
        controlnet=[controlnet_seg, controlnet_pose],   #lista = MultiControlNet
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    return pipe


def redimensionar_para_sd(img, mask, tamano_largo):
    w, h = img.size

    if w >= h:
        new_w = tamano_largo
        new_h = int(h * tamano_largo / w)
    else:
        new_h = tamano_largo
        new_w = int(w * tamano_largo / h)
    new_w = max(8, (new_w // 8) * 8)
    new_h = max(8, (new_h // 8) * 8)
    img_r = img.resize((new_w, new_h), Image.BILINEAR)
    mask_r = mask.resize((new_w, new_h), Image.NEAREST)
    return img_r, mask_r, new_w, new_h

def cargar_cond_densepose(stem):
    cond_path = DIR_MASCARAS / f"{stem}_cond.npz"
    if not cond_path.exists():
        return None
    d = np.load(cond_path)
    if "I_pred" not in d.files:
        return None
    return d["I_pred"].astype(np.int32)

#-----------------------------------------------------------------
def main():
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if device == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}  compute: {torch.cuda.get_device_capability(0)}")

    if not DIR_IMAGENES.exists() or not DIR_MASCARAS.exists():
        log.error("missing input dirs")
        return

    extensiones = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    imagenes = sorted([f for f in DIR_IMAGENES.iterdir() if f.suffix in extensiones])
    log.info(f"Images to process: {len(imagenes)}")

    pipe = cargar_pipeline_multi(device)
    log.info(f"ControlNet conditioning scales: seg={CONTROLNET_SCALE_SEG}, pose={CONTROLNET_SCALE_POSE}")

    procesadas, sin_mask, sin_cond, errores = 0, 0, 0, 0

    for img_path in tqdm(imagenes, desc="SD+MultiCN"):
        try:
            stem_limpio = img_path.stem
            for _ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                if stem_limpio.endswith(_ext):
                    stem_limpio = stem_limpio[:-len(_ext)]
                    break

            salida_path = DIR_SALIDA / (stem_limpio + "_sdcn.png")
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

            img_r, mask_r, new_w, new_h = redimensionar_para_sd(img, mask, TAMANO_LARGO)

            I_pred = cargar_cond_densepose(stem_limpio)
            if I_pred is None:
                sin_cond += 1
                #fallback: control images vacias (negras)
                seg_img = Image.new("RGB", (new_w, new_h), (0, 0, 0))
                pose_img = Image.new("RGB", (new_w, new_h), (0, 0, 0))
            else:
                seg_img = densepose_a_seg_image(I_pred, new_w, new_h)
                pose_img = densepose_a_openpose_image(I_pred, new_w, new_h)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = pipe(
                    prompt=PROMPT,
                    negative_prompt=NEGATIVE_PROMPT,
                    image=img_r,
                    mask_image=mask_r,
                    control_image=[seg_img, pose_img],
                    controlnet_conditioning_scale=[CONTROLNET_SCALE_SEG, CONTROLNET_SCALE_POSE],
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                ).images[0]

            output_resized = output.resize(tamano_original, Image.BILINEAR)
            output_resized.save(salida_path)
            procesadas += 1

        except Exception as e:
            errores += 1
            log.error(f"Error en {img_path.name}: {e}")

    log.info("SD + MULTI-CONTROLNET (SEG + POSE) COMPLETED")
    log.info(f"Processed: {procesadas}")
    log.info(f"Skipped (no mask): {sin_mask}")
    log.info(f"Sin DP cond (control vacio): {sin_cond}")
    log.info(f"Errors: {errores}")
    log.info(f"Output at: {DIR_SALIDA}")


if __name__ == "__main__":
    main()
