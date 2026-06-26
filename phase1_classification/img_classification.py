#Clasifica automaticamente imagenes de esculturas clasicas usando el modelo DeepLabv3+ fine-tuneado.
#Categorias:
#- whole_body -> Dominio B del CycleGAN. Todas las partes esperadas segun la region visible estan presentes (la cabeza puede faltar siempre).
#- broken_body -> Dominio A del CycleGAN. Partes humanas detectadas pero falta al menos una de las esperadas segun la region visible.
#- head_only -> Solo zona cabeza: Head(14) y/o Upper_Arm_L(10)/Upper_Arm_R(11), sin torso ni piernas.
#              Tambien bustos: torso presente pero sin ningun brazo ni pierna detectados (pecho/hombros visibles pero sin extremidades)
#              O brazos detectados pero con muy pocos pixeles (pliegues de tela confundidos con brazo)
#- no_human -> Sin partes corporales detectadas. Se descarta.
#NOTA: para las imagenes clasificadas como broken_body se guarda ademas la mascara de segmentacion
#de DeepLabv3+ en DIR_MASCARAS_CLASIFICACION --> se usa en la 3a pasada de clasificacion para
#detectar partes corporales cubiertas por ropa (misma tonalidad que la escultura pero clase 0)

import json
import logging
import shutil
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import (deeplabv3_resnet50,DeepLabV3_ResNet50_Weights,)


#CONFIGURACION:
PESOS_MODELO = Path("/home/pfc/cescuder/tfg/deeplabv3_esculturas_full.pth")

DIR_SALIDA = Path("/home/pfc/cescuder/tfg/dataset_classificado")

#carpeta donde se guardan las mascaras de segmentacion de DeepLabv3+ para imagenes broken_body
#--> se usan en la 3a pasada para detectar partes cubiertas por ropa

INPUT_DIRS = [
    "/home/pfc/cescuder/tfg/dataset_esculturas/sculpture_training",
    "/home/pfc/cescuder/tfg/dataset_esculturas/sculpture_validation",
    "/home/pfc/cescuder/tfg/dataset_esculturas/archive_2/images/apollo",
    "/home/pfc/cescuder/tfg/dataset_esculturas/archive_2/images/hera",
    "/home/pfc/cescuder/tfg/dataset_esculturas/archive_2/images/athena",
    "/home/pfc/cescuder/tfg/dataset_esculturas/archive_2/images/hermes",
    "/home/pfc/cescuder/tfg/dataset_esculturas/archive_2/images/hestia",
    "/home/pfc/cescuder/tfg/dataset_esculturas/archive_2/images/poseidon",
    "/home/pfc/cescuder/tfg/dataset_esculturas/archive_2/images/zeus",
    "/home/pfc/cescuder/tfg/dataset_esculturas/greek_sculptures",
    "/home/pfc/cescuder/tfg/dataset_esculturas/met_sculptures",]

NUM_CLASES = 15
TAMANO_IMG  = 520
UMBRAL_PRESENCIA = 0.002
UMBRAL_CONFIANZA = 0.2
#UMBRAL CONFIANZSA = DeepLabv3+ produce para cada pixel una distribución de probabilidades sobre las 15 clases (softmax)
# --> Si el valor maximo de esa distribución para un píxel es menor que 0.4, ese píxel se fuerza a clase 0 (background), aunque el modelo haya predicho algo

# CLASES Y GRUPOS

CLASES = {0:"Background", 1:"Torso", 2:"Right_Hand", 3:"Left_Hand",
    4:"Left_Foot", 5:"Right_Foot", 6:"Upper_Leg_Right", 7:"Upper_Leg_Left",
    8:"Lower_Leg_Right", 9:"Lower_Leg_Left", 10:"Upper_Arm_Left",
    11:"Upper_Arm_Right", 12:"Lower_Arm_Left", 13:"Lower_Arm_Right", 14:"Head",}

# Zona cabeza: cabeza + hombros (sin torso ni piernas)
GRUPO_ZONA_CABEZA_AMPLIADA = {14, 10, 11}

# Partes de piernas
GRUPO_PIERNAS = {4, 5, 6, 7, 8, 9}  # pies + upper legs + lower legs

# Partes de brazos (upper arms + lower arms + manos)
GRUPO_BRAZOS = {2, 3, 10, 11, 12, 13}

#umbral minimo de pixeles de brazos para considerarlos realmente presentes:
#--> si los pixeles de brazos son menos del 2% de la imagen se ignoran
#--> evita que pliegues de tela o hombros confundidos con brazo manden bustos a broken_body
UMBRAL_PRESENCIA_BRAZOS = 0.05

# Partes esperadas si hay cuerpo completo visible (piernas presentes)
# La cabeza (14) puede faltar siempre → no se incluye como obligatoria
PARTES_ESPERADAS_CUERPO_COMPLETO = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}

# Partes esperadas si solo hay torso superior visible (sin piernas)
# Torso + brazos + manos. La cabeza puede faltar siempre → no se incluye
PARTES_ESPERADAS_TORSO_SUPERIOR = {1, 2, 3, 10, 11, 12, 13}

# whole_body original (las 14 partes) — ya no se usa como criterio principal
TODAS_LAS_PARTES = set(range(1, 15))


# LOGGING:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("/home/pfc/cescuder/tfg/logs/clasificar_esculturas.log", encoding="utf-8"),])
log = logging.getLogger(__name__)


#MODELO:
def cargar_modelo(pesos_path, device):
    log.info("gettig DeepLabv3+ fine-tuned model!!!!!!!!!!")
    modelo = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    modelo.classifier[4] = nn.Conv2d(256, NUM_CLASES, kernel_size=1)
    if modelo.aux_classifier is not None:
        modelo.aux_classifier[4] = nn.Conv2d(256, NUM_CLASES, kernel_size=1)
    checkpoint = torch.load(pesos_path, map_location=device)
    if "state_dict" in checkpoint:
        modelo.load_state_dict(checkpoint["state_dict"])
        log.info(f"  Epoch: {checkpoint.get('epoca','?')}  --  mIoU: {checkpoint.get('miou_val',0.0):.4f}")
    else:
        modelo.load_state_dict(checkpoint)
    modelo = modelo.to(device)
    modelo.eval()
    log.info("model downloaded")
    return modelo


#PRE-PROCESADO:
_normalizar = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

def preprocesar_imagen(img_path, tamano):
    img = Image.open(img_path).convert("RGB")
    w_orig, h_orig = img.size
    img_r = TF.resize(img, [tamano, tamano], interpolation=Image.BILINEAR)
    t = _normalizar(TF.to_tensor(img_r))
    return t.unsqueeze(0), (w_orig, h_orig)


#INFERENCIA:
def inferir_partes(modelo, tensor_img, device, tamano_original):
    with torch.no_grad():
        salida = modelo(tensor_img.to(device))["out"]
        probs = torch.softmax(salida, dim=1)
        max_probs, pred = probs.max(dim=1)

    pred_np = pred[0].cpu().numpy().astype(np.uint8)
    max_prob_np = max_probs[0].cpu().numpy()
    pred_np[max_prob_np < UMBRAL_CONFIANZA] = 0

    w_orig, h_orig = tamano_original
    mascara_orig = np.array(Image.fromarray(pred_np).resize((w_orig, h_orig), Image.NEAREST))

    total = w_orig * h_orig
    partes_presentes = set()
    for c in range(1, NUM_CLASES):
        if int((mascara_orig == c).sum()) / total >= UMBRAL_PRESENCIA:
            partes_presentes.add(c)

    #calcular ratio de pixeles de brazos sobre el total --> para detectar bustos con pliegues de tela
    ratio_brazos = sum(int((mascara_orig == c).sum()) for c in GRUPO_BRAZOS) / total

    ratio_humano = float((mascara_orig > 0).sum()) / total
    partes_faltantes = TODAS_LAS_PARTES - partes_presentes
    #devuelve tambien mascara_orig para guardarla en disco para las broken_body
    return partes_presentes, ratio_humano, partes_faltantes, mascara_orig, ratio_brazos


#CLASIFICACION:
def clasificar_escultura(partes_presentes, ratio_humano, ratio_brazos):
    # Sin deteccion humana -> no_human
    if ratio_humano < 0.02 or not partes_presentes:
        return "no_human"

    # Solo cabeza/hombros sin torso ni piernas -> head_only
    partes_sin_cabeza = partes_presentes - GRUPO_ZONA_CABEZA_AMPLIADA
    if not partes_sin_cabeza:
        return "head_only"

    # Busto: torso presente pero sin ningun brazo ni pierna detectados
    # --> es un busto (cabeza + pecho/hombros) aunque DeepLabv3+ detecte algo de torso
    # --> sin extremidades no tiene sentido tratarlo como broken_body
    hay_piernas = bool(partes_presentes & GRUPO_PIERNAS)
    hay_brazos  = bool(partes_presentes & GRUPO_BRAZOS) and ratio_brazos >= UMBRAL_PRESENCIA_BRAZOS
    #--> si los pixeles de brazos son menos del 2% de la imagen se ignoran (pliegues de tela)
    if not hay_piernas and not hay_brazos:
        return "head_only"

    # Determinar region visible y partes esperadas segun si hay piernas o no
    if hay_piernas:
        # Cuerpo completo visible: se esperan todas las partes excepto cabeza
        partes_esperadas = PARTES_ESPERADAS_CUERPO_COMPLETO
    else:
        # Solo torso superior visible: se esperan torso + brazos + manos (sin cabeza, sin piernas)
        partes_esperadas = PARTES_ESPERADAS_TORSO_SUPERIOR

    # Comprobar si estan todas las partes esperadas
    # La cabeza (14) puede faltar siempre sin penalizar
    partes_faltantes_esperadas = partes_esperadas - partes_presentes

    if not partes_faltantes_esperadas:
        return "whole_body"
    else:
        return "broken_body"


#RECOPILACION:
def recopilar_imagenes(input_dirs):
    ext    = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG",".webp"}
    rutas  = []
    vistas = set()
    for d in input_dirs:
        p = Path(d)
        if not p.exists():
            log.warning(f"directory not found: {p}")
            continue
        for f in p.iterdir():
            if f.suffix in ext and f.name not in vistas:
                #comparar por nombre de archivo (no por ruta completa) para evitar duplicados
                #de la misma imagen descargada desde distintas fuentes con el mismo nombre
                rutas.append(f)
                vistas.add(f.name)
    log.info(f"total images found: {len(rutas)}")
    return rutas

#MAIN:

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    carpetas = {
        "whole_body": DIR_SALIDA / "whole_body",
        "broken_body": DIR_SALIDA / "broken_body",
        "head_only": DIR_SALIDA / "head_only",
        "no_human": DIR_SALIDA / "no_human",
    }
    for c in carpetas.values():
        c.mkdir(parents=True, exist_ok=True)

    if not PESOS_MODELO.exists():
        log.error(f"No weights found: {PESOS_MODELO}")
        return

    #construir set de imagenes ya clasificadas (checkpoint implicito)
    ya_procesadas = set()
    for cat, carpeta in carpetas.items():
        for f in carpeta.iterdir():
            ya_procesadas.add(f.name)

    modelo = cargar_modelo(PESOS_MODELO, device)
    rutas = recopilar_imagenes(INPUT_DIRS)
    if not rutas:
        log.error("there's no images")
        return

    log.info(f"Already classified (skipping): {len(ya_procesadas)}")

    cont = {"whole_body":0,"broken_body":0,"head_only":0,"no_human":0} #contador
    reg = []
    errores = 0

    for img_path in tqdm(rutas, desc="Clasificando"):
        try:
            if img_path.name in ya_procesadas:
                continue

            tensor, tam  = preprocesar_imagen(img_path, TAMANO_IMG)
            partes, ratio, faltantes, mascara_orig, ratio_brazos = inferir_partes(modelo, tensor, device, tam)
            cat = clasificar_escultura(partes, ratio, ratio_brazos)
            cont[cat] += 1

            shutil.copy2(img_path, carpetas[cat] / img_path.name)


            reg.append({
                "archivo": str(img_path),
                "nombre": img_path.name,
                "categoria": cat,
                "ratio_humano": round(ratio, 4),
                "partes_detectadas": sorted([CLASES[p] for p in partes]),
                "partes_faltantes": sorted([CLASES[f] for f in faltantes]),
            })
        except Exception as e:
            log.warning(f"Error en {img_path.name}: {e}")
            errores += 1

    resumen = {
        "total_processed": len(rutas),
        "errors": errores,
        "counters": cont,
        "criteria": {
            "whole_body": "Todas las partes esperadas segun region visible presentes (cabeza opcional siempre)",
            "broken_body": "Partes humanas detectadas pero falta al menos una de las esperadas segun region visible",
            "head_only": "Solo zona cabeza ampliada o busto sin extremidades (Head/Upper_Arms o torso sin brazos ni piernas)",
            "no_human": "ratio_humano < 0.02 or no part detected",
        },
        "parametros": {
            "umbral_presencia": UMBRAL_PRESENCIA,
            "umbral_confianza": UMBRAL_CONFIANZA,
            "umbral_presencia_brazos": UMBRAL_PRESENCIA_BRAZOS,
            "tamano_inferencia": TAMANO_IMG,
        },
        "imagenes": reg,
    }
    json_path = DIR_SALIDA / "clasificacion.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)


    log.info(f"whole_body: {cont['whole_body']} -> for domain B CycleGAN")
    log.info(f"broken_body: {cont['broken_body']} -> for domain A CycleGAN")
    log.info(f"head_only: {cont['head_only']}")
    log.info(f"no_human: {cont['no_human']}  -> discarted")




if __name__ == "__main__":
    main()
