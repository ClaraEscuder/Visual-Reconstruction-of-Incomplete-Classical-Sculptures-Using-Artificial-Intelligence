"""
Fine-tuning de DeepLabv3+ con ResNet50 para segmentación de partes corporales en esculturas de mármol (DE LOS ESTILOS QUE TENGA ZEUS).

Usa el dataset sintético generado por style_transfer.py:
-imgs = personas con aspecto de escultura de mármol (que vienen del style_transfer.py)
-masks = 15 clases (0=fondo, 1-14=partes corporales DensePose)

Clases (14 partes + fondo):
    0 Background
    1 Torso
    2 Right Hand
    3 Left Hand
    4 Left Foot
    5 Right Foot
    6 Upper Leg Right
    7 Upper Leg Left
    8 Lower Leg Right
    9 Lower Leg Left
    10 Upper Arm Left
    11 Upper Arm Right
    12 Lower Arm Left
    13 Lower Arm Right
    14 Head

Los pesos finales se guardan para usarlos en el pipeline de detección de partes faltantes y generación de máscaras de dps!!!!
VERSION FULL: usa todos los pares imagen/mascara disponibles (sin limite de 8000).
"""

import logging
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import (deeplabv3_resnet50,DeepLabV3_ResNet50_Weights,)


BASE = Path("/home/pfc/cescuder/tfg")

DIR_IMAGENES = BASE / "synthetic_dataset" / "images"
DIR_MASCARAS = BASE / "synthetic_dataset" / "masks"

PESOS_SALIDA = BASE / "deeplabv3_esculturas_full.pth"
CHECKPOINT_PATH = BASE / "logs" / "checkpoint_finetune_full.pth"

NUM_CLASES = 15 #0 (fondo) + 14 partes corporales
TAMANO_IMG = 520  #tamaño de entrada al modelo
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4 #learning rate bajo para fine-tuning (no destruir pesos preentrenados)
VAL_SPLIT = 0.15 #15% para validación
NUM_WORKERS = 4 #0 (pq estoy en Windows) para evitar problemas con multiprocessing


#LOGGING:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(BASE / "logs" / "finetune_deeplabv3_full.log", encoding="utf-8"),])
log = logging.getLogger(__name__)



#DATASET------------------------------------------------------------------------------------

class EsculturasDataset(Dataset):
    """
    Dataset para fine-tuning: pares (imagen estilizada, máscara de partes).

    Aplica augmentaciones completas en tiempo de entrenamiento:
    - Augmentaciones de color (solo imagen): blanco/negro, saturación,brillo, contraste, ruido, blur y combinaciones.
    - Augmentaciones geométricas (imagen Y máscara): flip, rotación, zoom.
      Las geométricas siempre se aplican de forma idéntica a ambas.
      El flip horizontal intercambia además las partes izq/der en la máscara.
    """

    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    #pares laterales DensePose que se intercambian al hacer flip horizontal
    PARES_LATERALES = [(2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]

    def __init__(self, pares: list, tamano: int, augmentar: bool = True):
        self.pares = pares
        self.tamano = tamano
        self.augmentar = augmentar
        self.norm = T.Normalize(mean=self.NORM_MEAN, std=self.NORM_STD)

    def __len__(self):
        return len(self.pares)

    def __getitem__(self, idx):
        img_path, mask_path = self.pares[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)  #valores 0-14, NO convertir a RGB

        #redimensionar (siempre, augmente o no)
        img = TF.resize(img,  [self.tamano, self.tamano], interpolation=Image.BILINEAR)
        mask = TF.resize(mask, [self.tamano, self.tamano], interpolation=Image.NEAREST)

        if self.augmentar:
            img, mask = self._aplicar_augmentaciones(img, mask)

        #convertir a tensores
        img_t = TF.to_tensor(img)
        img_t = self.norm(img_t)
        mask_t = torch.from_numpy(np.array(mask)).long()
        mask_t = mask_t.clamp(0, NUM_CLASES - 1)

        return img_t, mask_t

    def _aplicar_augmentaciones(self, img, mask):

        #aplica augmentaciones aleatorias
        #las de color solo modifican img
        #las geometricas modifican img Y mask de forma identica

        from PIL import ImageFilter, ImageEnhance

        #AUGMENTACIONES DE COLOR (probabilidad independiente cada una):

        #blanco y negro / desaturación
        if torch.rand(1) < 0.3:
            img = img.convert("L").convert("RGB")
        elif torch.rand(1) < 0.4:
            factor = float(torch.empty(1).uniform_(0.1, 0.6))
            img = ImageEnhance.Color(img).enhance(factor)

        #brillo
        if torch.rand(1) < 0.4:
            factor = float(torch.empty(1).uniform_(0.5, 1.6))
            img = ImageEnhance.Brightness(img).enhance(factor)

        #contraste
        if torch.rand(1) < 0.4:
            factor = float(torch.empty(1).uniform_(0.5, 1.8))
            img = ImageEnhance.Contrast(img).enhance(factor)

        #ruido gaussiano (simula textura granular de piedra)
        if torch.rand(1) < 0.3:
            intensidad = float(torch.empty(1).uniform_(10, 35))
            arr = np.array(img, dtype=np.float32)
            arr = np.clip(arr + np.random.normal(0, intensidad, arr.shape), 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        #blur suave (fotos de esculturas a veces desenfocadas)
        if torch.rand(1) < 0.25:
            radio = float(torch.empty(1).uniform_(0.5, 2.5))
            img = img.filter(ImageFilter.GaussianBlur(radius=radio))

        #AUGMENTACIONES GEOMÉTRICAS (imagen + máscara juntas):

        #flip horizontal
        if torch.rand(1) < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
            mask = self._intercambiar_lados(mask)

        #rotación aleatoria (esculturas fotografiadas en ángulos variados)
        if torch.rand(1) < 0.5:
            angulo = float(torch.empty(1).uniform_(-25, 25))
            img = TF.rotate(img,  angulo, fill=128, interpolation=Image.BILINEAR)
            mask = TF.rotate(mask, angulo, fill=0,   interpolation=Image.NEAREST)

        #zoom / crop aleatorio
        if torch.rand(1) < 0.35:
            factor = float(torch.empty(1).uniform_(0.78, 0.95))
            w, h = img.size
            cw, ch = int(w * factor), int(h * factor)
            l = random.randint(0, w - cw)
            t = random.randint(0, h - ch)
            img = img.crop((l, t, l + cw, t + ch)).resize((w, h), Image.LANCZOS)
            mask = mask.crop((l, t, l + cw, t + ch)).resize((w, h), Image.NEAREST)

        return img, mask

    @staticmethod
    def _intercambiar_lados(mask: Image.Image) -> Image.Image:
        """
        Al hacer flip horizontal, las partes izquierda/derecha se invierten.
        Intercambia los valores de los pares laterales en la máscara para
        que la anotación siga siendo correcta tras el espejo.

        Pares DensePose (right <-> left):
        -Right Hand (2) <-> Left Hand (3)
        -Left Foot (4) <-> Right Foot (5)
        -Upper Leg Right (6) <-> Upper Leg Left (7)
        -Lower Leg Right (8) <-> Lower Leg Left (9)
        -Upper Arm Left (10)<-> Upper Arm Right (11)
        -Lower Arm Left (12) <-> Lower Arm Right (13)
        """
        m = np.array(mask, dtype=np.uint8)
        m_nuevo = m.copy()
        for a, b in EsculturasDataset.PARES_LATERALES:
            m_nuevo[m == a] = b
            m_nuevo[m == b] = a
        return Image.fromarray(m_nuevo)


#ENTRENAMIENTO

def calcular_miou(pred: torch.Tensor, target: torch.Tensor, num_clases: int) -> float:
    #calcula mean Intersection over Union (mIoU) ignorando el fondo
    ious = []
    pred_flat   = pred.view(-1)
    target_flat = target.view(-1)

    for cls in range(1, num_clases):  # ignorar clase 0 (fondo)
        pred_cls = pred_flat == cls
        target_cls = target_flat == cls
        intersec = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union > 0:
            ious.append((intersec / union).item())
    return float(np.mean(ious)) if ious else 0.0


def entrenar_epoca(modelo, loader, optimizer, criterion, device):
    modelo.train()
    total_loss = 0.0
    n = 0

    for imgs, masks in tqdm(loader, desc="  Train", leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        salida = modelo(imgs)["out"]
        loss = criterion(salida, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / n if n > 0 else 0.0


def validar_epoca(modelo, loader, criterion, device, num_clases):
    modelo.eval()
    total_loss = 0.0
    total_miou = 0.0
    n = 0

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="  Val  ", leave=False):
            imgs = imgs.to(device)
            masks = masks.to(device)

            salida = modelo(imgs)["out"]
            loss = criterion(salida, masks)
            pred = salida.argmax(dim=1)

            total_loss += loss.item()
            total_miou += calcular_miou(pred.cpu(), masks.cpu(), num_clases)
            n += 1

    return (total_loss / n if n > 0 else 0.0,
            total_miou / n if n > 0 else 0.0)



#MAIN:----------------------------------------

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")


    #1.Construir lista de pares (imagen, mask):
    extensiones = {".jpg", ".jpeg", ".png"}
    imagenes = sorted([
        p for p in DIR_IMAGENES.glob("*")
        if p.suffix.lower() in extensiones
    ])

    pares = []
    for img_path in imagenes:
        mask_path = DIR_MASCARAS / (img_path.stem + ".png")
        if mask_path.exists():
            pares.append((img_path, mask_path))
    # SIN LIMITE - usa todos los pares disponibles

    if not pares:
        log.error(f"There was no pairs image/mask in {DIR_IMAGENES} -- {DIR_MASCARAS}")
        return

    log.info(f"Image/mask pairs found (ALL): {len(pares)}")


    # 2. Split train / validación:
    n_val = max(1, int(len(pares) * VAL_SPLIT))
    n_train = len(pares) - n_val

    ds_train = EsculturasDataset(pares[:n_train], TAMANO_IMG, augmentar=True)
    ds_val = EsculturasDataset(pares[n_train:], TAMANO_IMG, augmentar=False)

    log.info(f"Train: {len(ds_train)} -- Val: {len(ds_val)}")

    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,num_workers=NUM_WORKERS, pin_memory=(device == "cuda"), drop_last=True)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS, pin_memory=(device == "cuda"))


    # 3. Cargar DeepLabv3+ preentrenado y adaptar el head:
    log.info("Getting Deeplabv3+ with pretrained weights")
    modelo = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

    # Reemplazar el clasificador final para NUM_CLASES clases (el preentrenado tiene 21 clases de Pascal VOC)
    modelo.classifier[4] = nn.Conv2d(256, NUM_CLASES, kernel_size=1)
    if modelo.aux_classifier is not None:
        modelo.aux_classifier[4] = nn.Conv2d(256, NUM_CLASES, kernel_size=1)

    modelo = modelo.to(device)

    # 4. Optimizador con learning rates diferenciados:
    #backbone (ResNet50) ya está preentrenado: lr muy bajo!!!
    #cabezas de clasificación son nuevas: lr más alto!!!
    params_backbone = [p for n, p in modelo.named_parameters() if "backbone" in n]
    params_cabeza   = [p for n, p in modelo.named_parameters() if "backbone" not in n]

    optimizer = optim.Adam([{"params": params_backbone, "lr": LR * 0.1}, {"params": params_cabeza,   "lr": LR},])

    # Scheduler: reduce lr cuando la validación no mejora
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    criterion = nn.CrossEntropyLoss(ignore_index=255) #LOSS FUNCITON --> CrossEntropy ignorando píxeles con valor 255


    # 5. Cargar checkpoint si existe (para continuar entrenamiento previo):
    mejor_miou = 0.0
    historial = []
    epoca_inicio = 1

    if CHECKPOINT_PATH.exists():
        log.info(f"Checkpoint encontrado, continuando desde: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        modelo.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        epoca_inicio = ckpt["epoca"] + 1
        mejor_miou = ckpt["mejor_miou"]
        historial = ckpt["historial"]
        log.info(f"Continuando desde época {epoca_inicio} -- mejor mIoU hasta ahora: {mejor_miou:.4f}")
    else:
        log.info("No se encontró checkpoint, empezando desde cero")


    # 6. Bucle de entrenamiento:
    log.info(f"iniciando fine-tuning CON {EPOCHS} épocas")

    for epoca in range(epoca_inicio, EPOCHS + 1):
        log.info(f"\nepoch {epoca}/{EPOCHS}")

        loss_train = entrenar_epoca(modelo, loader_train, optimizer, criterion, device)
        loss_val, miou_val = validar_epoca(modelo, loader_val, criterion, device, NUM_CLASES)

        scheduler.step(loss_val)

        log.info(f"  Loss train: {loss_train:.4f}  |  "
                 f"Loss val: {loss_val:.4f}  -- mIoU val: {miou_val:.4f}")

        historial.append({
            "epoca": epoca,
            "loss_train": loss_train,
            "loss_val": loss_val,
            "miou_val": miou_val,
        })

        #Guardar mejor modelo:
        if miou_val > mejor_miou:
            mejor_miou = miou_val
            torch.save({
                "epoca": epoca,
                "state_dict": modelo.state_dict(),
                "miou_val": miou_val,
                "num_clases": NUM_CLASES,
                "clases": {
                    "0": "Background",
                    "1": "Torso",
                    "2": "Right_Hand",
                    "3": "Left_Hand",
                    "4": "Left_Foot",
                    "5": "Right_Foot",
                    "6": "Upper_Leg_Right",
                    "7": "Upper_Leg_Left",
                    "8": "Lower_Leg_Right",
                    "9": "Lower_Leg_Left",
                    "10": "Upper_Arm_Left",
                    "11": "Upper_Arm_Right",
                    "12": "Lower_Arm_Left",
                    "13": "Lower_Arm_Right",
                    "14": "Head",
                },
            }, PESOS_SALIDA)
            log.info(f"mejor modelo guardado (mIoU={mejor_miou:.4f}): {PESOS_SALIDA}")

        #Guardar checkpoint al final de cada época:
        torch.save({
            "epoca": epoca,
            "state_dict": modelo.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "mejor_miou": mejor_miou,
            "historial": historial,
        }, CHECKPOINT_PATH)


    # 7. Guardar historial de entrenamiento:
    hist_path = BASE / "logs" / "historial_entrenamiento_full.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(historial, f, indent=2)


    log.info("FINE-TUNING FULL COMPLETED!!!!!!!")
    log.info(f"Best mIoU validation: {mejor_miou:.4f}")
    log.info(f"Weights saved at: {PESOS_SALIDA}")


if __name__ == "__main__":
    main()