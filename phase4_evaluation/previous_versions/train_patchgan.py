#Entrena el discriminador PatchGAN del framework CycleGAN para evaluar la calidad visual
#de las reconstrucciones generadas por LaMa/MAT
#
#LOGICA DE ENTRENAMIENTO (fase unica):
#  positivos = whole_body reales (esculturas que el pipeline considera completas)
#  negativos = reconstrucciones LaMa/MAT (salida del generador sobre broken_body)
#  --> no se usa broken_body sin reconstruir como negativos porque eso entrena al discriminador
#      a detectar esculturas rotas, no a evaluar la calidad de una reconstruccion generada por IA
#      que es la tarea real que necesitamos
#
#PatchGAN clasifica parches NxN de la imagen (no la imagen entera):
#  --> mas sensible a la calidad local de textura, que es donde se notan las malas reconstrucciones
#  --> converge con pocos datos de entrenamiento, adecuado para el tamaño de nuestro dataset
#
#INPUT:
#  - positivos (esculturas completas reales): ~/tfg/background_removed/whole_body/
#  - negativos (reconstrucciones LaMa o MAT): ~/tfg/inpainting_results/lama/  (o mat/)
#OUTPUT:
#  - pesos del discriminador: ~/tfg/patchgan_discriminador.pth
#  - historial de entrenamiento: ~/tfg/logs/historial_patchgan.json

import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

DIR_POSITIVOS  = Path("/home/pfc/cescuder/tfg/background_removed/whole_body")
#directorio con las reconstrucciones del generador (lama o mat)
#cambiar a /mat/ si se usa MAT como generador final
DIR_INPAINTING = Path("/home/pfc/cescuder/tfg/inpainting_results/lama")
DIR_PESOS = Path("/home/pfc/cescuder/tfg/patchgan_discriminador.pth")
CHECKPOINT_PATH = Path("/home/pfc/cescuder/tfg/logs/checkpoint_patchgan.pth")

TAMANO_IMG = 256   #PatchGAN trabaja bien con 256x256
BATCH_SIZE = 8
LR = 2e-4  #learning rate estandar para discriminadores GAN (Adam)
BETA1 = 0.5 #beta1 estandar para Adam en GANs
EPOCHS = 30 #epocas de entrenamiento con reconstrucciones como negativos


#LOGGING:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("/home/pfc/cescuder/tfg/logs/train_patchgan.log", encoding="utf-8"),])
log = logging.getLogger(__name__)


#ARQUITECTURA PATCHGAN:
class BloquePatchGAN(nn.Module):
    #bloque convolucional estandar de PatchGAN: Conv -> (BatchNorm) -> LeakyReLU
    def __init__(self, in_ch, out_ch, stride=2, normalizar=True):
        super().__init__()
        capas = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=not normalizar)]
        if normalizar:
            capas.append(nn.BatchNorm2d(out_ch))
        capas.append(nn.LeakyReLU(0.2, inplace=True))
        self.bloque = nn.Sequential(*capas)

    def forward(self, x):
        return self.bloque(x)


class PatchGAN(nn.Module):
    #discriminador PatchGAN de 70x70 pixeles (arquitectura estandar de CycleGAN/pix2pix)
    #entrada: imagen RGB (3 canales)
    #salida: mapa de parches con scores de autenticidad (1=real, 0=falso)
    def __init__(self, in_channels=3):
        super().__init__()
        self.modelo = nn.Sequential(
            BloquePatchGAN(in_channels, 64,  stride=2, normalizar=False),   # 128x128
            BloquePatchGAN(64,128, stride=2, normalizar=True),    # 64x64
            BloquePatchGAN(128,256, stride=2, normalizar=True),    # 32x32
            BloquePatchGAN(256,512, stride=1, normalizar=True),    # 32x32
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),  # mapa de patchs
        )

    def forward(self, x):
        return self.modelo(x)


#DATASET:
class DatasetEsculturas(Dataset):
    #dataset que carga imagenes de dos carpetas: positivos (reales) y negativos (falsos)
    #etiqueta 1.0 para positivos (whole_body reales), 0.0 para negativos

    def __init__(self, dir_positivos: Path, dir_negativos: Path, tamano: int):
        extensiones = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".webp"}

        self.positivos = [f for f in dir_positivos.iterdir() if f.suffix in extensiones]
        self.negativos = [f for f in dir_negativos.iterdir() if f.suffix in extensiones]

        #equilibrar el numero de positivos y negativos para evitar sesgo de clase
        n = min(len(self.positivos), len(self.negativos))
        self.positivos = self.positivos[:n]
        self.negativos = self.negativos[:n]

        self.imagenes = [(p, 1.0) for p in self.positivos] + [(n, 0.0) for n in self.negativos]

        self.transform = T.Compose([
            T.Resize((tamano, tamano)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1]
        ])

        log.info(f"Dataset: {len(self.positivos)} positivos + {len(self.negativos)} negativos = {len(self.imagenes)} total")

    def __len__(self):
        return len(self.imagenes)

    def __getitem__(self, idx):
        ruta, etiqueta = self.imagenes[idx]
        img = Image.open(ruta).convert("RGB")
        return self.transform(img), torch.tensor(etiqueta, dtype=torch.float32)


#ENTRENAMIENTO DE UNA EPOCA:
def entrenar_epoca(modelo, loader, optimizador, criterio, device, epoca, total_epocas):
    modelo.train()
    losses = []

    for imgs, etiquetas in tqdm(loader, desc=f"Epoch {epoca}/{total_epocas}", leave=False):
        imgs = imgs.to(device)
        etiquetas = etiquetas.to(device)

        optimizador.zero_grad()
        predicciones = modelo(imgs)

        #expandir etiquetas al tamaño del mapa de parches de PatchGAN
        etiquetas_exp = etiquetas.view(-1, 1, 1, 1).expand_as(predicciones)
        loss = criterio(predicciones, etiquetas_exp)
        loss.backward()
        optimizador.step()
        losses.append(loss.item())

    return float(np.mean(losses))


#VALIDACION:
def validar(modelo, loader, criterio, device):
    modelo.eval()
    losses = []
    correctas = 0
    total = 0

    with torch.no_grad():
        for imgs, etiquetas in loader:
            imgs = imgs.to(device)
            etiquetas = etiquetas.to(device)
            predicciones = modelo(imgs)
            etiquetas_exp = etiquetas.view(-1, 1, 1, 1).expand_as(predicciones)
            loss = criterio(predicciones, etiquetas_exp)
            losses.append(loss.item())

            #accuracy: promedio del mapa de parches > 0.5 como prediccion final
            pred_bin = (torch.sigmoid(predicciones).mean(dim=[1, 2, 3]) > 0.5).float()
            correctas += (pred_bin == etiquetas).sum().item()
            total += len(etiquetas)

    return float(np.mean(losses)), float(correctas / total)


#MAIN:
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    if not DIR_POSITIVOS.exists():
        log.error(f"Positive samples directory not found: {DIR_POSITIVOS}")
        return
    if not DIR_INPAINTING.exists() or not any(DIR_INPAINTING.iterdir()):
        log.error(f"Inpainting results not found at: {DIR_INPAINTING}")
        log.error("Run lama_inpainting.py or mat_inpainting.py first")
        return

    modelo = PatchGAN(in_channels=3).to(device)
    optimizador = optim.Adam(modelo.parameters(), lr=LR, betas=(BETA1, 0.999))
    #BCEWithLogitsLoss = BCE + sigmoid en un solo paso (mas estable numericamente)
    criterio = nn.BCEWithLogitsLoss()

    historial = []
    epoca_inicio = 0

    #checkpoint: reanudar desde el ultimo checkpoint si existe
    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        modelo.load_state_dict(ckpt["state_dict"])
        optimizador.load_state_dict(ckpt["optimizer"])
        epoca_inicio = ckpt["epoca"]
        historial = ckpt.get("historial", [])
        log.info(f"Resuming from epoch {epoca_inicio}")
    else:
        log.info("Starting from scratch")

    #ENTRENAMIENTO: whole_body reales como positivos, reconstrucciones como negativos--------------------------------
    log.info("=== TRAINING: whole_body (positivos) vs reconstrucciones LaMa/MAT (negativos) ===")
    dataset = DatasetEsculturas(DIR_POSITIVOS, DIR_INPAINTING, TAMANO_IMG)
    n_val = max(1, int(len(dataset) * 0.15))
    n_train = len(dataset) - n_val
    ds_train, ds_val = torch.utils.data.random_split(dataset, [n_train, n_val])
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    for ep in range(max(0, epoca_inicio), EPOCHS):
        loss_train = entrenar_epoca(modelo, loader_train, optimizador, criterio, device, ep+1, EPOCHS)
        loss_val, acc_val = validar(modelo, loader_val, criterio, device)

        log.info(f"Epoch {ep+1}/{EPOCHS}  Loss train: {loss_train:.4f}  Loss val: {loss_val:.4f}  Acc val: {acc_val:.4f}")
        historial.append({"epoca": ep+1, "loss_train": loss_train, "loss_val": loss_val, "acc_val": acc_val})

        #guardar checkpoint al final de cada epoca
        torch.save({"state_dict": modelo.state_dict(), "optimizer": optimizador.state_dict(),
                    "epoca": ep+1, "historial": historial}, CHECKPOINT_PATH)

    #guardar pesos finales del discriminador
    torch.save(modelo.state_dict(), DIR_PESOS)
    log.info(f"Discriminator weights saved at: {DIR_PESOS}")

    #guardar historial completo
    hist_path = Path("/home/pfc/cescuder/tfg/logs/historial_patchgan.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(historial, f, indent=2, ensure_ascii=False)
    log.info(f"Training history saved at: {hist_path}")
    log.info("PATCHGAN TRAINING COMPLETED")


if __name__ == "__main__":
    main()
