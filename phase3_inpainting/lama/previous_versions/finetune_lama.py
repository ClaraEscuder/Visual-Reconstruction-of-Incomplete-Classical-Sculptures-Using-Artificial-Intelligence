"""
Fine-tuning de LaMa sobre el dataset sintetico bw-first (COCO con apariencia de marmol)

Estrategia:
    - Cargamos el generator de big-lama tal cual (FFCResNetGenerator del repo oficial
      advimman/lama, con sus pesos preentrenados sobre Places2).
    - NO entrenamos el discriminador. Hacer GAN-finetuning sin haber estabilizado los
      hiperparametros suele divergir; nos quedamos con un training supervisado L1 +
      perceptual (VGG16) que es mas estable y deja al generator deslizarse hacia la
      distribucion de marmol sin destruir lo que ya sabe sobre estructura humana.
    - Las mascaras de entrenamiento se generan al vuelo a partir de las mascaras
      DensePose: para cada imagen, escogemos 1-3 partes corporales aleatorias y las
      enmascaramos. Eso simula la distribucion real de "miembros amputados" que
      encontramos en broken_body.

INPUT:
    - imagenes: ~/tfg/synthetic_dataset_bw_first/images/   (estilo marmol B/N first)
    - mascaras: ~/tfg/synthetic_dataset_bw_first/masks/    (15 clases DensePose: 0=fondo, 1-14=partes)
    - pesos LaMa: ~/tfg/lama_repo/big-lama/models/best.ckpt
    - config LaMa: ~/tfg/lama_repo/big-lama/config.yaml

OUTPUT:
    - ~/tfg/lama_repo/big-lama/models/best_finetuned.ckpt   (pesos finetuneados)
    - ~/tfg/logs/finetune_lama.log
"""

import sys
import random
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision import models as tv_models
from PIL import Image
from tqdm import tqdm

#FFCResNetGenerator standalone para no depender de la libreria saicinpainting,
#que en cluster esta rota por incompatibilidad de albumentations 0.5 / pytorch_lightning 1.6 / pkg_resources
sys.path.insert(0, "/home/pfc/cescuder/tfg/scripts")
from ffc_standalone import FFCResNetGenerator, BIG_LAMA_GENERATOR_KWARGS, cargar_pesos_big_lama


BASE = Path("/home/pfc/cescuder/tfg")

DIR_IMAGENES = BASE / "synthetic_dataset_bw_first" / "images"
DIR_MASCARAS = BASE / "synthetic_dataset_bw_first" / "masks"

DIR_LAMA_REPO = BASE / "lama_repo"
DIR_BIG_LAMA = DIR_LAMA_REPO / "big-lama"
CONFIG_LAMA = DIR_BIG_LAMA / "config.yaml"
CKPT_LAMA = DIR_BIG_LAMA / "models" / "best.ckpt"
CKPT_FINETUNED = DIR_BIG_LAMA / "models" / "best_finetuned.ckpt"
#last_finetuned.ckpt = estado del fin de cada epoch (no solo cuando mejora la val). permite
#reanudar el training desde el epoch donde se quedo si el job se interrumpio por cualquier razon
CKPT_LAST = DIR_BIG_LAMA / "models" / "last_finetuned.ckpt"

#hiperparametros de fine-tune
TAMANO_IMG = 256 #LaMa puede trabajar a varias resoluciones; 256 va sobrado en GPU 8GB
BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-5 #muy bajo: estamos adaptando, no entrenando desde cero
VAL_SPLIT = 0.10
NUM_WORKERS = 4
PESO_L1 = 1.0
PESO_PERCEPTUAL = 0.1 #equilibrio empirico: el L1 marca la textura, el perceptual la coherencia

#mascarado sintetico: cuantas partes corporales quitamos por imagen
N_PARTES_MIN = 1
N_PARTES_MAX = 3
#dilatacion en pixeles de la mascara sintetica final (para que cubra bien el borde, como hace v6 con SD)
DILATACION_MASK_PX = 8

NUM_CLASES_DENSEPOSE = 15   #0=fondo + 14 partes


#LOGGING:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(BASE / "logs" / "finetune_lama.log", encoding="utf-8"),])
log = logging.getLogger(__name__)


#DATASET:
class SyntheticInpaintDataset(Dataset):
    """
    Para cada imagen del synthetic_dataset_bw_first, devuelve (img, mask) donde:
        - img es la imagen estilo-marmol redimensionada a TAMANO_IMG x TAMANO_IMG, normalizada a [0,1]
        - mask es una mascara binaria construida al vuelo escogiendo N partes corporales
          aleatorias (entre 1 y 3) de la anotacion DensePose, simulando "miembros amputados"
    """

    NORM_MEAN = [0.5, 0.5, 0.5]
    NORM_STD = [0.5, 0.5, 0.5]

    def __init__(self, pares: list, tamano: int):
        self.pares = pares
        self.tamano = tamano

    def __len__(self):
        return len(self.pares)

    def __getitem__(self, idx):
        #si la imagen esta corrupta, saltamos al siguiente sample valido. style_transfer.py
        #ocasionalmente deja jpgs invalidos (corrupciones esporadicas durante el guardado)
        #y un DataLoader worker que falla mata todo el training --> mejor saltar y seguir
        for intento in range(10):
            img_path, mask_path = self.pares[(idx + intento) % len(self.pares)]
            try:
                img = Image.open(img_path).convert("RGB").resize((self.tamano, self.tamano), Image.BILINEAR)
                mask_dp = np.array(Image.open(mask_path).resize((self.tamano, self.tamano), Image.NEAREST))
                break
            except Exception:
                continue
        else:
            #si despues de 10 reintentos todos fallan, lanzamos error real
            raise RuntimeError(f"10 imagenes consecutivas corruptas a partir de idx={idx}")

        #construir mascara sintetica: elegir N partes (clases 1-14) presentes en la imagen
        partes_presentes = [c for c in range(1, NUM_CLASES_DENSEPOSE) if (mask_dp == c).sum() > 0]
        if len(partes_presentes) == 0:
            #si no hay partes anotadas no podemos generar mascara; devolvemos una mascara central rectangular como fallback
            mask_bin = np.zeros((self.tamano, self.tamano), dtype=np.uint8)
            h, w = mask_bin.shape
            mask_bin[h//3:2*h//3, w//3:2*w//3] = 1
        else:
            n_partes = random.randint(N_PARTES_MIN, min(N_PARTES_MAX, len(partes_presentes)))
            elegidas = random.sample(partes_presentes, n_partes)
            mask_bin = np.zeros((self.tamano, self.tamano), dtype=np.uint8)
            for c in elegidas:
                mask_bin |= (mask_dp == c).astype(np.uint8)

            #dilatar un poco para que LaMa tenga un poco de margen alrededor del borde
            if DILATACION_MASK_PX > 0:
                from scipy.ndimage import binary_dilation
                mask_bin = binary_dilation(mask_bin, iterations=DILATACION_MASK_PX).astype(np.uint8)

        #img a tensor normalizado [-1, 1] (como espera LaMa por convencion de styleGAN-like)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1)
        for c in range(3):
            img_t[c] = (img_t[c] - self.NORM_MEAN[c]) / self.NORM_STD[c]

        mask_t = torch.from_numpy(mask_bin.astype(np.float32)).unsqueeze(0)  #(1, H, W)

        return img_t, mask_t


#VGG PERCEPTUAL LOSS:
class VGGPerceptualLoss(nn.Module):
    """
    Loss perceptual basada en VGG16: compara las features en varias capas intermedias
    entre la prediccion y el ground truth. Es lo que mantiene la textura del marmol
    coherente sin penalizar diferencias pixel-a-pixel intrascendentes.
    """
    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        #usamos las salidas de relu1_2, relu2_2, relu3_3
        self.bloques = nn.ModuleList([
            vgg[:4], #conv1_1 - relu1_2
            vgg[4:9], #conv2_1 - relu2_2
            vgg[9:16], #conv3_1 - relu3_3
        ])
        #vgg espera input normalizado con stats ImageNet
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1), requires_grad=False)

    def forward(self, pred, target):
        #pred y target estan en [-1, 1] tras la normalizacion del dataset. Pasamos a [0,1] y luego a stats ImageNet.
        pred01 = (pred * 0.5 + 0.5).clamp(0, 1)
        tgt01 = (target * 0.5 + 0.5).clamp(0, 1)
        pred_n = (pred01 - self.mean) / self.std
        tgt_n = (tgt01  - self.mean) / self.std

        loss = 0.0
        x_p, x_t = pred_n, tgt_n
        for bloque in self.bloques:
            x_p = bloque(x_p)
            x_t = bloque(x_t)
            loss = loss + F.l1_loss(x_p, x_t)
        return loss


#CARGAR LAMA:
def cargar_generator_lama(device):
    """
    Construye un FFCResNetGenerator con los parametros de big-lama (b18_ffc075) y carga
    pesos. Si existe last_finetuned.ckpt (de una corrida previa interrumpida) reanuda
    desde alli y devuelve (generator, start_epoch, best_val). Si no, carga big-lama
    vainilla y devuelve start_epoch=1, best_val=inf.
    """
    if not CKPT_LAMA.exists():
        raise FileNotFoundError(
            f"big-lama checkpoint not found at {CKPT_LAMA}. The slurm script should download big-lama.zip from HF.")

    log.info("building FFCResNetGenerator with big-lama params (b18_ffc075)")
    generator = FFCResNetGenerator(**BIG_LAMA_GENERATOR_KWARGS)

    #intento de reanudar desde last_finetuned.ckpt si existe
    if CKPT_LAST.exists():
        log.info(f"resuming from previous run: {CKPT_LAST}")
        state = torch.load(str(CKPT_LAST), map_location=device, weights_only=False)
        gen_sd = {k[len("generator."):]: v for k, v in state["state_dict"].items() if k.startswith("generator.")}
        missing, unexpected = generator.load_state_dict(gen_sd, strict=False)
        log.info(f"  state_dict loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")
        start_epoch = int(state.get("epoch", 0)) + 1
        best_val = float(state.get("best_val", float("inf")))
        log.info(f"  resuming at epoch {start_epoch} (previous best_val = {best_val:.4f})")
    else:
        log.info(f"no previous run found, loading big-lama base weights: {CKPT_LAMA}")
        missing, unexpected = cargar_pesos_big_lama(generator, str(CKPT_LAMA), device=device)
        log.info(f"  state_dict loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")
        start_epoch = 1
        best_val = float("inf")

    generator = generator.to(device).train()
    log.info("LaMa generator ready to fine-tune")
    return generator, start_epoch, best_val


#GUARDAR FINETUNEADO:
def guardar_checkpoint_finetuneado(model, generator, path, epoch=None, best_val=None):
    """
    Guarda los pesos del generator finetuneado en un .ckpt con el mismo formato que
    big-lama (claves con prefijo "generator."). Asi la inferencia v6 puede recargarlo
    via cargar_pesos_big_lama() sin cambios.

    Si epoch y best_val se pasan, tambien se guardan para soportar resume
    """
    state_full = {"state_dict": {}}
    for k, v in generator.state_dict().items():
        state_full["state_dict"][f"generator.{k}"] = v.detach().cpu()
    if epoch is not None:
        state_full["epoch"] = int(epoch)
    if best_val is not None:
        state_full["best_val"] = float(best_val)
    torch.save(state_full, str(path))
    log.info(f"checkpoint saved to: {path}")


#MAIN:
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    log.info(f"Hyperparams: BATCH={BATCH_SIZE}, LR={LR}, EPOCHS={EPOCHS}, IMG={TAMANO_IMG}, L1_W={PESO_L1}, PERC_W={PESO_PERCEPTUAL}")

    #recolectar pares (imagen, mascara DensePose)
    extensiones = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    pares = []
    for img_path in sorted(DIR_IMAGENES.iterdir()):
        if img_path.suffix not in extensiones:
            continue
        mask_path = DIR_MASCARAS / (img_path.stem + ".png")
        if mask_path.exists():
            pares.append((img_path, mask_path))
    log.info(f"image/mask pairs found: {len(pares)}")
    if len(pares) == 0:
        log.error("No pairs found. Did you run style_transfer_bw_first.py first?")
        return

    #split train/val
    random.seed(42)
    random.shuffle(pares)
    n_val = max(1, int(len(pares) * VAL_SPLIT))
    pares_val = pares[:n_val]
    pares_train = pares[n_val:]
    log.info(f"train: {len(pares_train)} | val: {len(pares_val)}")

    ds_train = SyntheticInpaintDataset(pares_train, TAMANO_IMG)
    ds_val   = SyntheticInpaintDataset(pares_val, TAMANO_IMG)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    dl_val   = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    generator, start_epoch, mejor_val = cargar_generator_lama(device)
    model = None  #compat
    perceptual = VGGPerceptualLoss().to(device).eval()

    optim_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))

    if start_epoch > EPOCHS:
        log.info(f"start_epoch ({start_epoch}) > EPOCHS ({EPOCHS}); nothing to do, exit")
        return
    if start_epoch > 1:
        log.info(f"RESUMING training at epoch {start_epoch}/{EPOCHS} (previous best_val={mejor_val:.4f})")

    for epoch in range(start_epoch, EPOCHS + 1):
        #--- TRAIN ---
        generator.train()
        loss_train_acc = 0.0
        n_train = 0
        for img_t, mask_t in tqdm(dl_train, desc=f"epoch {epoch} train"):
            img_t  = img_t.to(device, non_blocking=True)
            mask_t = mask_t.to(device, non_blocking=True)

            #LaMa recibe imagen con zona enmascarada a cero + mascara como cuarto canal
            masked_img = img_t * (1 - mask_t)
            entrada = torch.cat([masked_img, mask_t], dim=1)  #(B, 4, H, W)

            salida = generator(entrada)
            #LaMa devuelve la imagen completa; lo que importa es lo que generó dentro de la mascara
            pred_compuesto = masked_img + salida * mask_t

            loss_l1 = F.l1_loss(pred_compuesto * mask_t, img_t * mask_t)
            loss_perc = perceptual(pred_compuesto, img_t)
            loss = PESO_L1 * loss_l1 + PESO_PERCEPTUAL * loss_perc

            optim_g.zero_grad(set_to_none=True)
            loss.backward()
            optim_g.step()

            loss_train_acc += loss.item() * img_t.size(0)
            n_train += img_t.size(0)

        loss_train = loss_train_acc / max(n_train, 1)

        #--- VAL ---
        generator.eval()
        loss_val_acc = 0.0
        n_val_seen = 0
        with torch.no_grad():
            for img_t, mask_t in tqdm(dl_val, desc=f"epoch {epoch} val"):
                img_t  = img_t.to(device, non_blocking=True)
                mask_t = mask_t.to(device, non_blocking=True)

                masked_img = img_t * (1 - mask_t)
                entrada = torch.cat([masked_img, mask_t], dim=1)
                salida = generator(entrada)
                pred_compuesto = masked_img + salida * mask_t

                loss_l1 = F.l1_loss(pred_compuesto * mask_t, img_t * mask_t)
                loss_perc = perceptual(pred_compuesto, img_t)
                loss = PESO_L1 * loss_l1 + PESO_PERCEPTUAL * loss_perc

                loss_val_acc += loss.item() * img_t.size(0)
                n_val_seen += img_t.size(0)

        loss_val = loss_val_acc / max(n_val_seen, 1)
        log.info(f"epoch {epoch:02d} | train_loss={loss_train:.4f} | val_loss={loss_val:.4f}")

        #si la val mejora, guardamos best_finetuned.ckpt (es el que la inferencia usa)
        if loss_val < mejor_val:
            mejor_val = loss_val
            guardar_checkpoint_finetuneado(model, generator, CKPT_FINETUNED, epoch=epoch, best_val=mejor_val)
            log.info(f"   -> new best val ({loss_val:.4f}), saved")

        #ademas, al final de CADA epoch guardamos last_finetuned.ckpt para permitir
        #resume si el job se interrumpe (con epoch y best_val actuales)
        guardar_checkpoint_finetuneado(model, generator, CKPT_LAST, epoch=epoch, best_val=mejor_val)

    log.info(f"FINE-TUNE COMPLETED - best val loss: {mejor_val:.4f}")
    log.info(f"Fine-tuned checkpoint: {CKPT_FINETUNED}")


if __name__ == "__main__":
    main()
