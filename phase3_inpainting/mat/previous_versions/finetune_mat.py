"""
Fine-tuning EXPERIMENTAL de MAT sobre synthetic_dataset_bw_first.

Caveats que conviene saber:
    - MAT es StyleGAN-like: recibe img + mask + z (ruido) + c (clase) y su salida
      depende del z. para hacer L1 supervisado contra un ground truth tenemos que
      FIJAR el z en una semilla constante (sino la loss tiene varianza alta).
    - MAT esta entrenado adversarial. quitarle el discriminador puede llevar a que el
      output se vuelva borroso ("regression to the mean"). lo aceptamos como costo:
      lo unico que queremos verificar es que MAT, igual que LaMa, no es capaz de
      aprender estructura anatomica solo con L1+perceptual sobre marmol.
    - MAT tiene resolucion fija 512x512, no podemos reducirla para entrenar.
    - batch_size=2 es lo maximo que cabe en 8GB de VRAM (RTX 2070) tipicamente; si
      OOM, reducir a 1.

Estrategia: cargamos el Generator de MAT, descongelamos todo, entrenamos con z fijo
y L1+perceptual sobre el mismo dataset que LaMa v6/v7. mismos hyperparametros que
LaMa para comparabilidad.

INPUT:
    - imagenes: ~/tfg/synthetic_dataset_bw_first/images/
    - mascaras: ~/tfg/synthetic_dataset_bw_first/masks/  (DensePose 15 clases)
    - pesos: ~/tfg/MAT/Places_512_FullData_G.pkl

OUTPUT:
    - ~/tfg/MAT/Places_512_FullData_G_finetuned.pkl
    - ~/tfg/logs/finetune_mat.log
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
from torchvision import models as tv_models
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_dilation


BASE = Path("/home/pfc/cescuder/tfg")
DIR_IMAGENES = BASE / "synthetic_dataset_bw_first" / "images"
DIR_MASCARAS = BASE / "synthetic_dataset_bw_first" / "masks"
DIR_MAT_REPO = BASE / "MAT"
PESOS_MAT = DIR_MAT_REPO / "Places_512_FullData_G.pkl"
PESOS_FINETUNED = DIR_MAT_REPO / "Places_512_FullData_G_finetuned.pkl"

TAMANO_IMG = 512 #fijo por arquitectura MAT
BATCH_SIZE = 2 #ajustar a 1 si OOM
EPOCHS = 20
LR = 1e-5
VAL_SPLIT = 0.10
NUM_WORKERS = 4
PESO_L1 = 1.0
PESO_PERCEPTUAL = 0.1
SEED_SPLIT = 42
SEED_Z = 0 #z fijo durante todo el training --> PARA QUE LA LOSS TENGA SENTIDO

N_PARTES_MIN = 1
N_PARTES_MAX = 3
DILATACION_MASK_PX = 8
NUM_CLASES_DENSEPOSE_15 = 15


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(BASE / "logs" / "finetune_mat.log", encoding="utf-8"),])
log = logging.getLogger(__name__)


#DATASET (mismo formato que finetune_lama, batch_size 2 y resolution 512 para MAT):
class SyntheticInpaintDataset(Dataset):
    def __init__(self, pares, tamano):
        self.pares = pares
        self.tamano = tamano

    def __len__(self):
        return len(self.pares)

    def __getitem__(self, idx):
        #robustez ante imagenes corruptas (como en finetune_lama)
        for intento in range(10):
            img_path, mask_path = self.pares[(idx + intento) % len(self.pares)]
            try:
                img = Image.open(img_path).convert("RGB").resize((self.tamano, self.tamano), Image.BILINEAR)
                mask_dp = np.array(Image.open(mask_path).resize((self.tamano, self.tamano), Image.NEAREST))
                break
            except Exception:
                continue
        else:
            raise RuntimeError(f"10 imagenes corruptas a partir de idx={idx}")

        #construir mascara sintetica
        partes_presentes = [c for c in range(1, NUM_CLASES_DENSEPOSE_15) if (mask_dp == c).sum() > 0]
        if len(partes_presentes) == 0:
            mask_bin = np.zeros((self.tamano, self.tamano), dtype=np.uint8)
            h, w = mask_bin.shape
            mask_bin[h//3:2*h//3, w//3:2*w//3] = 1
        else:
            n_partes = random.randint(N_PARTES_MIN, min(N_PARTES_MAX, len(partes_presentes)))
            elegidas = random.sample(partes_presentes, n_partes)
            mask_bin = np.zeros((self.tamano, self.tamano), dtype=np.uint8)
            for c in elegidas:
                mask_bin |= (mask_dp == c).astype(np.uint8)
            if DILATACION_MASK_PX > 0:
                mask_bin = binary_dilation(mask_bin, iterations=DILATACION_MASK_PX).astype(np.uint8)

        #MAT espera img en [-1, 1] y mask como float (no booleana)
        img_np = np.array(img).astype(np.float32) / 127.5 - 1.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask_bin.astype(np.float32)).unsqueeze(0)
        return img_t, mask_t


#VGG PERCEPTUAL LOSS (mismo que finetune_lama):
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.bloques = nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:16]])
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1), requires_grad=False)

    def forward(self, pred, target):
        pred01 = (pred * 0.5 + 0.5).clamp(0, 1)
        tgt01 = (target * 0.5 + 0.5).clamp(0, 1)
        pred_n = (pred01 - self.mean) / self.std
        tgt_n = (tgt01 - self.mean) / self.std
        loss = 0.0
        x_p, x_t = pred_n, tgt_n
        for bloque in self.bloques:
            x_p = bloque(x_p)
            x_t = bloque(x_t)
            loss = loss + F.l1_loss(x_p, x_t)
        return loss


#CARGAR MAT:
def cargar_generator_mat(device):
    sys.path.insert(0, str(DIR_MAT_REPO))
    from networks.mat import Generator
    log.info(f"Loading MAT generator with weights from: {PESOS_MAT}")
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=TAMANO_IMG, img_channels=3)
    state_dict = torch.load(str(PESOS_MAT), map_location=device, weights_only=False)
    missing, unexpected = G.load_state_dict(state_dict, strict=False)
    log.info(f"  state_dict loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")
    G = G.to(device).train()
    #descongelar TODO el generator para que pueda finetunearse:
    for p in G.parameters():
        p.requires_grad = True
    log.info("MAT generator unfrozen, ready to fine-tune")
    return G


#MAIN:
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    log.info(f"Hyperparams: BATCH={BATCH_SIZE}, LR={LR}, EPOCHS={EPOCHS}, IMG={TAMANO_IMG}")

    extensiones = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    pares = []
    for img_path in sorted(DIR_IMAGENES.iterdir()):
        if img_path.suffix not in extensiones:
            continue
        mask_path = DIR_MASCARAS / (img_path.stem + ".png")
        if mask_path.exists():
            pares.append((img_path, mask_path))
    log.info(f"pairs found: {len(pares)}")
    if len(pares) == 0:
        log.error("no pairs found")
        return

    random.seed(SEED_SPLIT)
    random.shuffle(pares)
    n_val = max(1, int(len(pares) * VAL_SPLIT))
    pares_val = pares[:n_val]
    pares_train = pares[n_val:]
    log.info(f"train: {len(pares_train)} | val: {len(pares_val)}")

    ds_train = SyntheticInpaintDataset(pares_train, TAMANO_IMG)
    ds_val = SyntheticInpaintDataset(pares_val, TAMANO_IMG)
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    G = cargar_generator_mat(device)
    perceptual = VGGPerceptualLoss().to(device).eval()
    optim_g = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))

    #z fijo durante todo el entrenamiento
    torch.manual_seed(SEED_Z)
    z_fijo = torch.randn(BATCH_SIZE, G.z_dim, device=device)
    c_fijo = torch.zeros(BATCH_SIZE, G.c_dim, device=device)

    mejor_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        G.train()
        loss_train_acc = 0.0
        n_train = 0
        for img_t, mask_t in tqdm(dl_train, desc=f"epoch {epoch} train"):
            img_t = img_t.to(device, non_blocking=True)
            mask_t = mask_t.to(device, non_blocking=True)
            bs_actual = img_t.size(0)
            z = z_fijo[:bs_actual]
            c = c_fijo[:bs_actual]

            #MAT espera (img, mask, z, c). la mascara MAT 1=conservar 0=reconstruir
            #ojo: en nuestra convencion mask=1 es REGENERAR. invertimos para MAT
            mask_mat = 1.0 - mask_t

            output = G(img_t, mask_mat, z, c, truncation_psi=1, noise_mode="const")
            #componer: dentro mascara -> MAT output, fuera -> imagen original
            pred = img_t * (1 - mask_t) + output * mask_t

            loss_l1 = F.l1_loss(pred * mask_t, img_t * mask_t)
            loss_perc = perceptual(pred, img_t)
            loss = PESO_L1 * loss_l1 + PESO_PERCEPTUAL * loss_perc

            optim_g.zero_grad(set_to_none=True)
            loss.backward()
            optim_g.step()

            loss_train_acc += loss.item() * bs_actual
            n_train += bs_actual

        loss_train = loss_train_acc / max(n_train, 1)

        G.eval()
        loss_val_acc = 0.0
        n_val_seen = 0
        with torch.no_grad():
            for img_t, mask_t in tqdm(dl_val, desc=f"epoch {epoch} val"):
                img_t = img_t.to(device, non_blocking=True)
                mask_t = mask_t.to(device, non_blocking=True)
                bs_actual = img_t.size(0)
                z = z_fijo[:bs_actual]
                c = c_fijo[:bs_actual]
                mask_mat = 1.0 - mask_t
                output = G(img_t, mask_mat, z, c, truncation_psi=1, noise_mode="const")
                pred = img_t * (1 - mask_t) + output * mask_t
                loss_l1 = F.l1_loss(pred * mask_t, img_t * mask_t)
                loss_perc = perceptual(pred, img_t)
                loss = PESO_L1 * loss_l1 + PESO_PERCEPTUAL * loss_perc
                loss_val_acc += loss.item() * bs_actual
                n_val_seen += bs_actual

        loss_val = loss_val_acc / max(n_val_seen, 1)
        log.info(f"epoch {epoch:02d} | train_loss={loss_train:.4f} | val_loss={loss_val:.4f}")

        if loss_val < mejor_val:
            mejor_val = loss_val
            #guardamos en formato OrderedDict plano:
            torch.save(G.state_dict(), str(PESOS_FINETUNED))
            log.info(f"   -> new best val ({loss_val:.4f}), saved to {PESOS_FINETUNED}")

    log.info(f"FINE-TUNE MAT COMPLETED - best val loss: {mejor_val:.4f}")


if __name__ == "__main__":
    main()
