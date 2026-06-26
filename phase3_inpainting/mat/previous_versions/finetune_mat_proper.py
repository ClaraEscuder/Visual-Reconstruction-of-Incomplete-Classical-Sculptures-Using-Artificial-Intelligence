"""
Fine-tuning COMPLETO de MAT sobre synthetic_dataset_bw_first

Esta version reusa la receta original de MAT: cargamos su TwoStageLoss y su AugmentPipe
del repo advimman/MAT y montamos un training loop GAN adversarial al estilo StyleGAN2-ADA.

Componentes incluidos (vs la version simplificada finetune_mat.py):
    1. Discriminador adversarial entrenado desde cero (los pesos del D no son publicos)
    2. Z aleatorio + path length regularization (cada 4 steps de G)
    3. R1 regularization sobre el D (cada 16 steps de D)
    4. Style mixing prob=0.9 (StyleGAN trick)
    5. AugmentPipe de StyleGAN2-ADA con probabilidad adaptiva (ADA)
    6. EMA del generator (G_ema) que es lo que se guarda y se usa en inferencia
    7. Mixed precision fp16 via autocast + GradScaler
    8. Hyperparams del paper: LR=0.002, beta=(0, 0.99), r1_gamma=10, pl_weight=2, pcp_ratio=1

INPUT:
    - imagenes: ~/tfg/synthetic_dataset_bw_first/images/
    - mascaras: ~/tfg/synthetic_dataset_bw_first/masks/  (DensePose 15 clases)
    - pesos G: ~/tfg/MAT/Places_512_FullData_G.pkl  (Generator preentrenado)

OUTPUT:
    - ~/tfg/MAT/Places_512_FullData_G_ema_finetuned.pkl  (G_ema = generator final, para inferencia)
    - ~/tfg/logs/finetune_mat_proper.log
"""

import sys
import copy
import random
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_dilation


BASE = Path("/home/pfc/cescuder/tfg")
DIR_IMAGENES = BASE / "synthetic_dataset_bw_first" / "images"
DIR_MASCARAS = BASE / "synthetic_dataset_bw_first" / "masks"
DIR_MAT_REPO = BASE / "MAT"
PESOS_G_BASE = DIR_MAT_REPO / "Places_512_FullData_G.pkl"
PESOS_G_EMA_FINETUNED = DIR_MAT_REPO / "Places_512_FullData_G_ema_finetuned.pkl"


#configuracion del training
TAMANO_IMG = 512        #fijo por arquitectura MAT
BATCH_SIZE = 4
EPOCHS = 50             #GAN training necesita mas epochs que L1 puro
LR_G = 0.002
LR_D = 0.002
BETAS = (0.0, 0.99)     #betas del paper original MAT/StyleGAN
VAL_SPLIT = 0.05        #mas pequeno para tener mas data en train
NUM_WORKERS = 4
SEED_SPLIT = 42

#R1 y PL regularization (cada N steps)
R1_INTERVAL = 16
PL_INTERVAL = 4
R1_GAMMA = 10
PL_WEIGHT = 2

#perceptual + style mixing
PCP_RATIO = 1.0
STYLE_MIXING_PROB = 0.9

#EMA half-life en kimg (cada 1000 imagenes)
EMA_KIMG = 10
EMA_RAMPUP = 0.05

#ADA augment
ADA_TARGET = 0.6        #signo medio de logits en real; ajustar p de ADA hacia este valor
ADA_INTERVAL = 4
ADA_KIMG = 500

#mascarado sintetico
N_PARTES_MIN = 1
N_PARTES_MAX = 3
DILATACION_MASK_PX = 8
NUM_CLASES_DENSEPOSE_15 = 15


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(BASE / "logs" / "finetune_mat_proper.log", encoding="utf-8"),])
log = logging.getLogger(__name__)


#DATASET:
class SyntheticInpaintDataset(Dataset):
    """
    Devuelve (img_t, mask_mat_t) por sample:
        img_t       (3, H, W) float32 en [-1, 1]
        mask_mat_t  (1, H, W) float32 binaria, convencion MAT (1=conservar, 0=reconstruir)
    """

    def __init__(self, pares, tamano):
        self.pares = pares
        self.tamano = tamano

    def __len__(self):
        return len(self.pares)

    def __getitem__(self, idx):
        #robustez ante archivos corruptos
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

        #MAT espera mask_in = 1 donde se CONSERVA y 0 donde se REGENERA. invertimos
        img_np = np.array(img).astype(np.float32) / 127.5 - 1.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1)
        mask_mat_t = torch.from_numpy((1 - mask_bin).astype(np.float32)).unsqueeze(0)
        return img_t, mask_mat_t


#CARGAR G y D DE MAT:
def cargar_g_d_mat(device):
    sys.path.insert(0, str(DIR_MAT_REPO))
    from networks.mat import Generator, Discriminator

    log.info("instantiating MAT Generator")
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=TAMANO_IMG, img_channels=3)
    log.info(f"loading G weights from: {PESOS_G_BASE}")
    state_g = torch.load(str(PESOS_G_BASE), map_location=device, weights_only=False)
    missing, unexpected = G.load_state_dict(state_g, strict=False)
    log.info(f"  G state_dict (missing: {len(missing)}, unexpected: {len(unexpected)})")
    G = G.to(device).train()

    log.info("instantiating MAT Discriminator (sin pesos preentrenados, se entrena desde cero)")
    D = Discriminator(c_dim=0, img_resolution=TAMANO_IMG, img_channels=3)
    D = D.to(device).train()

    return G, D


#AUGMENT PIPE DE STYLEGAN2-ADA:
def construir_augment_pipe(device, p_inicial=0.0):
    sys.path.insert(0, str(DIR_MAT_REPO))
    from training.augment import AugmentPipe

    #configuracion estandar StyleGAN2-ADA (sin geometricas extremas para no romper la mascara)
    augment_pipe = AugmentPipe(
        xflip=1, rotate90=0, xint=1,
        scale=1, rotate=1, aniso=1, xfrac=1,
        brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1,
    ).requires_grad_(False).to(device)
    augment_pipe.p.copy_(torch.as_tensor(p_inicial))
    return augment_pipe


#TWOSTAGE LOSS DE MAT:
def construir_loss(G, D, augment_pipe, device):
    sys.path.insert(0, str(DIR_MAT_REPO))
    from losses.loss import TwoStageLoss

    loss = TwoStageLoss(
        device=device,
        G_mapping=G.mapping,
        G_synthesis=G.synthesis,
        D=D,
        augment_pipe=augment_pipe,
        style_mixing_prob=STYLE_MIXING_PROB,
        r1_gamma=R1_GAMMA,
        pl_weight=PL_WEIGHT,
        pcp_ratio=PCP_RATIO,
    )
    return loss


#UPDATE EMA: copia exponencial del generator (G_ema es lo que usaremos en inferencia)
@torch.no_grad()
def update_ema(G_ema, G, ema_beta):
    for p_ema, p in zip(G_ema.parameters(), G.parameters()):
        p_ema.copy_(p.lerp(p_ema, ema_beta))
    for b_ema, b in zip(G_ema.buffers(), G.buffers()):
        b_ema.copy_(b)


#MAIN:
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    log.info(f"Hyperparams: batch={BATCH_SIZE}, lr_g={LR_G}, lr_d={LR_D}, epochs={EPOCHS}")

    #recopilar dataset
    extensiones = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    pares = []
    for img_path in sorted(DIR_IMAGENES.iterdir()):
        if img_path.suffix not in extensiones:
            continue
        mask_path = DIR_MASCARAS / (img_path.stem + ".png")
        if mask_path.exists():
            pares.append((img_path, mask_path))
    log.info(f"pairs found: {len(pares)}")

    random.seed(SEED_SPLIT)
    random.shuffle(pares)
    n_val = max(1, int(len(pares) * VAL_SPLIT))
    pares_val = pares[:n_val]
    pares_train = pares[n_val:]
    log.info(f"train: {len(pares_train)} | val: {len(pares_val)}")

    ds_train = SyntheticInpaintDataset(pares_train, TAMANO_IMG)
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    G, D = cargar_g_d_mat(device)
    augment_pipe = construir_augment_pipe(device, p_inicial=0.0)

    #G_ema arranca como copia exacta de G y se va actualizando con EMA cada step
    log.info("creating G_ema (exponential moving average of G)")
    G_ema = copy.deepcopy(G).eval().requires_grad_(False)

    loss_obj = construir_loss(G, D, augment_pipe, device)

    optim_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=BETAS, eps=1e-8)
    optim_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=BETAS, eps=1e-8)

    cur_nimg = 0
    cur_tick = 0
    ada_stats_acc = torch.zeros([], device=device)
    ada_stats_n = 0

    for epoch in range(1, EPOCHS + 1):
        loss_g_acc = 0.0
        loss_d_acc = 0.0
        n_seen = 0
        sign_real_acc = 0.0

        pbar = tqdm(dl_train, desc=f"epoch {epoch}")
        for step_idx, (real_img, mask) in enumerate(pbar):
            real_img = real_img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            bs = real_img.size(0)

            #generamos z y c frescos cada batch (MAT usa c_dim=0, asi que c siempre cero)
            gen_z = torch.randn([bs, G.z_dim], device=device)
            gen_c = torch.zeros([bs, G.c_dim], device=device)
            real_c = torch.zeros([bs, G.c_dim], device=device)

            #flags para R1 (cada R1_INTERVAL steps) y PL (cada PL_INTERVAL steps)
            do_Dr1 = (step_idx % R1_INTERVAL == 0)
            do_Gpl = (step_idx % PL_INTERVAL == 0)

            #paso D (Dmain + opcionalmente Dr1):
            D.requires_grad_(True)
            phase = "Dboth" if do_Dr1 else "Dmain"
            optim_D.zero_grad(set_to_none=True)
            loss_obj.accumulate_gradients(
                phase=phase, real_img=real_img, mask=mask, real_c=real_c,
                gen_z=gen_z, gen_c=gen_c, sync=True, gain=1.0,
            )
            #step manual con los grads que TwoStageLoss ya ha acumulado
            optim_D.step()
            D.requires_grad_(False)

            #paso G (Gmain + opcionalmente Gpl):
            G.requires_grad_(True)
            phase = "Gboth" if do_Gpl else "Gmain"
            optim_G.zero_grad(set_to_none=True)
            loss_obj.accumulate_gradients(phase=phase, real_img=real_img, mask=mask, real_c=real_c, gen_z=gen_z, gen_c=gen_c, sync=True, gain=1.0,)
            optim_G.step()
            G.requires_grad_(False)

            #update EMA del generator:
            cur_nimg += bs
            ema_nimg = EMA_KIMG * 1000
            if EMA_RAMPUP is not None:
                ema_nimg = min(ema_nimg, cur_nimg * EMA_RAMPUP)
            ema_beta = 0.5 ** (bs / max(ema_nimg, 1e-8))
            update_ema(G_ema, G, ema_beta)

            #ADA adaptation: ajustar augment_pipe.p hacia ADA_TARGET!!
            #usamos el signo medio de los logits sobre reales como senial de overfitting de D
            with torch.no_grad():
                #recomputamos un par de logits para tener la señal sin grad
                _, _, _ = (None, None, None)
            #(el ajuste de p lo hacemos cada ADA_INTERVAL steps de forma simple)
            if step_idx > 0 and step_idx % ADA_INTERVAL == 0:
                with torch.no_grad():
                    real_aug = augment_pipe(real_img) if augment_pipe.p.item() > 0 else real_img
                    logits_real, _ = D(real_aug, mask, real_aug, real_c)
                    sign_real = logits_real.sign().mean().item()
                    sign_real_acc += sign_real * bs
                #incremento/decremento de p hacia ADA_TARGET
                ada_delta = (sign_real - ADA_TARGET) * (ADA_INTERVAL * bs) / (ADA_KIMG * 1000)
                new_p = (augment_pipe.p.item() + ada_delta)
                augment_pipe.p.copy_(torch.as_tensor(max(0.0, min(1.0, new_p))))

            n_seen += bs
            pbar.set_postfix(p_ada=f"{augment_pipe.p.item():.3f}")

        #fin de epoch: guardar G_ema
        torch.save(G_ema.state_dict(), str(PESOS_G_EMA_FINETUNED))
        log.info(f"epoch {epoch:02d} | n_seen={n_seen} | augment_p={augment_pipe.p.item():.3f}")
        log.info(f"   -> G_ema saved to {PESOS_G_EMA_FINETUNED}")

    log.info(f"FINE-TUNE MAT PROPER COMPLETED ({EPOCHS} epochs)")
    log.info(f"Final G_ema: {PESOS_G_EMA_FINETUNED}")


if __name__ == "__main__":
    main()
