"""
Fine-tuning de LaMa con conditioning DensePose (variante v7).

Diferencias con finetune_lama.py:
    - La primera convolucion del generator pasa de 4 a 7 canales:
      RGB + mask + part_id + U + V.
    - Para cada imagen, el dataset construye los canales extra leyendo el cache de
      DensePose generado por extract_densepose_synthetic.py.
    - Con probabilidad 0.5, el conditioning de la zona enmascarada se reemplaza
      por un gradiente sintetizado (mismo procedimiento que el mask generator
      v6.1 hace en inferencia). Esto entrena al modelo para tolerar conditioning
      "proyectado" y no solo "perfecto".

INPUT:
    - imagenes:   ~/tfg/synthetic_dataset_bw_first/images/
    - mascaras DensePose 15 clases: ~/tfg/synthetic_dataset_bw_first/masks/
    - cache UV:   ~/tfg/synthetic_dataset_bw_first/densepose_cache/images/
    - pesos LaMa: ~/tfg/lama_repo/big-lama/models/best.ckpt

OUTPUT:
    - ~/tfg/lama_repo/big-lama/models/best_finetuned_v7.ckpt
    - ~/tfg/logs/finetune_lama_v7.log
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

#standalone FFCResNetGenerator
sys.path.insert(0, "/home/pfc/cescuder/tfg/scripts")
from ffc_standalone import FFCResNetGenerator, BIG_LAMA_GENERATOR_KWARGS, cargar_pesos_big_lama


BASE = Path("/home/pfc/cescuder/tfg")

DIR_IMAGENES = BASE / "synthetic_dataset_bw_first" / "images_no_bg"
DIR_MASCARAS_DP15 = BASE / "synthetic_dataset_bw_first" / "masks"
DIR_DP_CACHE = BASE / "synthetic_dataset_bw_first" / "densepose_cache" / "images"

DIR_LAMA_REPO = BASE / "lama_repo"
DIR_BIG_LAMA = DIR_LAMA_REPO / "big-lama"
CONFIG_LAMA = DIR_BIG_LAMA / "config.yaml"
CKPT_LAMA = DIR_BIG_LAMA / "models" / "best.ckpt"
CKPT_FINETUNED = DIR_BIG_LAMA / "models" / "best_finetuned_v7_nobg.ckpt"
CKPT_LAST = DIR_BIG_LAMA / "models" / "last_finetuned_v7_nobg.ckpt"


#hiperparametros del fine-tune
TAMANO_IMG = 256
BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-5
VAL_SPLIT = 0.10
NUM_WORKERS = 4
PESO_L1 = 1.0
PESO_PERCEPTUAL = 0.1
SEED_SPLIT = 42

#mascarado sintetico
N_PARTES_MIN = 1
N_PARTES_MAX = 3
DILATACION_MASK_PX = 8
NUM_CLASES_DENSEPOSE_15 = 15

#probabilidad de simular conditioning "proyectado" en vez de usar el real
PROB_SIMULAR_PROYECCION = 0.5


#LOGGING:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(BASE / "logs" / "finetune_lama_v7_nobg.log", encoding="utf-8"),])
log = logging.getLogger(__name__)


#SINTESIS DE UV EN UNA REGION:
def sintetizar_uv_en_region(mascara_region: np.ndarray):
    """
    Misma logica que compute_mask_from_densepose_v6_1.sintetizar_uv_en_region:
    gradiente lineal a lo largo del eje principal PCA (U: 0->1 axial, V: 0->1 perp).
    Devuelve U y V (float32) del mismo shape que la mascara. Cero fuera de la mascara.
    """
    U = np.zeros(mascara_region.shape, dtype=np.float32)
    V = np.zeros(mascara_region.shape, dtype=np.float32)
    ys, xs = np.where(mascara_region)
    if len(ys) < 2:
        return U, V

    pts = np.column_stack([ys, xs]).astype(np.float64)
    centroide = pts.mean(axis=0)
    pts_c = pts - centroide
    cov = np.cov(pts_c.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx_p = np.argmax(eigvals)
    eje_principal = eigvecs[:, idx_p]
    eje_perp = np.array([-eje_principal[1], eje_principal[0]])

    proj_p = pts_c @ eje_principal
    proj_q = pts_c @ eje_perp
    p_min, p_max = proj_p.min(), proj_p.max()
    q_min, q_max = proj_q.min(), proj_q.max()
    proj_p_norm = (proj_p - p_min) / (p_max - p_min) if p_max - p_min > 1e-6 else np.full_like(proj_p, 0.5)
    proj_q_norm = (proj_q - q_min) / (q_max - q_min) if q_max - q_min > 1e-6 else np.full_like(proj_q, 0.5)

    U[ys, xs] = proj_p_norm.astype(np.float32)
    V[ys, xs] = proj_q_norm.astype(np.float32)
    return U, V


#DATASET:
class SyntheticInpaintCondDataset(Dataset):
    """
    Devuelve por imagen:
        img_t --> (3, H, W) float32 normalizada a [-1, 1]
        mask_t --> (1, H, W) float32 binaria  (1 = region a regenerar)
        cond_t --> (3, H, W) float32  (part_id_norm, U, V)

    El cond se construye partiendo del DensePose real cacheado en .npz; con
    probabilidad PROB_SIMULAR_PROYECCION, en la zona enmascarada el cond UV se
    reemplaza por un gradiente lineal --> para
    enseñar al modelo a tolerar PROEJCTED CONDITIONING.
    """

    NORM_MEAN = [0.5, 0.5, 0.5]
    NORM_STD  = [0.5, 0.5, 0.5]
    #normalizamos part_id a [0, 1] dividiendo por el numero total de clases SMPL (24)
    MAX_PART_ID = 24.0

    def __init__(self, tripletas: list, tamano: int):
        self.tripletas = tripletas
        self.tamano = tamano

    def __len__(self):
        return len(self.tripletas)

    def __getitem__(self, idx):
        #robustez frente a imagenes/.npz corruptos en synthetic_dataset_bw_first: saltamos
        #al siguiente sample valido si falla la lectura de cualquier parte de la tripleta
        for intento in range(10):
            img_path, mask_dp15_path, dp_cache_path = self.tripletas[(idx + intento) % len(self.tripletas)]
            try:
                img = Image.open(img_path).convert("RGB").resize((self.tamano, self.tamano), Image.BILINEAR)
                mask_dp15 = np.array(Image.open(mask_dp15_path).resize((self.tamano, self.tamano), Image.NEAREST))
                cache = np.load(dp_cache_path)
                I_full = cache["I"]
                U_full = cache["U"]
                V_full = cache["V"]
                break
            except Exception:
                continue
        else:
            raise RuntimeError(f"10 tripletas consecutivas corruptas a partir de idx={idx}")

        img_np = np.array(img).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1)
        for c in range(3):
            img_t[c] = (img_t[c] - self.NORM_MEAN[c]) / self.NORM_STD[c]
        #redimensionar al tamaño de entrenamiento
        I_pil = Image.fromarray(I_full).resize((self.tamano, self.tamano), Image.NEAREST)
        U_pil = Image.fromarray(U_full).resize((self.tamano, self.tamano), Image.BILINEAR)
        V_pil = Image.fromarray(V_full).resize((self.tamano, self.tamano), Image.BILINEAR)
        I_arr = np.array(I_pil).astype(np.uint8)
        U_arr = np.array(U_pil).astype(np.float32)
        V_arr = np.array(V_pil).astype(np.float32)

        #construir mascara sintetica de inpainting: 1-3 partes elegidas al azar de
        #las que aparecen en mask_dp15
        partes_presentes = [c for c in range(1, NUM_CLASES_DENSEPOSE_15) if (mask_dp15 == c).sum() > 0]
        if len(partes_presentes) == 0:
            mask_bin = np.zeros((self.tamano, self.tamano), dtype=np.uint8)
            h, w = mask_bin.shape
            mask_bin[h//3:2*h//3, w//3:2*w//3] = 1
        else:
            n_partes = random.randint(N_PARTES_MIN, min(N_PARTES_MAX, len(partes_presentes)))
            elegidas = random.sample(partes_presentes, n_partes)
            mask_bin = np.zeros((self.tamano, self.tamano), dtype=np.uint8)
            for c in elegidas:
                mask_bin |= (mask_dp15 == c).astype(np.uint8)
            if DILATACION_MASK_PX > 0:
                mask_bin = binary_dilation(mask_bin, iterations=DILATACION_MASK_PX).astype(np.uint8)

        mask_bool = mask_bin.astype(bool)

        #cond inicial: copia del DensePose real
        I_cond = I_arr.copy()
        U_cond = U_arr.copy()
        V_cond = V_arr.copy()

        #con probabilidad PROB_SIMULAR_PROYECCION, sustituir el UV dentro de la mascara
        #por el gradiente sintetizado (que es lo que vera en inferencia desde v6.1)
        if random.random() < PROB_SIMULAR_PROYECCION and mask_bool.sum() > 2:
            U_sim, V_sim = sintetizar_uv_en_region(mask_bool)
            U_cond[mask_bool] = U_sim[mask_bool]
            V_cond[mask_bool] = V_sim[mask_bool]
            #para el part_id: en proyeccion v6.1 se asigna el part_id canonico de la region
            #aqui aproximamos asignando el part_id real mas frecuente dentro de la mascara
            valores_id, conteos = np.unique(I_arr[mask_bool], return_counts=True)
            #ignorar el background si esta presente
            no_bg = valores_id > 0
            if no_bg.sum() > 0:
                idx_mejor = np.argmax(conteos[no_bg])
                part_id_canonico = int(valores_id[no_bg][idx_mejor])
                I_cond[mask_bool] = part_id_canonico

        #tensores finales
        mask_t = torch.from_numpy(mask_bin.astype(np.float32)).unsqueeze(0)
        cond_t = torch.from_numpy(np.stack([
            I_cond.astype(np.float32) / self.MAX_PART_ID,
            U_cond,
            V_cond,
        ], axis=0))

        return img_t, mask_t, cond_t


#VGG PERCEPTUAL LOSS:
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.bloques = nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:16]])
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1), requires_grad=False)
        self.std  = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1), requires_grad=False)

    def forward(self, pred, target):
        pred01 = (pred * 0.5 + 0.5).clamp(0, 1)
        tgt01  = (target * 0.5 + 0.5).clamp(0, 1)
        pred_n = (pred01 - self.mean) / self.std
        tgt_n  = (tgt01  - self.mean) / self.std

        loss = 0.0
        x_p, x_t = pred_n, tgt_n
        for bloque in self.bloques:
            x_p = bloque(x_p)
            x_t = bloque(x_t)
            loss = loss + F.l1_loss(x_p, x_t)
        return loss


#EXPANSION DEL PRIMER CONV (4 -> 7 CANALES):
def expandir_primer_conv(generator, in_orig=4, in_nuevo=7):
    """
    Busca recursivamente todas las nn.Conv2d con in_channels==in_orig y las
    reemplaza por una conv equivalente con in_channels=in_nuevo. Los primeros
    in_orig canales de los pesos nuevos se copian de la conv original; los
    canales nuevos se inicializan con desv. estandar 0.01 (casi-cero) para que
    al principio del fine-tune el modelo se comporte aprox como vanilla LaMa.

    Devuelve el numero de convs modificadas (debe ser >= 1).
    """
    modificadas = 0

    def reemplazar_en(parent: nn.Module):
        nonlocal modificadas
        for nombre, hijo in list(parent.named_children()):
            if isinstance(hijo, nn.Conv2d) and hijo.in_channels == in_orig:
                conv_nuevo = nn.Conv2d(
                    in_channels=in_nuevo,
                    out_channels=hijo.out_channels,
                    kernel_size=hijo.kernel_size,
                    stride=hijo.stride,
                    padding=hijo.padding,
                    dilation=hijo.dilation,
                    groups=hijo.groups,
                    bias=hijo.bias is not None,
                    padding_mode=hijo.padding_mode,
                )
                with torch.no_grad():
                    conv_nuevo.weight.zero_()
                    conv_nuevo.weight[:, :in_orig].copy_(hijo.weight)
                    nn.init.normal_(conv_nuevo.weight[:, in_orig:], mean=0.0, std=0.01)
                    if hijo.bias is not None:
                        conv_nuevo.bias.copy_(hijo.bias)
                setattr(parent, nombre, conv_nuevo)
                modificadas += 1
            else:
                reemplazar_en(hijo)

    reemplazar_en(generator)
    return modificadas


#CARGAR Y EXPANDIR LAMA:
def cargar_generator_lama_v7(device):
    if not CKPT_LAMA.exists():
        raise FileNotFoundError(f"big-lama checkpoint not found at {CKPT_LAMA}")

    log.info("building FFCResNetGenerator with big-lama params (input_nc=4)")
    generator = FFCResNetGenerator(**BIG_LAMA_GENERATOR_KWARGS)

    log.info(f"loading big-lama base weights (4-channel) from {CKPT_LAMA}")
    missing, unexpected = cargar_pesos_big_lama(generator, str(CKPT_LAMA), device=device)
    log.info(f"  big-lama loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")

    log.info("expanding first conv 4 -> 7 channels (RGB + mask + part_id + U + V)")
    n_modif = expandir_primer_conv(generator, in_orig=4, in_nuevo=7)
    log.info(f"  convs modified: {n_modif}")

    if CKPT_LAST.exists():
        log.info(f"resuming from previous nobg run: {CKPT_LAST}")
        state = torch.load(str(CKPT_LAST), map_location=device, weights_only=False)
        gen_sd = {k[len("generator."):]: v for k, v in state["state_dict"].items() if k.startswith("generator.")}
        missing, unexpected = generator.load_state_dict(gen_sd, strict=False)
        log.info(f"  state_dict loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")
        start_epoch = int(state.get("epoch", 0)) + 1
        best_val = float(state.get("best_val", float("inf")))
        log.info(f"  resuming at epoch {start_epoch} (previous best_val = {best_val:.4f})")
    else:
        log.info("training from clean expanded big-lama base (no warm-start)")
        start_epoch = 1
        best_val = float("inf")

    generator = generator.to(device).train()
    log.info("LaMa-v7 generator ready (7-channel input)")
    return generator, start_epoch, best_val


#GUARDAR FINETUNEADO:
def guardar_checkpoint_finetuneado(model, generator, path, epoch=None, best_val=None):
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
    log.info(f"Hyperparams: BATCH={BATCH_SIZE}, LR={LR}, EPOCHS={EPOCHS}, IMG={TAMANO_IMG}")
    log.info(f"             L1_W={PESO_L1}, PERC_W={PESO_PERCEPTUAL}, P_PROY={PROB_SIMULAR_PROYECCION}")

    #recolectar tripletas (img, mask_dp15, dp_cache)
    extensiones = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    tripletas = []
    for img_path in sorted(DIR_IMAGENES.iterdir()):
        if img_path.suffix not in extensiones:
            continue
        mask_dp15 = DIR_MASCARAS_DP15 / (img_path.stem + ".png")
        dp_cache = DIR_DP_CACHE / (img_path.stem + ".npz")
        if mask_dp15.exists() and dp_cache.exists():
            tripletas.append((img_path, mask_dp15, dp_cache))
    log.info(f"triplets found (image + 15-class mask + densepose .npz): {len(tripletas)}")
    if len(tripletas) == 0:
        log.error("No triplets found. Make sure style_transfer_bw_first.py and extract_densepose_synthetic.py have both run.")
        return

    #split train/val:
    random.seed(SEED_SPLIT)
    random.shuffle(tripletas)
    n_val = max(1, int(len(tripletas) * VAL_SPLIT))
    tripletas_val = tripletas[:n_val]
    tripletas_train = tripletas[n_val:]
    log.info(f"train: {len(tripletas_train)} | val: {len(tripletas_val)}")

    ds_train = SyntheticInpaintCondDataset(tripletas_train, TAMANO_IMG)
    ds_val= SyntheticInpaintCondDataset(tripletas_val, TAMANO_IMG)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    generator, start_epoch, mejor_val = cargar_generator_lama_v7(device)
    model = None
    perceptual = VGGPerceptualLoss().to(device).eval()
    optim_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))

    if start_epoch > EPOCHS:
        log.info(f"start_epoch ({start_epoch}) > EPOCHS ({EPOCHS}); nothing to do, exit")
        return
    if start_epoch > 1:
        log.info(f"RESUMING v7 training at epoch {start_epoch}/{EPOCHS} (previous best_val={mejor_val:.4f})")

    for epoch in range(start_epoch, EPOCHS + 1):
        generator.train()
        loss_train_acc = 0.0
        n_train = 0
        for img_t, mask_t, cond_t in tqdm(dl_train, desc=f"epoch {epoch} train"):
            img_t  = img_t.to(device, non_blocking=True)
            mask_t = mask_t.to(device, non_blocking=True)
            cond_t = cond_t.to(device, non_blocking=True)

            masked_img = img_t * (1 - mask_t)
            #7 canales: RGB + mask + part_id_norm + U + V
            entrada = torch.cat([masked_img, mask_t, cond_t], dim=1)

            salida = generator(entrada)
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

        generator.eval()
        loss_val_acc = 0.0
        n_val_seen = 0
        with torch.no_grad():
            for img_t, mask_t, cond_t in tqdm(dl_val, desc=f"epoch {epoch} val"):
                img_t  = img_t.to(device, non_blocking=True)
                mask_t = mask_t.to(device, non_blocking=True)
                cond_t = cond_t.to(device, non_blocking=True)

                masked_img = img_t * (1 - mask_t)
                entrada = torch.cat([masked_img, mask_t, cond_t], dim=1)
                salida = generator(entrada)
                pred_compuesto = masked_img + salida * mask_t

                loss_l1 = F.l1_loss(pred_compuesto * mask_t, img_t * mask_t)
                loss_perc = perceptual(pred_compuesto, img_t)
                loss = PESO_L1 * loss_l1 + PESO_PERCEPTUAL * loss_perc

                loss_val_acc += loss.item() * img_t.size(0)
                n_val_seen += img_t.size(0)

        loss_val = loss_val_acc / max(n_val_seen, 1)
        log.info(f"epoch {epoch:02d} | train_loss={loss_train:.4f} | val_loss={loss_val:.4f}")

        if loss_val < mejor_val:
            mejor_val = loss_val
            guardar_checkpoint_finetuneado(model, generator, CKPT_FINETUNED, epoch=epoch, best_val=mejor_val)
            log.info(f"   -> new best val ({loss_val:.4f}), saved")

        #al final de cada epoch guardamos last_finetuned_v7.ckpt para resume
        guardar_checkpoint_finetuneado(model, generator, CKPT_LAST, epoch=epoch, best_val=mejor_val)

    log.info(f"FINE-TUNE v7 COMPLETED - best val loss: {mejor_val:.4f}")
    log.info(f"Fine-tuned checkpoint: {CKPT_FINETUNED}")


if __name__ == "__main__":
    main()
