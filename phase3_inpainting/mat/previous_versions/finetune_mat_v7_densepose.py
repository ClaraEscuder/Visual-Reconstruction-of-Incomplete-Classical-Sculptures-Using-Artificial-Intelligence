"""
Fine-tuning de MAT con conditioning DensePose.

Se expande la entrada de imagen de MAT para aceptar 3 canales adicionales (I_pred, U_pred, V_pred) junto a los 3 RGB originales. Concretamente:

    FirstStage.conv_first: 4 -> 7 canales (mask + RGB*mask + I + U + V)
    Encoder.EncConv_Block_512x512: 7 -> 10 canales (mask + composed + RGB*mask + I + U + V)

Para cada imagen, el dataset construye los canales extra leyendo el cache de
DensePose. Con probabilidad PROB_SIMULAR_PROYECCION el conditioning de la zona
enmascarada se reemplaza por un gradiente sintetico (mismo procedimiento que el
mask generator v6.1 hace en inferencia). Esto entrena al modelo a tolerar
conditioning proyectado y no solo el real.

INPUT:
    - imagenes:   ~/tfg/synthetic_dataset_bw_first/images/
    - mascaras 15 clases: ~/tfg/synthetic_dataset_bw_first/masks/
    - cache DP:   ~/tfg/synthetic_dataset_bw_first/densepose_cache/images/
    - pesos MAT:  ~/tfg/MAT/Places_512_FullData_G.pkl

OUTPUT:
    - ~/tfg/MAT/checkpoints/best_finetuned_mat_v7.pt
    - ~/tfg/MAT/checkpoints/last_finetuned_mat_v7.pt
    - ~/tfg/logs/finetune_mat_v7.log
"""

import sys
import random
import logging
from pathlib import Path
from types import MethodType

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

DIR_MAT_REPO = BASE / "MAT"
sys.path.insert(0, str(DIR_MAT_REPO))
sys.path.insert(0, str(BASE / "scripts"))

from networks.mat import Generator, FirstStage, SynthesisNet
from networks.basic_module import Conv2dLayer
from networks.mat import Conv2dLayerPartial


DIR_IMAGENES = BASE / "synthetic_dataset_bw_first" / "images_no_bg"
DIR_MASCARAS_DP15 = BASE / "synthetic_dataset_bw_first" / "masks"
DIR_DP_CACHE = BASE / "synthetic_dataset_bw_first" / "densepose_cache" / "images"

PKL_MAT = DIR_MAT_REPO / "Places_512_FullData_G.pkl"
DIR_CKPT = DIR_MAT_REPO / "checkpoints"
DIR_CKPT.mkdir(parents=True, exist_ok=True)
CKPT_BEST = DIR_CKPT / "best_finetuned_mat_v7_nobg.pt"
CKPT_LAST = DIR_CKPT / "last_finetuned_mat_v7_nobg.pt"


TAMANO_IMG = 512
BATCH_SIZE = 2
EPOCHS = 20
LR = 1e-5
VAL_SPLIT = 0.10
NUM_WORKERS = 2
PESO_L1 = 1.0
PESO_PERCEPTUAL = 0.1
SEED_SPLIT = 42

EARLY_STOP_PATIENCE = 6
EARLY_STOP_MIN_DELTA = 1e-4

N_PARTES_MIN = 1
N_PARTES_MAX = 3
DILATACION_MASK_PX = 8
NUM_CLASES_DENSEPOSE_15 = 15
MAX_PART_ID = 24.0

PROB_SIMULAR_PROYECCION = 0.5


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(BASE / "logs" / "finetune_mat_v7_nobg.log", encoding="utf-8")])
log = logging.getLogger(__name__)


def sintetizar_uv_en_region(mascara_region: np.ndarray):
    """Gradiente lineal a lo largo del eje principal PCA.
    U: proyeccion sobre el eje principal normalizada 0-1.
    V: proyeccion sobre el eje perpendicular normalizada 0-1.
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
    if p_max - p_min > 1e-6:
        proj_p_norm = (proj_p - p_min) / (p_max - p_min)
    else:
        proj_p_norm = np.full_like(proj_p, 0.5)
    if q_max - q_min > 1e-6:
        proj_q_norm = (proj_q - q_min) / (q_max - q_min)
    else:
        proj_q_norm = np.full_like(proj_q, 0.5)
    U[ys, xs] = proj_p_norm.astype(np.float32)
    V[ys, xs] = proj_q_norm.astype(np.float32)
    return U, V


def construir_mascara_aleatoria(mask_dp15: np.ndarray):
    """A partir de la mascara DensePose de 15 clases, escoge entre 1 y 3 partes
    (excluyendo fondo) y devuelve mascara binaria dilatada.
    """
    partes_disponibles = [c for c in range(1, NUM_CLASES_DENSEPOSE_15) if (mask_dp15 == c).any()]
    if not partes_disponibles:
        return np.zeros_like(mask_dp15, dtype=np.uint8)
    n_partes = random.randint(N_PARTES_MIN, min(N_PARTES_MAX, len(partes_disponibles)))
    elegidas = random.sample(partes_disponibles, n_partes)
    mascara = np.zeros_like(mask_dp15, dtype=bool)
    for c in elegidas:
        mascara |= (mask_dp15 == c)
    if DILATACION_MASK_PX > 0:
        mascara = binary_dilation(mascara, iterations=DILATACION_MASK_PX)
    return mascara.astype(np.uint8)


class DatasetMATv7(Dataset):
    def __init__(self, stems, dir_imgs, dir_masks, dir_dp_cache, tamano=TAMANO_IMG):
        self.stems = stems
        self.dir_imgs = dir_imgs
        self.dir_masks = dir_masks
        self.dir_dp_cache = dir_dp_cache
        self.tamano = tamano

    def __len__(self):
        return len(self.stems)

    def _cargar_dp(self, stem):
        """Lee I, U, V crudos del cache DensePose. Si no existe devuelve None."""
        candidatos = [
            self.dir_dp_cache / f"{stem}.jpg.npz",
            self.dir_dp_cache / f"{stem}.png.npz",
            self.dir_dp_cache / f"{stem}.npz",
        ]
        for c in candidatos:
            if c.exists():
                d = np.load(c)
                I = d["I_pred"] if "I_pred" in d.files else d.get("labels", None)
                U = d["U_pred"] if "U_pred" in d.files else d.get("U", None)
                V = d["V_pred"] if "V_pred" in d.files else d.get("V", None)
                if I is None or U is None or V is None:
                    return None
                return I.astype(np.float32), U.astype(np.float32), V.astype(np.float32)
        return None

    def __getitem__(self, idx):
        stem = self.stems[idx]
        try:
            img = Image.open(self.dir_imgs / f"{stem}.jpg").convert("RGB")
        except (OSError, Image.UnidentifiedImageError):
            return self.__getitem__((idx + 1) % len(self))

        mask_dp15 = np.array(Image.open(self.dir_masks / f"{stem}.png"))
        if mask_dp15.ndim == 3:
            mask_dp15 = mask_dp15[..., 0]

        img = img.resize((self.tamano, self.tamano), Image.BILINEAR)
        mask_dp15 = np.array(Image.fromarray(mask_dp15).resize((self.tamano, self.tamano), Image.NEAREST))

        mascara_bin = construir_mascara_aleatoria(mask_dp15)

        dp_raw = self._cargar_dp(stem)
        if dp_raw is not None:
            I_full, U_full, V_full = dp_raw
            h0, w0 = I_full.shape
            if (h0, w0) != (self.tamano, self.tamano):
                I_full = np.array(Image.fromarray(I_full.astype(np.uint8)).resize((self.tamano, self.tamano), Image.NEAREST)).astype(np.float32)
                U_full = np.array(Image.fromarray(U_full).resize((self.tamano, self.tamano), Image.BILINEAR)).astype(np.float32)
                V_full = np.array(Image.fromarray(V_full).resize((self.tamano, self.tamano), Image.BILINEAR)).astype(np.float32)
        else:
            I_full = np.zeros((self.tamano, self.tamano), dtype=np.float32)
            U_full = np.zeros((self.tamano, self.tamano), dtype=np.float32)
            V_full = np.zeros((self.tamano, self.tamano), dtype=np.float32)

        if random.random() < PROB_SIMULAR_PROYECCION and mascara_bin.any():
            I_sim = I_full.copy()
            U_sim = U_full.copy()
            V_sim = V_full.copy()
            region_bool = mascara_bin.astype(bool)
            u_grad, v_grad = sintetizar_uv_en_region(region_bool)
            U_sim[region_bool] = u_grad[region_bool]
            V_sim[region_bool] = v_grad[region_bool]
            partes_visibles = I_full[(mask_dp15 > 0) & (mask_dp15 < NUM_CLASES_DENSEPOSE_15)]
            partes_visibles = partes_visibles[partes_visibles > 0]
            if partes_visibles.size > 0:
                I_sim[region_bool] = float(np.median(partes_visibles))
            I_full, U_full, V_full = I_sim, U_sim, V_sim

        img_np = np.array(img).astype(np.float32) / 127.5 - 1.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float()

        mask_t = torch.from_numpy(1.0 - mascara_bin.astype(np.float32)).unsqueeze(0)

        I_norm = (I_full / MAX_PART_ID).clip(0, 1) * 2.0 - 1.0
        U_norm = U_full.clip(0, 1) * 2.0 - 1.0
        V_norm = V_full.clip(0, 1) * 2.0 - 1.0
        dp_t = torch.from_numpy(np.stack([I_norm, U_norm, V_norm], axis=0)).float()

        return img_t, mask_t, dp_t


def expandir_conv2d_layer(conv_orig: Conv2dLayer, in_nuevo: int) -> Conv2dLayer:
    """Crea un nuevo Conv2dLayer con mas canales de entrada, copiando los pesos
    originales en los primeros in_orig canales y rellenando los nuevos con ruido peque"""
    in_orig = conv_orig.weight.shape[1]
    kernel_size = conv_orig.weight.shape[-1]
    out_channels = conv_orig.weight.shape[0]
    conv_nuevo = Conv2dLayer(
        in_channels=in_nuevo,
        out_channels=out_channels,
        kernel_size=kernel_size,
        bias=conv_orig.bias is not None,
        activation=conv_orig.activation,
        up=conv_orig.up,
        down=conv_orig.down,
        conv_clamp=conv_orig.conv_clamp,
        trainable=True,
    )
    with torch.no_grad():
        conv_nuevo.weight.zero_()
        conv_nuevo.weight[:, :in_orig].copy_(conv_orig.weight)
        nn.init.normal_(conv_nuevo.weight[:, in_orig:], mean=0.0, std=0.01)
        if conv_orig.bias is not None:
            conv_nuevo.bias.copy_(conv_orig.bias)
    return conv_nuevo


def expandir_conv2d_partial(conv_partial_orig: Conv2dLayerPartial, in_nuevo: int) -> Conv2dLayerPartial:
    """Sustituye el Conv2dLayer interno de un Conv2dLayerPartial por uno con mas
    canales de entrada. La parte de actualizacion de mascara no cambia
    """
    in_orig = conv_partial_orig.conv.weight.shape[1]
    nueva = Conv2dLayerPartial(
        in_channels=in_nuevo,
        out_channels=conv_partial_orig.conv.weight.shape[0],
        kernel_size=conv_partial_orig.conv.weight.shape[-1],
        bias=conv_partial_orig.conv.bias is not None,
        activation=conv_partial_orig.conv.activation,
    )
    with torch.no_grad():
        nueva.conv.weight.zero_()
        nueva.conv.weight[:, :in_orig].copy_(conv_partial_orig.conv.weight)
        nn.init.normal_(nueva.conv.weight[:, in_orig:], mean=0.0, std=0.01)
        if conv_partial_orig.conv.bias is not None:
            nueva.conv.bias.copy_(conv_partial_orig.conv.bias)
    return nueva


def first_stage_forward_con_dp(self, images_in, masks_in, ws, dp_in, noise_mode='random'):
    """Reemplaza el forward original de FirstStage para concatenar dp_in (3 canales)
    como entrada extra del conv_first
    """
    x = torch.cat([masks_in - 0.5, images_in * masks_in, dp_in], dim=1)
    skips = []
    x, mask = self.conv_first(x, masks_in)
    skips.append(x)
    for i, block in enumerate(self.enc_conv):
        x, mask = block(x, mask)
        if i != len(self.enc_conv) - 1:
            skips.append(x)
    x_size = x.size()[-2:]
    from networks.mat import feature2token, token2feature
    x = feature2token(x)
    mask = feature2token(mask)
    mid = len(self.tran) // 2
    for i, block in enumerate(self.tran):
        if i < mid:
            x, x_size, mask = block(x, x_size, mask)
            skips.append(x)
        elif i > mid:
            x, x_size, mask = block(x, x_size, None)
            x = x + skips[mid - i]
        else:
            x, x_size, mask = block(x, x_size, None)
            mul_map = torch.ones_like(x) * 0.5
            mul_map = F.dropout(mul_map, training=True)
            ws_ = self.ws_style(ws[:, -1])
            add_n = self.to_square(ws_).unsqueeze(1)
            add_n = F.interpolate(add_n, size=x.size(1), mode='linear', align_corners=False).squeeze(1).unsqueeze(-1)
            x = x * mul_map + add_n * (1 - mul_map)
            gs = self.to_style(self.down_conv(token2feature(x, x_size)).flatten(start_dim=1))
            style = torch.cat([gs, ws_], dim=1)
    x = token2feature(x, x_size).contiguous()
    img = None
    for i, block in enumerate(self.dec_conv):
        x, img = block(x, img, style, skips[len(self.dec_conv)-i-1], noise_mode=noise_mode)
    img = img * (1 - masks_in) + images_in * masks_in
    return img


def synthesis_forward_con_dp(self, images_in, masks_in, ws, dp_in, noise_mode='random', return_stg1=False):
    """Reemplaza el forward original de SynthesisNet para pasar dp_in tanto a
    FirstStage como concatenarlo en el encoder del segundo stage
    """
    out_stg1 = self.first_stage(images_in, masks_in, ws, dp_in, noise_mode=noise_mode)
    x = images_in * masks_in + out_stg1 * (1 - masks_in)
    x = torch.cat([masks_in - 0.5, x, images_in * masks_in, dp_in], dim=1)
    E_features = self.enc(x)
    fea_16 = E_features[4]
    mul_map = torch.ones_like(fea_16) * 0.5
    mul_map = F.dropout(mul_map, training=True)
    add_n = self.to_square(ws[:, 0]).view(-1, 16, 16).unsqueeze(1)
    add_n = F.interpolate(add_n, size=fea_16.size()[-2:], mode='bilinear', align_corners=False)
    fea_16 = fea_16 * mul_map + add_n * (1 - mul_map)
    E_features[4] = fea_16
    gs = self.to_style(fea_16)
    img = self.dec(fea_16, ws, gs, E_features, noise_mode=noise_mode)
    img = img * (1 - masks_in) + images_in * masks_in
    if not return_stg1:
        return img
    return img, out_stg1


def generator_forward_con_dp(self, images_in, masks_in, dp_in, z, c, truncation_psi=1, truncation_cutoff=None,
                              skip_w_avg_update=False, noise_mode='random', return_stg1=False):
    ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                      skip_w_avg_update=skip_w_avg_update)
    if not return_stg1:
        return self.synthesis(images_in, masks_in, ws, dp_in, noise_mode=noise_mode)
    return self.synthesis(images_in, masks_in, ws, dp_in, noise_mode=noise_mode, return_stg1=True)


def cargar_mat_expandido(device):
    """Carga MAT con pesos del pkl, expande los convs de entrada para aceptar 3
    canales DensePose extra, y monkey-patcha los forwards. Devuelve Generator listo
    para entrenar
    """
    log.info(f"Loading MAT weights from: {PKL_MAT}")
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=TAMANO_IMG, img_channels=3)
    state_dict = torch.load(str(PKL_MAT), map_location="cpu", weights_only=False)
    missing, unexpected = G.load_state_dict(state_dict, strict=False)
    log.info(f"Loaded base MAT (missing: {len(missing)}, unexpected: {len(unexpected)})")

    G.synthesis.first_stage.conv_first = expandir_conv2d_partial(
        G.synthesis.first_stage.conv_first, in_nuevo=7)
    log.info("Expanded FirstStage.conv_first: 4 -> 7 channels")

    enc_attr = f"EncConv_Block_{TAMANO_IMG}x{TAMANO_IMG}"
    enc_first = getattr(G.synthesis.enc, enc_attr)
    enc_first.conv0 = expandir_conv2d_layer(enc_first.conv0, in_nuevo=10)
    log.info(f"Expanded Encoder.{enc_attr}.conv0: 7 -> 10 channels")

    G.synthesis.first_stage.forward = MethodType(first_stage_forward_con_dp, G.synthesis.first_stage)
    G.synthesis.forward = MethodType(synthesis_forward_con_dp, G.synthesis)
    G.forward = MethodType(generator_forward_con_dp, G)

    return G.to(device)


class PerdidaPerceptual(nn.Module):
    """VGG16 features. Calcula L1 entre activaciones en varias capas"""
    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(4)])
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(4, 9)])
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(9, 16)])
        self.slice4 = nn.Sequential(*[vgg[i] for i in range(16, 23)])
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _norm(self, x):
        x_01 = (x + 1.0) / 2.0
        return (x_01 - self.mean) / self.std

    def forward(self, pred, target):
        p, t = self._norm(pred), self._norm(target)
        loss = 0.0
        for slc in (self.slice1, self.slice2, self.slice3, self.slice4):
            p, t = slc(p), slc(t)
            loss = loss + F.l1_loss(p, t)
        return loss


def calcular_perdida(pred, target, mask, perceptual_fn):
    """L1 + perceptual sobre toda la imagen (igual que en LaMa v7)"""
    l1 = F.l1_loss(pred, target)
    perc = perceptual_fn(pred, target)
    return PESO_L1 * l1 + PESO_PERCEPTUAL * perc, l1.item(), perc.item()


def listar_stems_validos():
    """Stems que tienen imagen + mascara dp15 disponibles"""
    stems_img = {p.stem for p in DIR_IMAGENES.glob("*.jpg")}
    stems_mask = {p.stem for p in DIR_MASCARAS_DP15.glob("*.png")}
    return sorted(stems_img & stems_mask)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if device == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}  compute: {torch.cuda.get_device_capability(0)}")

    random.seed(SEED_SPLIT)
    np.random.seed(SEED_SPLIT)
    torch.manual_seed(SEED_SPLIT)

    stems = listar_stems_validos()
    log.info(f"Total stems disponibles: {len(stems)}")
    if not stems:
        log.error("Sin datos de entrenamiento. Saliendo.")
        return

    random.shuffle(stems)
    n_val = max(1, int(len(stems) * VAL_SPLIT))
    stems_val = stems[:n_val]
    stems_train = stems[n_val:]
    log.info(f"train: {len(stems_train)}  val: {len(stems_val)}")

    ds_train = DatasetMATv7(stems_train, DIR_IMAGENES, DIR_MASCARAS_DP15, DIR_DP_CACHE)
    ds_val = DatasetMATv7(stems_val, DIR_IMAGENES, DIR_MASCARAS_DP15, DIR_DP_CACHE)
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, drop_last=False, pin_memory=True)

    G = cargar_mat_expandido(device)
    G.train()
    perceptual_fn = PerdidaPerceptual().to(device).eval()

    params_entrenables = [p for p in G.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params_entrenables, lr=LR, betas=(0.5, 0.999))
    log.info(f"Parametros entrenables: {sum(p.numel() for p in params_entrenables):,}")

    start_epoch = 0
    best_val = float("inf")
    epochs_sin_mejora = 0
    if CKPT_LAST.exists():
        log.info(f"Reanudando desde {CKPT_LAST}")
        state = torch.load(str(CKPT_LAST), map_location=device, weights_only=False)
        G.load_state_dict(state["generator"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best_val = float(state.get("best_val", float("inf")))
        epochs_sin_mejora = int(state.get("epochs_sin_mejora", 0))
        log.info(f"Reanudando epoch={start_epoch}  best_val={best_val:.4f}  sin_mejora={epochs_sin_mejora}")
    else:
        log.info("entrenando desde MAT expandido limpio (sin warm-start desde v7 anterior)")

    for epoch in range(start_epoch, EPOCHS):
        G.train()
        train_l1, train_perc, train_total, n_batches = 0.0, 0.0, 0.0, 0
        for img_t, mask_t, dp_t in tqdm(dl_train, desc=f"train e{epoch}"):
            img_t = img_t.to(device, non_blocking=True)
            mask_t = mask_t.to(device, non_blocking=True)
            dp_t = dp_t.to(device, non_blocking=True)

            z = torch.randn(img_t.size(0), G.z_dim, device=device)
            c = torch.zeros(img_t.size(0), G.c_dim, device=device)

            pred = G(img_t, mask_t, dp_t, z, c, truncation_psi=1, noise_mode="const")
            loss, l1_v, perc_v = calcular_perdida(pred, img_t, mask_t, perceptual_fn)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_entrenables, max_norm=5.0)
            optimizer.step()

            train_total += loss.item()
            train_l1 += l1_v
            train_perc += perc_v
            n_batches += 1

        train_l1 /= max(n_batches, 1)
        train_perc /= max(n_batches, 1)
        train_total /= max(n_batches, 1)
        log.info(f"epoch {epoch} TRAIN  total={train_total:.4f}  l1={train_l1:.4f}  perc={train_perc:.4f}")

        G.eval()
        val_total, n_val_batches = 0.0, 0
        with torch.no_grad():
            for img_t, mask_t, dp_t in tqdm(dl_val, desc=f"val e{epoch}"):
                img_t = img_t.to(device, non_blocking=True)
                mask_t = mask_t.to(device, non_blocking=True)
                dp_t = dp_t.to(device, non_blocking=True)
                z = torch.randn(img_t.size(0), G.z_dim, device=device)
                c = torch.zeros(img_t.size(0), G.c_dim, device=device)
                pred = G(img_t, mask_t, dp_t, z, c, truncation_psi=1, noise_mode="const")
                loss, _, _ = calcular_perdida(pred, img_t, mask_t, perceptual_fn)
                val_total += loss.item()
                n_val_batches += 1
        val_total /= max(n_val_batches, 1)
        log.info(f"epoch {epoch} VAL    total={val_total:.4f}")

        mejora = val_total < (best_val - EARLY_STOP_MIN_DELTA)
        if mejora:
            best_val = val_total
            epochs_sin_mejora = 0
        else:
            epochs_sin_mejora += 1

        state_save = {
            "epoch": epoch,
            "best_val": best_val,
            "epochs_sin_mejora": epochs_sin_mejora,
            "generator": G.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(state_save, str(CKPT_LAST))

        if mejora:
            torch.save(state_save, str(CKPT_BEST))
            log.info(f"epoch {epoch} new best val {val_total:.4f}, saved to {CKPT_BEST.name}")
        else:
            log.info(f"epoch {epoch} sin mejora ({epochs_sin_mejora}/{EARLY_STOP_PATIENCE})  best={best_val:.4f}")

        if epochs_sin_mejora >= EARLY_STOP_PATIENCE:
            log.info(f"Early stopping en epoch {epoch}: {EARLY_STOP_PATIENCE} epochs sin mejora. "
                     f"Mejor val loss: {best_val:.4f}")
            break


if __name__ == "__main__":
    main()

