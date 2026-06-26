"""
Sanity check rapido del MAT v7 expandido. Carga MAT, expande los convs de
entrada, hace un forward dummy y verifica que las dimensiones cuadran.
"""

import sys
import logging
from pathlib import Path

import torch

BASE = Path("/home/pfc/cescuder/tfg")
sys.path.insert(0, str(BASE / "MAT"))
sys.path.insert(0, str(BASE / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if device == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"Compute: {torch.cuda.get_device_capability(0)}")

    from finetune_mat_v7_densepose import cargar_mat_expandido

    log.info("Cargando MAT expandido...")
    G = cargar_mat_expandido(device)
    G.eval()

    log.info("Probando forward con tensores dummy...")
    B = 1
    H = W = 512
    img = torch.randn(B, 3, H, W, device=device)
    mask = torch.ones(B, 1, H, W, device=device)
    mask[:, :, 100:300, 100:300] = 0.0
    dp = torch.zeros(B, 3, H, W, device=device)
    z = torch.randn(B, G.z_dim, device=device)
    c = torch.zeros(B, G.c_dim, device=device)

    with torch.no_grad():
        out = G(img, mask, dp, z, c, truncation_psi=1, noise_mode="const")

    log.info(f"Forward OK. Output shape: {tuple(out.shape)}")
    log.info(f"Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")

    log.info("Verificando shapes de los convs expandidos:")
    cf_w = G.synthesis.first_stage.conv_first.conv.weight.shape
    log.info(f"  FirstStage.conv_first.weight: {tuple(cf_w)}  (esperado: [180, 7, 3, 3])")
    enc_w = G.synthesis.enc.EncConv_Block_512x512.conv0.weight.shape
    log.info(f"  Encoder.EncConv_Block_512x512.conv0.weight: {tuple(enc_w)}  (esperado: [..., 10, 1, 1])")

    log.info("SANITY CHECK OK")


if __name__ == "__main__":
    main()
