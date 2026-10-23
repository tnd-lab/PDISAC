"""Training hyperparameters and the default networks, built from configs/pdisac_system.json.

model      : DenoiseModel(RDPDNet), models/bases/pdnet.py (version label "pdnet")
afm_model  : MaskNet of AFM (4 input channels: [Z_rd_hat, Z_rd_alt - Z_rd])
The DnCNN baseline is built in its own training scripts from depth, n_filters,
kernel_size and img_channels.
"""
import warnings

import torch

from models.bases import pdnet, afm
from models.model import DenoiseModel
from configs.isac import N_block, N_chip
from configs.config_loader import cfg


def _resolve_device(name: str) -> str:
    """Fall back to CPU when the requested accelerator is unavailable.

    This module builds the model on import (``from configs.ml import model``),
    so a hard-coded ``cfg["device"]`` (e.g. "mps") would crash the import on a
    CUDA/CPU-only host before the caller can select a device. Resolve it safely.
    """
    try:
        if name == "cuda" and not torch.cuda.is_available():
            fallback = "cpu"
        elif name == "mps" and not (
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ):
            fallback = "cpu"
        else:
            return name
    except Exception:
        fallback = "cpu"
    warnings.warn(
        f"Requested device '{name}' is unavailable; falling back to '{fallback}'.",
        RuntimeWarning,
    )
    return fallback


batch_size = cfg["batch_size"]
num_workers = cfg["num_workers"]
pin_memory = cfg["pin_memory"]
device = _resolve_device(cfg["device"])

epochs = cfg["epochs"]
lr = cfg["lr"]
betas = tuple(cfg["betas"])
n_splits = cfg["n_splits"]

# Training protocol (identical for the four scripts of scripts/train/)
n_folds_run = cfg["folds_run"]                 # GroupKFold folds actually trained
kl_warmup_epochs = cfg["kl_warmup_epochs"]     # lambda_KL grows linearly to kls_thresh (0 = off)
lambda_rec = cfg["lambda_rec"]                 # weight of the target-support term of L_rec
lambda_inf = cfg["lambda_inf"]                 # weight of the inference-path loss L_rec_inf
target_loss = cfg["target_loss"]               # "rel" (target_rel_mse) or "nmse" (target_nmse)

# RDPDNet hyperparameters
input_shape = (2, N_chip, N_block)
hidden_channel = cfg["hidden_channel"]
level = cfg["level"]
mask_thresh = cfg["mask_thresh"]               # lambda_mask: energy penalty of the AFM mask
rec_thresh = cfg["rec_thresh"]                 # weight of L_rec_mix (AFM)
kls_thresh = cfg["kls_thresh"]                 # lambda_KL

# DnCNN hyperparameters
depth = cfg["depth"]
img_channels = cfg["img_channels"]
n_filters = cfg["n_filters"]
kernel_size = cfg["kernel_size"]

# Version label stored in every RDPDNet checkpoint (models/bases/pdnet.py).
# Checkpoints of earlier runs carry "v4" (same network); both labels are accepted.
MODEL_VERSION = "pdnet"
RDPDNET_VERSIONS = ("pdnet", "v4")
current_model = pdnet.RDPDNet(
    input_shape=input_shape, hidden_channel=hidden_channel, level=level
).to(device)
model = DenoiseModel(model=current_model).to(device)

# AFM adversary
afm_model = afm.MaskNet(in_ch=4).to(device)
