import warnings

import torch

from models.bases import pdnet, dncnn, afm
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

# Denoise model hyperparameters
input_shape = (2, N_chip, N_block)
hidden_channel = cfg["hidden_channel"]
level = cfg["level"]
mask_thresh = cfg["mask_thresh"]
rec_thresh = cfg["rec_thresh"]
kls_thresh = cfg["kls_thresh"]

depth = cfg["depth"]
img_channels = cfg["img_channels"]
n_filters = cfg["n_filters"]
kernel_size = cfg["kernel_size"]

current_model = pdnet.RDPDNet(
    input_shape=input_shape, hidden_channel=hidden_channel, level=level
).to(device)
# current_model = dncnn.DnCNN(
#    depth=depth,
#    img_channels=img_channels,
#    n_filters=n_filters,
#    kernel_size=kernel_size,
# ).to(device)
model = DenoiseModel(model=current_model).to(device)

# training model
afm_model = afm.MaskNet(in_ch=2).to(device)
