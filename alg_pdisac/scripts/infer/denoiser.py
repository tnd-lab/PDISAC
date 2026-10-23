"""
Denoisers of configs/inference.json: one loader shared by the MATLAB bridge
(scripts/infer/denoise_mat.py) and scripts/test/eval_detection.py.

Each entry of configs/inference.json gives
    name        chain label used by MATLAB (letters, digits, '_')
    version     "pdnet" -> models/bases/pdnet.py (RDPDNet), or "dncnn"
    checkpoint  path relative to alg_pdisac
The architecture is built from the declared version (RDPDNet: hidden_channel and level
from the shared config; DnCNN: the hyperparameters stored in its checkpoint). A
checkpoint whose stored model_version / arch contradicts the declared version is
rejected, so a DnCNN checkpoint can never be loaded into an RDPDNet and vice versa.

denoise() applies the per-map normalisation of RDMapDataset (real and imaginary
channels, mean / std over H x W), runs f_den^infer (model.predict), and maps the output
back with the input statistics: peak positions and CA-CFAR (a relative threshold) are
unaffected, the absolute power of the clean map is not recovered.
"""
from __future__ import annotations

import json
import os
import re
import warnings

import numpy as np
import torch

from configs.config_loader import cfg, resolve_path
from configs.isac import N_chip, N_block
from models.bases import pdnet
from models.bases.dncnn import DnCNN
from models.model import DenoiseModel

INFERENCE_JSON = resolve_path("configs/inference.json")
_RDPDNET = {"pdnet": pdnet}
# model_version labels accepted in an RDPDNet checkpoint (earlier runs stored "v4")
_ALIASES = {"pdnet": ("pdnet", "v4")}


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def read_inference_config(path: str = INFERENCE_JSON) -> dict:
    with open(path) as fh:
        conf = json.load(fh)
    for m in conf["models"]:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", m["name"]):
            raise ValueError(f"inference.json: model name '{m['name']}' must be letters, digits, '_'")
        if m["version"] not in (*_RDPDNET, "dncnn"):
            raise ValueError(f"inference.json: unknown version '{m['version']}' of model '{m['name']}'")
    return conf


def build_model(version: str, ckpt: dict) -> DenoiseModel:
    """Network of the declared version, with the weights of ckpt (strict)."""
    arch = ckpt.get("arch")
    if version == "dncnn":
        if arch != "dncnn":
            raise ValueError("declared version 'dncnn' but the checkpoint is not a DnCNN checkpoint")
        net = DnCNN(**ckpt["dncnn_cfg"])
    else:
        stored = ckpt.get("model_version")
        if arch == "dncnn" or (stored is not None and stored not in _ALIASES[version]):
            raise ValueError(f"declared version '{version}' but the checkpoint is "
                             f"{'a DnCNN' if arch == 'dncnn' else 'model_version ' + str(stored)}")
        net = _RDPDNET[version].RDPDNet(input_shape=(2, N_chip, N_block),
                                        hidden_channel=cfg["hidden_channel"], level=cfg["level"])
    model = DenoiseModel(model=net)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


def load_models(names=None, device: torch.device | str = "auto", path: str = INFERENCE_JSON) -> list[dict]:
    """Models of inference.json (all, or the given names in the given order).
    A missing checkpoint is skipped with a warning, unless it was requested by name."""
    device = select_device(device) if isinstance(device, str) else device
    conf = read_inference_config(path)
    entries = {m["name"]: m for m in conf["models"]}
    if names:
        unknown = [n for n in names if n not in entries]
        if unknown:
            raise KeyError(f"models {unknown} are not in {path}")
        chosen = [entries[n] for n in names]
    else:
        chosen = conf["models"]
    out = []
    for m in chosen:
        ckpt_path = resolve_path(m["checkpoint"])
        if not os.path.isfile(ckpt_path):
            if names:
                raise FileNotFoundError(f"checkpoint of '{m['name']}' not found: {ckpt_path}")
            warnings.warn(f"skipping '{m['name']}': checkpoint not found ({ckpt_path})")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        out.append(dict(name=m["name"], version=m["version"], checkpoint=ckpt_path,
                        epoch=int(ckpt.get("epoch", -1)), val_nmse=float(ckpt.get("val_nmse", np.nan)),
                        net=build_model(m["version"], ckpt).to(device)))
    if not out:
        raise RuntimeError(f"no model of {path} could be loaded")
    return out


def denoise(net: DenoiseModel, Z: np.ndarray, device: torch.device, batch_size: int = 8) -> np.ndarray:
    """Complex B x H x W -> complex64 B x H x W (f_den^infer, input scale)."""
    Z = np.asarray(Z)
    if Z.shape[1:] != (N_chip, N_block):
        raise ValueError(f"maps must be {N_chip} x {N_block} (N_chip x N_block); got {Z.shape[1:]}")
    ch = np.stack((Z.real, Z.imag), axis=1).astype(np.float32)
    mean = ch.mean(axis=(2, 3), keepdims=True)
    std = ch.std(axis=(2, 3), keepdims=True)
    x = (ch - mean) / (std + 1e-8)
    outs = []
    with torch.inference_mode():
        for b in range(0, len(x), batch_size):
            outs.append(net.predict(torch.from_numpy(x[b:b + batch_size]).to(device)).float().cpu().numpy())
    y = np.concatenate(outs) * (std + 1e-8) + mean
    return (y[:, 0] + 1j * y[:, 1]).astype(np.complex64)
