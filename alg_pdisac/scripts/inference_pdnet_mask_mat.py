"""File-based RDPDNet + AFM-mask inference bridge for MATLAB (main9b).

Extends scripts/inference_pdnet_mat.py: in addition to the denoised RD map,
it reconstructs the adversarial frequency mask exactly as in training
(scripts/train.py):

    Z_rd_hat = model.predict(Z_rd)              # normalized space
    M_mask   = afm_model(Z_rd_hat)              # MaskNet, in [0,1]
    M_tars   = model.create_mask(Z_rd_prbs)     # target-protection mask
    M_afm    = M_mask * (1 - M_tars)

MATLAB input variables:
    RD_map_input_inference : complex H x W, or complex B x H x W (noisy, with data)
    RD_map_prbs_inference  : complex, same shape (no-data map, for M_tars).
                             Optional; if absent, M_afm = M_mask.
    trial_id_inference     : optional identifier(s)

Python output variables:
    RD_map_denoised_inference : complex, same shape as input
    M_afm_inference           : float32 (B x) 2 x H x W in [0,1]
    M_mask_inference          : float32 (B x) 2 x H x W in [0,1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat, savemat

from configs.ml import model, afm_model
from configs.isac import N_block, N_chip


INPUT_KEY = "RD_map_input_inference"
PRBS_KEY = "RD_map_prbs_inference"
OUTPUT_KEY = "RD_map_denoised_inference"
TRIAL_KEY = "trial_id_inference"


def _as_batched_complex(value: np.ndarray) -> tuple[np.ndarray, bool]:
    value = np.asarray(value)
    if value.ndim == 2:
        return value[None, ...], True
    if value.ndim == 3:
        return value, False
    raise ValueError(f"input must have shape HxW or BxHxW; received {value.shape}.")


def _normalize(value: np.ndarray, eps: float = 1e-8):
    """Match RDMapDataset: independent real/imag normalization over H,W."""
    channels = np.stack((value.real, value.imag), axis=1).astype(np.float32)
    mean = channels.mean(axis=(2, 3), keepdims=True)
    std = channels.std(axis=(2, 3), keepdims=True)
    normalized = (channels - mean) / (std + eps)
    return normalized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _restore(value: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    channels = value.detach().cpu().numpy()
    channels = channels * (std + 1e-8) + mean
    return (channels[:, 0] + 1j * channels[:, 1]).astype(np.complex64)


def _select_device(requested: str) -> torch.device:
    def _mps_available() -> bool:
        backend = getattr(torch.backends, "mps", None)
        return bool(backend and backend.is_available())

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def run(args: argparse.Namespace) -> None:
    device = _select_device(args.device)
    input_data = loadmat(args.input)
    if INPUT_KEY not in input_data:
        raise KeyError(f"MAT file does not contain required variable '{INPUT_KEY}'.")

    rd_map, remove_batch_dimension = _as_batched_complex(input_data[INPUT_KEY])
    normalized, mean, std = _normalize(rd_map)

    have_prbs = PRBS_KEY in input_data
    if have_prbs:
        rd_prbs, _ = _as_batched_complex(input_data[PRBS_KEY])
        if rd_prbs.shape != rd_map.shape:
            raise ValueError(
                f"{PRBS_KEY} shape {rd_prbs.shape} differs from "
                f"{INPUT_KEY} shape {rd_map.shape}."
            )
        prbs_normalized, _, _ = _normalize(rd_prbs)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if "afm_model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint lacks 'afm_model_state_dict'; cannot rebuild M_afm. "
            "Use a checkpoint saved by train.py (the AFM trainer)."
        )
    afm_model.load_state_dict(checkpoint["afm_model_state_dict"])
    afm_model.to(device)
    afm_model.eval()

    expected_shape = (2, N_chip, N_block)
    if tuple(normalized.shape[1:]) != expected_shape:
        raise ValueError(
            f"Checkpoint/model expects CxHxW={expected_shape}, but input gives "
            f"{tuple(normalized.shape[1:])}."
        )

    outputs, masks_afm, masks_raw, masks_tars = [], [], [], []
    with torch.inference_mode():
        for start in range(0, len(normalized), args.batch_size):
            stop = start + args.batch_size
            batch = torch.from_numpy(normalized[start:stop]).to(device)
            z_rd_hat = model.predict(batch)
            m_mask = afm_model(z_rd_hat)
            if have_prbs:
                prbs_batch = torch.from_numpy(prbs_normalized[start:stop]).to(device)
                m_tars = model.create_mask(prbs_batch)
                m_afm = m_mask * (1 - m_tars)
                masks_tars.append(m_tars.cpu())
            else:
                m_afm = m_mask
            outputs.append(z_rd_hat.cpu())
            masks_afm.append(m_afm.cpu())
            masks_raw.append(m_mask.cpu())

    prediction = torch.cat(outputs, dim=0)
    denoised = _restore(prediction, mean, std)
    m_afm = torch.cat(masks_afm, dim=0).numpy().astype(np.float32)
    m_mask = torch.cat(masks_raw, dim=0).numpy().astype(np.float32)
    if have_prbs:
        m_tars = torch.cat(masks_tars, dim=0).numpy().astype(np.float32)
    else:
        m_tars = np.zeros_like(m_mask)
    if remove_batch_dimension:
        denoised = denoised[0]
        m_afm = m_afm[0]
        m_mask = m_mask[0]
        m_tars = m_tars[0]

    output = {
        OUTPUT_KEY: denoised,
        "M_afm_inference": m_afm,
        "M_mask_inference": m_mask,
        "M_tars_inference": m_tars,
        "normalization_mean_inference": mean.squeeze(-1).squeeze(-1),
        "normalization_std_inference": std.squeeze(-1).squeeze(-1),
        "checkpoint_inference": str(Path(args.checkpoint).resolve()),
    }
    if TRIAL_KEY in input_data:
        output[TRIAL_KEY] = input_data[TRIAL_KEY]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(output_path, output, do_compression=True)
    print(f"RDPDNet+AFM inference complete on {device}: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run trained RDPDNet + AFM MaskNet on MATLAB RD maps."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


if __name__ == "__main__":
    run(parse_args())
