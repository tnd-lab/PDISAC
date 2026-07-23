"""File-based RDPDNet inference bridge for MATLAB.

MATLAB input variables (all names use the requested ``_inference`` suffix):
    RD_map_input_inference : complex H x W, or complex B x H x W
    trial_id_inference     : optional identifier(s)

Python output variables:
    RD_map_denoised_inference : complex H x W, or complex B x H x W
    trial_id_inference        : copied when present
    normalization_*_inference : statistics used to restore physical scale

Run from ``alg_pdisac`` with its repository root on PYTHONPATH, for example:

    PYTHONPATH=. .venv/bin/python scripts/inference_pdnet_mat.py \
      --checkpoint exps/full_training/fold_1/checkpoints/best_checkpoint.pth \
      --input /tmp/dataset_inference_input.mat \
      --output /tmp/dataset_inference_output.mat --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat, savemat

from configs.ml import model
from configs.isac import N_block, N_chip


INPUT_KEY = "RD_map_input_inference"
OUTPUT_KEY = "RD_map_denoised_inference"
TRIAL_KEY = "trial_id_inference"


def _as_batched_complex(value: np.ndarray) -> tuple[np.ndarray, bool]:
    value = np.asarray(value)
    if value.ndim == 2:
        return value[None, ...], True
    if value.ndim == 3:
        return value, False
    raise ValueError(
        f"{INPUT_KEY} must have shape HxW or BxHxW; received {value.shape}."
    )


def _normalize(value: np.ndarray, eps: float = 1e-8):
    """Match RDMapDataset: independent real/imag normalization over H,W."""
    channels = np.stack((value.real, value.imag), axis=1).astype(np.float32)
    mean = channels.mean(axis=(2, 3), keepdims=True)
    std = channels.std(axis=(2, 3), keepdims=True)
    normalized = (channels - mean) / (std + eps)
    return normalized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _restore(value: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    # NOTE: the network output lives in the per-map normalized space it was
    # trained in. We de-normalize with the *input (noisy)* map's mean/std,
    # because the clean map's statistics are not available at inference. The
    # absolute amplitude of the returned map therefore matches the noisy input
    # scale, not the physical clean scale. This is harmless for CA-CFAR (a
    # relative threshold) and for range/velocity peak locations, but do not
    # read absolute power / SNR off the denoised map.
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

    # weights_only=False: this is our own trusted training checkpoint, which is
    # a full dict (model + AFM + optimizer states + scalars). PyTorch >= 2.6
    # defaults weights_only=True and would reject those non-tensor objects.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    expected_shape = (2, N_chip, N_block)
    actual_shape = tuple(normalized.shape[1:])
    if actual_shape != expected_shape:
        raise ValueError(
            f"Checkpoint/model expects CxHxW={expected_shape}, but input gives "
            f"{actual_shape}. Check MATLAB/Python N_chip and N_block settings."
        )

    outputs = []
    with torch.inference_mode():
        for start in range(0, len(normalized), args.batch_size):
            batch = torch.from_numpy(normalized[start : start + args.batch_size]).to(device)
            outputs.append(model.predict(batch).cpu())

    prediction = torch.cat(outputs, dim=0)
    denoised = _restore(prediction, mean, std)
    if remove_batch_dimension:
        denoised = denoised[0]

    output = {
        OUTPUT_KEY: denoised,
        "normalization_mean_inference": mean.squeeze(-1).squeeze(-1),
        "normalization_std_inference": std.squeeze(-1).squeeze(-1),
        "checkpoint_inference": str(Path(args.checkpoint).resolve()),
    }
    if TRIAL_KEY in input_data:
        output[TRIAL_KEY] = input_data[TRIAL_KEY]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(output_path, output, do_compression=True)
    print(f"RDPDNet inference complete on {device}: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained RDPDNet on MATLAB RD maps.")
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
