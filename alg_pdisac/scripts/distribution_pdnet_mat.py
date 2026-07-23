"""File-based RDPDNet distribution-export bridge for MATLAB (main9b).

Companion to scripts/inference_pdnet_mat.py and scripts/inference_pdnet_mask_mat.py.
Instead of returning the denoised RD map, it calls

    model.distribution(Z_rd=..., Z_rd_prbs=..., distributions={})

which populates the full latent hierarchy:

    p(Z_rd), p(Z_rd^prbs)          — observed / clean inputs
    p(z_i | z_{i-1})               — bottom-up (encoder) latents
    p(z_L)                         — top latent
    p(z_{i-1} | z_i)               — top-down (decoder) latents
    p(Z_rd_hat)                    — reconstruction

Every distribution is aggregated over the batch (Monte-Carlo trials) and over
the channel dimension with mean aggregation, giving one H x W map per
distribution that MATLAB renders
as a 3-D surface.

MATLAB input variables:
    RD_map_input_inference : complex H x W, or complex B x H x W (noisy, with data)
    RD_map_prbs_inference  : complex, same shape (clean / data-free reference)
    trial_id_inference     : optional identifier(s)

Python output variables:
    dist_maps_inference   : float32 K x H x W, batch/channel-averaged maps
    dist_labels_inference : 1 x K cell of LaTeX label strings
    dist_counts_inference : int32 1 x K, number of samples averaged per map
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
PRBS_KEY = "RD_map_prbs_inference"
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
    return normalized.astype(np.float32)


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
    for key in (INPUT_KEY, PRBS_KEY):
        if key not in input_data:
            raise KeyError(f"MAT file does not contain required variable '{key}'.")

    rd_map, _ = _as_batched_complex(input_data[INPUT_KEY])
    rd_prbs, _ = _as_batched_complex(input_data[PRBS_KEY])
    if rd_prbs.shape != rd_map.shape:
        raise ValueError(
            f"{PRBS_KEY} shape {rd_prbs.shape} differs from "
            f"{INPUT_KEY} shape {rd_map.shape}."
        )

    normalized = _normalize(rd_map)
    prbs_normalized = _normalize(rd_prbs)

    expected_shape = (2, N_chip, N_block)
    if tuple(normalized.shape[1:]) != expected_shape:
        raise ValueError(
            f"Checkpoint/model expects CxHxW={expected_shape}, but input gives "
            f"{tuple(normalized.shape[1:])}."
        )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Running batch-sum of the channel-averaged map, per distribution key.
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}

    with torch.inference_mode():
        for start in range(0, len(normalized), args.batch_size):
            stop = start + args.batch_size
            z_rd = torch.from_numpy(normalized[start:stop]).to(device)
            z_prbs = torch.from_numpy(prbs_normalized[start:stop]).to(device)

            # Pass a fresh dict so the mutable default is never reused.
            distributions = model.distribution(
                Z_rd=z_rd, Z_rd_prbs=z_prbs, distributions={}
            )

            for key, value in distributions.items():
                if not torch.is_tensor(value) or value.dim() != 4:
                    continue
                # mean over channels (dim=1), sum over the batch (dim=0)
                partial = value.detach().float().mean(dim=1).sum(dim=0)
                partial = partial.cpu().numpy()
                if key in sums:
                    sums[key] += partial
                else:
                    sums[key] = partial
                counts[key] = counts.get(key, 0) + int(value.shape[0])

    if not sums:
        raise RuntimeError("model.distribution() returned no 4-D tensors to aggregate.")

    labels = list(sums.keys())
    maps = np.stack([sums[k] / max(counts[k], 1) for k in labels]).astype(np.float32)

    output = {
        "dist_maps_inference": maps,
        "dist_labels_inference": np.array(labels, dtype=object),
        "dist_counts_inference": np.array(
            [counts[k] for k in labels], dtype=np.int32
        ),
        "checkpoint_inference": str(Path(args.checkpoint).resolve()),
    }
    if TRIAL_KEY in input_data:
        output[TRIAL_KEY] = input_data[TRIAL_KEY]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(output_path, output, do_compression=True)
    print(
        f"RDPDNet distribution export complete on {device}: "
        f"{len(labels)} distributions, {maps.shape[1]}x{maps.shape[2]} -> {output_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export RDPDNet latent/RD distributions for MATLAB 3-D plotting."
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
