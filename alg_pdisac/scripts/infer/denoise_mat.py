"""
MATLAB bridge: denoise a batch of RD maps with the models of configs/inference.json.

Input  MAT (from simulation_matlab/inference_run_pdnet.m):
    RD_map_input_inference      complex B x H x W (or H x W)
Output MAT:
    RD_map_denoised_inference   complex64 K x B x H x W  (K models, same order as model_name_inference)
    model_name_inference        1 x K cell   (chain labels)
    model_version_inference     1 x K cell   (pdnet or dncnn)
    checkpoint_inference        1 x K cell   (absolute paths)
    epoch_inference, val_nmse_inference   1 x K
Every model sees the same maps; torch and the models are loaded once per call.

Run (from alg_pdisac, or with alg_pdisac on PYTHONPATH):
    python scripts/infer/denoise_mat.py --input in.mat --output out.mat [--models pdnet,dncnn]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ALG = str(Path(__file__).resolve().parents[2])          # alg_pdisac
if _ALG not in sys.path:
    sys.path.insert(0, _ALG)

import numpy as np
from scipy.io import loadmat, savemat

from scripts.infer.denoiser import INFERENCE_JSON, denoise, load_models, select_device

INPUT_KEY, OUTPUT_KEY = "RD_map_input_inference", "RD_map_denoised_inference"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--models", default="", help="comma-separated names of inference.json ('' = all available)")
    ap.add_argument("--device", default="", help="auto | cpu | cuda | mps ('' = inference.json)")
    ap.add_argument("--batch-size", type=int, default=0, help="0 = inference.json")
    args = ap.parse_args()

    with open(INFERENCE_JSON) as fh:
        conf = json.load(fh)
    device = select_device(args.device or conf.get("device", "auto"))
    batch_size = args.batch_size or int(conf.get("batch_size", 8))
    names = [n.strip() for n in args.models.split(",") if n.strip()]

    data = loadmat(args.input)
    if INPUT_KEY not in data:
        raise KeyError(f"{args.input} has no variable '{INPUT_KEY}'")
    Z = np.asarray(data[INPUT_KEY])
    single = Z.ndim == 2
    if single:
        Z = Z[None]
    if Z.ndim != 3:
        raise ValueError(f"{INPUT_KEY} must be H x W or B x H x W; got {Z.shape}")

    models = load_models(names or None, device=device)
    out = np.stack([denoise(m["net"], Z, device, batch_size) for m in models])   # K x B x H x W
    if single:
        out = out[:, 0]
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("non-finite values in the denoised maps")

    cell = lambda xs: np.array(list(xs), dtype=object)[None, :]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    savemat(args.output, {
        OUTPUT_KEY: out,
        "model_name_inference": cell(m["name"] for m in models),
        "model_version_inference": cell(m["version"] for m in models),
        "checkpoint_inference": cell(m["checkpoint"] for m in models),
        "epoch_inference": np.array([[m["epoch"] for m in models]], dtype=float),
        "val_nmse_inference": np.array([[m["val_nmse"] for m in models]], dtype=float),
    }, do_compression=True)
    print(f"[denoise_mat] {len(models)} model(s) {[m['name'] for m in models]} on {device}: "
          f"{Z.shape[0]} map(s) -> {args.output}")


if __name__ == "__main__":
    main()
