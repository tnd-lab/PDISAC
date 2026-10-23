"""
NMSE of the trained denoisers on a validation fold (same data, normalisation and
GroupKFold split as scripts/train/*).

For every validation map and every chain
    input   : Z_rd itself (no denoiser)
    <name>  : f_den^infer(Z_rd) of a model of configs/inference.json
it computes NMSE = ||Zhat - Z_prbs||^2 / ||Z_prbs||^2 on the normalised maps (the
validation metric used for checkpoint selection) and reports the mean, the median and
the mean in dB per chain.

Run (from alg_pdisac/):
    python -m scripts.test.test_nmse                                  # all models, fold 1
    python -m scripts.test.test_nmse --models pdnet,dncnn --fold 1
    PDISAC_SMOKE=1 python -m scripts.test.test_nmse --max-samples 4   # smoke database
Output: <out>/nmse_per_map.csv and <out>/nmse_summary.csv.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from configs.config_loader import resolve_path
from configs.data import db_path, n_records
from configs.ml import n_splits
from data.dataloader import RDMapDataset
from data.read_data import load_radar_results
from scripts.infer.denoiser import load_models, select_device


def nmse_per_map(Zhat: np.ndarray, Z_prbs: np.ndarray) -> np.ndarray:
    """B x 2 x H x W -> B values of ||Zhat - Z_prbs||^2 / ||Z_prbs||^2."""
    err = ((Zhat - Z_prbs) ** 2).reshape(len(Zhat), -1).sum(1)
    ref = (Z_prbs ** 2).reshape(len(Z_prbs), -1).sum(1)
    return err / np.maximum(ref, 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="all", help="comma-separated names of configs/inference.json, or 'all'")
    ap.add_argument("--fold", type=int, default=1, help="validation fold (1-based, as in training)")
    ap.add_argument("--max-samples", type=int, default=None, help="evaluate only the first N validation maps")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--out", default="exps/test_nmse", help="output folder (relative to alg_pdisac)")
    args = ap.parse_args()
    device = select_device(args.device)

    # ---- data: same filtering, normalisation and GroupKFold split as training ----
    df = load_radar_results(db_path, n_records=n_records)
    df = df[df["RD_map"].notna() & df["RD_map_no_noise"].notna()].reset_index(drop=True)
    ds = RDMapDataset.build(df)
    _, val_idx = list(GroupKFold(n_splits=n_splits).split(np.arange(len(ds)), groups=ds.gids))[args.fold - 1]
    if args.max_samples:
        val_idx = val_idx[:args.max_samples]
    print(f"[test_nmse] fold {args.fold}: {len(val_idx)} validation maps on {device}")

    models = load_models(None if args.models == "all" else [m.strip() for m in args.models.split(",")],
                         device=device)

    rows = []
    for start in range(0, len(val_idx), args.batch_size):
        idx = val_idx[start:start + args.batch_size]
        Z_in, Z_ref = ds.P_rd[idx], ds.P_rd_prbs[idx]
        outs = {"input": Z_in}
        with torch.inference_mode():
            x = torch.from_numpy(Z_in).to(device)
            for m in models:
                outs[m["name"]] = m["net"].predict(x).float().cpu().numpy()
        for name, Zhat in outs.items():
            for i, v in zip(idx, nmse_per_map(Zhat, Z_ref)):
                rows.append(dict(chain=name, sample=int(i), scene=int(ds.gids[i]), nmse=float(v)))

    per_map = pd.DataFrame(rows)
    summary = (per_map.groupby("chain", sort=False)["nmse"]
               .agg(n="count", nmse_mean="mean", nmse_median="median").reset_index())
    summary["nmse_mean_db"] = 10 * np.log10(summary["nmse_mean"])
    ckpt = {m["name"]: m for m in models}
    summary["epoch"] = [ckpt[c]["epoch"] if c in ckpt else np.nan for c in summary["chain"]]
    summary["val_nmse_ckpt"] = [ckpt[c]["val_nmse"] if c in ckpt else np.nan for c in summary["chain"]]

    out_dir = resolve_path(args.out)
    os.makedirs(out_dir, exist_ok=True)
    per_map.to_csv(os.path.join(out_dir, "nmse_per_map.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "nmse_summary.csv"), index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"[test_nmse] -> {out_dir}")


if __name__ == "__main__":
    main()
