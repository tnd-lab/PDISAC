"""
Detection-oriented evaluation of the trained denoisers (RDPDNet, DnCNN) on the validation fold.

For every validation map and every chain
    input   : CA-CFAR on the noisy map Z_rd                (conventional chain)
    clean   : CA-CFAR on the clean map Z_rd_prbs           (reference, upper bound)
    <label> : CA-CFAR on f_den^infer(Z_rd) of a model (one per --models name or --ckpt)
it computes, per target,
    hit     : detected after peak selection, one-to-one within +-GATE bins (as main5b)
    PBR_dB  : peak-to-background ratio, 10 log10( max P in the +-1 box / median P of the background )
and, per map,
    n_fa, n_cell : raw CFAR crossings / cells in the test region (P_fa, as main5b)
    spur_dB      : largest background cell above the background median (worst spurious peak).

With --pfa-sweep (default 1e-6,...,1e-2) the nominal CFAR P_fa is swept and, for every
chain, the measured P_d (overall and per target-strength class) is reported against the
measured P_fa: this compares the chains at equal false-alarm rate (eval_roc.csv).

CA-CFAR, peak selection, test region and P_d association replicate
simulation_matlab/func_ca_cfar_adaptive_threshold.m and main5b_detection_vs_nbit.m.
Targets in the removed zero-Doppler column are not counted (invisible in the RD map).

Run (from alg_pdisac/):
    python -m scripts.test.eval_detection --models all --fold 1 --out exps/eval_fold1
    python -m scripts.test.eval_detection --models pdnet,pdnet_noafm --fold 1
    python -m scripts.test.eval_detection \
        --ckpt AFM=exps/pdnet_afm/fold_1/checkpoints/best_checkpoint.pth --fold 1
--models takes names of configs/inference.json; --ckpt takes LABEL=path. RDPDNet checkpoints must
carry a model_version of RDPDNET_VERSIONS in configs/ml.py; DnCNN checkpoints (arch = "dncnn", from
train_dncnn_afm.py / train_dncnn_no_afm.py) are rebuilt from the hyperparameters they store.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import convolve, maximum_filter
from sklearn.model_selection import GroupKFold

from configs.config_loader import system_cfg
from configs.data import db_path, n_records
from configs.isac import range_resolution, vel_axis
from configs.ml import model, device, n_splits, MODEL_VERSION, RDPDNET_VERSIONS
from models.bases.dncnn import DnCNN
from models.model import DenoiseModel
from data.read_data import load_radar_results
from data.dataloader import RDMapDataset


# --------------------------------------------------------------------------- #
#  CA-CFAR (func_ca_cfar_adaptive_threshold.m)
# --------------------------------------------------------------------------- #
def cfar_stats(P, g_r, g_d, t_r, t_d):
    """Training-cell average, training-cell count and peak-selection mask (independent of P_fa)."""
    R_out, D_out = g_r + t_r, g_d + t_d
    mask = np.ones((2 * R_out + 1, 2 * D_out + 1))
    mask[R_out - g_r:R_out + g_r + 1, D_out - g_d:D_out + g_d + 1] = 0
    noise_sum = convolve(P, mask, mode="constant", cval=0.0)                  # conv2(...,'same')
    count = convolve(np.ones_like(P), mask, mode="constant", cval=0.0)
    noise_avg = noise_sum / np.maximum(count, 1)
    local_max = P >= maximum_filter(P, size=(2 * R_out + 1, 2 * D_out + 1), mode="constant", cval=0.0)
    return noise_avg, count, local_max


def cfar_detect(P, stats, p_fa, peak_select=True):
    """Raw crossings and detections [K, 2] (sorted by power) for one nominal P_fa."""
    noise_avg, count, local_max = stats
    alpha = np.zeros_like(P)
    for n in np.unique(count):
        if n >= 1:
            alpha[count == n] = n * (p_fa ** (-1.0 / n) - 1)
    raw = P > alpha * noise_avg
    det = raw & local_max if peak_select else raw
    rows, cols = np.nonzero(det)
    order = np.argsort(-P[rows, cols])
    return raw, np.stack([rows[order], cols[order]], axis=1)


def associate(det, cells, gate):
    """One-to-one association, closest (target, detection) pairs first (score_map in main5b)."""
    hit = np.zeros(len(cells), dtype=bool)
    if len(det) == 0:
        return hit
    pairs = []
    for m, (k, d) in enumerate(cells):
        dk, dd = det[:, 0] - k, det[:, 1] - d
        for a in np.nonzero((np.abs(dk) <= gate) & (np.abs(dd) <= gate))[0]:
            pairs.append((dk[a] ** 2 + dd[a] ** 2, m, a))
    used = set()
    for _, m, a in sorted(pairs):
        if hit[m] or a in used:
            continue
        hit[m] = True
        used.add(a)
    return hit


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", default=[], help="LABEL=path/to/best_checkpoint.pth (repeatable)")
    ap.add_argument("--models", default=None,
                    help="models of configs/inference.json: comma-separated names, or 'all' (version from the json)")
    ap.add_argument("--fold", type=int, default=1, help="validation fold (1-based, as in training)")
    ap.add_argument("--out", default="exps/eval_fold1", help="output folder for the CSV files")
    ap.add_argument("--max-samples", type=int, default=None, help="evaluate only the first N validation maps")
    ap.add_argument("--batch-size", type=int, default=8)
    # CFAR settings: must match PDISAC_cfg.sensing.cfar in main1_matlab_config.m
    ap.add_argument("--guard-range", type=int, default=2)
    ap.add_argument("--guard-doppler", type=int, default=2)
    ap.add_argument("--train-range", type=int, default=4)
    ap.add_argument("--train-doppler", type=int, default=4)
    ap.add_argument("--pfa", type=float, default=1e-4)
    ap.add_argument("--no-peak-select", action="store_true")
    ap.add_argument("--gate", type=int, default=1, help="P_d gate in bins (main5b: GATE = 1)")
    ap.add_argument("--pfa-sweep", default="1e-6,1e-5,1e-4,1e-3,1e-2",
                    help="comma-separated nominal P_fa values for the P_d-vs-P_fa comparison ('' = off)")
    args = ap.parse_args()

    g_r, g_d, t_r, t_d = args.guard_range, args.guard_doppler, args.train_range, args.train_doppler
    peak_select = not args.no_peak_select
    sweep = [float(x) for x in args.pfa_sweep.split(",") if x.strip()]
    CLASSES = ["weak (<-20 dB)", "mid (-20..-10)", "strong (-10..0)", "strongest"]

    def strength_class(rel_db):
        return CLASSES[0] if rel_db < -20 else CLASSES[1] if rel_db < -10 else CLASSES[2] if rel_db < -1e-9 else CLASSES[3]

    # ---- data: same filtering, normalisation and GroupKFold split as training ----
    df = load_radar_results(db_path, n_records=n_records)
    df = df[df["RD_map"].notna() & df["RD_map_no_noise"].notna()].reset_index(drop=True)
    ds = RDMapDataset.build(df)
    _, val_idx = list(GroupKFold(n_splits=n_splits).split(np.arange(len(ds)), groups=ds.gids))[args.fold - 1]
    if args.max_samples:
        val_idx = val_idx[:args.max_samples]
    print(f"[eval] fold {args.fold}: {len(val_idx)} validation maps | model {MODEL_VERSION}")

    # ---- geometry: zero-Doppler column, test region (main5b) ----
    N_r, N_d = ds.P_rd.shape[2], ds.P_rd.shape[3]
    d0 = int(np.argmin(np.abs(vel_axis)))
    roi = np.asarray(system_cfg["scene"]["roi_m"], dtype=float)
    R_scene = float(np.sqrt(np.max(np.abs(roi[0])) ** 2 + np.max(np.abs(roi[1])) ** 2))
    region = np.zeros((N_r, N_d), dtype=bool)
    region[np.arange(N_r) * range_resolution <= R_scene + (g_r + t_r) * range_resolution, :] = True
    region[:, d0] = False

    # ---- chains ----
    chains = {"input": None, "clean": None}
    for spec in args.ckpt:
        label, path = spec.split("=", 1)
        ck = torch.load(path, map_location=device, weights_only=False)
        if ck.get("arch") == "dncnn":                    # DnCNN baseline: own network per checkpoint
            net = DenoiseModel(model=DnCNN(**ck["dncnn_cfg"])).to(device)
        else:
            mv = ck.get("model_version", MODEL_VERSION)
            if mv not in RDPDNET_VERSIONS:
                raise ValueError(f"{label}: checkpoint model_version={mv} is not an RDPDNet ({RDPDNET_VERSIONS})")
            net = model
        chains[label] = (net, ck["model_state_dict"])
    if args.models:
        from scripts.infer.denoiser import load_models
        for m in load_models(None if args.models == "all" else args.models.split(","), device=device):
            chains[m["name"]] = (m["net"], None)            # weights already loaded

    tgt_rows, map_rows = [], []
    # ROC accumulators: roc[(chain, pfa)] = {"n_fa", "n_cell", "hit_<class>", "n_<class>"}
    roc = {}
    for label, state in chains.items():
        if state is not None:
            net, sd = state
            if sd is not None:
                net.load_state_dict(sd)
            net.to(device).eval()
        for start in range(0, len(val_idx), args.batch_size):
            idx = val_idx[start:start + args.batch_size]
            Z_in = torch.from_numpy(ds.P_rd[idx]).to(device)
            if label == "input":
                Z_out = ds.P_rd[idx]
            elif label == "clean":
                Z_out = ds.P_rd_prbs[idx]
            else:
                with torch.no_grad():
                    Z_out = net.predict(Z_in).cpu().numpy()
            for j, i in enumerate(idx):
                P = Z_out[j, 0].astype(np.float64) ** 2 + Z_out[j, 1].astype(np.float64) ** 2
                P_clean = ds.P_rd_prbs[i, 0] ** 2 + ds.P_rd_prbs[i, 1] ** 2
                truth = np.asarray(ds.truth[i]).reshape(-1, 2)
                cells, clean_pk = [], []
                for r, v in truth:
                    k, d = int(round(r / range_resolution)), int(np.argmin(np.abs(vel_axis - v)))
                    if d == d0 or not (0 <= k < N_r):
                        continue                                    # invisible in the RD map
                    cells.append((k, d))
                    clean_pk.append(P_clean[max(0, k - 1):k + 2, max(0, d - 1):d + 2].max())
                test = region.copy()
                for k, d in cells:
                    test[max(0, k - g_r - 1):k + g_r + 2, max(0, d - g_d - 1):d + g_d + 2] = False
                bg = np.median(P[test]) + 1e-20

                stats = cfar_stats(P, g_r, g_d, t_r, t_d)
                raw, det = cfar_detect(P, stats, args.pfa, peak_select)
                hit = associate(det, cells, args.gate)
                rel_db = 10 * np.log10(np.asarray(clean_pk) / (max(clean_pk) + 1e-20) + 1e-20) if cells else []
                cls = [strength_class(x) for x in rel_db]
                for pf in sweep:
                    raw_s, det_s = cfar_detect(P, stats, pf, peak_select)
                    hit_s = associate(det_s, cells, args.gate)
                    acc = roc.setdefault((label, pf), {"n_fa": 0, "n_cell": 0,
                                                       **{f"hit_{c}": 0 for c in CLASSES},
                                                       **{f"n_{c}": 0 for c in CLASSES}})
                    acc["n_fa"] += int(np.count_nonzero(raw_s & test)); acc["n_cell"] += int(test.sum())
                    for c, h in zip(cls, hit_s):
                        acc[f"hit_{c}"] += int(h); acc[f"n_{c}"] += 1
                row = df.iloc[i]
                for m, (k, d) in enumerate(cells):
                    pk = P[max(0, k - 1):k + 2, max(0, d - 1):d + 2].max()
                    tgt_rows.append(dict(chain=label, sample=int(i), SNR_dB=row["SNR_dB"],
                                         N_sym=row["N_sym_per_block"], target=m,
                                         rel_clean_dB=float(rel_db[m]), hit=bool(hit[m]),
                                         PBR_dB=float(10 * np.log10(pk / bg))))
                map_rows.append(dict(chain=label, sample=int(i), SNR_dB=row["SNR_dB"],
                                     N_sym=row["N_sym_per_block"],
                                     n_fa=int(np.count_nonzero(raw & test)), n_cell=int(test.sum()),
                                     spur_dB=float(10 * np.log10(P[test].max() / bg))))
        print(f"[eval] chain '{label}' done")

    os.makedirs(args.out, exist_ok=True)
    T, M = pd.DataFrame(tgt_rows), pd.DataFrame(map_rows)
    T.to_csv(os.path.join(args.out, "eval_targets.csv"), index=False)
    M.to_csv(os.path.join(args.out, "eval_maps.csv"), index=False)

    # ---- summaries ----
    T["class"] = pd.cut(T["rel_clean_dB"], [-np.inf, -20, -10, -1e-9, np.inf],
                        labels=["weak (<-20 dB)", "mid (-20..-10)", "strong (-10..0)", "strongest"])
    pd.set_option("display.width", 200)
    print("\n=== P_d (overall) and P_fa ===")
    s = T.groupby("chain")["hit"].mean().rename("P_d").to_frame()
    s["P_fa"] = M.groupby("chain").apply(lambda g: g["n_fa"].sum() / g["n_cell"].sum())
    s["PBR_dB_median"] = T.groupby("chain")["PBR_dB"].median()
    s["spur_dB_median"] = M.groupby("chain")["spur_dB"].median()
    print(s.round(4))
    print("\n=== P_d by target strength (clean peak relative to the strongest target of the map) ===")
    print(T.pivot_table(index="class", columns="chain", values="hit", aggfunc="mean", observed=False).round(3))
    print("\n=== median PBR_dB by target strength ===")
    print(T.pivot_table(index="class", columns="chain", values="PBR_dB", aggfunc="median", observed=False).round(2))
    print("\n=== P_d by SNR_dB ===")
    print(T.pivot_table(index="SNR_dB", columns="chain", values="hit", aggfunc="mean").round(3))
    print("\n=== P_d by N_sym_per_block ===")
    print(T.pivot_table(index="N_sym", columns="chain", values="hit", aggfunc="mean").round(3))
    if roc:
        rows = []
        for (label, pf), acc in sorted(roc.items()):
            n_t = sum(acc[f"n_{c}"] for c in CLASSES)
            r = dict(chain=label, pfa_nominal=pf, P_fa=acc["n_fa"] / max(acc["n_cell"], 1),
                     P_d=sum(acc[f"hit_{c}"] for c in CLASSES) / max(n_t, 1))
            for c in CLASSES:
                r[f"P_d {c}"] = acc[f"hit_{c}"] / max(acc[f"n_{c}"], 1)
            rows.append(r)
        R = pd.DataFrame(rows)
        R.to_csv(os.path.join(args.out, "eval_roc.csv"), index=False)
        print("\n=== P_d versus measured P_fa (sweep of the nominal CFAR P_fa) ===")
        print(R.round(4).to_string(index=False))
        print("Compare chains at equal MEASURED P_fa (interpolate along each chain), not at equal nominal P_fa.")
    print(f"\n[eval] per-target, per-map and ROC results written to {args.out}")


if __name__ == "__main__":
    main()
