"""
RD maps and AFM quantities for the figures of simulation_matlab/main5d_fig_maps.m: one scene, several denoisers,
and the AFM quantities of the proposed model, exactly as in training
(scripts/train/train_pdnet_afm.py).

Input MAT
    Z_rd, Z_rd_alt, Z_rd_prbs   complex H x W   two realisations of the scene and its clean map
    truth                       N x 2           [range_m, radial velocity m/s] of every target
Output MAT
    den, model_name             K x H x W complex (input scale), 1 x K cell
    M_mask, M_tars, M_afm       H x W in [0, 1] (channel 0)
    Z_mix, den_mix              H x W complex (scale of Z_rd): mixed input and its denoised map
    err_gain                    H x W: |f(Z_mix) - Z_prbs|^2 - |f(Z_rd) - Z_prbs|^2 (normalised maps)
    coverage                    realised mean of M_afm over the non-target region
    evidence_csv                afm_evidence.csv of the proposed run ('' if absent)

Run: python scripts/infer/afm_maps_mat.py --input in.mat --output out.mat --models a,b --mask-model a
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ALG = str(Path(__file__).resolve().parents[2])          # alg_pdisac
if _ALG not in sys.path:
    sys.path.insert(0, _ALG)

import numpy as np
import torch
from scipy.io import loadmat, savemat

from data.read_data import target_support_mask
from models.bases.afm import MaskNet
from scripts.infer.denoiser import load_models, select_device
from utils.afm_util import mask_coverage, mix_realizations


def _norm(Z):
    """RDMapDataset normalisation: per channel (real, imag), mean / std over H x W."""
    ch = np.stack((Z.real, Z.imag))[None].astype(np.float32)          # 1 x 2 x H x W
    m = ch.mean(axis=(2, 3), keepdims=True)
    s = ch.std(axis=(2, 3), keepdims=True)
    return (ch - m) / (s + 1e-8), m, s


def _restore(x, m, s):
    y = x.detach().float().cpu().numpy() * (s + 1e-8) + m
    return (y[0, 0] + 1j * y[0, 1]).astype(np.complex64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--models", required=True, help="comma-separated names of configs/inference.json")
    ap.add_argument("--mask-model", required=True, help="AFM model whose MaskNet is shown")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = select_device(args.device)

    d = loadmat(args.input)
    Z, Z_alt, Z_prbs = (np.asarray(d[k]) for k in ("Z_rd", "Z_rd_alt", "Z_rd_prbs"))
    truth = np.asarray(d["truth"], dtype=float).reshape(-1, 2)

    names = [n.strip() for n in args.models.split(",") if n.strip()]
    if args.mask_model not in names:
        names.append(args.mask_model)
    models = []
    for n in names:                                     # a missing checkpoint is skipped (not the mask model)
        try:
            models += load_models([n], device=device)
        except FileNotFoundError as e:
            if n == args.mask_model:
                raise
            print(f"[afm_maps_mat] skipping '{n}': {e}")
    names = [m["name"] for m in models]
    by_name = {m["name"]: m for m in models}

    Zn, mZ, sZ = _norm(Z)
    Zan, _, _ = _norm(Z_alt)
    Zpn, _, _ = _norm(Z_prbs)
    t = lambda a: torch.from_numpy(a).to(device)
    Zn_t, Zan_t, Zpn_t = t(Zn), t(Zan), t(Zpn)

    out = {"model_name": np.array(names, dtype=object)[None, :]}
    with torch.inference_mode():
        out["den"] = np.stack([_restore(by_name[n]["net"].predict(Zn_t), mZ, sZ) for n in names])

        # AFM of the proposed model, as in Step 2 of the training loop
        pm = by_name[args.mask_model]
        ckpt = torch.load(pm["checkpoint"], map_location=device, weights_only=False)
        if "afm_model_state_dict" not in ckpt:
            raise KeyError(f"'{args.mask_model}' has no MaskNet (not an AFM checkpoint)")
        masknet = MaskNet(in_ch=int(ckpt.get("masknet_in_ch", 4))).to(device)
        masknet.load_state_dict(ckpt["afm_model_state_dict"])
        masknet.eval()
        net = pm["net"]
        M_tars = t(target_support_mask(Zpn[0], truth)[None])                  # 1 x 2 x H x W
        Z_hat = net.predict(Zn_t)
        M_mask = masknet(torch.cat([Z_hat, Zan_t - Zn_t], dim=1))
        M_afm = M_mask * (1 - M_tars)
        Z_mix = mix_realizations(Zn_t, Zan_t, Zpn_t, M_afm)
        Z_mix_hat = net.predict(Z_mix)
        err = lambda X: ((X - Zpn_t) ** 2).sum(dim=1)[0]
        out.update(
            M_mask=M_mask[0, 0].float().cpu().numpy(), M_tars=M_tars[0, 0].float().cpu().numpy(),
            M_afm=M_afm[0, 0].float().cpu().numpy(),
            Z_mix=_restore(Z_mix, mZ, sZ), den_mix=_restore(Z_mix_hat, mZ, sZ),
            err_gain=(err(Z_mix_hat) - err(Z_hat)).float().cpu().numpy(),
            coverage=float(mask_coverage(M_afm, M_tars)))
    ev = os.path.join(os.path.dirname(os.path.dirname(pm["checkpoint"])), "afm_evidence.csv")
    out["evidence_csv"] = ev if os.path.isfile(ev) else ""
    out["mask_model"] = args.mask_model
    savemat(args.output, out, do_compression=True)
    print(f"[afm_maps_mat] models {names}, MaskNet of {args.mask_model} "
          f"(coverage {out['coverage']:.3f}) -> {args.output}")


if __name__ == "__main__":
    main()
