"""
Distributions of RDPDNet for simulation_matlab/main5e_fig_dist.m: one scene, the encoder chain
(training) and the inference chain, as in Sec. IV-B (eq_q_hier, eq_p_hier, eq_infer_denoising_process).

  encoder   q(z_1 | z_0), q(z_l | z_{l-1}), l = 2..L     means of models/bases/pdnet.py Encoder,
            z_0 = Z_rd_prbs
  inference p(z_L | Z_rd)  (extractor),  p(z_l | z_{l+1}), l = L-1..1  (decoder priors),
            Zhat = mean of p(Zhat | z_1) = f_den^infer(Z_rd)
Every latent is its mean (as model.predict with sample_at_inference = False); the inference chain is
NOT the teacher-forced decoder of model.distribution() (that is the training path f_den^train).

Input MAT
    Z_rd, Z_rd_prbs      complex H x W   the scene with data and noise, and its clean map
Output MAT
    maps                 K x H x W float32, per-pixel magnitude:
                           RD maps (Z_rd_prbs, Z_rd, Zhat; input scale)  |Re + j Im|
                           latents (C channels)                          RMS over the C channels
    keys                 1 x K cell: Z_prbs, q1..qL, Z_rd, pL..p1, Zhat
    model_name, version, level

Run: python scripts/infer/distribution_mat.py --input in.mat --output out.mat --model pdnet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ALG = str(Path(__file__).resolve().parents[2])          # alg_pdisac
if _ALG not in sys.path:
    sys.path.insert(0, _ALG)

import numpy as np
import torch
from scipy.io import loadmat, savemat

from scripts.infer.denoiser import load_models, select_device


def _norm(Z):
    """RDMapDataset normalisation: per channel (real, imag), mean / std over H x W."""
    ch = np.stack((Z.real, Z.imag))[None].astype(np.float32)          # 1 x 2 x H x W
    m = ch.mean(axis=(2, 3), keepdims=True)
    s = ch.std(axis=(2, 3), keepdims=True)
    return (ch - m) / (s + 1e-8), m, s


def _rd_mag(x, m, s):
    """|Re + j Im| of a normalised 1 x 2 x H x W map, back at input scale."""
    y = x.detach().float().cpu().numpy() * (s + 1e-8) + m
    return np.abs(y[0, 0] + 1j * y[0, 1]).astype(np.float32)


def _latent_mag(mu):
    """RMS over the C latent channels of a 1 x C x H x W mean."""
    return mu.detach().float().pow(2).mean(dim=1).sqrt()[0].cpu().numpy().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="pdnet", help="name in configs/inference.json (RDPDNet, version pdnet)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = select_device(args.device)

    d = loadmat(args.input)
    Z, Z_prbs = np.asarray(d["Z_rd"]), np.asarray(d["Z_rd_prbs"])
    entry = load_models([args.model], device=device)[0]
    if entry["version"] != "pdnet":
        raise ValueError(f"'{args.model}' is version {entry['version']}; only RDPDNet (pdnet) is supported")
    net = entry["net"].model                                           # RDPDNet

    Zn, mZ, sZ = _norm(Z)
    Zpn, _, _ = _norm(Z_prbs)
    t = lambda a: torch.from_numpy(a).to(device)
    Zn_t, Zpn_t = t(Zn), t(Zpn)

    keys, maps = ["Z_prbs"], [np.abs(Z_prbs).astype(np.float32)]
    with torch.inference_mode():
        # encoder chain q(z_l | z_{l-1}), means
        q_params, _ = net.encoder(Zpn_t, sample=False)
        for l, (mu, _) in enumerate(q_params, start=1):
            keys.append(f"q{l}");  maps.append(_latent_mag(mu))
        L = len(q_params)
        # inference chain: extractor prior p(z_L | Z_rd), decoder priors p(z_l | z_{l+1}), Zhat
        keys.append("Z_rd");  maps.append(np.abs(Z).astype(np.float32))
        z, _ = net.extractor(Zn_t)
        keys.append(f"p{L}");  maps.append(_latent_mag(z))
        for i in range(L - 1):
            z, _ = net.decoder.prior(i, z)
            keys.append(f"p{L - 1 - i}");  maps.append(_latent_mag(z))
        Zhat_n = net.decoder.reconstruct(z)
        # consistency with f_den^infer (model.predict)
        err = (Zhat_n - net.predict(Zn_t)).abs().max().item()
        if err > 1e-4:
            raise RuntimeError(f"inference chain differs from model.predict (max abs error {err:.2e})")
        keys.append("Zhat");  maps.append(_rd_mag(Zhat_n, mZ, sZ))

    savemat(args.output, {"maps": np.stack(maps), "keys": np.array(keys, dtype=object)[None, :],
                          "model_name": args.model, "version": entry["version"], "level": L},
            do_compression=True)
    print(f"[distribution_mat] {args.model} ({entry['version']}, L = {L}): {', '.join(keys)} -> {args.output}")


if __name__ == "__main__":
    main()
