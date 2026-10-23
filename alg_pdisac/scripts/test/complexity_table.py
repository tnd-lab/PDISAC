"""
Per-frame complexity of the sensing chains (complexity table of the paper), all on ONE device.

Chains (input: received frame Y_sen and transmitted frame D_isac, N_chip x N_prbs each)
    conv     : MF + Doppler FFT + zero-Doppler removal (main4)  ->  CA-CFAR
    rdpdnet  : MF + Doppler FFT -> RDPDNet f_den^infer (extractor + decoder)  ->  CA-CFAR
    dncnn    : MF + Doppler FFT -> DnCNN                                       ->  CA-CFAR
    ls       : LS static-clutter cancellation (main5c rd_map_ls) + MF + Doppler FFT -> CA-CFAR
and the two networks alone (forward pass of a normalized 2 x N_chip x N_prbs map).

CA-CFAR replicates simulation_matlab/func_ca_cfar_adaptive_threshold.m (zero-padded training
window with the actual number of training cells at the border, alpha = n (P_fa^(-1/n) - 1),
peak selection by local maximum over the (2 R_out + 1) x (2 D_out + 1) window). --check compares
it with the NumPy/SciPy replica of scripts/test/eval_detection.py on synthetic maps.
The sub-bin read-out (sinc / Dirichlet interpolation) is not timed: it costs O(K N_sinc) per
frame for K detections, the same for every chain.

Reported per chain and batch size B
    params_total / params_infer : all trainable parameters / those executed at inference
    flops_frame                 : FLOPs per frame (networks: 2 x MACs of Conv/Linear via hooks;
                                  FFT 5 L log2 L per complex length-L transform; other operations
                                  counted per element; formulas in flops_* below)
    lat_batch_ms                : median wall time of one batch (device synchronized)
    lat_frame_ms                : lat_batch_ms / B
    fps                         : B / lat_batch
    cpi_ms                      : T_D = N_prbs N_chip / B_w (one frame)
    load                        : lat_frame_ms / cpi_ms  (< 1: the chain keeps up with the frame rate)
Network weights come from configs/inference.json (pdnet, dncnn); without a checkpoint the
architecture of the shared config is used with random weights (same cost), flagged in 'weights'.

Run (from alg_pdisac):
    Windows PowerShell:  $env:PYTHONPATH = (Get-Location).Path
                         python -m scripts.test.complexity_table --device cuda --batches 1,8,32
    macOS / Linux:       PYTHONPATH=$(pwd) python -m scripts.test.complexity_table --device mps
    consistency check:   python -m scripts.test.complexity_table --check --device cpu
"""
from __future__ import annotations

import argparse
import math
import os
import platform
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from configs.config_loader import cfg, resolve_path
from configs.isac import N_chip, N_block, B as BANDWIDTH
from models.bases import pdnet
from models.bases.dncnn import DnCNN
from models.model import DenoiseModel, count_parameters, count_flops, synchronize


# --------------------------------------------------------------------------- #
#  Building blocks (batched, on any device)
# --------------------------------------------------------------------------- #
def rd_map(Y, D):
    """MF + Doppler FFT of main4: B x N_chip x N_prbs complex -> RD map (zero-Doppler removed)."""
    mf = torch.fft.fft(Y, dim=1) * torch.conj(torch.fft.fft(D, dim=1))   # fast-time correlation
    mf = torch.fft.fft(mf, dim=2)                                          # Doppler
    mf[:, :, 0] = 0                                                        # zero-Doppler column
    return torch.fft.fftshift(torch.fft.ifft(mf, dim=1), dim=2)


def ls_cancel(Y, D, taps=(-64, 256), solve_device=None):
    """LS static-clutter cancellation of main5c rd_map_ls on the chip stream of the frame."""
    Bsz, Nc, Nb = Y.shape
    n_fr = Nc * Nb
    x = D.transpose(1, 2).reshape(Bsz, n_fr)            # blocks concatenated (MATLAB column order)
    y = Y.transpose(1, 2).reshape(Bsz, n_fr)
    lags = torch.arange(taps[0], taps[1] + 1, device=Y.device)
    K = lags.numel()
    nfft = 1 << math.ceil(math.log2(2 * n_fr + K))
    X = torch.fft.fft(x, nfft)
    cc = torch.fft.ifft(torch.fft.fft(y, nfft) * torch.conj(X))
    ac = torch.fft.ifft(X.abs() ** 2).real
    b = cc[:, lags % nfft]                               # B x K
    idx = (torch.arange(K, device=Y.device)[:, None] - torch.arange(K, device=Y.device)[None, :]).abs()
    R = ac[:, idx].to(Y.dtype)                           # B x K x K, Toeplitz (x real)
    if solve_device is not None and solve_device != Y.device:
        h = torch.linalg.solve(R.to(solve_device), b.to(solve_device).unsqueeze(-1)).to(Y.device)
    else:
        h = torch.linalg.solve(R, b.unsqueeze(-1))
    hp = torch.zeros(Bsz, nfft, dtype=Y.dtype, device=Y.device)
    hp[:, lags % nfft] = h.squeeze(-1)
    yc = torch.fft.ifft(X * torch.fft.fft(hp))[:, :n_fr]
    return (y - yc).reshape(Bsz, Nb, Nc).transpose(1, 2)


class CACFAR:
    """func_ca_cfar_adaptive_threshold.m on a batch of power maps (B x N_chip x N_prbs)."""

    def __init__(self, shape, g_r, g_d, t_r, t_d, p_fa, peak_select, device):
        self.R_out, self.D_out = g_r + t_r, g_d + t_d
        k = torch.ones(2 * self.R_out + 1, 2 * self.D_out + 1)
        k[t_r:t_r + 2 * g_r + 1, t_d:t_d + 2 * g_d + 1] = 0
        self.n_train = int(k.sum())
        self.kernel = k[None, None].to(device)
        count = F.conv2d(torch.ones(1, 1, *shape, device=device), self.kernel,
                         padding=(self.R_out, self.D_out))[0, 0]
        self.count = count
        self.alpha = count * (p_fa ** (-1.0 / count) - 1)          # per-cell threshold factor
        self.peak_select = peak_select

    def __call__(self, Z):
        P = Z.real ** 2 + Z.imag ** 2                               # B x H x W
        noise_avg = F.conv2d(P[:, None], self.kernel, padding=(self.R_out, self.D_out))[:, 0] \
            / self.count.clamp(min=1)
        det = P > self.alpha * noise_avg
        if self.peak_select:                                        # separable window maximum
            m = F.max_pool2d(P[:, None], (2 * self.R_out + 1, 1), 1, (self.R_out, 0))
            m = F.max_pool2d(m, (1, 2 * self.D_out + 1), 1, (0, self.D_out))[:, 0]
            det = det & (P >= m)
        return det

    def detections(self, Z):
        return torch.nonzero(self(Z))                               # K x 3 (map, range, Doppler)


def normalize(Z):
    """Per-map normalisation of scripts/infer/denoiser.py (real/imag channels, mean/std)."""
    x = torch.stack((Z.real, Z.imag), dim=1)
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True, correction=0)
    return (x - mean) / (std + 1e-8), mean, std


def denormalize(y, mean, std):
    y = y * (std + 1e-8) + mean
    return torch.complex(y[:, 0], y[:, 1])


# --------------------------------------------------------------------------- #
#  FLOPs per frame (analytic, non-network operations)
# --------------------------------------------------------------------------- #
def fft_flops(L):
    return 5 * L * math.log2(L)


def flops_rd(Nc, Nb):
    # fast-time FFT of Y and D, product, Doppler FFT, fast-time IFFT
    return 2 * Nb * fft_flops(Nc) + 6 * Nc * Nb + Nc * fft_flops(Nb) + Nb * fft_flops(Nc)


def flops_cfar(Nc, Nb, cfar):
    ww = (2 * cfar.R_out) + (2 * cfar.D_out)                   # separable max comparisons
    # power (3), training-cell sum (2 n_train), average (1), threshold (1), compare (1),
    # window maximum (ww), peak compare (1), AND (1)
    return Nc * Nb * (3 + 2 * cfar.n_train + 1 + 1 + 1 + ww + 1 + 1)


def flops_norm(Nc, Nb):
    # 2 channels: mean, variance, subtract, divide; output: multiply, add
    return 2 * Nc * Nb * (1 + 3 + 1 + 1) + 2 * Nc * Nb * 2


def flops_ls(Nc, Nb, taps=(-64, 256)):
    n_fr = Nc * Nb
    K = taps[1] - taps[0] + 1
    nfft = 1 << math.ceil(math.log2(2 * n_fr + K))
    # 6 FFTs of nfft, 3 spectral products (6 each), |X|^2 (3), solve (8/3 K^3 complex LU),
    # subtraction (2 n_fr)
    return 6 * fft_flops(nfft) + 3 * 6 * nfft + 3 * nfft + 8 / 3 * K ** 3 + 2 * n_fr


# --------------------------------------------------------------------------- #
#  Timing
# --------------------------------------------------------------------------- #
def time_call(fn, device, runs, warmup):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        t = []
        for _ in range(runs):
            synchronize(device)
            t0 = time.perf_counter()
            fn()
            synchronize(device)
            t.append(time.perf_counter() - t0)
    return 1000 * float(np.median(t))


def load_networks(device):
    """pdnet and dncnn of configs/inference.json; config architecture if no checkpoint."""
    nets = {}
    try:
        from scripts.infer.denoiser import load_models
        for m in load_models(["pdnet", "dncnn"], device=str(device)):
            nets[m["name"]] = (m["net"].eval(), "trained")
    except Exception as err:                                        # missing checkpoint(s)
        print(f"[weights] {err}; missing networks use random weights (same cost)")
    if "pdnet" not in nets:
        net = pdnet.RDPDNet(input_shape=(2, N_chip, N_block),
                            hidden_channel=cfg["hidden_channel"], level=cfg["level"])
        nets["pdnet"] = (DenoiseModel(model=net).to(device).eval(), "random")
    if "dncnn" not in nets:
        net = DnCNN(depth=cfg["depth"], n_filters=cfg["n_filters"],
                    kernel_size=cfg["kernel_size"], img_channels=cfg["img_channels"])
        nets["dncnn"] = (DenoiseModel(model=net).to(device).eval(), "random")
    return nets


def device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return f"Apple MPS ({platform.machine()})"
    return f"CPU ({platform.processor() or platform.machine()}, {torch.get_num_threads()} threads)"


# --------------------------------------------------------------------------- #
#  Consistency check of the CA-CFAR against the NumPy replica (eval_detection.py)
# --------------------------------------------------------------------------- #
def check_cfar(args, device, n_maps=20, seed=0):
    from scipy.ndimage import convolve, maximum_filter
    rng = np.random.default_rng(seed)
    cfar = CACFAR((N_chip, N_block), args.guard_range, args.guard_doppler, args.train_range,
                  args.train_doppler, args.pfa, not args.no_peak_select, device)
    R_out, D_out = cfar.R_out, cfar.D_out
    mask = np.ones((2 * R_out + 1, 2 * D_out + 1))
    mask[R_out - args.guard_range:R_out + args.guard_range + 1,
         D_out - args.guard_doppler:D_out + args.guard_doppler + 1] = 0
    n_bad, n_det = 0, 0
    for _ in range(n_maps):
        Z = (rng.standard_normal((N_chip, N_block)) + 1j * rng.standard_normal((N_chip, N_block))) / np.sqrt(2)
        for _ in range(8):                                          # point targets, 10..40 dB
            k, i = rng.integers(0, N_chip), rng.integers(0, N_block)
            Z[k, i] += 10 ** (rng.uniform(10, 40) / 20)
        P = np.abs(Z) ** 2
        count = convolve(np.ones_like(P), mask, mode="constant", cval=0.0)
        noise_avg = convolve(P, mask, mode="constant", cval=0.0) / np.maximum(count, 1)
        ref = P > count * (args.pfa ** (-1.0 / count) - 1) * noise_avg
        if not args.no_peak_select:
            ref &= P >= maximum_filter(P, size=(2 * R_out + 1, 2 * D_out + 1), mode="constant", cval=0.0)
        out = cfar(torch.from_numpy(Z.astype(np.complex64))[None].to(device))[0].cpu().numpy()
        n_bad += int((out != ref).sum())
        n_det += int(ref.sum())
    print(f"[check] CA-CFAR torch vs NumPy replica on {n_maps} synthetic maps: "
          f"{n_det} reference detections, {n_bad} differing cells")
    return n_bad == 0


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    ap.add_argument("--batches", default="1,8,32", help="comma-separated batch sizes")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="exps/complexity", help="output folder (relative to alg_pdisac)")
    ap.add_argument("--check", action="store_true", help="only the CA-CFAR consistency check")
    # CA-CFAR settings: PDISAC_cfg.sensing.cfar in main1_matlab_config.m
    ap.add_argument("--guard-range", type=int, default=2)
    ap.add_argument("--guard-doppler", type=int, default=2)
    ap.add_argument("--train-range", type=int, default=4)
    ap.add_argument("--train-doppler", type=int, default=4)
    ap.add_argument("--pfa", type=float, default=1e-4)
    ap.add_argument("--no-peak-select", action="store_true")
    args = ap.parse_args()

    if args.device == "auto":
        from scripts.infer.denoiser import select_device
        device = select_device("auto")
    else:
        device = torch.device(args.device)
    torch.manual_seed(0)

    if args.check:
        check_cfar(args, device)
        return

    Nc, Nb = N_chip, N_block
    cpi_ms = 1000 * Nb * Nc / BANDWIDTH
    cfar = CACFAR((Nc, Nb), args.guard_range, args.guard_doppler, args.train_range,
                  args.train_doppler, args.pfa, not args.no_peak_select, device)
    nets = load_networks(device)
    x1 = torch.randn(1, 2, Nc, Nb, device=device)

    info = {}
    for name, (net, weights) in nets.items():
        info[name] = dict(weights=weights,
                          params_total=count_parameters(net),
                          params_infer=count_parameters(net, {"Z_rd": x1}),
                          flops_net=count_flops(net, {"Z_rd": x1})[1])

    f_rd, f_cf, f_nm, f_ls = flops_rd(Nc, Nb), flops_cfar(Nc, Nb, cfar), flops_norm(Nc, Nb), flops_ls(Nc, Nb)
    solve_dev = None
    rows = []
    for Bsz in [int(b) for b in args.batches.split(",")]:
        Y = torch.randn(Bsz, Nc, Nb, dtype=torch.complex64, device=device)
        D = (torch.randint(0, 2, (Bsz, Nc, Nb), device=device) * 2 - 1).to(torch.complex64)
        xin = torch.randn(Bsz, 2, Nc, Nb, device=device)

        def run_net(name):
            return lambda: nets[name][0].predict(Z_rd=xin)

        def run_chain(name):
            def fn():
                Z = rd_map(Y, D)
                if name in nets:
                    x, mu, sd = normalize(Z)
                    Z = denormalize(nets[name][0].predict(Z_rd=x), mu, sd)
                return cfar.detections(Z)
            return fn

        def run_ls():
            return cfar.detections(rd_map(ls_cancel(Y, D, solve_device=solve_dev), D))

        cases = [
            ("MF + CA-CFAR", run_chain("conv"), 0, 0, f_rd + f_cf, "-"),
            ("RDPDNet (network)", run_net("pdnet"), info["pdnet"]["params_total"],
             info["pdnet"]["params_infer"], info["pdnet"]["flops_net"], info["pdnet"]["weights"]),
            ("MF + RDPDNet + CA-CFAR", run_chain("pdnet"), info["pdnet"]["params_total"],
             info["pdnet"]["params_infer"], f_rd + f_nm + info["pdnet"]["flops_net"] + f_cf,
             info["pdnet"]["weights"]),
            ("DnCNN (network)", run_net("dncnn"), info["dncnn"]["params_total"],
             info["dncnn"]["params_infer"], info["dncnn"]["flops_net"], info["dncnn"]["weights"]),
            ("MF + DnCNN + CA-CFAR", run_chain("dncnn"), info["dncnn"]["params_total"],
             info["dncnn"]["params_infer"], f_rd + f_nm + info["dncnn"]["flops_net"] + f_cf,
             info["dncnn"]["weights"]),
            ("LS + MF + CA-CFAR", run_ls, 0, 0, f_ls + f_rd + f_cf, "-"),
        ]
        for label, fn, p_tot, p_inf, flops, weights in cases:
            note = ""
            try:
                if label.startswith("LS") and solve_dev is None:
                    try:
                        fn()
                    except (RuntimeError, NotImplementedError) as err:   # e.g. complex solve on MPS
                        solve_dev = torch.device("cpu")
                        note = f"K x K solve on CPU ({type(err).__name__})"
                lat = time_call(fn, device, args.runs, args.warmup)
            except RuntimeError as err:                                  # e.g. out of memory
                lat, note = float("nan"), f"failed: {str(err).splitlines()[0][:80]}"
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if label.startswith("LS") and solve_dev is not None and not note:
                note = "K x K solve on CPU"
            rows.append(dict(chain=label, batch=Bsz, device=device.type, device_name=device_name(device),
                             weights=weights, params_total=p_tot, params_infer=p_inf,
                             gflops_frame=flops / 1e9, lat_batch_ms=lat, lat_frame_ms=lat / Bsz,
                             fps=1000 * Bsz / lat, cpi_ms=cpi_ms, load=lat / Bsz / cpi_ms, note=note))

    df = pd.DataFrame(rows)
    out_dir = resolve_path(args.out)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"complexity_{device.type}.csv")
    df.to_csv(out_csv, index=False)

    pd.set_option("display.width", 200)
    print(f"\nMap {Nc} x {Nb} | CPI T_D = {cpi_ms:.3f} ms (1/T_D = {1000 / cpi_ms:.0f} frames/s) | "
          f"{device_name(device)} | torch {torch.__version__} | fp32")
    print(f"CA-CFAR: guard {args.guard_range}/{args.guard_doppler}, train {args.train_range}/"
          f"{args.train_doppler}, P_fa {args.pfa:g}, peak select {not args.no_peak_select}")
    print(df[["chain", "batch", "weights", "params_total", "params_infer", "gflops_frame",
              "lat_batch_ms", "lat_frame_ms", "fps", "load", "note"]].to_string(
        index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nsaved {out_csv}")


if __name__ == "__main__":
    main()
