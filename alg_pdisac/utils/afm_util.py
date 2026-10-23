"""
afm_util — shared pieces of the AFM training scripts
(scripts/train/train_pdnet_afm.py, scripts/train/train_dncnn_afm.py) and of
scripts/infer/afm_maps_mat.py.

mix_realizations : variance-preserving cellwise mixing of two realisations of the same scene,
                   Z_mix = C + sqrt(1 - M_afm) D + sqrt(M_afm) D'.
mask_coverage    : realised mean of M_afm over the non-target region.
AfmLogger        : per-epoch CSV (afm_evidence.csv) with the mask coverage and
                   L_rec_mix - L_rec_inf (> 0: the mixed input is harder than the nominal one).
apply_budget     : optional projection of the mask onto a fixed mean coverage rho over the
                   non-target region (not used by the training scripts).
"""
from __future__ import annotations

import csv
import os

import torch


def apply_budget(M_raw: torch.Tensor, M_tars: torch.Tensor, rho: float,
                 eps: float = 1e-8) -> torch.Tensor:
    """
    Project the raw MaskNet output onto { M in [0,1]^Omega : mean_Omega(M) = rho },
    where Omega is the non-target region (1 - M_tars).

    Per sample: confine the mask to Omega, rescale so its mean over Omega equals
    rho, then clip to [0,1].  Clipping can leave the realised coverage slightly
    below rho, which is why the achieved value is logged rather than assumed.
    """
    W = 1.0 - M_tars                                   # non-target weights in [0,1]
    M = M_raw * W
    denom = W.flatten(1).sum(1).clamp_min(eps)         # size of Omega, per sample
    cur = M.flatten(1).sum(1) / denom                  # current mean coverage
    scale = (rho / cur.clamp_min(eps)).view(-1, *([1] * (M.dim() - 1)))
    return (M * scale).clamp(0.0, 1.0)


def mask_coverage(M_afm: torch.Tensor, M_tars: torch.Tensor,
                  eps: float = 1e-8) -> torch.Tensor:
    """Realised mean coverage of M_afm over the non-target region (scalar)."""
    W = 1.0 - M_tars
    return (M_afm.flatten(1).sum(1) / W.flatten(1).sum(1).clamp_min(eps)).mean()


class AfmLogger:
    """Per-epoch CSV of the AFM evidence (mask coverage, L_rec_mix - L_rec_inf)."""

    FIELDS = [
        "epoch",
        "coverage",        # realised mean mask coverage over the non-target region
        "L_rec",           # reconstruction loss of f_den^train (teacher forcing)
        "L_rec_inf",       # reconstruction loss of f_den^infer on the nominal input
        "L_rec_mix",       # reconstruction loss of f_den^infer on the adversarial input
        "mix_gap",         # L_rec_mix - L_rec_inf  (> 0 => the adversary is biting)
        "L_kl",
        "L_denoise",
        "nmse_train",
        "nmse_val",
    ]

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as fh:
            csv.writer(fh).writerow(self.FIELDS)

    def log(self, **kw) -> None:
        with open(self.path, "a", newline="") as fh:
            csv.writer(fh).writerow([kw.get(k, "") for k in self.FIELDS])

    @staticmethod
    def report(epoch: int, coverage: float, l_rec: float, l_mix: float) -> str:
        """l_rec: nominal loss on the SAME path as l_mix (f_den^infer)."""
        gap = l_mix - l_rec
        flag = "OK" if (coverage > 1e-3 and gap > 0) else "DEGENERATE?"
        return (f"    [AFM] coverage={coverage:.4f}  L_rec_inf={l_rec:.4f}  "
                f"L_rec_mix={l_mix:.4f}  gap={gap:+.4f}  -> {flag}")


# AFM: mixing of two realisations of the same scene.
def clean_component(Z, Z_prbs, eps=1e-12):
    """Least-squares projection of Z onto Z_prbs per sample and channel, a * Z_prbs:
    the clean part of Z in Z's own normalisation (disturbance zero-mean, uncorrelated)."""
    a = (Z * Z_prbs).sum(dim=(2, 3), keepdim=True) / (Z_prbs * Z_prbs).sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    return a * Z_prbs


def mix_realizations(Z_rd, Z_rd_alt, Z_rd_prbs, M_afm, eps=1e-6):
    """Variance-preserving cellwise mixing Z_mix = C + sqrt(1-M) D + sqrt(M) D'."""
    C = clean_component(Z_rd, Z_rd_prbs)
    D = Z_rd - C
    D_alt = Z_rd_alt - clean_component(Z_rd_alt, Z_rd_prbs)
    return (C + torch.sqrt((1 - M_afm).clamp_min(eps)) * D
            + torch.sqrt(M_afm.clamp_min(eps)) * D_alt)
