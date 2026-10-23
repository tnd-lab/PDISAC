"""JSON-driven configuration.

All settings live in configs/pdisac_system.json (single source of truth).
Real training uses the machine_learning defaults; set the PDISAC_SMOKE
environment variable to apply the fast smoke-test overrides:

    python -m scripts.train.train_pdnet_afm                  # real training  (dataset_train.db)
    PDISAC_SMOKE=1 python -m scripts.train.train_pdnet_afm    # quick smoke test (dataset_smoke.db)

PDISAC_SMOKE is the same switch the MATLAB dataset generator uses, so one
variable flips the whole pipeline between train and smoke. Paths inside the
config (db_path, output_dir) are resolved relative to the repository root
(the parent of this configs/ directory).
"""

import json
import os

_configs_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_configs_dir)
_workspace_root = os.path.dirname(_repo_root)

_system_cfg_path = os.environ.get(
    "PDISAC_SYSTEM_CONFIG",
    os.path.join(_configs_dir, "pdisac_system.json"),
)
if not os.path.isabs(_system_cfg_path):
    _system_cfg_path = os.path.join(_workspace_root, _system_cfg_path)
with open(_system_cfg_path) as _f:
    system_cfg = json.load(_f)

_ml = system_cfg["machine_learning"]
cfg = {
    "description": "Derived from configs/pdisac_system.json",
    "output_dir": os.path.relpath(_ml["output_dir"], "alg_pdisac")
        if _ml["output_dir"].startswith("alg_pdisac/") else _ml["output_dir"],
    "db_path": os.path.relpath(system_cfg["pdisac_dataset"]["dataset_train"]["output_db"], "alg_pdisac")
        if system_cfg["pdisac_dataset"]["dataset_train"]["output_db"].startswith("alg_pdisac/")
        else system_cfg["pdisac_dataset"]["dataset_train"]["output_db"],
    "n_records": _ml["n_records"], "num_bins": _ml["num_bins"],
    "batch_size": _ml["batch_size"], "num_workers": _ml["num_workers"],
    "pin_memory": _ml["pin_memory"], "device": _ml["device"],
    "epochs": _ml["epochs"], "lr": _ml["learning_rate"],
    "betas": _ml["betas"], "n_splits": _ml["folds"],
    "hidden_channel": _ml["hidden_channels"], "level": _ml["latent_levels"],
    "mask_thresh": _ml["mask_weight"], "rec_thresh": _ml["reconstruction_weight"],
    "kls_thresh": _ml["kl_weight"], "depth": _ml["dncnn_depth"],
    "img_channels": _ml["image_channels"], "n_filters": _ml["filters"],
    "kernel_size": _ml["kernel_size"],
    # training protocol shared by the four scripts of scripts/train/
    "folds_run": _ml.get("folds_run", 1),
    "kl_warmup_epochs": _ml.get("kl_warmup_epochs", 10),
    "lambda_rec": _ml.get("target_weight", 1.0),
    "lambda_inf": _ml.get("inference_weight", 1.0),
    "target_loss": _ml.get("target_loss", "rel"),
}

# Single switch: PDISAC_SMOKE applies the fast smoke-test overrides
# (same env var the MATLAB dataset generator uses).
if os.environ.get("PDISAC_SMOKE"):
    _smoke = dict(system_cfg.get("smoke", {}))
    # Normalize repo-relative paths the same way as the train paths above,
    # so the block works whether or not it is written with the alg_pdisac/ prefix.
    for _k in ("db_path", "output_dir"):
        if isinstance(_smoke.get(_k), str) and _smoke[_k].startswith("alg_pdisac/"):
            _smoke[_k] = os.path.relpath(_smoke[_k], "alg_pdisac")
    cfg.update(_smoke)
    _mode = "smoke"
else:
    _mode = "train"
print(f"[config] pdisac_system.json ({_mode} mode)")


def resolve_path(p: str) -> str:
    """Resolve a config-relative path against the repo root."""
    return p if os.path.isabs(p) else os.path.join(_repo_root, p)
