"""
DnCNN trained without AFM (DnCNN baseline; counterpart of train_pdnet_no_afm.py).

Same protocol as train_pdnet_no_afm.py with the RDPDNet replaced by a residual DnCNN
(models/bases/dncnn.py: depth, n_filters, kernel_size, img_channels from the config).
DnCNN has no latent branch, so there is no L_KL and the training path equals the inference path:
        L_den = (1 + lambda_inf) * L_rec + elastic ,
        L_rec = ||f_den(Z_rd) - Z_prbs||^2 + lambda_rec * target term .
The checkpoint stores arch = "dncnn" and the DnCNN hyperparameters.

Run (from alg_pdisac/):
    python -m scripts.train.train_dncnn_no_afm                  # dataset_train.db
    PDISAC_SMOKE=1 python -m scripts.train.train_dncnn_no_afm   # smoke test (dataset_smoke.db)
Settings: configs/pdisac_system.json (machine_learning block).
Checkpoint: exps/dncnn_no_afm/fold_<k>/checkpoints/best_checkpoint.pth (best validation NMSE of the
inference map), read by scripts/infer/* through configs/inference.json.
"""
import csv
import os

import torch
from tqdm import tqdm

from data.dataloader import RDMapDataloader
from data.read_data import load_radar_results
from utils.utils import compute_nmse
from utils.plots import plot_denoise_loss, plot_nmse, plot_matrix
from configs.data import output_dir, db_path, n_records
from configs.ml import (
    n_folds_run, kl_warmup_epochs, lambda_rec, lambda_inf, target_loss, num_workers, pin_memory,
    epochs, device, batch_size, n_splits, lr, betas, depth, n_filters, kernel_size, img_channels,
)
from models.bases.dncnn import DnCNN
from models.model import DenoiseModel
from losses.losses import MSELoss, elastic_regularization, target_nmse, target_rel_mse

N_FOLDS_RUN = n_folds_run
LAMBDA_REC = lambda_rec
LAMBDA_INF = lambda_inf
TARGET_LOSS = target_loss
target_term = target_rel_mse if TARGET_LOSS == "rel" else target_nmse

ARCH = "dncnn"
DNCNN_CFG = dict(depth=depth, n_filters=n_filters, kernel_size=kernel_size, img_channels=img_channels)
model = DenoiseModel(model=DnCNN(**DNCNN_CFG)).to(device)

output_dir = os.path.join(output_dir, "dncnn_no_afm")
LOG_FIELDS = ["epoch", "L_rec", "L_denoise", "nmse_train", "val_mse", "nmse_val"]


if __name__ == "__main__":
    print(f"DnCNN {DNCNN_CFG}: {sum(p.numel() for p in model.parameters()):,} parameters")
    df = load_radar_results(db_path, n_records=n_records)
    loader = RDMapDataloader(
        df=df, n_splits=n_splits, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    fold_dataloaders = loader.get_fold_dataloaders()

    for fold_idx, (train_loader, val_loader) in enumerate(fold_dataloaders):
        print(f"\n{'=' * 50}\nFold {fold_idx + 1} / {len(fold_dataloaders)}  (DnCNN, no AFM)\n{'=' * 50}")

        fold_output_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
        checkpoint_dir = os.path.join(fold_output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        log_path = os.path.join(fold_output_dir, "train_log.csv")
        with open(log_path, "w", newline="") as fh:
            csv.writer(fh).writerow(LOG_FIELDS)

        model.model._initialize_weights()            # DnCNN init (Kaiming normal), per fold
        model_opt = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
        mse_loss = MSELoss().to(device)
        best_val_nmse = float("inf")

        # Warm start: if this fold already has a best checkpoint, load it and keep training.
        ckpt_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            if ckpt.get("arch") != ARCH or ckpt.get("dncnn_cfg") != DNCNN_CFG:
                print(f"Checkpoint {ckpt_path} has another architecture: training from scratch.")
            else:
                model.load_state_dict(ckpt["model_state_dict"])
                model_opt.load_state_dict(ckpt["model_optimizer_state_dict"])
                best_val_nmse = ckpt["val_nmse"]
                print(f"Loaded {ckpt_path} (epoch {ckpt['epoch']}, val NMSE {best_val_nmse:.4f})")

        train_denoise_losses, train_nmse_list = [], []
        val_denoise_losses, val_nmse_list = [], []

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            model.train()
            tot_den = tot_rec = tot_nmse = 0.0
            nb = 0

            for Z_rd, Z_rd_prbs, M_tars in tqdm(train_loader):
                Z_rd = Z_rd.to(device)
                Z_rd_prbs = Z_rd_prbs.to(device)
                M_tars = M_tars.to(device)                 # target support of eq_L_rec

                model_opt.zero_grad()
                Z_rd_hat = model.predict(Z_rd)             # training path = inference path
                L_rec = mse_loss(Z_rd_hat, Z_rd_prbs) + LAMBDA_REC * target_term(Z_rd_hat, Z_rd_prbs, M_tars)
                L_denoise = (1 + LAMBDA_INF) * L_rec + elastic_regularization(model)
                L_denoise.backward()
                model_opt.step()

                nmse = compute_nmse(Z_rd_hat.detach().flatten(start_dim=1),
                                    Z_rd_prbs.flatten(start_dim=1), dim=1)
                tot_den += L_denoise.item(); tot_rec += L_rec.item()
                tot_nmse += nmse.mean().item()
                nb += 1

            avg = lambda x: x / nb
            train_denoise_losses.append(avg(tot_den))
            train_nmse_list.append(avg(tot_nmse))
            print(f"Train — Denoise {avg(tot_den):.4f}  L_rec {avg(tot_rec):.4f}  NMSE {avg(tot_nmse):.4f}")

            # ---------- Validation ----------
            model.eval()
            v_den = v_nmse = 0.0; vb = 0
            with torch.no_grad():
                for Z_rd, Z_rd_prbs, M_tars in val_loader:
                    Z_rd = Z_rd.to(device); Z_rd_prbs = Z_rd_prbs.to(device)
                    M_tars = M_tars.to(device)
                    Z_rd_hat = model.predict(Z_rd)
                    nmse = compute_nmse(Z_rd_hat.flatten(start_dim=1),
                                        Z_rd_prbs.flatten(start_dim=1), dim=1)
                    v_den += mse_loss(Z_rd_hat, Z_rd_prbs).item()
                    v_nmse += nmse.mean().item(); vb += 1

            v_den /= vb; v_nmse /= vb
            val_denoise_losses.append(v_den)
            val_nmse_list.append(v_nmse)
            print(f"Val   — Denoise {v_den:.4f}  NMSE {v_nmse:.4f}")

            with open(log_path, "a", newline="") as fh:
                csv.writer(fh).writerow([epoch + 1, avg(tot_rec), avg(tot_den), avg(tot_nmse), v_den, v_nmse])

            if v_nmse < best_val_nmse:
                best_val_nmse = v_nmse
                torch.save({"fold": fold_idx + 1, "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "model_optimizer_state_dict": model_opt.state_dict(),
                            "val_nmse": best_val_nmse, "variant": "no_afm",
                            "arch": ARCH, "dncnn_cfg": DNCNN_CFG, "model_version": ARCH,
                            "lambda_inf": LAMBDA_INF, "lambda_rec": LAMBDA_REC, "target_loss": TARGET_LOSS},
                           ckpt_path)
                print(f"Saved checkpoint — best val NMSE: {best_val_nmse:.4f}")
                plot_matrix({"Z_rd": Z_rd[0], "Z_rd_hat": Z_rd_hat[0],
                             "Z_rd_prbs": Z_rd_prbs[0], "M_tars": M_tars[0]},
                            save_path=os.path.join(fold_output_dir, "visualize_matrix.png"))

        epoch_range = range(1, len(train_denoise_losses) + 1)
        plot_denoise_loss(epoch_range, train_denoise_losses, val_denoise_losses, fold_output_dir)
        plot_nmse(epoch_range, train_nmse_list, val_nmse_list, fold_output_dir)

        if fold_idx + 1 >= N_FOLDS_RUN:
            break
