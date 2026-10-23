"""
RDPDNet trained without AFM (baseline of the AFM ablation).

The denoiser is trained on the objective of P2 only (no MaskNet, no Z_mix, no L_rec_mix):
        L_den = L_rec + lambda_inf * L_rec_inf + lambda_KL * L_KL + elastic ,
        L_rec     = ||f_den^train(Z_rd) - Z_prbs||^2 + lambda_rec * target term ,
        L_rec_inf = the same loss on f_den^infer(Z_rd).
M_tars is the target support of L_rec (part of P2, not of AFM). Data loader, GroupKFold split,
loss weights, KL warm-up, checkpoint selection and checkpoint keys are those of
train_pdnet_afm.py; the only difference is the absence of the AFM terms.

Run (from alg_pdisac/):
    python -m scripts.train.train_pdnet_no_afm                  # dataset_train.db
    PDISAC_SMOKE=1 python -m scripts.train.train_pdnet_no_afm   # smoke test (dataset_smoke.db)
Settings: configs/pdisac_system.json (machine_learning block).
Checkpoint: exps/pdnet_no_afm/fold_<k>/checkpoints/best_checkpoint.pth (best validation NMSE of the
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
    epochs, device, model, batch_size, n_splits, lr, betas, kls_thresh, MODEL_VERSION, RDPDNET_VERSIONS,
)
from losses.losses import MSELoss, elastic_regularization, target_nmse, target_rel_mse

N_FOLDS_RUN = n_folds_run
LAMBDA_REC = lambda_rec
LAMBDA_INF = lambda_inf
KL_WARMUP_EPOCHS = kl_warmup_epochs

# Target term of eq_L_rec: "rel" = target_rel_mse (per-cell, each target weighted by its
# own peak), "nmse" = target_nmse (per-map, dominated by the strongest target).
TARGET_LOSS = target_loss
target_term = target_rel_mse if TARGET_LOSS == "rel" else target_nmse

output_dir = os.path.join(output_dir, "pdnet_no_afm")

LOG_FIELDS = ["epoch", "L_rec", "L_rec_inf", "L_kl", "L_denoise", "nmse_train", "val_mse", "nmse_val"]


if __name__ == "__main__":
    df = load_radar_results(db_path, n_records=n_records)
    loader = RDMapDataloader(
        df=df, n_splits=n_splits, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    fold_dataloaders = loader.get_fold_dataloaders()

    for fold_idx, (train_loader, val_loader) in enumerate(fold_dataloaders):
        print(f"\n{'=' * 50}\nFold {fold_idx + 1} / {len(fold_dataloaders)}  (no AFM)\n{'=' * 50}")

        fold_output_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
        checkpoint_dir = os.path.join(fold_output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        log_path = os.path.join(fold_output_dir, "train_log.csv")
        with open(log_path, "w", newline="") as fh:
            csv.writer(fh).writerow(LOG_FIELDS)

        model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
        model_opt = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
        mse_loss = MSELoss().to(device)
        best_val_nmse = float("inf")

        # Warm start: if this fold already has a best checkpoint, load it and keep training.
        ckpt_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            if ckpt.get("model_version", MODEL_VERSION) not in RDPDNET_VERSIONS:
                print(f"Checkpoint {ckpt_path} is model {ckpt.get('model_version')}, "
                      f"not {MODEL_VERSION}: training from scratch.")
            else:
                model.load_state_dict(ckpt["model_state_dict"])
                model_opt.load_state_dict(ckpt["model_optimizer_state_dict"])
                best_val_nmse = ckpt["val_nmse"]   # overwrite the checkpoint only if the new run beats it
                print(f"Loaded {ckpt_path} (epoch {ckpt['epoch']}, val NMSE {best_val_nmse:.4f})")

        train_denoise_losses, train_nmse_list = [], []
        val_denoise_losses, val_nmse_list = [], []

        for epoch in range(epochs):
            kl_w = kls_thresh * (min(1.0, (epoch + 1) / KL_WARMUP_EPOCHS) if KL_WARMUP_EPOCHS > 0 else 1.0)
            print(f"\nEpoch {epoch + 1}/{epochs}")
            model.train()
            tot_den = tot_rec = tot_inf = tot_kl = tot_nmse = 0.0
            nb = 0

            for Z_rd, Z_rd_prbs, M_tars in tqdm(train_loader):
                Z_rd = Z_rd.to(device)
                Z_rd_prbs = Z_rd_prbs.to(device)
                M_tars = M_tars.to(device)                 # target support of eq_L_rec

                model_opt.zero_grad()
                # f_den^train (teacher forcing): L_rec + L_KL is the ELBO of Prop. 1
                Z_rd_hat, kls = model(Z_rd, Z_rd_prbs)
                # f_den^infer (with gradient): the map used at run time
                Z_rd_hat_inf = model.predict(Z_rd)

                L_rec = mse_loss(Z_rd_hat, Z_rd_prbs) + LAMBDA_REC * target_term(Z_rd_hat, Z_rd_prbs, M_tars)
                L_rec_inf = mse_loss(Z_rd_hat_inf, Z_rd_prbs) + LAMBDA_REC * target_term(Z_rd_hat_inf, Z_rd_prbs, M_tars)
                L_kl = sum(kls) / len(kls)

                L_denoise = L_rec + LAMBDA_INF * L_rec_inf + kl_w * L_kl + elastic_regularization(model)
                L_denoise.backward()
                model_opt.step()

                nmse = compute_nmse(Z_rd_hat_inf.detach().flatten(start_dim=1),   # NMSE of f_den^infer
                                    Z_rd_prbs.flatten(start_dim=1), dim=1)

                tot_den += L_denoise.item(); tot_rec += L_rec.item()
                tot_inf += L_rec_inf.item(); tot_kl += float(L_kl)
                tot_nmse += nmse.mean().item()
                nb += 1

            avg = lambda x: x / nb
            train_denoise_losses.append(avg(tot_den))
            train_nmse_list.append(avg(tot_nmse))
            print(f"Train — Denoise {avg(tot_den):.4f}  L_rec_inf {avg(tot_inf):.4f}  NMSE {avg(tot_nmse):.4f}")

            # ---------- Validation (f_den^infer, as in the AFM scripts) ----------
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
                csv.writer(fh).writerow([epoch + 1, avg(tot_rec), avg(tot_inf), avg(tot_kl),
                                         avg(tot_den), avg(tot_nmse), v_den, v_nmse])

            if v_nmse < best_val_nmse:
                best_val_nmse = v_nmse
                torch.save({"fold": fold_idx + 1, "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "model_optimizer_state_dict": model_opt.state_dict(),
                            "val_nmse": best_val_nmse, "variant": "no_afm",
                            "model_version": MODEL_VERSION, "lambda_inf": LAMBDA_INF,
                            "lambda_rec": LAMBDA_REC, "kl_warmup_epochs": KL_WARMUP_EPOCHS, "target_loss": TARGET_LOSS},
                           os.path.join(checkpoint_dir, "best_checkpoint.pth"))
                print(f"Saved checkpoint — best val NMSE: {best_val_nmse:.4f}")
                plot_matrix({"Z_rd": Z_rd[0], "Z_rd_hat": Z_rd_hat[0],
                             "Z_rd_prbs": Z_rd_prbs[0], "M_tars": M_tars[0]},
                            save_path=os.path.join(fold_output_dir, "visualize_matrix.png"))

        epoch_range = range(1, len(train_denoise_losses) + 1)
        plot_denoise_loss(epoch_range, train_denoise_losses, val_denoise_losses, fold_output_dir)
        plot_nmse(epoch_range, train_nmse_list, val_nmse_list, fold_output_dir)

        if fold_idx + 1 >= N_FOLDS_RUN:
            break
