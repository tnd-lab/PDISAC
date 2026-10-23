"""
DnCNN trained with AFM (DnCNN baseline; counterpart of train_pdnet_afm.py).

Same AFM, data loader, split, loss terms, weights, checkpoint selection and keys as
train_pdnet_afm.py; the denoiser is a residual DnCNN (models/bases/dncnn.py) instead of the
RDPDNet. DnCNN has no latent branch, so there is no L_KL and the training path equals the
inference path; the RDPDNet objective then reads
        L_den = (1 + lambda_inf) * L_rec + lambda_mix * L_rec_mix + elastic ,
lambda_mix = reconstruction_weight (rec_thresh).
The checkpoint stores arch = "dncnn" and the DnCNN hyperparameters, so the inference and
test scripts rebuild the right network.

Run (from alg_pdisac/):
    python -m scripts.train.train_dncnn_afm                  # dataset_train.db
    PDISAC_SMOKE=1 python -m scripts.train.train_dncnn_afm   # smoke test (dataset_smoke.db)
Settings: configs/pdisac_system.json (machine_learning block).
Checkpoint: exps/dncnn_afm/fold_<k>/checkpoints/best_checkpoint.pth (best validation NMSE of the
inference map), read by scripts/infer/* through configs/inference.json.
"""
import os

import torch
from tqdm import tqdm

from data.dataloader_afm import RDMapDataloaderAFM, load_radar_results_afm
from utils.utils import compute_nmse
from utils.plots import plot_denoise_loss, plot_afm_and_nmse, plot_matrix
from configs.data import output_dir, db_path, n_records
from configs.ml import (
    n_folds_run, kl_warmup_epochs, lambda_rec, lambda_inf, target_loss, num_workers, pin_memory,
    epochs, device, batch_size, n_splits, lr, betas, rec_thresh, mask_thresh,
    depth, n_filters, kernel_size, img_channels,
)
from models.bases.afm import MaskNet
from models.bases.dncnn import DnCNN
from models.model import DenoiseModel
from losses.losses import MSELoss, elastic_regularization, target_nmse, target_rel_mse
from utils.afm_util import mask_coverage, AfmLogger, mix_realizations

N_FOLDS_RUN = n_folds_run
LAMBDA_REC = lambda_rec
LAMBDA_INF = lambda_inf
LAMBDA_MASK = mask_thresh
TARGET_LOSS = target_loss
target_term = target_rel_mse if TARGET_LOSS == "rel" else target_nmse

ARCH = "dncnn"
DNCNN_CFG = dict(depth=depth, n_filters=n_filters, kernel_size=kernel_size, img_channels=img_channels)
model = DenoiseModel(model=DnCNN(**DNCNN_CFG)).to(device)
afm_model = MaskNet(in_ch=4).to(device)

output_dir = os.path.join(output_dir, "dncnn_afm")


if __name__ == "__main__":
    print(f"DnCNN {DNCNN_CFG}: {sum(p.numel() for p in model.parameters()):,} parameters")
    df = load_radar_results_afm(db_path, n_records=n_records)
    loader = RDMapDataloaderAFM(
        df=df, n_splits=n_splits, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    fold_dataloaders = loader.get_fold_dataloaders()

    for fold_idx, (train_loader, val_loader) in enumerate(fold_dataloaders):
        print(f"\n{'=' * 50}\nFold {fold_idx + 1} / {len(fold_dataloaders)}  "
              f"(DnCNN, data-symbol adversary, lambda_mask={LAMBDA_MASK})\n{'=' * 50}")

        fold_output_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
        checkpoint_dir = os.path.join(fold_output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        evidence = AfmLogger(os.path.join(fold_output_dir, "afm_evidence.csv"))

        model.model._initialize_weights()            # DnCNN init (Kaiming normal), per fold
        afm_model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
        model_opt = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
        afm_model_opt = torch.optim.Adam(afm_model.parameters(), lr=lr, betas=betas)
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
                afm_model.load_state_dict(ckpt["afm_model_state_dict"])
                model_opt.load_state_dict(ckpt["model_optimizer_state_dict"])
                afm_model_opt.load_state_dict(ckpt["afm_optimizer_state_dict"])
                best_val_nmse = ckpt["val_nmse"]
                print(f"Loaded {ckpt_path} (epoch {ckpt['epoch']}, val NMSE {best_val_nmse:.4f})")

        train_denoise_losses, train_afm_losses, train_nmse_list = [], [], []
        val_denoise_losses, val_afm_losses, val_nmse_list = [], [], []

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            model.train(); afm_model.train()
            tot_den = tot_afm = tot_nmse = tot_rec = tot_mix = tot_cov = 0.0
            nb = 0

            for Z_rd, Z_rd_prbs, Z_rd_alt, M_tars in tqdm(train_loader):
                Z_rd = Z_rd.to(device)
                Z_rd_prbs = Z_rd_prbs.to(device)
                Z_rd_alt = Z_rd_alt.to(device)
                M_tars = M_tars.to(device)

                # ---------- Step 1: denoiser ----------------------------------
                model_opt.zero_grad()
                Z_rd_hat = model.predict(Z_rd)             # training path = inference path
                with torch.no_grad():
                    M_mask = afm_model(torch.cat([Z_rd_hat.detach(), Z_rd_alt - Z_rd], dim=1))
                M_afm = M_mask * (1 - M_tars)
                Z_rd_mix = mix_realizations(Z_rd, Z_rd_alt, Z_rd_prbs, M_afm)
                Z_rd_mix_hat = model.predict(Z_rd_mix)

                L_rec = mse_loss(Z_rd_hat, Z_rd_prbs) + LAMBDA_REC * target_term(Z_rd_hat, Z_rd_prbs, M_tars)
                L_rec_mix = mse_loss(Z_rd_mix_hat, Z_rd_prbs) + LAMBDA_REC * target_term(Z_rd_mix_hat, Z_rd_prbs, M_tars)
                L_denoise = ((1 + LAMBDA_INF) * L_rec + rec_thresh * L_rec_mix
                             + elastic_regularization(model))
                L_denoise.backward()
                model_opt.step()

                nmse = compute_nmse(Z_rd_hat.detach().flatten(start_dim=1),
                                    Z_rd_prbs.flatten(start_dim=1), dim=1)

                # ---------- Step 2: adversary ---------------------------------
                afm_model_opt.zero_grad(); model_opt.zero_grad()
                with torch.no_grad():
                    Z_rd_hat_d = model.predict(Z_rd)
                M_mask_adv = afm_model(torch.cat([Z_rd_hat_d, Z_rd_alt - Z_rd], dim=1))
                M_afm_adv = M_mask_adv * (1 - M_tars)
                Z_rd_mix_adv = mix_realizations(Z_rd, Z_rd_alt, Z_rd_prbs, M_afm_adv)
                Z_rd_mix_hat_adv = model.predict(Z_rd_mix_adv)

                L_mask = (-mse_loss(Z_rd_mix_hat_adv, Z_rd_prbs)
                          + LAMBDA_MASK * M_mask_adv.pow(2).mean())
                L_mask.backward()
                afm_model_opt.step()
                model_opt.zero_grad()

                tot_den += L_denoise.item(); tot_afm += L_mask.item()
                tot_rec += L_rec.item();     tot_mix += L_rec_mix.item()
                tot_nmse += nmse.mean().item()
                tot_cov += mask_coverage(M_afm_adv.detach(), M_tars).item()
                nb += 1

            avg = lambda x: x / nb
            train_denoise_losses.append(avg(tot_den))
            train_afm_losses.append(avg(tot_afm))
            train_nmse_list.append(avg(tot_nmse))
            print(f"Train — Denoise {avg(tot_den):.4f}  AFM {avg(tot_afm):.4f}  NMSE {avg(tot_nmse):.4f}")
            print(AfmLogger.report(epoch + 1, avg(tot_cov), avg(tot_rec), avg(tot_mix)))

            # ---------- Validation ---------------------------------------
            model.eval(); afm_model.eval()
            v_den = v_afm = v_nmse = 0.0; vb = 0
            with torch.no_grad():
                for Z_rd, Z_rd_prbs, Z_rd_alt, M_tars in val_loader:
                    Z_rd = Z_rd.to(device); Z_rd_prbs = Z_rd_prbs.to(device)
                    Z_rd_alt = Z_rd_alt.to(device)
                    M_tars = M_tars.to(device)

                    Z_rd_hat = model.predict(Z_rd)
                    nmse = compute_nmse(Z_rd_hat.flatten(start_dim=1),
                                        Z_rd_prbs.flatten(start_dim=1), dim=1)
                    M_mask = afm_model(torch.cat([Z_rd_hat, Z_rd_alt - Z_rd], dim=1))
                    M_afm = M_mask * (1 - M_tars)
                    Z_rd_mix = mix_realizations(Z_rd, Z_rd_alt, Z_rd_prbs, M_afm)
                    Z_rd_mix_hat = model.predict(Z_rd_mix)

                    v_den += mse_loss(Z_rd_hat, Z_rd_prbs).item()
                    v_afm += (-mse_loss(Z_rd_mix_hat, Z_rd_prbs)
                              + LAMBDA_MASK * M_mask.pow(2).mean()).item()
                    v_nmse += nmse.mean().item(); vb += 1

            v_den /= vb; v_afm /= vb; v_nmse /= vb
            val_denoise_losses.append(v_den); val_afm_losses.append(v_afm)
            val_nmse_list.append(v_nmse)
            print(f"Val   — Denoise {v_den:.4f}  AFM {v_afm:.4f}  NMSE {v_nmse:.4f}")

            # training path = inference path: L_rec and L_rec_inf are the same loss here
            evidence.log(epoch=epoch + 1, coverage=avg(tot_cov), L_rec=avg(tot_rec),
                         L_rec_inf=avg(tot_rec), L_rec_mix=avg(tot_mix),
                         mix_gap=avg(tot_mix) - avg(tot_rec), L_kl=0.0,
                         L_denoise=avg(tot_den), nmse_train=avg(tot_nmse), nmse_val=v_nmse)

            if v_nmse < best_val_nmse:
                best_val_nmse = v_nmse
                torch.save({"fold": fold_idx + 1, "epoch": epoch + 1,
                            "afm_model_state_dict": afm_model.state_dict(),
                            "model_state_dict": model.state_dict(),
                            "afm_optimizer_state_dict": afm_model_opt.state_dict(),
                            "model_optimizer_state_dict": model_opt.state_dict(),
                            "val_nmse": best_val_nmse, "lambda_mask": LAMBDA_MASK,
                            "masknet_in_ch": 4, "mixing": "variance_preserving",
                            "variant": "data_symbol_adversary",
                            "arch": ARCH, "dncnn_cfg": DNCNN_CFG, "model_version": ARCH,
                            "lambda_inf": LAMBDA_INF, "lambda_rec": LAMBDA_REC, "target_loss": TARGET_LOSS},
                           ckpt_path)
                print(f"Saved checkpoint — best val NMSE: {best_val_nmse:.4f}")
                plot_matrix({"Z_rd": Z_rd[0], "Z_rd_alt": Z_rd_alt[0],
                             "Z_rd_hat": Z_rd_hat[0], "Z_rd_prbs": Z_rd_prbs[0],
                             "Z_rd_mix": Z_rd_mix[0], "Z_rd_mix_hat": Z_rd_mix_hat[0],
                             "M_mask": M_mask[0], "M_tars": M_tars[0], "M_afm": M_afm[0]},
                            save_path=os.path.join(fold_output_dir, "visualize_matrix.png"))

        epoch_range = range(1, epochs + 1)
        plot_denoise_loss(epoch_range, train_denoise_losses, val_denoise_losses, fold_output_dir)
        plot_afm_and_nmse(epoch_range, train_afm_losses, val_afm_losses,
                          train_nmse_list, val_nmse_list, fold_output_dir)

        if fold_idx + 1 >= N_FOLDS_RUN:
            break
