import torch
import os
from data.dataloader import RDMapDataloader
from data.read_data import load_radar_results
from utils.utils import compute_nmse
from utils.plots import plot_denoise_loss, plot_afm_and_nmse, plot_matrix
from configs.data import output_dir, db_path, n_records
from configs.ml import (
    epochs,
    device,
    model,
    afm_model,
    batch_size,
    n_splits,
    lr,
    betas,
    rec_thresh,
    mask_thresh,
    kls_thresh,
)
from losses.losses import MSELoss, elastic_regularization
from tqdm import tqdm


if __name__ == "__main__":
    df = load_radar_results(db_path, n_records=n_records)
    loader = RDMapDataloader(
        df=df,
        n_splits=n_splits,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
    )

    fold_dataloaders = loader.get_fold_dataloaders()

    for fold_idx, (train_loader, val_loader) in enumerate(fold_dataloaders):
        print(f"\n{'=' * 50}")
        print(f"Fold {fold_idx + 1} / {len(fold_dataloaders)}")
        print(f"{'=' * 50}")

        fold_output_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
        checkpoint_dir = os.path.join(output_dir, "checkpoints")  # no fold subfolder (matches MATLAB inference path)
        os.makedirs(fold_output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        model.apply(
            lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None
        )
        afm_model.apply(
            lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None
        )
        model_opt = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
        afm_model_opt = torch.optim.Adam(afm_model.parameters(), lr=lr, betas=betas)
        mse_loss = MSELoss().to(device)

        best_val_nmse = float("inf")
        
        # Resume from a checkpoint for this fold if one already exists
        checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            afm_model.load_state_dict(checkpoint["afm_model_state_dict"])
            model_opt.load_state_dict(checkpoint["model_optimizer_state_dict"])
            afm_model_opt.load_state_dict(checkpoint["afm_optimizer_state_dict"])
            best_val_nmse = checkpoint["val_nmse"]
            print(f"Resumed from checkpoint: {checkpoint_path} (best val NMSE: {best_val_nmse:.4f})")

        train_denoise_losses, train_afm_losses, train_nmse_list = [], [], []
        val_denoise_losses, val_afm_losses, val_nmse_list = [], [], []

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # ----------------------------------------------------------------
            #  Training
            # ----------------------------------------------------------------
            model.train()
            afm_model.train()
            total_loss_denoise, total_loss_afm, total_nmse, num_batches = (
                0.0,
                0.0,
                0.0,
                0,
            )

            for Z_rd, Z_rd_prbs in tqdm(train_loader):
                # Z_rd      : noisy range-Doppler map  (B, 2, H, W)
                # Z_rd_prbs : clean reference          (B, 2, H, W)
                Z_rd = Z_rd.to(device)
                Z_rd_prbs = Z_rd_prbs.to(device)

                # Binary mask marking known pilot (PRBS) positions
                M_tars = model.create_mask(Z_rd_prbs)

                # --------------------------------------------------------
                # Step 1 — Update denoiser (model)
                # --------------------------------------------------------
                model_opt.zero_grad()

                # Forward pass on noisy input
                Z_rd_hat, kls = model(Z_rd, Z_rd_prbs)

                # Adversarial mask from frozen afm_model output
                # FIX: detach so afm_model gets no gradient from this step
                M_mask = afm_model(Z_rd_hat).detach()
                M_afm = M_mask * (1 - M_tars)

                # FIX: mix denoised output with NOISY Z_rd (not clean Z_rd_prbs)
                # Creates a partially-denoised / partially-noisy adversarial image
                Z_rd_mix = M_afm * Z_rd_hat + (1 - M_afm) * Z_rd

                # Denoise the adversarially mixed image
                Z_rd_mix_hat, _ = model(Z_rd_mix, Z_rd_prbs)

                # Reconstruction losses against clean reference Z_rd_prbs
                L_rec = mse_loss(Z_rd_hat, Z_rd_prbs) + mse_loss(
                    M_tars * Z_rd_hat, M_tars * Z_rd_prbs
                )
                L_rec_mix = mse_loss(Z_rd_mix_hat, Z_rd_prbs) + mse_loss(
                    M_tars * Z_rd_mix_hat, M_tars * Z_rd_prbs
                )

                # FIX: safer KL averaging — works for both scalar and tensor KLs
                L_kl = sum(kls) / len(kls)

                L_denoise = (
                    L_rec
                    + rec_thresh * L_rec_mix
                    + L_kl * kls_thresh
                    + elastic_regularization(model)
                )
                L_denoise.backward()
                model_opt.step()

                # NMSE: denoised vs. clean reference  ← correct direction
                nmse = compute_nmse(
                    Z_rd_hat.detach().flatten(start_dim=1),
                    Z_rd_prbs.flatten(start_dim=1),
                    dim=1,
                )

                # --------------------------------------------------------
                # Step 2 — Update mask model (afm_model)
                # --------------------------------------------------------
                # Zero BOTH optimizers:
                #   - afm_model_opt : we will update this
                #   - model_opt     : we will NOT update this, but backward()
                #                     accumulates grads in model params too;
                #                     zeroing here prevents silent accumulation.
                afm_model_opt.zero_grad()
                model_opt.zero_grad()

                # Denoiser output on noisy input — no gradient w.r.t. model params
                # needed here (the output itself is used as a fixed feature map).
                with torch.no_grad():
                    Z_rd_hat = model.predict(Z_rd)

                # Mask model forward — has gradients
                M_mask = afm_model(Z_rd_hat)
                M_afm = M_mask * (1 - M_tars)

                # FIX: mix denoised output with NOISY Z_rd (not clean Z_rd_prbs)
                Z_rd_mix = M_afm * Z_rd_hat + (1 - M_afm) * Z_rd

                # FIX: do NOT wrap in torch.no_grad()
                # Gradient must flow:  Z_rd_mix_hat → Z_rd_mix → M_afm → M_mask
                #                      → afm_model parameters
                # The denoiser weights are safe because model_opt.step() is
                # never called in this block.
                Z_rd_mix_hat = model.predict(Z_rd_mix)

                # Adversarial mask loss:
                #   - Maximize denoising error on the mixed image (adversarial)
                #   - Minimize mask density (sparsity constraint)
                # FIX: removed the constant `-mse_loss(Z_rd_hat, Z_rd_prbs)` term
                # which had zero gradient (Z_rd_hat came from no_grad above).
                L_mask = (
                    -mse_loss(Z_rd_mix_hat, Z_rd_prbs)
                    + mask_thresh * M_mask.pow(2).sum()
                )
                L_mask.backward()

                afm_model_opt.step()          # update mask model
                model_opt.zero_grad()         # discard any grads on denoiser weights

                total_loss_denoise += L_denoise.item()
                total_loss_afm += L_mask.item()
                total_nmse += nmse.mean().item()
                num_batches += 1

            avg_loss_denoise = total_loss_denoise / num_batches
            avg_loss_afm = total_loss_afm / num_batches
            avg_nmse = total_nmse / num_batches
            train_denoise_losses.append(avg_loss_denoise)
            train_afm_losses.append(avg_loss_afm)
            train_nmse_list.append(avg_nmse)
            print(
                f"Train — Denoise Loss: {avg_loss_denoise:.4f}  "
                f"AFM Loss: {avg_loss_afm:.4f}  NMSE: {avg_nmse:.4f}"
            )

            # ----------------------------------------------------------------
            #  Validation
            # ----------------------------------------------------------------
            model.eval()
            afm_model.eval()
            (
                total_val_loss_denoise,
                total_val_loss_afm,
                total_val_nmse,
                num_val_batches,
            ) = 0.0, 0.0, 0.0, 0

            with torch.no_grad():
                for Z_rd, Z_rd_prbs in val_loader:
                    Z_rd = Z_rd.to(device)
                    Z_rd_prbs = Z_rd_prbs.to(device)

                    M_tars = model.create_mask(Z_rd_prbs)

                    # Denoiser prediction on noisy input
                    Z_rd_hat = model.predict(Z_rd)
                    loss_denoise = mse_loss(Z_rd_hat, Z_rd_prbs)

                    # NMSE: denoised vs. clean reference
                    nmse = compute_nmse(
                        Z_rd_hat.flatten(start_dim=1),
                        Z_rd_prbs.flatten(start_dim=1),
                        dim=1,
                    )

                    # AFM mask and mix (with noisy Z_rd)
                    M_mask = afm_model(Z_rd_hat)
                    M_afm = M_mask * (1 - M_tars)

                    # FIX: mix with noisy Z_rd (consistent with training)
                    Z_rd_mix = M_afm * Z_rd_hat + (1 - M_afm) * Z_rd
                    Z_rd_mix_hat = model.predict(Z_rd_mix)

                    loss_afm = (
                        -mse_loss(Z_rd_mix_hat, Z_rd_prbs)
                        + mask_thresh * M_mask.pow(2).sum()
                    )

                    total_val_loss_denoise += loss_denoise.item()
                    total_val_loss_afm += loss_afm.item()
                    total_val_nmse += nmse.mean().item()
                    num_val_batches += 1

            avg_val_loss_denoise = total_val_loss_denoise / num_val_batches
            avg_val_loss_afm = total_val_loss_afm / num_val_batches
            avg_val_nmse = total_val_nmse / num_val_batches
            val_denoise_losses.append(avg_val_loss_denoise)
            val_afm_losses.append(avg_val_loss_afm)
            val_nmse_list.append(avg_val_nmse)
            print(
                f"Val   — Denoise Loss: {avg_val_loss_denoise:.4f}  "
                f"AFM Loss: {avg_val_loss_afm:.4f}  NMSE: {avg_val_nmse:.4f}"
            )

            # ----------------------------------------------------------------
            #  Checkpoint if best val NMSE
            # ----------------------------------------------------------------
            if avg_val_nmse < best_val_nmse:
                best_val_nmse = avg_val_nmse
                torch.save(
                    {
                        "fold": fold_idx + 1,
                        "epoch": epoch + 1,
                        "afm_model_state_dict": afm_model.state_dict(),
                        "model_state_dict": model.state_dict(),
                        "afm_optimizer_state_dict": afm_model_opt.state_dict(),
                        "model_optimizer_state_dict": model_opt.state_dict(),
                        "val_nmse": best_val_nmse,
                    },
                    os.path.join(checkpoint_dir, "best_checkpoint.pth"),
                )
                print(f"Saved checkpoint — best val NMSE: {best_val_nmse:.4f}")

                plot_matrix(
                    {
                        "Z_rd": Z_rd[0],
                        "Z_rd_hat": Z_rd_hat[0],
                        "Z_rd_prbs": Z_rd_prbs[0],
                        "Z_rd_mix": Z_rd_mix[0],
                        "Z_rd_mix_hat": Z_rd_mix_hat[0],
                        "M_mask": M_mask[0],
                        "M_tars": M_tars[0],
                        "M_afm": M_afm[0],
                    },
                    save_path=os.path.join(fold_output_dir, "visualize_matrix.png"),
                )

        # ----------------------------------------------------------------
        #  Per-fold plots
        # ----------------------------------------------------------------
        epoch_range = range(1, epochs + 1)
        plot_denoise_loss(
            epoch_range, train_denoise_losses, val_denoise_losses, fold_output_dir
        )
        plot_afm_and_nmse(
            epoch_range,
            train_afm_losses,
            val_afm_losses,
            train_nmse_list,
            val_nmse_list,
            fold_output_dir,
        )
        break  # Remove this to run all folds
