import os

import torch
from scipy.io import savemat
from tqdm import tqdm

from data.dataloader import RDMapDataloader
from data.read_data import load_radar_results
from utils.utils import compute_nmse
from utils.plots import plot_matrix
from configs.data import output_dir, db_path, n_records
from configs.ml import device, model, afm_model, batch_size, n_splits, mask_thresh


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

    for fold_idx, (_, val_loader) in enumerate(fold_dataloaders):
        print(f"\n{'=' * 50}")
        print(f"Fold {fold_idx + 1} / {len(fold_dataloaders)}")
        print(f"{'=' * 50}")

        fold_output_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
        checkpoint_dir = os.path.join(output_dir, "checkpoints")  # no fold subfolder (matches MATLAB inference path)
        fold_eval_dir = os.path.join(fold_output_dir, "eval")

        # ----------------------------------------------------------------
        #  Load checkpoint for this fold (same pattern as train.py resume)
        # ----------------------------------------------------------------
        checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            afm_model.load_state_dict(checkpoint["afm_model_state_dict"])
            print(
                f"Loaded checkpoint: {checkpoint_path} "
                f"(val NMSE: {checkpoint['val_nmse']:.4f})"
            )
        else:
            print(f"Checkpoint not found at {checkpoint_path}, skipping fold.")
            continue

        model.eval()
        afm_model.eval()

        complex_Z_rd_prbs_list = []
        complex_Z_rd_hat_list = []
        complex_Z_rd_mix_hat_list = []

        total_nmse, total_mix_nmse, num_batches = 0.0, 0.0, 0

        with torch.no_grad():
            for Z_rd, Z_rd_prbs in tqdm(val_loader):
                # Z_rd      : noisy range-Doppler map  (B, 2, H, W)
                # Z_rd_prbs : clean reference          (B, 2, H, W)
                Z_rd = Z_rd.to(device)
                Z_rd_prbs = Z_rd_prbs.to(device)

                M_tars = model.create_mask(Z_rd_prbs)

                # Denoiser prediction on noisy input
                Z_rd_hat = model.predict(Z_rd)

                # NMSE: denoised vs. clean reference
                nmse = compute_nmse(
                    Z_rd_hat.flatten(start_dim=1),
                    Z_rd_prbs.flatten(start_dim=1),
                    dim=1,
                )

                # Adversarial mask + mix (with noisy Z_rd, consistent with train.py)
                M_mask = afm_model(Z_rd_hat)
                M_afm = M_mask * (1 - M_tars)
                Z_rd_mix = M_afm * Z_rd_hat + (1 - M_afm) * Z_rd
                Z_rd_mix_hat = model.predict(Z_rd_mix)

                mix_nmse = compute_nmse(
                    Z_rd_mix_hat.flatten(start_dim=1),
                    Z_rd_prbs.flatten(start_dim=1),
                    dim=1,
                )

                total_nmse += nmse.mean().item()
                total_mix_nmse += mix_nmse.mean().item()
                num_batches += 1

                # Convert real/imag channels into complex tensors for .mat export
                complex_Z_rd_prbs_list.append(
                    torch.complex(Z_rd_prbs[:, 0], Z_rd_prbs[:, 1]).cpu()
                )
                complex_Z_rd_hat_list.append(
                    torch.complex(Z_rd_hat[:, 0], Z_rd_hat[:, 1]).cpu()
                )
                complex_Z_rd_mix_hat_list.append(
                    torch.complex(Z_rd_mix_hat[:, 0], Z_rd_mix_hat[:, 1]).cpu()
                )

        avg_nmse = total_nmse / num_batches
        avg_mix_nmse = total_mix_nmse / num_batches
        print(f"Fold {fold_idx + 1} — NMSE: {avg_nmse:.4f}  Mixed NMSE: {avg_mix_nmse:.4f}")

        # Stack collected complex tensors across the val set
        complex_Z_rd_prbs = torch.cat(complex_Z_rd_prbs_list, dim=0).numpy()
        complex_Z_rd_hat = torch.cat(complex_Z_rd_hat_list, dim=0).numpy()
        complex_Z_rd_mix_hat = torch.cat(complex_Z_rd_mix_hat_list, dim=0).numpy()

        output_path = os.path.join(fold_eval_dir, "data.mat")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
            save_path=os.path.join(fold_eval_dir, "visualize_matrix.png"),
        )

        savemat(
            output_path,
            {
                "complex_Z_rd_prbs": complex_Z_rd_prbs,
                "complex_Z_rd_hat": complex_Z_rd_hat,
                "complex_Z_rd_mix_hat": complex_Z_rd_mix_hat,
            },
        )

        print(f"Data saved to {output_path}")

        break  # Remove this to evaluate all folds