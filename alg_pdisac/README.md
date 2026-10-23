# RDPDNet: Python training, test and inference (PDISAC)

This folder contains the Python side of PDISAC: the **RDPDNet** denoiser, the **DnCNN** baseline, their training with and without **adversarial frequency-mixup (AFM)**, the test scripts, and the inference bridge that MATLAB calls during evaluation.

## Method

RDPDNet (`models/bases/pdnet.py`, version label `pdnet`) is a lightweight hierarchical latent-variable network inserted between the MF-based range-Doppler (RD) map and CA-CFAR detection. It suppresses the data-induced sidelobes (`Z_rd^ran`) and the thermal noise (`N_rd^mf`) that embedding data in the PMCW waveform adds to the RD map.

- **Training path** `f_den^train(Z_rd, Z_rd^prbs)`: an encoder `q(z_l | z_{l-1})` on the clean map `Z_rd^prbs`, the extractor prior `p(z_L | Z_rd)` and the decoder `p(z_l | z_{l+1})`, `p(Ẑ_rd | z_1)`, trained with a reconstruction loss and per-level KL terms.
- **Inference path** `f_den^infer(Z_rd) = f_dec(f_ext(Z_rd))`: the encoder is dropped, so the denoiser needs no knowledge of the embedded symbols.
- **AFM**: two realisations of the same scene, `Z_rd` and `Z_rd_alt` (other data symbols and noise), are mixed cell by cell outside the target support, `Z_mix = C + sqrt(1 − M_afm) D + sqrt(M_afm) D'`. MaskNet (`models/bases/afm.py`) chooses `M_afm` to increase the error of the denoiser on `Z_mix`, while the denoiser is trained to decrease it; the two are updated alternately.

The DnCNN baseline (`models/bases/dncnn.py`) is trained with the same data, losses, AFM and checkpoint selection.

## Requirements

Python 3.13.14 in the conda environment `afm` (the environment MATLAB looks for in `simulation_matlab/main1_matlab_config.m`) and the packages of `requirements.txt`:

```bash
conda create -n afm python=3.13.14 -y
conda activate afm
pip install -r requirements.txt
```

`requirements.txt` installs the CUDA 13.0 build of PyTorch; on another platform, install `torch` and `torchvision` from <https://pytorch.org> first.

## Layout

```
alg_pdisac/
├── configs/
│   ├── pdisac_system.json    system and training settings (single source of truth), smoke overrides
│   ├── inference.json        trained denoisers used by MATLAB and scripts/test (name, version, checkpoint)
│   ├── config_loader.py      loads pdisac_system.json, applies the PDISAC_SMOKE overrides
│   ├── data.py               db_path, output_dir
│   ├── ml.py                 hyperparameters, RDPDNet and the AFM MaskNet
│   └── isac.py               waveform-derived constants (N_chip, N_block, axes)
├── data/
│   ├── read_data.py          SQLite database -> dataframe (RD maps, target truth, target support M_tars)
│   ├── dataloader.py         (Z_rd, Z_rd_prbs, M_tars), GroupKFold on the scene index
│   ├── dataloader_afm.py     (Z_rd, Z_rd_prbs, Z_rd_alt, M_tars) for the AFM scripts
│   ├── dataset_train.db      training set (written by MATLAB)
│   └── dataset_smoke.db      smoke set (written by MATLAB)
├── models/
│   ├── model.py              DenoiseModel wrapper; parameter, FLOP and latency helpers
│   └── bases/                pdnet.py (RDPDNet), dncnn.py (DnCNN), afm.py (MaskNet)
├── losses/                   reconstruction, target-support and regularisation terms
├── utils/                    AFM mixing and logging (afm_util.py), NMSE, plots
├── scripts/
│   ├── train/                train_pdnet_afm.py, train_pdnet_no_afm.py, train_dncnn_afm.py, train_dncnn_no_afm.py
│   ├── test/                 test_nmse.py, eval_detection.py, complexity_table.py
│   └── infer/                denoiser.py (shared loader), denoise_mat.py, afm_maps_mat.py, distribution_mat.py
└── exps/                     checkpoints, logs and figures of the runs
```

## Configuration

All training settings are in the `machine_learning` block of **`configs/pdisac_system.json`**:

| key | role | symbol in the paper |
|---|---|---|
| `epochs`, `batch_size`, `learning_rate`, `betas` | optimiser | |
| `folds`, `folds_run` | GroupKFold folds, folds actually trained | |
| `hidden_channels`, `latent_levels` | RDPDNet width and number of levels `L` | `L` |
| `target_weight`, `target_loss` | target-support term of `L_rec` (`rel` or `nmse`) | `λ_rec` |
| `inference_weight` | weight of `L_rec^inf` | `λ_inf` |
| `reconstruction_weight` | weight of `L_rec^mix` (AFM) | `λ_mix` |
| `kl_weight`, `kl_warmup_epochs` | KL weight and its linear warm-up | `λ_KL` |
| `mask_weight` | energy penalty of the AFM mask | `λ_mask` |
| `dncnn_depth`, `filters`, `kernel_size`, `image_channels` | DnCNN size | |

The system values (`carrier_hz`, `bandwidth_hz`, `chips`, `blocks`, `roi_m`, maximum speeds) must equal those of `simulation_matlab/main1_matlab_config.m`. The environment variable **`PDISAC_SMOKE`** applies the `smoke` block (smoke database, 2 epochs, 2 folds, CPU); it is the same switch the MATLAB dataset generator uses.

**`configs/inference.json`** lists the trained denoisers by name; the names are the chain labels used by MATLAB:

| name | model | training script | checkpoint |
|---|---|---|---|
| `pdnet` | RDPDNet + AFM (proposed) | `train_pdnet_afm.py` | `exps/pdnet_afm/fold_1/checkpoints/best_checkpoint.pth` |
| `pdnet_noafm` | RDPDNet without AFM | `train_pdnet_no_afm.py` | `exps/pdnet_no_afm/fold_1/…` |
| `dncnn` | DnCNN + AFM | `train_dncnn_afm.py` | `exps/dncnn_afm/fold_1/…` |
| `dncnn_noafm` | DnCNN without AFM | `train_dncnn_no_afm.py` | `exps/dncnn_no_afm/fold_1/…` |

## Running

All commands are run from this `alg_pdisac/` folder, so that `configs`, `data`, `models` and `scripts` resolve as packages.

**Train**

```bash
python -m scripts.train.train_pdnet_afm
python -m scripts.train.train_pdnet_no_afm
python -m scripts.train.train_dncnn_afm
python -m scripts.train.train_dncnn_no_afm
PDISAC_SMOKE=1 python -m scripts.train.train_pdnet_afm      # smoke check
```

The AFM scripts need the second realisation `Z_rd_alt` of every scene (column `RD_map_alt_compressed`), which `generate_radar_dataset.m` writes. An existing checkpoint is used as a warm start and is overwritten only when the run reaches a lower validation NMSE of the inference map. The AFM runs log the mask coverage and `L_rec_mix − L_rec_inf` per epoch to `afm_evidence.csv`.

**Test**

```bash
python -m scripts.test.test_nmse --models all --fold 1                   # NMSE on the validation fold
python -m scripts.test.eval_detection --models all --fold 1              # CA-CFAR P_d / P_fa, ROC, peak-to-background ratio
python -m scripts.test.complexity_table --device cuda --batches 1,8,32   # parameters, FLOPs, latency per frame
```

**Inference (MATLAB bridge)**

The `scripts/infer/*_mat.py` scripts are launched by MATLAB through `PDISAC_cfg.inference.python_executable`; they can also be run by hand:

```bash
python scripts/infer/denoise_mat.py --input in.mat --output out.mat --models pdnet,dncnn
python scripts/infer/afm_maps_mat.py --input in.mat --output out.mat --models pdnet,dncnn --mask-model pdnet
python scripts/infer/distribution_mat.py --input in.mat --output out.mat --model pdnet
```

| script | called by | input MAT | output MAT |
|---|---|---|---|
| `denoise_mat.py` | `inference_run_pdnet.m` (`main5b`, `main5c`) | `RD_map_input_inference` (B × H × W complex) | `RD_map_denoised_inference` (K × B × H × W), model names and checkpoints |
| `afm_maps_mat.py` | `main5d_fig_maps.m` | `Z_rd`, `Z_rd_alt`, `Z_rd_prbs`, `truth` | denoised maps, `M_mask`, `M_tars`, `M_afm`, `Z_mix` and its denoised map |
| `distribution_mat.py` | `main5e_fig_dist.m` | `Z_rd`, `Z_rd_prbs` | encoder and inference-chain distributions of RDPDNet |

## Link to the MATLAB code

1. **Data (MATLAB → Python).** `simulation_matlab/generate_radar_dataset.m` writes `data/dataset_train.db` (or `data/dataset_smoke.db` under `PDISAC_SMOKE`), read by `data/read_data.py`.
2. **Checkpoints (Python → MATLAB).** The trainers write `exps/<model>_<afm|no_afm>/fold_1/checkpoints/best_checkpoint.pth`; `configs/inference.json` maps them to the chain names used by MATLAB.
3. **Inference bridge (MATLAB → Python).** The MATLAB evaluation calls `scripts/infer/*_mat.py` to denoise RD maps, rebuild the AFM masks and export the RDPDNet distributions.

See [`../README.md`](../README.md) for the full pipeline and [`../simulation_matlab/README.md`](../simulation_matlab/README.md) for the MATLAB side.
