# PDISAC: MATLAB simulation and evaluation

This folder contains the MATLAB side of PDISAC. It simulates the multi-bit slot-partitioned PMCW waveform and the geometry-based channel, runs the conventional sensing and communication receivers, writes the training database of the denoisers, and produces the statistics and figures of the paper.

The `main1`–`main4` and `main6` scripts run in the MATLAB base workspace and share variables. The analysis and figure drivers (`main5a`–`main5e`, `main7`, `channel_com`, `check_gamma_gof`) build their own scenes and run on their own.

## Requirements

- MATLAB R2022b or newer.
- Phased Array System Toolbox: `phased.Platform` (target and UE motion), `physconst`.
- Statistics and Machine Learning Toolbox: distribution fits of `check_gamma_gof`.
- Database Toolbox: `sqlite` (training database for Python).
- For the learned denoisers: the Python 3.13.14 environment `afm` of `alg_pdisac/` (see [`../alg_pdisac/README.md`](../alg_pdisac/README.md)) and trained checkpoints.

## Configuration: `main1_matlab_config.m`

Single configuration file, sourced by every other script. It defines the raw settings, derives all constants and exposes the profile handles `dataset_*_cfg` and `*_statistics_cfg`.

- **System and waveform**: carrier, bandwidth, `chips` (`N_chip`), `blocks` (`N_prbs`), `symbols_per_block` (`N_bit^prbs`).
- **Scene**: ROI, UE, targets, number of scatterers, RCS ranges, maximum speeds, `communication_reflectors` (`N_tars^ref`).
- **CA-CFAR**: guard and training half-widths, nominal false-alarm probability `P_fa^nom`, peak selection.
- **Dataset profiles**: `dataset_train`, `dataset_test`, `dataset_smoke` (written into `alg_pdisac/data/`); the test profile also sets the scene of `main5d_fig_maps` and `main5e_fig_dist`.
- **Statistics profiles**: scenes, Monte Carlo runs, SNR grids and scene sizes of the analysis drivers.
- **Denoiser bridge**: `PDISAC_cfg.inference.python_executable`, the first existing interpreter of the conda environment `afm` in a list of common install paths; add yours to `py_candidates` if it is elsewhere. Models, checkpoints, device and batch size are set in `alg_pdisac/configs/inference.json`.

The system values shared with Python (carrier, bandwidth, chips, blocks, ROI, maximum speeds) must equal those of `alg_pdisac/configs/pdisac_system.json`.

## Scripts

**Single-scene pipeline**

| Script | Role |
|---|---|
| `main1_matlab_config.m` | Configuration. |
| `main2_topology.m` | Scene: UE, moving targets, static scatterers; plan view `fig_topology`. |
| `main3_signal_channel_model.m` | ISAC waveform (MLS with pilot and data slots) and the sensing and communication channels (band-limited fractional delays). |
| `main4_sensing_process.m` | Matched filter, RD map, zero-Doppler removal; optional CA-CFAR and interpolation figures. |
| `main6_comm_process.m` | UE receiver: delay estimation, pilot-slot channel estimation, BPSK detection. |
| `main8_main.m` | Runs the pipeline (STEP 1–5), renders the result figures from `exported_statistics/` (STEP 6a–6g) and writes the table of RMSE sweeps `exported_statistics/table_rmse_sweep.csv` (STEP 6h). |

**Analysis drivers** (write `exported_statistics/`)

| Script | Output | Used by `main8_main` |
|---|---|---|
| `main5a_sensing_sweep.m` | `statistics_sensing_sweep_nchip.csv`, `statistics_sensing_sweep_nprbs.csv` | STEP 6c, 6h |
| `main5b_detection_vs_nbit.m` | `statistics_detection_vs_nbit_ml.csv`, `statistics_detection_vs_nbit_ml_roc.csv` | STEP 6e |
| `main5c_bcrb_check.m` | `statistics_bcrb_check<tag>.csv` (runs below) | STEP 6f, 6g, 6h |
| `main7_comm_analysis.m` | `statistics_communication.csv` | STEP 6a |
| `channel_com.m` | `statistics_channel.mat`, `statistics_channel_summary.csv` | STEP 6b |
| `check_gamma_gof.m` | `statistics_gamma_gof.mat`, `statistics_gamma_gof.csv` | STEP 6d |

`main5b_detection_vs_nbit.m` appends `_ml` when `USE_DENOISER = true` (default); `main8_main` reads the file without `_ml` when the `_ml` file is absent.

`main5c_bcrb_check.m` builds the output tag from its flags. The runs read by `main8_main`:

| Run | Flags | File |
|---|---|---|
| conventional, sinc | `READOUT = "sinc"`, `LS_CANCEL = true`, `USE_DENOISER = false` | `statistics_bcrb_check_ls.csv` |
| conventional, Dirichlet | `READOUT = "dirichlet"`, `LS_CANCEL = true`, `USE_DENOISER = false` | `statistics_bcrb_check_dirichlet_ls.csv` |
| learned denoisers | `READOUT = "dirichlet"`, `LS_CANCEL = true`, `USE_DENOISER = true` (default) | `statistics_bcrb_check_dirichlet_ls_ml.csv` |
| RMSE vs `N_bit^prbs` | defaults, `for NBIT_EXT = [4 16 64 128], main5c_bcrb_check; end` (denoisers off) | `statistics_bcrb_check_dirichlet_ls_nbit<N>.csv` |

**Figure drivers** (call the Python denoisers, write `fig_exported/`)

| Script | Figures |
|---|---|
| `main5d_fig_maps.m` | RD maps of one test scene (with data, with LS cancellation, without data, clean), the maps denoised by RDPDNet and DnCNN with and without AFM, and the AFM masks and mixed map. |
| `main5e_fig_dist.m` | Distributions of RDPDNet on the same scene: encoder chain and inference chain. |

**Dataset and bridge**

| Script | Role |
|---|---|
| `generate_radar_dataset.m` | Writes `alg_pdisac/data/dataset_train.db` (`dataset_test.db` with `DATASET = 'test'`, `dataset_smoke.db` with `SMOKE = true` or `PDISAC_SMOKE`): the RD map `Z_rd`, a second realisation `Z_rd_alt` of the same scene (AFM), the clean map `Z_rd_prbs` and the target truth. |
| `inference_run_pdnet.m` | Denoises RD maps with the models of `alg_pdisac/configs/inference.json` (launches `alg_pdisac/scripts/infer/denoise_mat.py`). |

**Helper functions**: `func_bpsk_mod`, `func_bpsk_demod`, `func_ca_cfar_adaptive_threshold`, `func_frac_shift`, `func_generate_static_scatterers`, `func_get_array_response`, `func_get_x_delay_frac`, `func_mi_bpsk`, `func_sinc_interpolation`, `h_sen_tars`.

## Typical order

```matlab
% 1) training database
generate_radar_dataset

% 2) train the denoisers in Python (alg_pdisac/README.md)

% 3) statistics
main5a_sensing_sweep          % RMSE vs N_chip, N_prbs
main5b_detection_vs_nbit      % P_fa, P_d, ROC
main5c_bcrb_check             % RMSE vs BCRB: the runs of the table above
main7_comm_analysis           % BER and achievable rate
channel_com                   % LOS / NLOS channel distribution
check_gamma_gof               % goodness of fit of the NLOS power

% 4) figures
main5d_fig_maps               % RD maps, denoised maps, AFM
main5e_fig_dist               % RDPDNet distributions
main8_main                    % pipeline figures, result figures, table of RMSE sweeps
```

To run `main8_main` without the plots of the single-scene pipeline, define the flags before the call, e.g. `NO_SIGNAL_PLOT = true; NO_RADAR_PLOT = true; NO_CFAR_PLOT = true; NO_BPSK_PLOT = true; TOPO_NO_PLOT = true; main8_main`; `NO_RESULT_PLOTS = true` skips STEP 6.

## Outputs

- `exported_statistics/`: CSV and MAT statistics read by `main8_main.m` (not tracked by git).
- `fig_exported/`: figures of all scripts (not tracked by git).
