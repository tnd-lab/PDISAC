# PDISAC: Probabilistic Denoising-Enhanced ISAC for Stochastic Cluttered Mobile Environments

This work is currently under review for publication in the IEEE Transactions on Wireless Communications.

Reference implementation of **PDISAC**, a monostatic PMCW integrated sensing and communication (ISAC) framework with a learned range-Doppler (RD) denoiser.

![PDISAC scene](figures/topology.png)

## Abstract

We propose Probabilistic Denoising ISAC (PDISAC) for stochastic, cluttered mobile environments characterized by a geometry-based channel model. The approach integrates a probabilistic denoising stage with a **multi-bit slot-partitioned ISAC waveform**, where multiple bits are embedded by partitioning each subsequence into alternating pilot and data slots for symbol-level spreading. Embedding multiple bits increases throughput but introduces data-dependent sidelobes into the RD heatmap, degrading sensing performance. Instead of modifying the matched-filter (MF) receiver, we develop a lightweight RD probabilistic denoising network (**RDPDNet**) with an **adversarial frequency-mixup (AFM)** mechanism that encourages the network to suppress these sidelobes and thermal noise. The system performance is further characterized through an analytical lower bound, a semi-analytical bit error rate (BER), and the average achievable rate, linking the slot allocation and sequence length to sensing and communication performance. Results in an urban geometry show that RDPDNet suppresses data-induced sidelobes and reduces the root mean square error (RMSE) at medium SNR, and at high SNR when data are embedded. At high SNR, allocating more data slots improves the achievable rate but degrades the BER, whereas embedding data, regardless of the allocation, lowers the detection probability and increases the RMSE relative to a data-free waveform.

## Repository layout

```
PDISAC/
├── simulation_matlab/   MATLAB: waveform and channel simulation, conventional receivers,
│                        dataset generation, statistics and figures (simulation_matlab/README.md)
├── alg_pdisac/          Python: RDPDNet / DnCNN training with and without AFM, tests,
│                        and the inference bridge called by MATLAB (alg_pdisac/README.md)
├── figures/             images of this README
├── LICENSE
└── README.md
```

The two sides exchange files on disk: MATLAB writes the training database into `alg_pdisac/data/`, Python writes the checkpoints into `alg_pdisac/exps/`, and MATLAB calls the trained denoisers through the Python inference bridge during evaluation.

## Requirements

**MATLAB** R2022b or newer, with

- Phased Array System Toolbox (`phased.Platform`, `physconst`),
- Statistics and Machine Learning Toolbox (distribution fits of `check_gamma_gof`),
- Database Toolbox (`sqlite`, writes the training database).

**Python 3.13.14** in a conda environment named `afm` (the environment `simulation_matlab/main1_matlab_config.m` looks for), with the packages of [`alg_pdisac/requirements.txt`](alg_pdisac/requirements.txt):

```bash
conda create -n afm python=3.13.14 -y
conda activate afm
pip install -r alg_pdisac/requirements.txt
```

`requirements.txt` installs the CUDA 13.0 build of PyTorch; on another platform, install `torch` and `torchvision` from <https://pytorch.org> first.

## End-to-end pipeline

**1. Training database (MATLAB).**

```matlab
% in simulation_matlab/
generate_radar_dataset                                % -> alg_pdisac/data/dataset_train.db
% setenv('PDISAC_SMOKE','1'); generate_radar_dataset  % -> dataset_smoke.db (quick check)
```

**2. Denoisers (Python).** From `alg_pdisac/`:

```bash
python -m scripts.train.train_pdnet_afm      # RDPDNet + AFM (proposed)
python -m scripts.train.train_pdnet_no_afm   # RDPDNet without AFM
python -m scripts.train.train_dncnn_afm      # DnCNN + AFM
python -m scripts.train.train_dncnn_no_afm   # DnCNN without AFM
```

Each run writes `exps/<model>_<afm|no_afm>/fold_1/checkpoints/best_checkpoint.pth`, listed by name in `configs/inference.json`.

**3. Statistics (MATLAB).** Check `PDISAC_cfg.inference.python_executable` in `simulation_matlab/main1_matlab_config.m`, then run the analysis drivers; `main5b` and `main5c` call the trained denoisers through the bridge.

```matlab
% in simulation_matlab/
main5a_sensing_sweep          % RMSE vs N_chip, N_prbs
main5b_detection_vs_nbit      % P_fa, P_d, ROC
main5c_bcrb_check             % RMSE vs BCRB (several runs, see simulation_matlab/README.md)
main7_comm_analysis           % BER and achievable rate
channel_com                   % LOS / NLOS channel distribution
check_gamma_gof               % goodness of fit of the NLOS power
```

**4. Figures (MATLAB).** `main5d` and `main5e` also call the trained denoisers.

```matlab
main5d_fig_maps               % RD maps, denoised maps, AFM masks
main5e_fig_dist               % RDPDNet distributions
main8_main                    % pipeline figures, result figures, table of RMSE sweeps
```

`PDISAC_SMOKE` is shared by MATLAB (dataset generation) and Python (training configuration), so one variable switches the whole pipeline to the smoke run.

## Figures and tables of the paper

| Result | Produced by | Rendered by |
|---|---|---|
| Scene geometry | `main2_topology` | `main8_main` STEP 1 (`fig_topology`) |
| RD maps, denoised maps, AFM masks | `main5d_fig_maps` | same script (`fig_exported/fig_maps/`) |
| RDPDNet distributions | `main5e_fig_dist` | same script (`fig_exported/fig_dist/`) |
| `P_fa`, `P_d` vs `N_bit^prbs`, `P_d` at matched `P_fa`, ROC | `main5b_detection_vs_nbit` | `main8_main` STEP 6e |
| RMSE vs BCRB | `main5c_bcrb_check` | `main8_main` STEP 6f, 6g |
| Table of RMSE sweeps | `main5a_sensing_sweep`, `main5c_bcrb_check` (`NBIT_EXT`) | `main8_main` STEP 6h |
| Complexity table | `alg_pdisac/scripts/test/complexity_table.py` | same script |
| Goodness of fit of the NLOS power | `check_gamma_gof` | `main8_main` STEP 6d |
| LOS / NLOS channel distribution | `channel_com` | `main8_main` STEP 6b |
| BER and achievable rate | `main7_comm_analysis` | `main8_main` STEP 6a |

## Details

- MATLAB side: [`simulation_matlab/README.md`](simulation_matlab/README.md).
- Python side: [`alg_pdisac/README.md`](alg_pdisac/README.md).

## License

GNU General Public License v3.0, see [`LICENSE`](LICENSE).
