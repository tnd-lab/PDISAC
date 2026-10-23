% MAIN5B_DETECTION_VS_NBIT  Detection and false-alarm probability of the
% CA-CFAR detector versus N_bit^prbs
% -------------------------------------------------------------------------
% Conventional chain of Sec. III on the RD map with the zero-Doppler column
% removed (main3 + main4), CA-CFAR (func_ca_cfar_adaptive_threshold) with the
% settings of main1 (guard / training cells, nominal P_fa, local-maximum
% selection). Two maps per run, three with LS_CANCEL:
%   'data'    RD_map_shifted                MF with the data-bearing waveform
%   'nodata'  RD_map_noise_no_data_shifted  MF with the data-free (MLS-only)
%                                           waveform, same noise (reference)
%   'data_ls' (LS_CANCEL = true) the data map after LS cancellation of the
%             static clutter with the known transmitted stream (as main5c):
%             h_hat = (X^T X)^{-1} X^T y, y_ls = y - X h_hat, then the MF of main4.
%             It removes the data-induced residual of the static clutter only;
%             the data residual of the moving targets (and the UE) remains.
% USE_DENOISER = true adds, for every model selected by DENOISER_MODELS from
% alg_pdisac/configs/inference.json (RDPDNet / DnCNN, with or without AFM), the chains
% '<model>_data', '<model>_nodata' and, with LS_CANCEL, '<model>_data_ls': the data,
% data-free and LS-cancelled maps denoised by the model, then the same CA-CFAR.
% One Python call per (scene, N_bit) for all SNRs and MC runs (inference_run_pdnet);
% outputs get the suffix '_ml'. The baseline chains are unchanged (Python uses no MATLAB RNG).
%
% SCENES   random, as in generate_radar_dataset: cfg.fixed_targets targets + the UE
%          uniform in the ROI, cfg.scatterers clustered static scatterers, one scene
%          per RAN index and fixed over N_bit^prbs, SNR and Monte Carlo. Data symbols
%          and noise are redrawn per run, with the same seed for every SNR of a
%          (scene, N_bit, MC) triple (common random numbers across SNR).
%
% P_fa     per cell, from the RAW threshold crossings P_rd > lambda_rd (before
%          the local-maximum selection, which is not a per-cell test):
%              P_fa = #crossings in the test region / #cells in the test region
%          Test region: range bins up to the farthest point of the ROI (where the
%          targets and the clutter live) plus one CFAR half-window
%          (N_guard + N_train bins), all Doppler columns except the zeroed
%          zero-Doppler column, minus a +-(N_guard+1) box around every target.
%          Target sidelobes outside that box count as false alarms.
% P_d      per target, from the FINAL detections (after the local-maximum
%          selection of Sec. III): target m is detected if a detection lies within
%          +-1 range bin and +-1 Doppler bin of its true cell, one-to-one
%          (a detection is credited to one target only).
%          A target whose true cell is the removed zero-Doppler column cannot be
%          detected and is not scored.
%          P_d_raw (CSV only): any raw crossing within the same +-1 box, i.e.
%          before the local-maximum selection; P_d_raw - P_d is the loss caused
%          by that selection when two targets are closer than the CFAR window.
% Both come with 95% Wilson intervals.
%
% ROC = true also sweeps the design P_fa of the same CA-CFAR (LOG10_PFA_ROC = log10 of the design
% P_fa, nominal included) for every chain, so the chains can be compared at the same EMPIRICAL P_fa:
% a cell crosses at design P_fa iff its CFAR p-value (1 + P_rd/(N_eff * noise_avg))^(-N_eff) < P_fa,
% and the local-maximum selection does not depend on the threshold, so one CFAR pass serves every
% P_fa (score_map_roc). The sweep is uniform in alpha [dB] (ALPHA_DB_ROC, interior N_eff), and the test is
% done in log10 (log10 p-value < log10 P_fa), so it can go far below realmin: the denoised maps have a
% near-zero background, and their residual peaks only stop crossing at very large alpha.
% The P_fa scale alpha(N_eff, P_fa) assumes exponential cell powers, which holds for the raw maps
% but not for the denoised maps; the sweep makes the comparison independent of that assumption.
%
% Outputs (fig_exported / exported_statistics in simulation_matlab; file names get
% the suffix '_smoke' with SMOKE and then '_ml' with USE_DENOISER)
%   fig_pd_vs_nbit.(pdf|fig), fig_pfa_vs_nbit.(pdf|fig): data (solid), data + LS
%        (dash-dot) and no data (dashed, pooled over N_bit^prbs) per SNR; the
%        denoiser chains are in the CSV / .mat only
%   statistics_detection_vs_nbit.csv (path cfg.output_csv; Chain, SNR, N_bit, P_d with
%        Wilson interval, P_d_raw, P_fa with Wilson interval, counts, nominal P_fa),
%   statistics_detection_vs_nbit.mat (counts and CFAR settings)
%   ROC: statistics_detection_vs_nbit_roc.csv (Chain, SNR, N_bit, design P_fa and its log10, P_d,
%        empirical P_fa, counts) and r_hit / r_fa / LOG10_PFA_ROC / ALPHA_DB_ROC in the .mat; the
%        console prints P_d at the empirical P_fa = PFA_MATCH (pooled over N_bit^prbs, linear
%        interpolation of the ROC in log10 P_fa).
%
% Settings: PDISAC_cfg.statistics.detection in main1_matlab_config.m (detection_statistics_cfg),
% flags SMOKE, USE_DENOISER, DENOISER_MODELS, LS_CANCEL, LS_TAPS, ROC below.
% Runs = random_scenes x numel(symbols_per_block) x numel(snr_db) x monte_carlo
% (main3 + main4 each). SMOKE = true replaces the sweep by a short check
% (1 scene, 1 MC run, N_bit^prbs = [4 128], SNR = [0 20] dB).

clear; clc; close all;
script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(script_dir));
NO_SIGNAL_PLOT = true;  NO_BPSK_PLOT = true;      % headless: main3/main4 plot
NO_RADAR_PLOT  = true;  NO_CFAR_PLOT = true;      % nothing inside the sweep
main1_matlab_config;

SMOKE      = false;       % true: 1 scene, 1 MC, N_bit [4 128], SNR [0 20]
USE_DENOISER    = true;  % true: add the denoiser chains (Python, configs/inference.json)
DENOISER_MODELS = "pdnet";     % "" = every model of inference.json with a checkpoint, or "pdnet,dncnn"
LS_CANCEL  = true;        % true: add the chain 'data_ls' (LS static-clutter cancellation)
LS_TAPS    = [-64 256];   % static-clutter taps [chips], as main5c
SEED       = 5150000;

cfg = detection_statistics_cfg;
G_tx = 10^(cfg.system_override.gain_tx_db/10);
G_rx = 10^(cfg.system_override.gain_rx_db/10);
G_ue = 10^(cfg.system_override.gain_ue_db/10);

N_ran        = cfg.random_scenes;
Mc           = cfg.monte_carlo;
N_bit_list   = cfg.symbols_per_block(:).';
SNR_db       = cfg.snr_db(:).';
N_tars_fixed = cfg.fixed_targets;
N_scats_cfg  = cfg.scatterers;
N_ref_tars   = PDISAC_cfg.scene.communication_reflectors;   % needed by main3 (comm part)
if SMOKE
    N_ran = 1;  Mc = 1;  N_bit_list = [4 128];  SNR_db = [0 20];
end
n_bit = numel(N_bit_list);  n_snr = numel(SNR_db);

% ---- detector: exactly the settings of Sec. III (main1) -----------------
N_g_r = PDISAC_cfg.sensing.cfar.guard_range;
N_g_d = PDISAC_cfg.sensing.cfar.guard_doppler;
N_t_r = PDISAC_cfg.sensing.cfar.training_range;
N_t_d = PDISAC_cfg.sensing.cfar.training_doppler;
P_fa  = PDISAC_cfg.sensing.cfar.false_alarm_probability;
peak_select = PDISAC_cfg.sensing.cfar.peak_select;
GATE  = 1;                               % P_d gate, bins (range and Doppler)
ROC     = true;                          % true: also sweep the design P_fa (ROC, P_d at matched empirical P_fa)
N_EFF_ROC = (2*(N_g_r+N_t_r)+1)*(2*(N_g_d+N_t_d)+1) - (2*N_g_r+1)*(2*N_g_d+1);   % training cells (interior)
ALPHA_DB_ROC = 3:0.25:150;               % CFAR scale alpha of the sweep [dB], uniform, so the ROC points are evenly spread;
                                         % large values: the denoised maps (near-zero background) need them to lower their P_fa
LOG10_PFA_ROC = unique([-N_EFF_ROC * log10(1 + 10.^(ALPHA_DB_ROC/10) / N_EFF_ROC), log10(P_fa)]);   % log10 design P_fa (nominal included)
PFA_ROC = 10.^LOG10_PFA_ROC;             % 0 below realmin; LOG10_PFA_ROC is the reference
PFA_MATCH = [1e-3 3e-3 1e-2];            % empirical P_fa levels at which the chains are compared

% ---- RD axes and test region (N_chip, N_block fixed by main1) ------------
range_res  = c / (2 * B);
range_axis = (0 : N_chip-1) * range_res;
PRF        = 1 / T_pmcw;
fd_axis    = (-N_block/2 : N_block/2-1) * (PRF / N_block);
vel_axis   = -(Lambda / 2) * fd_axis;
col0       = N_block/2 + 1;              % zero-Doppler column after fftshift
R_scene    = sqrt(max(abs(Region_of_interest(1,:)))^2 + max(abs(Region_of_interest(2,:)))^2);
region     = false(N_chip, N_block);
region(range_axis <= R_scene + (N_g_r + N_t_r)*range_res, :) = true;
region(:, col0) = false;

chains = {'data', 'nodata'};
ch_ls = 0;  ml_names = strings(1, 0);  ml_meta = struct();
if LS_CANCEL
    chains{end+1} = 'data_ls';  ch_ls = numel(chains);
end
n_ch = numel(chains);
n_in = 2 + LS_CANCEL;                    % maps per run passed to the denoiser (data, data-free, LS)
ml_suffix = {'_data', '_nodata', '_data_ls'};   % chain = <model><suffix>, same order as RD_keep

n_hit = zeros(n_bit, n_snr, n_ch);  n_hit_raw = n_hit;  n_tar = n_hit;
n_fa  = zeros(n_bit, n_snr, n_ch);  n_cell    = n_fa;
n_roc = numel(LOG10_PFA_ROC);
r_hit = zeros(n_bit, n_snr, n_ch, n_roc);  r_fa = r_hit;    % ROC counts (n_tar, n_cell as above)

n_runs = N_ran * n_bit * n_snr * Mc;
fprintf('*** main5b | %d scenes x %d N_bit x %d SNR x %d MC = %d runs (main3+main4)\n', ...
    N_ran, n_bit, n_snr, Mc, n_runs);
fprintf('*** CFAR guard %d/%d, train %d/%d, nominal P_fa = %.0e, peak_select = %d | test range <= %.0f m\n', ...
    N_g_r, N_g_d, N_t_r, N_t_d, P_fa, peak_select, R_scene);
t0 = tic;  run_count = 0;

% =====================================================================
%  LOOP 1 — scene (drawn once, fixed over N_bit, SNR, MC)
% =====================================================================
for idx_ran = 1:N_ran
    rng(SEED + idx_ran, 'twister');

    % ---- UE ----
    N_ue        = 1;
    UE_position = rand_pos_in_roi(Region_of_interest, N_ue);
    UE_vel      = VEL_RANGE_UE * (2*rand(3,1) - 1);   UE_vel(3) = 0;
    UE_rcs      = 10^( (10*log10(10) + (10*log10(20)-10*log10(10))*rand()) / 10 );
    UE_motion   = phased.Platform('InitialPosition', UE_position, 'Velocity', UE_vel);

    % ---- moving targets, UE appended as the last one ----
    N_tars        = N_tars_fixed;
    Tars_position = rand_pos_in_roi(Region_of_interest, N_tars);
    Tars_vel      = VEL_RANGE_TAR * (2*rand(3, N_tars) - 1);   Tars_vel(3,:) = 0;
    Tars_rcs      = 10.^( (10*log10(10) + (10*log10(20)-10*log10(10))*rand(1, N_tars)) / 10 );
    N_tars        = N_tars + 1;
    Tars_position = [Tars_position, UE_position];
    Tars_vel      = [Tars_vel,      UE_vel];
    Tars_rcs      = [Tars_rcs,      UE_rcs];
    Tars_motion   = phased.Platform('InitialPosition', Tars_position, 'Velocity', Tars_vel);

    % ---- static scatterers ----
    N_scats        = N_scats_cfg;
    Scats_position = func_generate_static_scatterers(N_scats, Region_of_interest);
    Scats_vel      = zeros(size(Scats_position));
    Scats_rcs      = 10.^( (10*log10(5) + (10*log10(10)-10*log10(5))*rand(1, N_scats)) / 10 );
    Scats_motion   = phased.Platform('InitialPosition', Scats_position, 'Velocity', Scats_vel);

    % ---- true cells and the per-scene test mask ----
    rel         = Tars_position - Tx_position;
    truth_range = sqrt(sum(rel.^2, 1));
    truth_vel   = sum(Tars_vel .* rel ./ (truth_range + eps), 1);
    k_true = round(truth_range / range_res) + 1;
    d_true = zeros(1, N_tars);
    for m = 1:N_tars, [~, d_true(m)] = min(abs(vel_axis - truth_vel(m))); end
    vis = d_true ~= col0;     % a target in the removed zero-Doppler column is invisible: not scored
    test_mask = region;
    for m = 1:N_tars
        rr = max(1, k_true(m)-(N_g_r+1)) : min(N_chip,  k_true(m)+(N_g_r+1));
        dd = max(1, d_true(m)-(N_g_d+1)) : min(N_block, d_true(m)+(N_g_d+1));
        test_mask(rr, dd) = false;
    end

    % ==================================================================
    %  LOOP 2/3/4 — N_bit^prbs, SNR, Monte Carlo
    % ==================================================================
    for ib = 1:n_bit
        N_sym_per_block = N_bit_list(ib);
        L_slot_per_sym  = N_chip / (2 * N_sym_per_block);
        N_slot          = N_chip / L_slot_per_sym;
        if USE_DENOISER, RD_keep = complex(zeros(n_in*n_snr*Mc, N_chip, N_block, 'single')); end
        for is = 1:n_snr
            P_tx = Noise_power_sen * 10^(SNR_db(is)/10);
            for mc = 1:Mc
                rng(SEED + 1e6*idx_ran + 1e3*ib + mc, 'twister');   % same data/noise for every SNR
                main3_signal_channel_model;
                main4_sensing_process;

                maps = {RD_map_shifted, RD_map_noise_no_data_shifted};
                if LS_CANCEL, maps{ch_ls} = rd_map_ls(Data_Rx, Data_Tx, LS_TAPS); end
                for ch = 1:numel(maps)
                    [h, hr, fa, nc] = score_map(maps{ch}, test_mask, k_true(vis), d_true(vis), ...
                        N_g_r, N_g_d, N_t_r, N_t_d, P_fa, peak_select, GATE);
                    n_hit(ib,is,ch) = n_hit(ib,is,ch) + nnz(h);
                    n_hit_raw(ib,is,ch) = n_hit_raw(ib,is,ch) + nnz(hr);
                    n_tar(ib,is,ch) = n_tar(ib,is,ch) + nnz(vis);
                    n_fa(ib,is,ch)  = n_fa(ib,is,ch)  + fa;
                    n_cell(ib,is,ch)= n_cell(ib,is,ch)+ nc;
                    if ROC
                        [rh, rf] = score_map_roc(maps{ch}, test_mask, k_true(vis), d_true(vis), ...
                            N_g_r, N_g_d, N_t_r, N_t_d, LOG10_PFA_ROC, peak_select, GATE);
                        r_hit(ib,is,ch,:) = r_hit(ib,is,ch,:) + reshape(rh, 1, 1, 1, []);
                        r_fa(ib,is,ch,:)  = r_fa(ib,is,ch,:)  + reshape(rf, 1, 1, 1, []);
                    end
                end
                if USE_DENOISER                       % data / data-free / LS map, kept for one call
                    j = (is-1)*Mc + mc;
                    RD_keep(j, :, :)            = reshape(RD_map_shifted, [1 N_chip N_block]);
                    RD_keep(n_snr*Mc + j, :, :) = reshape(RD_map_noise_no_data_shifted, [1 N_chip N_block]);
                    if LS_CANCEL
                        RD_keep(2*n_snr*Mc + j, :, :) = reshape(maps{ch_ls}, [1 N_chip N_block]);
                    end
                end
                run_count = run_count + 1;
            end
        end
        % ---- denoisers: all SNRs and MC runs of this (scene, N_bit), one Python call ----
        if USE_DENOISER
            [RD_den, meta] = inference_run_pdnet(RD_keep, Models = DENOISER_MODELS);   % K x n_in n_snr Mc x H x W
            if isempty(ml_names)                      % first call: n_in chains per model
                ml_names = meta.name;  ml_meta = meta;  n_new = n_in * numel(ml_names);
                for k = 1:numel(ml_names)
                    chains(end+1:end+n_in) = cellstr(ml_names(k) + string(ml_suffix(1:n_in)));
                end
                grow = @(A) cat(3, A, zeros(n_bit, n_snr, n_new));
                n_hit = grow(n_hit);  n_hit_raw = grow(n_hit_raw);  n_tar = grow(n_tar);
                n_fa  = grow(n_fa);   n_cell    = grow(n_cell);   n_ch  = numel(chains);
                r_hit = cat(3, r_hit, zeros(n_bit, n_snr, n_new, n_roc));
                r_fa  = cat(3, r_fa,  zeros(n_bit, n_snr, n_new, n_roc));
                fprintf('  denoisers: %s\n', strjoin(compose("%s (%s, epoch %d)", ml_names(:), ...
                    meta.version(:), meta.epoch(:)), ', '));
            elseif ~isequal(meta.name, ml_names) || ~isequal(meta.checkpoint, ml_meta.checkpoint)
                error('main5b:DenoiserChanged', 'alg_pdisac/configs/inference.json changed during the run.');
            end
            n_base = n_ch - n_in * numel(ml_names);
            for k = 1:numel(ml_names)
                for h = 1:n_in                        % 1: data map, 2: data-free map, 3: LS map
                    ch = n_base + n_in*(k-1) + h;
                    for is = 1:n_snr
                        for mc = 1:Mc
                            j  = (h-1)*n_snr*Mc + (is-1)*Mc + mc;
                            RD = double(reshape(RD_den(k, j, :, :), N_chip, N_block));
                            [hh, hr, fa, nc] = score_map(RD, test_mask, k_true(vis), d_true(vis), ...
                                N_g_r, N_g_d, N_t_r, N_t_d, P_fa, peak_select, GATE);
                            n_hit(ib,is,ch)     = n_hit(ib,is,ch)     + nnz(hh);
                            n_hit_raw(ib,is,ch) = n_hit_raw(ib,is,ch) + nnz(hr);
                            n_tar(ib,is,ch)     = n_tar(ib,is,ch)     + nnz(vis);
                            n_fa(ib,is,ch)      = n_fa(ib,is,ch)      + fa;
                            n_cell(ib,is,ch)    = n_cell(ib,is,ch)    + nc;
                            if ROC
                                [rh, rf] = score_map_roc(RD, test_mask, k_true(vis), d_true(vis), ...
                                    N_g_r, N_g_d, N_t_r, N_t_d, LOG10_PFA_ROC, peak_select, GATE);
                                r_hit(ib,is,ch,:) = r_hit(ib,is,ch,:) + reshape(rh, 1, 1, 1, []);
                                r_fa(ib,is,ch,:)  = r_fa(ib,is,ch,:)  + reshape(rf, 1, 1, 1, []);
                            end
                        end
                    end
                end
            end
        end
    end
    el = toc(t0);
    fprintf('scene %2d/%d done | %d/%d runs | %.1f min elapsed, ~%.1f min left\n', ...
        idx_ran, N_ran, run_count, n_runs, el/60, el/60 * (n_runs - run_count) / run_count);
end

% =====================================================================
%  Rates, 95% Wilson intervals, export
% =====================================================================
Pd  = n_hit ./ n_tar;   Pd_raw = n_hit_raw ./ n_tar;   Pfa = n_fa ./ n_cell;
[Pd_lo,  Pd_hi]  = wilson_ci(n_hit, n_tar);
[Pfa_lo, Pfa_hi] = wilson_ci(n_fa,  n_cell);

fig_dir  = fullfile(script_dir, 'fig_exported');         if ~isfolder(fig_dir),  mkdir(fig_dir);  end
rows = cell(0, 15);
for ch = 1:n_ch
    for is = 1:n_snr
        for ib = 1:n_bit
            rows(end+1, :) = {chains{ch}, SNR_db(is), N_bit_list(ib), ...
                Pd(ib,is,ch), Pd_lo(ib,is,ch), Pd_hi(ib,is,ch), Pd_raw(ib,is,ch), ...
                Pfa(ib,is,ch), Pfa_lo(ib,is,ch), Pfa_hi(ib,is,ch), ...
                n_hit(ib,is,ch), n_tar(ib,is,ch), n_fa(ib,is,ch), n_cell(ib,is,ch), P_fa}; %#ok<SAGROW>
        end
    end
end
T = cell2table(rows, 'VariableNames', {'Chain','SNR_dB','N_bit','Pd','Pd_lo','Pd_hi','Pd_raw', ...
    'Pfa','Pfa_lo','Pfa_hi','n_hit','n_tar','n_fa','n_cell','Pfa_nominal'});
csv_path = fullfile(PDISAC_repo_root, cfg.output_csv);
if SMOKE, csv_path = strrep(csv_path, '.csv', '_smoke.csv'); end
if USE_DENOISER, csv_path = strrep(csv_path, '.csv', '_ml.csv'); end
if ~isfolder(fileparts(csv_path)), mkdir(fileparts(csv_path)); end
writetable(T, csv_path);
save(strrep(csv_path, '.csv', '.mat'), 'chains', 'SNR_db', 'N_bit_list', 'n_hit', 'n_hit_raw', ...
    'n_tar', 'n_fa', 'n_cell', 'P_fa', 'N_g_r', 'N_g_d', 'N_t_r', 'N_t_d', 'peak_select', 'R_scene', 'ml_meta');
fprintf('\nWrote %s\n', csv_path);
fprintf('\nP_d / P_fa per SNR, pooled over N_bit^prbs:\n%8s', 'SNR');
fprintf(' | %-20s', chains{:});  fprintf('\n');
for is = 1:n_snr
    fprintf('%6d dB', SNR_db(is));
    for ch = 1:n_ch
        fprintf(' | %6.3f / %8.2e   ', sum(n_hit(:,is,ch))/sum(n_tar(:,is,ch)), ...
            sum(n_fa(:,is,ch))/sum(n_cell(:,is,ch)));
    end
    fprintf('\n');
end

% =====================================================================
%  ROC: export, consistency with the nominal CFAR, P_d at matched empirical P_fa
% =====================================================================
if ROC
    % one row per (chain, SNR, N_bit, design P_fa), design P_fa fastest (vectorized: ~10^5 rows)
    [ip_g, ib_g, is_g, ch_g] = ndgrid(1:n_roc, 1:n_bit, 1:n_snr, 1:n_ch);
    li = sub2ind([n_bit n_snr n_ch], ib_g(:), is_g(:), ch_g(:));
    l4 = sub2ind([n_bit n_snr n_ch n_roc], ib_g(:), is_g(:), ch_g(:), ip_g(:));
    col_ = @(x) reshape(x, [], 1);
    T_roc = table(col_(chains(ch_g(:))), col_(SNR_db(is_g(:))), col_(N_bit_list(ib_g(:))), ...
        col_(PFA_ROC(ip_g(:))), col_(LOG10_PFA_ROC(ip_g(:))), r_hit(l4) ./ n_tar(li), r_fa(l4) ./ n_cell(li), ...
        r_hit(l4), n_tar(li), r_fa(l4), n_cell(li), 'VariableNames', {'Chain','SNR_dB','N_bit', ...
        'Pfa_design','log10_Pfa_design','Pd','Pfa','n_hit','n_tar','n_fa','n_cell'});
    roc_path = strrep(csv_path, '.csv', '_roc.csv');
    writetable(T_roc, roc_path);
    save(strrep(csv_path, '.csv', '.mat'), 'r_hit', 'r_fa', 'LOG10_PFA_ROC', 'ALPHA_DB_ROC', '-append');
    fprintf('\nWrote %s\n', roc_path);

    % the sweep point at the nominal P_fa must reproduce the nominal CFAR counts
    ip0 = find(LOG10_PFA_ROC == log10(P_fa), 1);
    d_hit = max(abs(r_hit(:,:,:,ip0) - n_hit), [], 'all');
    d_fa  = max(abs(r_fa(:,:,:,ip0)  - n_fa),  [], 'all');
    if d_hit > 0 || d_fa > 0
        warning('main5b:roc', 'ROC at the nominal P_fa differs from the nominal CFAR (max |diff| hits %d, false alarms %d).', d_hit, d_fa);
    end

    for pm = PFA_MATCH
        fprintf('\nP_d at the empirical P_fa = %.0e (ROC, pooled over N_bit^prbs; NaN = outside the swept range):\n%8s', pm, 'SNR');
        fprintf(' | %-15s', chains{:});  fprintf('\n');
        for is = 1:n_snr
            fprintf('%6d dB', SNR_db(is));
            for ch = 1:n_ch
                pd  = squeeze(sum(r_hit(:,is,ch,:), 1)) / sum(n_tar(:,is,ch));
                pfa = squeeze(sum(r_fa(:,is,ch,:),  1)) / sum(n_cell(:,is,ch));
                fprintf(' | %15.3f', pd_at_pfa(pd, pfa, pm));
            end
            fprintf('\n');
        end
    end
end

% =====================================================================
%  Figures: x = N_bit^prbs, one colour per SNR; data = solid + markers,
%  data + LS = dash-dot (LS_CANCEL), no data = dashed (pooled over N_bit^prbs:
%  it does not depend on it); denoiser chains are not plotted

% =====================================================================
set(groot, 'defaultTextInterpreter','latex', 'defaultAxesTickLabelInterpreter','latex', ...
    'defaultLegendInterpreter','latex', 'defaultAxesFontName','Times New Roman', ...
    'defaultTextFontName','Times New Roman', 'defaultAxesFontSize',14);
C = parula(n_snr + 1);  C = C(1:n_snr, :);
Pd_nd  = sum(n_hit(:,:,2),1) ./ sum(n_tar(:,:,2),1);     % pooled no-data reference
Pfa_nd = sum(n_fa(:,:,2),1)  ./ sum(n_cell(:,:,2),1);
tag = '';  if SMOKE, tag = '_smoke'; end
if USE_DENOISER, tag = [tag '_ml']; end

% ---- P_d ----
figure('Color','w','Units','inches','Position',[1 1 7 5]);
ax = gca; hold(ax,'on'); box(ax,'on'); grid(ax,'on');
for is = 1:n_snr
    plot(ax, N_bit_list, Pd(:,is,1), '-o', 'Color', C(is,:), 'MarkerFaceColor', C(is,:), ...
        'LineWidth', 1.6, 'DisplayName', sprintf('$%d$ dB', SNR_db(is)));
    yline(ax, Pd_nd(is), '--', 'Color', C(is,:), 'LineWidth', 1.2, 'HandleVisibility','off');
    if ch_ls
        plot(ax, N_bit_list, Pd(:,is,ch_ls), '-.s', 'Color', C(is,:), 'MarkerFaceColor','w', ...
            'LineWidth', 1.4, 'HandleVisibility','off');
    end
end
plot(ax, nan, nan, 'k-o', 'MarkerFaceColor','k', 'DisplayName','With data');
if ch_ls, plot(ax, nan, nan, 'k-.s', 'MarkerFaceColor','w', 'DisplayName','With data + LS'); end
plot(ax, nan, nan, 'k--', 'DisplayName','Without data');
set(ax, 'XScale','log');  xticks(ax, N_bit_list);  xticklabels(ax, string(N_bit_list));
xlim(ax, [min(N_bit_list) max(N_bit_list)]);  ylim(ax, [0 1]);
xlabel(ax, '$N_{\mathrm{bit}}^{\mathrm{prbs}}$');  ylabel(ax, 'Detection probability $P_{\mathrm d}$');
legend(ax, 'Location','eastoutside', 'Box','on', 'FontSize',11);
savefig(gcf, fullfile(fig_dir, ['fig_pd_vs_nbit' tag '.fig']));
exportgraphics(gcf, fullfile(fig_dir, ['fig_pd_vs_nbit' tag '.pdf']), 'ContentType','vector');

% ---- P_fa ----
figure('Color','w','Units','inches','Position',[1 1 7 5]);
ax = gca; hold(ax,'on'); box(ax,'on'); grid(ax,'on');
for is = 1:n_snr
    y = Pfa(:,is,1);  y(y == 0) = NaN;                    % log axis
    plot(ax, N_bit_list, y, '-o', 'Color', C(is,:), 'MarkerFaceColor', C(is,:), ...
        'LineWidth', 1.6, 'DisplayName', sprintf('$%d$ dB', SNR_db(is)));
    if Pfa_nd(is) > 0
        yline(ax, Pfa_nd(is), '--', 'Color', C(is,:), 'LineWidth', 1.2, 'HandleVisibility','off');
    end
    if ch_ls
        y = Pfa(:,is,ch_ls);  y(y == 0) = NaN;
        plot(ax, N_bit_list, y, '-.s', 'Color', C(is,:), 'MarkerFaceColor','w', ...
            'LineWidth', 1.4, 'HandleVisibility','off');
    end
end
yline(ax, P_fa, 'k:', 'LineWidth', 1.6, 'HandleVisibility','off');
plot(ax, nan, nan, 'k-o', 'MarkerFaceColor','k', 'DisplayName','With data');
if ch_ls, plot(ax, nan, nan, 'k-.s', 'MarkerFaceColor','w', 'DisplayName','With data + LS'); end
plot(ax, nan, nan, 'k--', 'DisplayName','Without data');
plot(ax, nan, nan, 'k:', 'LineWidth', 1.6, 'DisplayName','Nominal $P_{\mathrm{fa}}$');
set(ax, 'XScale','log', 'YScale','log');
xticks(ax, N_bit_list);  xticklabels(ax, string(N_bit_list));
xlim(ax, [min(N_bit_list) max(N_bit_list)]);
xlabel(ax, '$N_{\mathrm{bit}}^{\mathrm{prbs}}$');  ylabel(ax, 'False-alarm probability $P_{\mathrm{fa}}$');
legend(ax, 'Location','eastoutside', 'Box','on', 'FontSize',11);
savefig(gcf, fullfile(fig_dir, ['fig_pfa_vs_nbit' tag '.fig']));
exportgraphics(gcf, fullfile(fig_dir, ['fig_pfa_vs_nbit' tag '.pdf']), 'ContentType','vector');
fprintf('Figures written to %s\n', fig_dir);


% =====================================================================
%  LOCAL FUNCTIONS
% =====================================================================
function [hit, hit_raw, n_fa, n_cell] = score_map(RD, test_mask, k_true, d_true, ...
        N_g_r, N_g_d, N_t_r, N_t_d, P_fa, peak_select, gate)
% CA-CFAR of Sec. III on one RD map. Raw crossings -> P_fa (per cell);
% final detections (after local-maximum selection) -> P_d (per target).
    [thr, ~, det_pos, P] = func_ca_cfar_adaptive_threshold(RD, N_g_r, N_g_d, N_t_r, N_t_d, ...
                                                           P_fa, peak_select);
    raw    = P > thr;
    n_fa   = nnz(raw & test_mask);
    n_cell = nnz(test_mask);

    [N_r, N_d] = size(RD);
    N = numel(k_true);
    hit = false(N, 1);  hit_raw = false(N, 1);
    for m = 1:N
        rr = max(1, k_true(m)-gate) : min(N_r, k_true(m)+gate);
        dd = max(1, d_true(m)-gate) : min(N_d, d_true(m)+gate);
        hit_raw(m) = any(raw(rr, dd), 'all');
    end
    hit = match_one_to_one(det_pos, k_true, d_true, gate);
end

function hit = match_one_to_one(det_pos, k_true, d_true, gate)
% target m is detected if a detection lies within +-gate bins of its true cell, one-to-one
% (closest (target, detection) pairs first; a detection is credited to one target only)
    N = numel(k_true);
    hit = false(N, 1);
    if isempty(det_pos), return; end
    pairs = zeros(0, 3);
    for m = 1:N
        dk = det_pos(:,1) - k_true(m);  dd = det_pos(:,2) - d_true(m);
        cand = find(abs(dk) <= gate & abs(dd) <= gate);
        pairs = [pairs; [repmat(m, numel(cand), 1), cand(:), dk(cand).^2 + dd(cand).^2]]; %#ok<AGROW>
    end
    if isempty(pairs), return; end
    [~, order] = sort(pairs(:,3), 'ascend');
    used_det = false(size(det_pos, 1), 1);
    for q = order.'
        m = pairs(q,1);  a = pairs(q,2);
        if hit(m) || used_det(a), continue; end
        hit(m) = true;  used_det(a) = true;
    end
end

function [n_hit, n_fa] = score_map_roc(RD, test_mask, k_true, d_true, N_g_r, N_g_d, N_t_r, N_t_d, lpfa_list, peak_select, gate)
% The CA-CFAR of score_map (func_ca_cfar_adaptive_threshold, same peak_select) for every design
% P_fa = 10^lpfa_list in one pass. With N = N_eff and alpha(N, P_fa) = N (P_fa^(-1/N) - 1),
%   P_rd > alpha * noise_avg  <=>  log10 pval = -N log10(1 + P_rd / (N noise_avg)) < log10 P_fa,
% computed in log10 so that P_fa far below realmin can be swept,
% and the local-maximum test (cell = max of its (2R_out+1) x (2D_out+1) window) does not depend
% on the threshold. n_hit(ip): targets detected (one-to-one, as score_map); n_fa(ip): raw
% crossings in test_mask.
    P = abs(RD).^2;
    [N_r, N_d] = size(P);
    R_out = N_g_r + N_t_r;  D_out = N_g_d + N_t_d;
    mask = ones(2*R_out + 1, 2*D_out + 1);
    mask(R_out+1-N_g_r : R_out+1+N_g_r, D_out+1-N_g_d : D_out+1+N_g_d) = 0;
    cnt  = conv2(ones(N_r, N_d), mask, 'same');
    navg = conv2(P, mask, 'same') ./ max(cnt, 1);
    lpval = -cnt .* log10(1 + P ./ (cnt .* navg));
    lpval(isnan(lpval)) = 0;                           % P = 0 and navg = 0: never crosses
    if peak_select                                     % clipped window, as the CFAR
        lmax = P >= movmax(movmax(P, [R_out R_out], 1), [D_out D_out], 2);
    else
        lmax = true(N_r, N_d);
    end
    % only detections within +-gate of a target can be matched, so the others are dropped first
    % (the matching depends only on the relative order of the kept detections, which is unchanged)
    near = false(N_r, N_d);
    for m = 1:numel(k_true)
        near(max(1, k_true(m)-gate) : min(N_r, k_true(m)+gate), max(1, d_true(m)-gate) : min(N_d, d_true(m)+gate)) = true;
    end
    v = lpval(test_mask);                              % false alarms: raw crossings in the test region
    c = find(lmax & near);
    [~, o] = sort(P(c), 'descend');  c = c(o);         % same order as the CFAR output
    [rc, dc] = ind2sub([N_r N_d], c);  lc = lpval(c);
    n_roc = numel(lpfa_list);
    n_hit = zeros(n_roc, 1);  n_fa = zeros(n_roc, 1);
    for ip = 1:n_roc
        n_fa(ip) = nnz(v < lpfa_list(ip));
        s = lc < lpfa_list(ip);
        if any(s), n_hit(ip) = nnz(match_one_to_one([rc(s) dc(s)], k_true, d_true, gate)); end
    end
end

function pd = pd_at_pfa(pd_roc, pfa_roc, pfa0)
% P_d at the empirical P_fa = pfa0 by linear interpolation in log10(P_fa) along the ROC
% (points with P_fa = 0 dropped; NaN if pfa0 is outside the swept range)
    k = pfa_roc > 0 & isfinite(pfa_roc);
    x = log10(pfa_roc(k));  y = pd_roc(k);
    [x, iu] = unique(x);  y = y(iu);
    if numel(x) < 2 || log10(pfa0) < x(1) || log10(pfa0) > x(end), pd = NaN; return; end
    pd = interp1(x, y, log10(pfa0));
end

function RD = rd_map_ls(Data_Rx, Data_Tx, taps)
% LS cancellation of the static clutter, then the RD map of main4 (MF, Doppler FFT,
% zero-Doppler removal, fftshift).
% Model over the whole frame (stream index m = 0..N_fr-1, blocks concatenated):
%   y[m] = sum_k h_k x[m-k] + (moving targets + noise),   k = taps(1)..taps(2),
% with x the known transmitted chip stream (x = 0 before the frame) and h the unknown
% static clutter channel at chip spacing (band-limited, so chip-spaced taps suffice;
% negative taps hold the sinc tails). Normal equations R h = b with
%   R_{k,k'} = sum_m x[m-k] x[m-k']   (Toeplitz up to K/N_fr edge terms),
%   b_k      = sum_m y[m] x[m-k],
% both by FFT. Moving targets project onto h only through their mean over the blocks.
    [N_block, N_chip, ~] = size(Data_Rx);
    x = reshape(Data_Tx(:, :, 1).', [], 1);
    y = reshape(Data_Rx(:, :, 1).', [], 1);
    N_fr = numel(x);
    lags = (taps(1):taps(2)).';
    Nfft = 2^nextpow2(2*N_fr + numel(lags));
    X = fft(x, Nfft);  Yf = fft(y, Nfft);
    cc = ifft(Yf .* conj(X));                       % cc(1+k) = sum_m y[m] x[m-k]
    ac = real(ifft(abs(X).^2));                     % ac(1+d) = sum_m x[m] x[m+d]
    b  = cc(mod(lags, Nfft) + 1);
    R  = toeplitz(ac(1:numel(lags)));
    h  = R \ b;
    hp = zeros(Nfft, 1);  hp(mod(lags, Nfft) + 1) = h;
    yc = ifft(X .* fft(hp));                        % x * h_hat (linear: Nfft >= N_fr + span)
    y_ls = y - yc(1:N_fr);
    Rx_mat = reshape(y_ls, N_chip, N_block);        % N_chip x N_block, as main4
    Tx_mat = reshape(x,    N_chip, N_block);
    mf = fft(Rx_mat) .* conj(fft(Tx_mat));
    mf_fft = fft(mf, [], 2);
    mf_fft(:, 1) = 0;                               % same zero-Doppler removal as main4
    RD = fftshift(ifft(mf_fft, [], 1), 2);
end

function [lo, hi] = wilson_ci(k, n)
% 95% Wilson score interval for a binomial proportion (elementwise).
    z  = 1.959964;
    p  = k ./ n;
    dn = 1 + z^2 ./ n;
    ce = (p + z^2 ./ (2*n)) ./ dn;
    hw = z * sqrt(p .* (1 - p) ./ n + z^2 ./ (4*n.^2)) ./ dn;
    lo = max(ce - hw, 0);  hi = min(ce + hw, 1);
end

function pos = rand_pos_in_roi(roi, N)
    pos = zeros(3, N);
    for dim = 1:3
        lo = roi(dim,1);  hi = roi(dim,2);
        if lo == hi, pos(dim,:) = lo; else, pos(dim,:) = lo + (hi - lo) .* rand(1, N); end
    end
end
