% MAIN5A_SENSING_SWEEP  —  Waveform-parameter RMSE sweeps
% -------------------------------------------------------------------------
% Estimation RMSE of the conventional sensing chain vs. SNR_Tx for several
% waveform lengths. Uses the sensing scene overrides of main1
% (sensing_statistics_cfg), the sinc read-out and a percentile-bootstrap CI on
% the RMSE. Detection statistics (P_d, P_fa) are in main5b_detection_vs_nbit;
% this script evaluates estimation only. Two sweeps, one figure each:
%
%   Figure A   fig_rmse_range_vs_snr_nchip_sweep
%       Numerical range RMSE vs SNR for N_chip = [64 128 256 512] with
%       N_prbs (= N_block in the code) and T_prbs FIXED at the config values.
%       T_chip = T_prbs/N_chip and B = 1/T_chip, so only the range
%       resolution c/(2B) changes; the unambiguous range c*T_prbs/2 and the
%       velocity resolution stay fixed. P_tx is set from the noise power at
%       the config bandwidth, so the SNR axis is the SNR at that bandwidth and
%       the matched-filter output SNR P_tx*T_prbs/(k*T_ref*N_F_sen) is the
%       same for every N_chip.
%
%   Figure B   fig_rmse_vel_vs_snr_nprbs_sweep
%       Numerical velocity RMSE vs SNR for N_prbs (= N_block) =
%       [64 128 256 512] with N_chip FIXED at the config value.
%
% Each sweep line is drawn as a bootstrap 95% confidence band (lo, hi), shown as
% a shaded fill, plus the empirical mid line, one (band + line) per N_chip / N_prbs value.
% Two chains per trial (same scene, noise and Monte-Carlo run, from main4):
%   data   : RD_map_shifted               (data-carrying PRBS, solid)
%   nodata : RD_map_noise_no_data_shifted (data-free MLS, dashed, same colour)
%
% "Numerical RMSE" = estimation stage of the conventional pipeline:
% matched filter -> RD map (main4) -> windowed peak search -> sinc
% interpolation. Target m owns a fixed window of +-WIN bins in range and
% Doppler around its true cell (the truth only places the window); its
% estimate is the strongest cell of that window, with no CFAR threshold.
% One-to-one: targets are served in decreasing order of their window peak
% and a cell taken by one target is excluded for the others. A CFAR
% detection of target m is the local maximum of a +-(N_g+N_t) window, so it
% is also the strongest cell of the +-WIN window: detected targets keep
% their detection, and a missed target takes the strongest remaining peak
% near it. Every target has an estimate in every trial (no NaN), so the
% RMSE is unconditional; at low SNR it saturates near (2*WIN+1)*Delta/sqrt(12).
%
% Outputs (fig_exported / exported_statistics):
%   fig_rmse_range_vs_snr_nchip_sweep.(fig|png)
%   fig_rmse_vel_vs_snr_nprbs_sweep.(fig|png)
%   statistics_sensing_sweep_nchip.csv   (RMSE CI per N_chip, SNR; data + *_nodata columns)
%   statistics_sensing_sweep_nprbs.csv   (RMSE CI per N_prbs, SNR; data + *_nodata columns)
%
% The two CSVs are read by main8_main (STEP 6c figures, STEP 6h table).
%
% Run standalone: main5a_sensing_sweep. Calls main2_topology (which runs
% main1_matlab_config), main3_signal_channel_model, main4_sensing_process and
% func_sinc_interpolation. Long Monte-Carlo sweep (numel(sweep) x numel(SNR) x Mc
% full scene simulations per figure, Mc set below).

clear; clc; close all;

fig_dir = fullfile(fileparts(mfilename('fullpath')), 'fig_exported');
if ~exist(fig_dir,'dir'), mkdir(fig_dir); end
stat_dir = fullfile(fileparts(mfilename('fullpath')), 'exported_statistics');
if ~exist(stat_dir,'dir'), mkdir(stat_dir); end

NO_SIGNAL_PLOT = true;
NO_RADAR_PLOT  = true;
NO_CFAR_PLOT   = true;

rng("default");

% =====================================================================
%  Scene + system setup
% =====================================================================
TOPO_NO_PLOT = true; main2_topology;
UE_position = sensing_statistics_cfg.scene_override.ue_position_m(:);
UE_vel = sensing_statistics_cfg.scene_override.ue_velocity_mps(:);
UE_motion = phased.Platform('InitialPosition',UE_position,'Velocity',UE_vel);
Tars_vel(:,1:end-1) = sensing_statistics_cfg.scene_override.target_velocities_mps;
Tars_position(:,end) = UE_position;
Tars_vel(:,end) = UE_vel;
Tars_motion = phased.Platform('InitialPosition',Tars_position,'Velocity',Tars_vel);
G_tx_db = sensing_statistics_cfg.system_override.gain_tx_db;
G_rx_db = sensing_statistics_cfg.system_override.gain_rx_db;
G_ue_db = sensing_statistics_cfg.system_override.gain_ue_db;
N_F_com_db = sensing_statistics_cfg.system_override.noise_figure_com_db;
G_tx = 10^(G_tx_db/10); G_rx = 10^(G_rx_db/10); G_ue = 10^(G_ue_db/10);
N_F_com = 10^(N_F_com_db/10);
Noise_power_com = physconst('Boltzmann')*T_ref*N_F_com*B;
SNR_Tx_db = sensing_statistics_cfg.snr_db(:).';
SNR_Tx = 10.^(SNR_Tx_db / 10);
Mc = 50;   % Monte-Carlo runs per (sweep value, SNR); overrides sensing_statistics_cfg.monte_carlo

% Ground truth (fixed initial geometry)
truth_range = sqrt(sum((Tars_position - Tx_position).^2, 1)).';
unit_vec    = (Tars_position - Tx_position) ./ ...
              (sqrt(sum((Tars_position - Tx_position).^2, 1)) + eps);
truth_vel   = sum(Tars_vel .* unit_vec, 1).';

% =====================================================================
%  Sweep settings
% =====================================================================
N_chip_fix   = N_chip;     % config value (fixed during Sweep B)
N_block_fix  = N_block;    % config value = N_prbs (fixed during Sweep A)
T_chip_fix    = T_chip;                  % config chip duration (= 1/B)
B_fix         = B;                       % config bandwidth
Noise_sen_fix = Noise_power_sen;         % sensing noise power at the config bandwidth
T_prbs_fix    = N_chip_fix * T_chip_fix; % PRBS duration, fixed during Sweep A

N_chip_sweep  = [64 128 256 512];   % Figure A (range RMSE)
N_prbs_sweep  = [64 128 256 512];   % Figure B (velocity RMSE), N_prbs = N_block

WIN      = 1;              % half-width of each target's search window, bins (range and Doppler)
n_boot   = 2000;
alpha_ci = 0.05;           % 95% CI

n_snr = numel(SNR_Tx);

% =====================================================================
%  SWEEP A — range RMSE vs SNR for each N_chip (N_prbs fixed)
% =====================================================================
rmse_R_lo_A  = nan(n_snr, numel(N_chip_sweep));
rmse_R_hi_A  = nan(n_snr, numel(N_chip_sweep));
rmse_R_mid_A = nan(n_snr, numel(N_chip_sweep));
rmse_R_lo_A_nd  = nan(n_snr, numel(N_chip_sweep));   % no-data chain
rmse_R_hi_A_nd  = nan(n_snr, numel(N_chip_sweep));
rmse_R_mid_A_nd = nan(n_snr, numel(N_chip_sweep));

fprintf('===== Sweep A: N_chip in [%s], N_prbs = %d fixed =====\n', ...
    num2str(N_chip_sweep), N_block_fix);

for idx_cfg = 1:numel(N_chip_sweep)
    % ---- fixed T_prbs: T_chip = T_prbs/N_chip, B = 1/T_chip ----
    N_chip  = N_chip_sweep(idx_cfg);
    N_block = N_block_fix;
    if mod(N_chip, 2 * N_sym_per_block) ~= 0
        error("PDISAC:InvalidSweep", ...
            "N_chip = %d must be divisible by 2*N_sym_per_block = %d.", ...
            N_chip, 2 * N_sym_per_block);
    end
    T_chip          = T_prbs_fix / N_chip;
    B               = 1 / T_chip;
    Noise_power_sen = physconst('Boltzmann') * T_ref * N_F_sen * B;  % receiver noise at the new bandwidth
    T_pmcw          = T_prbs_fix;
    L_slot_per_sym  = N_chip / (2 * N_sym_per_block);
    N_slot          = N_chip / L_slot_per_sym;

    err_R_store    = cell(n_snr, 1);   % pooled range residuals per SNR, data
    err_R_store_nd = cell(n_snr, 1);   % same, no data

    for idx_snr = 1:n_snr
        P_tx = Noise_sen_fix * SNR_Tx(idx_snr);   % SNR defined at the config bandwidth
        all_err_R_cfar = nan(Mc, N_tars);
        all_err_R_nd   = nan(Mc, N_tars);

        for idx_mc = 1:Mc
            main3_signal_channel_model;
            main4_sensing_process;

            % RD-map axes for the CURRENT (N_chip, N_block)
            range_res  = c / (2 * B);
            range_axis = (0 : N_chip-1) * range_res;
            PRF        = 1 / T_pmcw;
            fd_axis    = (-N_block/2 : N_block/2-1) * (PRF / N_block);
            vel_axis   = -(Lambda / 2) * fd_axis;

            % windowed one-to-one peak search on the CURRENT grid + sinc read-out
            peak_pos = local_window_peaks(RD_map_shifted, truth_range, truth_vel, ...
                                          range_res, vel_axis, WIN);
            [det_range, det_vel] = func_sinc_interpolation( ...
                RD_map_shifted, peak_pos, range_axis, vel_axis);
            all_err_R_cfar(idx_mc, :) = det_range - truth_range;

            % same estimator on the no-data map
            peak_pos = local_window_peaks(RD_map_noise_no_data_shifted, truth_range, truth_vel, ...
                                          range_res, vel_axis, WIN);
            [det_range, ~] = func_sinc_interpolation( ...
                RD_map_noise_no_data_shifted, peak_pos, range_axis, vel_axis);
            all_err_R_nd(idx_mc, :) = det_range - truth_range;
        end

        err_R_store{idx_snr}    = all_err_R_cfar;
        err_R_store_nd{idx_snr} = all_err_R_nd;
        fprintf('  N_chip=%4d (B=%6.2f MHz) | SNR=%+3d dB | RMSE_R data / nodata = %.4f / %.4f m\n', ...
            N_chip, B/1e6, SNR_Tx_db(idx_snr), ...
            sqrt(mean(all_err_R_cfar(:).^2)), sqrt(mean(all_err_R_nd(:).^2)));
    end

    rng(1);  % reproducible bootstrap
    [lo, hi, mid] = local_bootstrap_rmse_ci(err_R_store, n_boot, alpha_ci);
    rmse_R_lo_A(:, idx_cfg)  = lo;
    rmse_R_hi_A(:, idx_cfg)  = hi;
    rmse_R_mid_A(:, idx_cfg) = mid;
    rng(1);
    [lo, hi, mid] = local_bootstrap_rmse_ci(err_R_store_nd, n_boot, alpha_ci);
    rmse_R_lo_A_nd(:, idx_cfg)  = lo;
    rmse_R_hi_A_nd(:, idx_cfg)  = hi;
    rmse_R_mid_A_nd(:, idx_cfg) = mid;
end

% =====================================================================
%  SWEEP B — velocity RMSE vs SNR for each N_prbs (= N_block), N_chip fixed
% =====================================================================
rmse_v_lo_B  = nan(n_snr, numel(N_prbs_sweep));
rmse_v_hi_B  = nan(n_snr, numel(N_prbs_sweep));
rmse_v_mid_B = nan(n_snr, numel(N_prbs_sweep));
rmse_v_lo_B_nd  = nan(n_snr, numel(N_prbs_sweep));   % no-data chain
rmse_v_hi_B_nd  = nan(n_snr, numel(N_prbs_sweep));
rmse_v_mid_B_nd = nan(n_snr, numel(N_prbs_sweep));

% restore the config chip length, chip duration, bandwidth and noise power
N_chip          = N_chip_fix;
T_chip          = T_chip_fix;
B               = B_fix;
Noise_power_sen = Noise_sen_fix;
T_pmcw          = N_chip * T_chip;
L_slot_per_sym = N_chip / (2 * N_sym_per_block);
N_slot         = N_chip / L_slot_per_sym;

fprintf('===== Sweep B: N_prbs (N_block) in [%s], N_chip = %d fixed =====\n', ...
    num2str(N_prbs_sweep), N_chip_fix);

for idx_cfg = 1:numel(N_prbs_sweep)
    N_block = N_prbs_sweep(idx_cfg);

    err_v_store    = cell(n_snr, 1);   % pooled velocity residuals per SNR, data
    err_v_store_nd = cell(n_snr, 1);   % same, no data

    for idx_snr = 1:n_snr
        P_tx = Noise_sen_fix * SNR_Tx(idx_snr);   % SNR defined at the config bandwidth
        all_err_v_cfar = nan(Mc, N_tars);
        all_err_v_nd   = nan(Mc, N_tars);

        for idx_mc = 1:Mc
            main3_signal_channel_model;
            main4_sensing_process;

            % RD-map axes for the CURRENT (N_chip, N_block)
            range_res  = c / (2 * B);
            range_axis = (0 : N_chip-1) * range_res;
            PRF        = 1 / T_pmcw;
            fd_axis    = (-N_block/2 : N_block/2-1) * (PRF / N_block);
            vel_axis   = -(Lambda / 2) * fd_axis;

            % windowed one-to-one peak search on the CURRENT grid + sinc read-out
            peak_pos = local_window_peaks(RD_map_shifted, truth_range, truth_vel, ...
                                          range_res, vel_axis, WIN);
            [det_range, det_vel] = func_sinc_interpolation( ...
                RD_map_shifted, peak_pos, range_axis, vel_axis);
            all_err_v_cfar(idx_mc, :) = det_vel - truth_vel;

            % same estimator on the no-data map
            peak_pos = local_window_peaks(RD_map_noise_no_data_shifted, truth_range, truth_vel, ...
                                          range_res, vel_axis, WIN);
            [~, det_vel] = func_sinc_interpolation( ...
                RD_map_noise_no_data_shifted, peak_pos, range_axis, vel_axis);
            all_err_v_nd(idx_mc, :) = det_vel - truth_vel;
        end

        err_v_store{idx_snr}    = all_err_v_cfar;
        err_v_store_nd{idx_snr} = all_err_v_nd;
        fprintf('  N_prbs=%4d | SNR=%+3d dB | RMSE_v data / nodata = %.4f / %.4f m/s\n', ...
            N_block, SNR_Tx_db(idx_snr), ...
            sqrt(mean(all_err_v_cfar(:).^2)), sqrt(mean(all_err_v_nd(:).^2)));
    end

    rng(1);  % reproducible bootstrap
    [lo, hi, mid] = local_bootstrap_rmse_ci(err_v_store, n_boot, alpha_ci);
    rmse_v_lo_B(:, idx_cfg)  = lo;
    rmse_v_hi_B(:, idx_cfg)  = hi;
    rmse_v_mid_B(:, idx_cfg) = mid;
    rng(1);
    [lo, hi, mid] = local_bootstrap_rmse_ci(err_v_store_nd, n_boot, alpha_ci);
    rmse_v_lo_B_nd(:, idx_cfg)  = lo;
    rmse_v_hi_B_nd(:, idx_cfg)  = hi;
    rmse_v_mid_B_nd(:, idx_cfg) = mid;
end

% restore the fixed block count for anything run after this script
N_block = N_block_fix;

% =====================================================================
%  Export sweep statistics (CSV)
% =====================================================================
tbl_A = local_sweep_table('N_chip', N_chip_sweep, SNR_Tx_db, ...
    rmse_R_lo_A, rmse_R_mid_A, rmse_R_hi_A, 'RMSE_Range_m');
tbl_A_nd = local_sweep_table('N_chip', N_chip_sweep, SNR_Tx_db, ...
    rmse_R_lo_A_nd, rmse_R_mid_A_nd, rmse_R_hi_A_nd, 'RMSE_Range_m_nodata');
tbl_A = [tbl_A, tbl_A_nd(:, 3:end)];
writetable(tbl_A, fullfile(stat_dir, 'statistics_sensing_sweep_nchip.csv'));

tbl_B = local_sweep_table('N_prbs', N_prbs_sweep, SNR_Tx_db, ...
    rmse_v_lo_B, rmse_v_mid_B, rmse_v_hi_B, 'RMSE_Vel_mps');
tbl_B_nd = local_sweep_table('N_prbs', N_prbs_sweep, SNR_Tx_db, ...
    rmse_v_lo_B_nd, rmse_v_mid_B_nd, rmse_v_hi_B_nd, 'RMSE_Vel_mps_nodata');
tbl_B = [tbl_B, tbl_B_nd(:, 3:end)];
writetable(tbl_B, fullfile(stat_dir, 'statistics_sensing_sweep_nprbs.csv'));

fprintf('\nSweep statistics saved to %s\n', stat_dir);

% =====================================================================
%  FIGURE A — Range RMSE vs SNR, one (lo, mid, hi) band per N_chip
%  Style: 95% CI fill + mid line; no-data chain dashed, same colour
% =====================================================================
c_lines = [0.00 0.45 0.74;   % blue
           0.85 0.33 0.10;   % orange
           0.47 0.67 0.19;   % green
           0.49 0.18 0.56];  % purple
markers = {'o','s','^','d'};

x = SNR_Tx_db(:);

fig_nchip = figure('Name','Range RMSE vs SNR for different N_chip (95% CI)', ...
    'Position',[140 140 800 520],'Color','w');
hold on;
for idx_cfg = 1:numel(N_chip_sweep)
    ci = c_lines(mod(idx_cfg-1, size(c_lines,1)) + 1, :);
    lo = rmse_R_lo_A(:, idx_cfg);
    hi = rmse_R_hi_A(:, idx_cfg);
    vd = ~isnan(lo) & ~isnan(hi);
    fill([x(vd); flipud(x(vd))], [lo(vd); flipud(hi(vd))], ci, ...
        'FaceAlpha', 0.25, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(x, rmse_R_mid_A(:, idx_cfg), ['-' markers{idx_cfg}], ...
        'LineWidth', 2.0, 'Color', ci, 'MarkerFaceColor', ci, ...
        'DisplayName', sprintf('$N_{\\rm chip} = %d$', N_chip_sweep(idx_cfg)));
    % no-data chain: same colour, dashed line, lighter band
    lo = rmse_R_lo_A_nd(:, idx_cfg);
    hi = rmse_R_hi_A_nd(:, idx_cfg);
    vd = ~isnan(lo) & ~isnan(hi);
    fill([x(vd); flipud(x(vd))], [lo(vd); flipud(hi(vd))], ci, ...
        'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(x, rmse_R_mid_A_nd(:, idx_cfg), ['--' markers{idx_cfg}], ...
        'LineWidth', 1.6, 'Color', ci, 'MarkerFaceColor', 'w', 'HandleVisibility', 'off');
end
plot(nan, nan, 'k-',  'LineWidth', 2.0, 'DisplayName', 'With data');
plot(nan, nan, 'k--', 'LineWidth', 1.6, 'DisplayName', 'Without data');
set(gca, 'YScale', 'log');
grid on; box on;
xlabel('SNR (dB)');
ylabel('RMSE range (m)');
title(sprintf('Numerical Range RMSE vs SNR ($N_{\\rm prbs} = %d$, $T_{\\rm prbs} = %.2f\\,\\mu$s fixed, 95\\%% CI bands)', ...
    N_block_fix, T_prbs_fix*1e6), 'Interpreter','latex');
legend('Location','best', 'Interpreter','latex');

savefig(fig_nchip, fullfile(fig_dir,'fig_rmse_range_vs_snr_nchip_sweep.fig'));
exportgraphics(fig_nchip, fullfile(fig_dir,'fig_rmse_range_vs_snr_nchip_sweep.png'),'Resolution',300);

% =====================================================================
%  FIGURE B — Velocity RMSE vs SNR, one (lo, mid, hi) band per N_prbs
% =====================================================================
fig_nprbs = figure('Name','Velocity RMSE vs SNR for different N_prbs (95% CI)', ...
    'Position',[160 160 800 520],'Color','w');
hold on;
for idx_cfg = 1:numel(N_prbs_sweep)
    ci = c_lines(mod(idx_cfg-1, size(c_lines,1)) + 1, :);
    lo = rmse_v_lo_B(:, idx_cfg);
    hi = rmse_v_hi_B(:, idx_cfg);
    vd = ~isnan(lo) & ~isnan(hi);
    fill([x(vd); flipud(x(vd))], [lo(vd); flipud(hi(vd))], ci, ...
        'FaceAlpha', 0.25, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(x, rmse_v_mid_B(:, idx_cfg), ['-' markers{idx_cfg}], ...
        'LineWidth', 2.0, 'Color', ci, 'MarkerFaceColor', ci, ...
        'DisplayName', sprintf('$N_{\\rm prbs} = %d$', N_prbs_sweep(idx_cfg)));
    % no-data chain: same colour, dashed line, lighter band
    lo = rmse_v_lo_B_nd(:, idx_cfg);
    hi = rmse_v_hi_B_nd(:, idx_cfg);
    vd = ~isnan(lo) & ~isnan(hi);
    fill([x(vd); flipud(x(vd))], [lo(vd); flipud(hi(vd))], ci, ...
        'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(x, rmse_v_mid_B_nd(:, idx_cfg), ['--' markers{idx_cfg}], ...
        'LineWidth', 1.6, 'Color', ci, 'MarkerFaceColor', 'w', 'HandleVisibility', 'off');
end
plot(nan, nan, 'k-',  'LineWidth', 2.0, 'DisplayName', 'With data');
plot(nan, nan, 'k--', 'LineWidth', 1.6, 'DisplayName', 'Without data');
set(gca, 'YScale', 'log');
grid on; box on;
xlabel('SNR (dB)');
ylabel('RMSE velocity (m/s)');
title(sprintf('Numerical Velocity RMSE vs SNR ($N_{\\rm chip} = %d$ fixed, 95\\%% CI bands)', ...
    N_chip_fix), 'Interpreter','latex');
legend('Location','best', 'Interpreter','latex');

savefig(fig_nprbs, fullfile(fig_dir,'fig_rmse_vel_vs_snr_nprbs_sweep.fig'));
exportgraphics(fig_nprbs, fullfile(fig_dir,'fig_rmse_vel_vs_snr_nprbs_sweep.png'),'Resolution',300);

fprintf('Figures saved to %s\n', fig_dir);

% =====================================================================
function [lo, hi, mid] = local_bootstrap_rmse_ci(err_store, n_boot, alpha_ci)
% Percentile bootstrap CI on the RMSE statistic, pooling trials across
% targets at each SNR point in err_store (a cell array, one Mc x N_tars
% error matrix per SNR).
    n_snr = numel(err_store);
    lo  = nan(n_snr, 1);
    hi  = nan(n_snr, 1);
    mid = nan(n_snr, 1);
    for idx_snr = 1:n_snr
        err_vec = err_store{idx_snr}(:);
        err_vec = err_vec(~isnan(err_vec));
        n = numel(err_vec);
        if n == 0, continue; end

        boot_rmse = zeros(n_boot, 1);
        for b = 1:n_boot
            samp = err_vec(randi(n, n, 1));
            boot_rmse(b) = sqrt(mean(samp.^2));
        end
        boot_rmse_sorted = sort(boot_rmse);
        lo_idx = max(1, round(n_boot * (alpha_ci/2)));
        hi_idx = min(n_boot, round(n_boot * (1 - alpha_ci/2)));

        lo(idx_snr)  = boot_rmse_sorted(lo_idx);
        hi(idx_snr)  = boot_rmse_sorted(hi_idx);
        mid(idx_snr) = sqrt(mean(err_vec.^2));
    end
end

function tbl = local_sweep_table(param_name, param_vals, snr_db, lo, mid, hi, metric_name)
% Long-format table: one row per (sweep value, SNR) with lo/mid/hi RMSE.
    n_snr = numel(snr_db);
    n_cfg = numel(param_vals);
    rows  = n_snr * n_cfg;
    P  = zeros(rows,1); S = zeros(rows,1);
    L  = zeros(rows,1); M = zeros(rows,1); H = zeros(rows,1);
    r = 1;
    for idx_cfg = 1:n_cfg
        for idx_snr = 1:n_snr
            P(r) = param_vals(idx_cfg);
            S(r) = snr_db(idx_snr);
            L(r) = lo(idx_snr, idx_cfg);
            M(r) = mid(idx_snr, idx_cfg);
            H(r) = hi(idx_snr, idx_cfg);
            r = r + 1;
        end
    end
    tbl = table(P, S, L, M, H, 'VariableNames', ...
        {param_name, 'SNR_dB', [metric_name '_CI_lo'], ...
         [metric_name '_mid'], [metric_name '_CI_hi']});
end

function peak_pos = local_window_peaks(RD, truth_range, truth_vel, range_res, vel_axis, win)
% Windowed one-to-one peak search. True cell of target m: k = round(r_m/range_res)+1
% (range_axis starts at 0) and the Doppler column whose velocity is closest
% to v_m. Window W_m = [k-win, k+win] x [d-win, d+win] (clipped to the map).
% Targets are served in decreasing order of their window peak power; each
% takes the strongest cell of its window not already taken by another target.
% peak_pos(m,:) = [range_bin, doppler_bin] of the estimate of target m.
    P = abs(RD).^2;
    [N_r, N_d] = size(P);
    N = numel(truth_range);
    k_true = round(truth_range(:) / range_res) + 1;
    [~, d_true] = min(abs(vel_axis(:).' - truth_vel(:)), [], 2);
    rr = cell(N, 1);  dd = cell(N, 1);  pk = zeros(N, 1);
    for m = 1:N
        rr{m} = max(1, k_true(m)-win) : min(N_r, k_true(m)+win);
        dd{m} = max(1, d_true(m)-win) : min(N_d, d_true(m)+win);
        pk(m) = max(P(rr{m}, dd{m}), [], 'all');
    end
    [~, order] = sort(pk, 'descend');
    taken = false(N_r, N_d);
    peak_pos = zeros(N, 2);
    for m = order.'
        Pw = P(rr{m}, dd{m});
        Pw(taken(rr{m}, dd{m})) = -Inf;
        [~, idx] = max(Pw(:));
        [ir, id] = ind2sub(size(Pw), idx);
        peak_pos(m, :) = [rr{m}(ir), dd{m}(id)];
        taken(peak_pos(m,1), peak_pos(m,2)) = true;
    end
end
