% MAIN8_MAIN — PDISAC end-to-end driver and result figures
% -------------------------------------------------------------------------
% STEP 1-5 run the single-scene pipeline (main1 -> main4, main6) and export
% the per-stage figures. STEP 6 renders the result figures of the paper from
% the statistics in exported_statistics/ (each sub-step is skipped with a
% warning if its file is missing):
%   6a  BER and BPSK-input achievable rate          <- main7_comm_analysis
%   6b  LOS / NLOS channel distribution             <- channel_com
%   6c  RMSE vs N_chip / N_prbs                     <- main5a_sensing_sweep
%   6d  goodness of fit of P_nlos                   <- check_gamma_gof
%   6e  P_fa, P_d vs N_bit^prbs, matched-P_fa P_d, ROC  <- main5b_detection_vs_nbit
%   6f  RMSE vs BCRB, conventional chains           <- main5c_bcrb_check (sinc, Dirichlet)
%   6g  RMSE vs BCRB, learned denoisers             <- main5c_bcrb_check (Dirichlet, LS, denoisers)
%   6h  table of RMSE sweeps (console + CSV)        <- main5a_sensing_sweep, main5c_bcrb_check (NBIT_EXT)
% The sub-steps run in the order 6a, 6c, 6b, 6d-6h. Figures are written to
% fig_exported/ (.fig + .pdf), the 6h table to exported_statistics/.
% Run: main8_main (adds its folder and subfolders to the path).
% To skip STEP 6 or run the pipeline headless, define NO_RESULT_PLOTS = true or
% the plot flags (NO_SIGNAL_PLOT, NO_RADAR_PLOT, NO_CFAR_PLOT, NO_BPSK_PLOT,
% TOPO_NO_PLOT) in the workspace before the run; STEP 0 keeps them.

% =====================================================================
%  STEP 0 — Setup: paths, GLOBAL figure style (set ONCE), output folder
% =====================================================================
clearvars -except NO_RESULT_PLOTS NO_SIGNAL_PLOT NO_RADAR_PLOT NO_CFAR_PLOT NO_BPSK_PLOT TOPO_NO_PLOT
clc; close all;

script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(script_dir));   % script folder and its subfolders on the path

% Global LaTeX + Times New Roman rendering for EVERY figure. Set once here;
% downstream steps inherit these and must NOT repeat set(groot,...).
set(groot, 'defaultTextInterpreter',           'latex');
set(groot, 'defaultAxesTickLabelInterpreter',  'latex');
set(groot, 'defaultLegendInterpreter',         'latex');
set(groot, 'defaultColorbarTickLabelInterpreter','latex');
set(groot, 'defaultAxesFontName',              'Times New Roman');
set(groot, 'defaultTextFontName',              'Times New Roman');
set(groot, 'defaultAxesFontSize',              14);
set(groot, 'defaultLineLineWidth',             1.6);

fig_dir = fullfile(script_dir, 'fig_exported');
if ~exist(fig_dir,'dir'), mkdir(fig_dir); end

% =====================================================================
%  STEP 1 — System & scene configuration + geometry            (Sec. II)
% =====================================================================
% main2_topology calls main1_matlab_config internally (carrier, waveform,
% ROI, targets, scatterers, UE) and draws the scene plan view.
if ~exist('TOPO_NO_PLOT', 'var'), TOPO_NO_PLOT = false; end
main2_topology;

% =====================================================================
%  STEP 2 — Transmit waveform & propagation channels           (Sec. II)
% =====================================================================
% ISAC waveform + sensing/comm channels -> received Y_sen (radar) and Y_com (UE).
main3_signal_channel_model;

% =====================================================================
%  STEP 3 — Radar sensing: MF -> RD map -> CA-CFAR -> interp   (Sec. III)
% =====================================================================
main4_sensing_process;

% =====================================================================
%  STEP 4 — Communication: delay/channel est. -> BPSK detect    (Sec. V)
% =====================================================================
main6_comm_process;

% =====================================================================
%  STEP 5 — Export the per-stage pipeline figures
% =====================================================================
figs = flipud(findall(0,'Type','figure'));
for k = 1:numel(figs)
    nm = get(figs(k),'Name');
    if isempty(nm), nm = sprintf('figure_%02d', k); end
    nm = matlab.lang.makeValidName(nm);
    savefig(figs(k),       fullfile(fig_dir, [nm '.fig']));
    exportgraphics(figs(k), fullfile(fig_dir, [nm '.pdf']), 'ContentType','vector');
end
fprintf('Saved %d pipeline figure(s) to %s\n', numel(figs), fig_dir);

% =====================================================================
%  STEP 6 — Aggregate result figures (from exported statistics)
%  Rendered only if the statistics files exist (produced by
%  main7_comm_analysis / channel_com / check_gamma_gof / main5a / main5b / main5c). Skip with NO_RESULT_PLOTS=true.
% =====================================================================
if ~exist('NO_RESULT_PLOTS','var') || ~NO_RESULT_PLOTS

% Okabe-Ito colour-blind-safe palette (shared by all STEP 6 figures)
col.black  = [0 0 0];       col.orange = [230 159   0]/255;
col.green  = [  0 158 115]/255; col.yellow = [240 228  66]/255;
col.blue   = [  0 114 178]/255; col.vermil = [213  94   0]/255;
col.purple = [204 121 167]/255; col.gray   = [0.5 0.5 0.5];

% Line styles: theory = solid (LS_THEO, LW_THEO), numerical = dashed + marker
% (LS_NUM, LW_NUM); STEP 6a, and the theory / line widths of STEP 6e-6g
LS_THEO = '-';   LW_THEO = 1.8;
LS_NUM  = '--';  LW_NUM  = 1.3;

% ---------------------------------------------------------------------
%  STEP 6a — Communication BER & achievable rate                     (Sec. V)
%  From exported_statistics/statistics_communication.csv (main7_comm_analysis):
%    Fig 1  fig_ber_nsym      BER vs. N_bit^prbs (avg over UE, MC, SNR)
%    Fig 2  fig_ber_snr_nsym  BER vs. SNR per N_bit^prbs (perfect tau_ue and h_com, + theory)
%    Fig 3  fig_ber_snr       BER vs. SNR (avg over UE, MC, N_bit^prbs)
%    Fig 4  fig_cap_nsym      BPSK-input achievable rate vs. N_bit^prbs (bits/sequence + bits/s)
%    Fig 5  fig_cap_snr_nsym  BPSK-input achievable rate vs. SNR per N_bit^prbs (bits/sequence + bits/s)
%  Numerical BER in four variants: estimated or perfect UE delay tau_ue and
%  estimated or perfect channel h_com,i.
%  Style: theory = solid line; numerical = dashed line + markers.
% ---------------------------------------------------------------------
comm_csv = fullfile(script_dir, 'exported_statistics', 'statistics_communication.csv');
if isfile(comm_csv)
    T = rmmissing(readtable(comm_csv));

    % Numerical BER variants: colour, marker, label
    numCol    = {col.blue, col.orange, col.green, col.vermil};
    numMk     = {'o','s','^','d'};
    numLbl    = { ...
        'Num.: $\hat{\tau}_{\mathrm{ue}}, \hat{h}_{\mathrm{com},i}$', ...
        'Num.: $\hat{\tau}_{\mathrm{ue}}, h_{\mathrm{com},i}$', ...
        'Num.: $\tau_{\mathrm{ue}}, \hat{h}_{\mathrm{com},i}$', ...
        'Num.: $\tau_{\mathrm{ue}}, h_{\mathrm{com},i}$'};
    numVarBER = {'Numerical_BER_Est_Tau_Est_h', 'Numerical_BER_Est_Tau_Perf_h', ...
                 'Numerical_BER_Perf_Tau_Est_h', 'Numerical_BER_Perf_Tau_Perf_h'};

    N_sym_list = unique(T.N_sym_per_block);
    N_sym_col  = {col.blue, col.orange, col.green, col.vermil, col.purple, col.yellow};

    % ---- Fig 1: BER vs. N_bit^prbs (avg over UE, MC, SNR) ------------
    G1 = groupsummary(T, 'N_sym_per_block', 'mean', ...
        [numVarBER, {'Theoretical_BER'}]);
    G1 = sortrows(G1, 'N_sym_per_block');

    figure('Color','w','Units','inches','Position',[1 1 7 5]);
    hold on; box on; grid on;
    for k = 1:4
        plot(G1.N_sym_per_block, G1.(['mean_' numVarBER{k}]), ...
            'LineStyle', LS_NUM, 'LineWidth', LW_NUM, 'Color', numCol{k}, ...
            'Marker', numMk{k}, 'MarkerFaceColor', numCol{k}, 'MarkerSize', 7, ...
            'DisplayName', numLbl{k});
    end
    plot(G1.N_sym_per_block, G1.mean_Theoretical_BER, ...
        LS_THEO, 'Color', col.gray, 'LineWidth', LW_THEO, ...
        'DisplayName', 'Theo.: Analytical form');
    set(gca, 'XScale', 'log');
    xticks(N_sym_list); xticklabels(string(N_sym_list));
    xlabel('$N_{\mathrm{bit}}^{\mathrm{prbs}}$');
    ylabel('Average Bit Error Rate (BER)');
    title('BER vs. $N_{\mathrm{bit}}^{\mathrm{prbs}}$');
    legend('Location','southeast','Box','on','FontSize',11);
    xlim([min(N_sym_list) max(N_sym_list)]);
    savefig(gcf, fullfile(fig_dir,'fig_ber_nsym.fig'));
    exportgraphics(gcf, fullfile(fig_dir,'fig_ber_nsym.pdf'), 'ContentType','vector');

    % ---- Fig 2: BER vs. SNR per N_bit^prbs (perfect tau_ue, h_com + theory)
    G2 = groupsummary(T, {'N_sym_per_block','SNR_dB'}, 'mean', ...
        {'Numerical_BER_Perf_Tau_Perf_h', 'Theoretical_BER'});

    figure('Color','w','Units','inches','Position',[1 1 7 5]);
    hold on; box on; grid on;
    for n = 1:numel(N_sym_list)
        Ns = N_sym_list(n);
        Gs = sortrows(G2(G2.N_sym_per_block == Ns, :), 'SNR_dB');
        c  = N_sym_col{mod(n-1, numel(N_sym_col)) + 1};
        % Numerical: dashed line + markers
        plot(Gs.SNR_dB, Gs.mean_Numerical_BER_Perf_Tau_Perf_h, ...
            'LineStyle', LS_NUM, 'LineWidth', LW_NUM, 'Color', c, ...
            'Marker', 'o', 'MarkerFaceColor', c, 'MarkerSize', 6, ...
            'DisplayName', sprintf('$N_{\\mathrm{bit}}^{\\mathrm{prbs}} = %d$', Ns));
        % Theoretical: solid line
        plot(Gs.SNR_dB, Gs.mean_Theoretical_BER, ...
            LS_THEO, 'Color', c, 'LineWidth', LW_THEO, 'HandleVisibility', 'off');
    end
    % Dummy handles for the style legend (Num = dashed+marker, Theo = solid)
    plot(nan, nan, 'LineStyle', LS_NUM, 'LineWidth', LW_NUM, 'Color', col.black, ...
        'Marker', 'o', 'MarkerFaceColor', col.black, ...
        'DisplayName', 'Num.: $\tau_{\mathrm{ue}}, h_{\mathrm{com},i}$');
    plot(nan, nan, LS_THEO, 'Color', col.black, 'LineWidth', LW_THEO, ...
        'DisplayName', 'Theo.: Analytical form');
    set(gca, 'YScale', 'log');
    xlabel('$\mathrm{SNR}_{\mathrm{Tx}}$ (dB)');
    ylabel('Average Bit Error Rate (BER)');
    title('BER vs. SNR for Different $N_{\mathrm{bit}}^{\mathrm{prbs}}$');
    legend('Location','southwest','Box','on','FontSize',10,'NumColumns',2);
    savefig(gcf, fullfile(fig_dir,'fig_ber_snr_nsym.fig'));
    exportgraphics(gcf, fullfile(fig_dir,'fig_ber_snr_nsym.pdf'), 'ContentType','vector');

    % ---- Fig 3: BER vs. SNR (avg over UE, MC, N_bit^prbs) ------------
    G3 = groupsummary(T, 'SNR_dB', 'mean', ...
        [numVarBER, {'Theoretical_BER'}]);
    G3 = sortrows(G3, 'SNR_dB');

    figure('Color','w','Units','inches','Position',[1 1 7 5]);
    hold on; box on; grid on;
    for k = 1:4
        plot(G3.SNR_dB, G3.(['mean_' numVarBER{k}]), ...
            'LineStyle', LS_NUM, 'LineWidth', LW_NUM, 'Color', numCol{k}, ...
            'Marker', numMk{k}, 'MarkerFaceColor', numCol{k}, 'MarkerSize', 7, ...
            'DisplayName', numLbl{k});
    end
    plot(G3.SNR_dB, G3.mean_Theoretical_BER, ...
        LS_THEO, 'Color', col.gray, 'LineWidth', LW_THEO, ...
        'DisplayName', 'Theo.: Analytical form');
    set(gca, 'YScale', 'log');
    xlabel('$\mathrm{SNR}_{\mathrm{Tx}}$ (dB)');
    ylabel('Average Bit Error Rate (BER)');
    title('BER vs. SNR');
    legend('Location','southwest','Box','on','FontSize',11);
    savefig(gcf, fullfile(fig_dir,'fig_ber_snr.fig'));
    exportgraphics(gcf, fullfile(fig_dir,'fig_ber_snr.pdf'), 'ContentType','vector');

    % ---- Fig 4: BPSK-input achievable rate vs. N_bit^prbs --------------
    %  Ensemble-average rate, both units on a dual y-axis. x-axis and style
    %  as Fig 1 (fig_ber_nsym): log x with ticks at N_sym_list,
    %  numerical = dashed + marker, theory = solid.
    %  Left  axis: Cbar_seq [bits/sequence] (the plotted curves)
    %  Right axis: Cbar_net = Cbar_seq / T_prbs [bits/s] (left axis rescaled)
    %  Averaged over UE, MC and SNR (grouped by N_bit^prbs), as Fig 1.
    capVars = {'Cap_Seq_Numerical','Cap_Seq_Theoretical'};
    GC = groupsummary(T, 'N_sym_per_block', 'mean', capVars);
    GC = sortrows(GC, 'N_sym_per_block');

    % Recover T_prbs from the exact proportionality Cbar_net = Cbar_seq/T_prbs.
    ratio  = T.Cap_Seq_Theoretical ./ T.Cap_Net_Theoretical;
    ratio  = ratio(isfinite(ratio) & ratio > 0);
    T_prbs = median(ratio);

    figure('Color','w','Units','inches','Position',[1 1 7 5]);
    ax = gca; hold(ax,'on'); box(ax,'on'); grid(ax,'on');

    % --- left axis: bits/sequence ---
    yyaxis(ax,'left');
    plot(ax, GC.N_sym_per_block, GC.mean_Cap_Seq_Numerical, ...
        'LineStyle', LS_NUM, 'LineWidth', LW_NUM, 'Color', col.blue, ...
        'Marker', 'o', 'MarkerFaceColor', col.blue, 'MarkerSize', 7, ...
        'DisplayName', 'Num.: Monte-Carlo');
    plot(ax, GC.N_sym_per_block, GC.mean_Cap_Seq_Theoretical, ...
        LS_THEO, 'Color', col.gray, 'LineWidth', LW_THEO, ...
        'DisplayName', 'Theo.: Analytical form');
    ylabel(ax, '$\bar{\mathcal{C}}_{\mathrm{seq}}$ (bits/sequence)');
    ax.YAxis(1).Color = col.black;

    % --- right axis: bits/second (left axis rescaled by 1/T_prbs) ---
    yyaxis(ax,'right');
    ylabel(ax, '$\bar{\mathcal{C}}_{\mathrm{net}}$ (bits/s)');
    ax.YAxis(2).Color = col.black;

    yyaxis(ax,'left');  yl = ylim(ax);
    yyaxis(ax,'right'); ylim(ax, yl / T_prbs);
    yyaxis(ax,'left');   % keep left active for the curves/legend

    set(ax, 'XScale', 'log');
    xticks(ax, N_sym_list); xticklabels(ax, string(N_sym_list));
    xlim(ax, [min(N_sym_list) max(N_sym_list)]);
    xlabel(ax, '$N_{\mathrm{bit}}^{\mathrm{prbs}}$');
    title(ax, 'BPSK-Input Achievable Rate vs. $N_{\mathrm{bit}}^{\mathrm{prbs}}$');
    legend(ax, 'Location','northwest','Box','on','FontSize',11);
    savefig(gcf, fullfile(fig_dir,'fig_cap_nsym.fig'));
    exportgraphics(gcf, fullfile(fig_dir,'fig_cap_nsym.pdf'), 'ContentType','vector');

    % ---- Fig 5: BPSK-input achievable rate vs. SNR per N_bit^prbs -----
    %  Same quantities and dual y-axis as Fig 4, x-axis = SNR; one colour
    %  per N_bit^prbs as in Fig 2 (numerical = dashed + marker, theory = solid,
    %  dotted = ceiling N_bit^prbs). Log y-axis on both sides.
    %  Averaged over UE and MC (grouped by N_bit^prbs and SNR).
    G5 = groupsummary(T, {'N_sym_per_block','SNR_dB'}, 'mean', ...
        {'Cap_Seq_Numerical','Cap_Seq_Theoretical'});

    figure('Color','w','Units','inches','Position',[1 1 7 5]);
    ax = gca; hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    yyaxis(ax,'left');
    for n = 1:numel(N_sym_list)
        Ns = N_sym_list(n);
        Gs = sortrows(G5(G5.N_sym_per_block == Ns, :), 'SNR_dB');
        c  = N_sym_col{mod(n-1, numel(N_sym_col)) + 1};
        plot(ax, Gs.SNR_dB, Gs.mean_Cap_Seq_Numerical, ...
            'LineStyle', LS_NUM, 'LineWidth', LW_NUM, 'Color', c, ...
            'Marker', 'o', 'MarkerFaceColor', c, 'MarkerSize', 6, ...
            'DisplayName', sprintf('$N_{\\mathrm{bit}}^{\\mathrm{prbs}} = %d$', Ns));
        plot(ax, Gs.SNR_dB, Gs.mean_Cap_Seq_Theoretical, ...
            LS_THEO, 'Color', c, 'LineWidth', LW_THEO, 'HandleVisibility', 'off');
        % ceiling: BPSK carries at most 1 bit per data slot -> C_seq <= N_bit^prbs
        yline(ax, Ns, ':', 'Color', c, 'LineWidth', 1.2, 'HandleVisibility', 'off');
    end
    plot(ax, nan, nan, 'LineStyle', LS_NUM, 'LineWidth', LW_NUM, 'Color', col.black, ...
        'Marker', 'o', 'MarkerFaceColor', col.black, 'DisplayName', 'Num.: Monte-Carlo');
    plot(ax, nan, nan, LS_THEO, 'Color', col.black, 'LineWidth', LW_THEO, ...
        'DisplayName', 'Theo.: Analytical form');
    plot(ax, nan, nan, ':', 'Color', col.black, 'LineWidth', 1.2, ...
        'DisplayName', 'Limit: $N_{\mathrm{bit}}^{\mathrm{prbs}}$');
    ylabel(ax, '$\bar{\mathcal{C}}_{\mathrm{seq}}$ (bits/sequence)');
    ax.YAxis(1).Color = col.black;
    yyaxis(ax,'right');
    ylabel(ax, '$\bar{\mathcal{C}}_{\mathrm{net}}$ (bits/s)');
    ax.YAxis(2).Color = col.black;
    % log y: the curves span about 1e-2 (low SNR) to 1e2 bits/sequence
    c_all = [G5.mean_Cap_Seq_Numerical; G5.mean_Cap_Seq_Theoretical];
    c_all = c_all(c_all > 0);
    yl = [10^floor(log10(min(c_all))), 1.5*max(N_sym_list)];   % room above the top ceiling
    yyaxis(ax,'left');  set(ax, 'YScale','log');  ylim(ax, yl);
    % left ticks: decades below 1, then the ceilings N_bit^prbs themselves
    yt_dec = 10.^(floor(log10(yl(1))):-1);
    yticks(ax, [yt_dec, 1, N_sym_list(:).']);
    yticklabels(ax, [string(compose('$10^{%d}$', log10(yt_dec))), "1", string(N_sym_list(:).')]);
    yyaxis(ax,'right'); set(ax, 'YScale','log');  ylim(ax, yl / T_prbs);
    yyaxis(ax,'left');
    xlabel(ax, '$\mathrm{SNR}_{\mathrm{Tx}}$ (dB)');
    xlim(ax, [min(T.SNR_dB) max(T.SNR_dB)]);
    title(ax, 'BPSK-Input Achievable Rate vs. SNR');
    legend(ax, 'Location','southeast','Box','on','FontSize',10,'NumColumns',2);
    savefig(gcf, fullfile(fig_dir,'fig_cap_snr_nsym.fig'));
    exportgraphics(gcf, fullfile(fig_dir,'fig_cap_snr_nsym.pdf'), 'ContentType','vector');

    fprintf('STEP 6a: 3 BER figures + 2 capacity figures saved to %s (T_prbs = %.4g s)\n', ...
        fig_dir, T_prbs);
else
    warning('main8_main:skipCommPerf', ...
        'STEP 6a skipped: %s not found (run main7_comm_analysis first).', comm_csv);
end

% ---------------------------------------------------------------------
%  STEP 6c — Sensing RMSE vs. waveform length (from main5a CSVs)  (Sec. IV)
%    Range RMSE    vs. N_chip (statistics_sensing_sweep_nchip.csv)
%    Velocity RMSE vs. N_prbs (statistics_sensing_sweep_nprbs.csv)
%  Figures fig_rmse_range_vs_nchip and fig_rmse_vel_vs_nprbs (log-log);
%  with data = blue solid, without data = orange dashed with open markers.
%  RMSE averaged over SNR_Tx = RMSE of all pooled trials:
%      MSE = mean_s RMSE_s^2   (every SNR point s has the same number of trials).
%  Shaded band: 95% CI of that pooled RMSE, from the per-SNR bootstrap CIs
%  [lo_s, hi_s] of main5a: sd(MSE_s) ~ (hi_s^2 - lo_s^2)/(2*1.96), independent
%  over s, so sd(MSE) = sqrt(sum_s sd(MSE_s)^2)/n_s and
%      CI = sqrt(MSE -+ 1.96*sd(MSE)).
%  RMSE_SNR_AVG = [lo hi] restricts the average to lo <= SNR_Tx <= hi.
% ---------------------------------------------------------------------
RMSE_SNR_AVG = [-Inf Inf];   % SNR_Tx range (dB) of the average; [-Inf Inf] = whole grid
FIG_WIDE     = [1 1 7 5];    % [left bottom width height] in inches (also STEP 6e)

sweep_nchip_csv = fullfile(script_dir, 'exported_statistics', 'statistics_sensing_sweep_nchip.csv');
sweep_nprbs_csv = fullfile(script_dir, 'exported_statistics', 'statistics_sensing_sweep_nprbs.csv');
% {csv, swept column, metric prefix, x label, y label, file name}
sweep_spec = { ...
    sweep_nchip_csv, 'N_chip', 'RMSE_Range_m', '$N_{\mathrm{chip}}$', 'Range RMSE (m)',      'fig_rmse_range_vs_nchip'; ...
    sweep_nprbs_csv, 'N_prbs', 'RMSE_Vel_mps', '$N_{\mathrm{prbs}}$', 'Velocity RMSE (m/s)', 'fig_rmse_vel_vs_nprbs'};
if all(isinf(RMSE_SNR_AVG))
    snr_ttl = 'averaged over $\mathrm{SNR}_{\mathrm{Tx}}$';
else
    snr_ttl = sprintf('averaged over $%g \\le \\mathrm{SNR}_{\\mathrm{Tx}} \\le %g$ dB', RMSE_SNR_AVG);
end

for q = 1:size(sweep_spec, 1)
    [csv_q, xname, mname, xlab, ylab, fname] = sweep_spec{q, :};
    if ~isfile(csv_q)
        warning('main8_main:skipSweep', ...
            'STEP 6c skipped %s: %s not found (run main5a_sensing_sweep first).', fname, csv_q);
        continue
    end
    S = rmmissing(readtable(csv_q));
    S = S(S.SNR_dB >= RMSE_SNR_AVG(1) & S.SNR_dB <= RMSE_SNR_AVG(2), :);
    xv = unique(S.(xname));

    figure('Color','w','Units','inches','Position',FIG_WIDE);
    hold on; box on; grid on;
    y_all = [];
    % {column suffix, colour, line style, marker face, legend}
    chains_q = {'',        col.blue,   '-',  col.blue, 'With data'; ...
                '_nodata', col.orange, '--', 'w',      'Without data'};
    for k = 1:size(chains_q, 1)
        [suf, c, ls, mf, lab] = chains_q{k, :};
        if ~ismember([mname suf '_mid'], S.Properties.VariableNames), continue; end
        rm = nan(size(xv));  lo = rm;  hi = rm;
        for n = 1:numel(xv)
            Sn  = S(S.(xname) == xv(n), :);
            mse = mean(Sn.([mname suf '_mid']).^2);
            sdm = sqrt(sum(((Sn.([mname suf '_CI_hi']).^2 - Sn.([mname suf '_CI_lo']).^2) ...
                  / (2*1.96)).^2)) / height(Sn);
            rm(n) = sqrt(mse);
            lo(n) = sqrt(max(mse - 1.96*sdm, 0));
            hi(n) = sqrt(mse + 1.96*sdm);
        end
        fill([xv; flipud(xv)], [lo; flipud(hi)], c, ...
            'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        plot(xv, rm, 'LineStyle', ls, 'LineWidth', 1.6, 'Color', c, 'Marker', 'o', ...
            'MarkerFaceColor', mf, 'MarkerSize', 7, 'DisplayName', lab);
        y_all = [y_all; lo; hi]; %#ok<AGROW>
    end
    set(gca, 'XScale', 'log', 'YScale', 'log');
    % tight y-range and 1-2-5 ticks, so the short log axis keeps its labels
    yl = [0.85*min(y_all(y_all > 0)), 1.15*max(y_all)];
    yt = reshape([1; 2; 5] * 10.^(floor(log10(yl(1))):ceil(log10(yl(2)))), 1, []);
    ylim(yl);  yticks(yt(yt >= yl(1) & yt <= yl(2)));
    xticks(xv);  xticklabels(string(xv));  xlim([min(xv) max(xv)]);
    xlabel(xlab);  ylabel(ylab);
    title(['RMSE ' snr_ttl]);
    legend('Location','northeast','Box','on','FontSize',11);
    savefig(gcf, fullfile(fig_dir, [fname '.fig']));
    exportgraphics(gcf, fullfile(fig_dir, [fname '.pdf']), 'ContentType','vector');
end

% ---------------------------------------------------------------------
%  STEP 6b — Channel LOS/NLOS distribution                     (Sec. II)
%  From exported_statistics/statistics_channel.mat (channel_com; does NOT
%  re-run the simulation). Histograms of the real and imaginary parts of the
%  LOS and aggregate NLOS channels, each trial normalized to unit variance,
%  with the arcsine (LOS) and N(0,1) (NLOS) PDFs -> fig_com_channel_dis.
% ---------------------------------------------------------------------
chan_mat = fullfile(script_dir, 'exported_statistics', 'statistics_channel.mat');
if isfile(chan_mat)
    S = load(chan_mat, 'los_channel_all', 'nlos_channel_all', 'N_block');

    % Normalize PER TRIAL (one scene = N_block+1 consecutive samples): the
    % model CN(0, 2 sigma_d^2) of Sec. II-C is conditional on the scene, since
    % 2 sigma_d^2 is the sum of that scene's path powers; a single global std
    % would mix scenes of very different NLOS power (a scale mixture: peaked,
    % heavy tails). Each trial is divided by its own rms, and real/imag parts
    % are scaled by sqrt(2) to unit variance: LOS -> arcsine on [-sqrt2, sqrt2],
    % NLOS -> N(0,1).
    n_per  = S.N_block + 1;
    los_z  = per_trial_unit_parts(S.los_channel_all,  n_per);
    nlos_z = per_trial_unit_parts(S.nlos_channel_all, n_per);

    % Theoretical shapes in z-units: LOS arcsine (Var = A^2/2 = 1 -> A = sqrt2), NLOS N(0,1)
    x_grid  = linspace(-4.2, 4.2, 4000)';
    y_arc   = arcsine_pdf(x_grid, sqrt(2));
    y_gauss = gaussian_pdf(x_grid, 0, 1);

    col_los_fit  = [0 40 90]/255;    % dark blue (arcsine fit)
    col_nlos_fit = [90 30 0]/255;    % dark vermillion (gaussian fit)

    figure('Color','w','Units','inches','Position',[1 1 7 5]);
    ax = gca; hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    histogram(ax, los_z,  'Normalization','pdf', 'NumBins',150, ...
        'FaceColor', col.blue,   'EdgeColor','none', 'FaceAlpha',0.55, ...
        'DisplayName','LOS (empirical)');
    histogram(ax, nlos_z, 'Normalization','pdf', 'NumBins',150, ...
        'FaceColor', col.vermil, 'EdgeColor','none', 'FaceAlpha',0.55, ...
        'DisplayName','NLOS (empirical)');
    plot(ax, x_grid, y_arc,   '--', 'Color', col_los_fit,  'LineWidth', 2.2, ...
        'DisplayName','LOS fit: arcsine');
    plot(ax, x_grid, y_gauss, '--', 'Color', col_nlos_fit, 'LineWidth', 2.2, ...
        'DisplayName','Aggregate NLOS fit: Gaussian');
    set(ax, 'YScale','log'); ylim(ax,[1e-3 1e1]); xlim(ax,[-4.2 4.2]);
    xlabel(ax, 'Amplitude');
    ylabel(ax, 'PDF (log scale)');
    title(ax, '$h_{\mathrm{com}}(t) = h^{\mathrm{los}}_{\rm com}(t) + h^{\mathrm{nlos}}_{\rm com}(t)$');
    legend(ax, 'Location','north','Box','on','FontSize',11,'NumColumns',2);
    savefig(gcf, fullfile(fig_dir,'fig_com_channel_dis.fig'));
    exportgraphics(gcf, fullfile(fig_dir,'fig_com_channel_dis.pdf'), 'ContentType','vector');
    fprintf('STEP 6b: channel-distribution figure saved to %s\n', fig_dir);
else
    warning('main8_main:skipChannelDist', ...
        'STEP 6b skipped: %s not found (run channel_com first).', chan_mat);
end

% ---------------------------------------------------------------------
%  STEP 6d — Goodness of fit of the NLOS power P_nlos           (Sec. V)
%  From exported_statistics/statistics_gamma_gof.mat (written by
%  check_gamma_gof.m; does NOT re-run the draws). One figure per
%  N_bit^prbs: histogram of P_nlos/Pbar_nlos, the Gamma model of the NLOS
%  power with the closed-form (K_nlos, Omega), and Lognormal / Weibull /
%  Inverse Gaussian / Exponential fitted by MLE. Curves are drawn from the
%  worst to the best KS distance (stored by check_gamma_gof), best on top.
%  Output: fig_gof_nbit<N> (fig_gof_ue<U>_nbit<N> if several UEs).
% ---------------------------------------------------------------------
gof_mat = fullfile(script_dir, 'exported_statistics', 'statistics_gamma_gof.mat');
if isfile(gof_mat)
    Gf = load(gof_mat, 'gof');
    gof_lab = {'Gamma', 'Lognormal', 'Weibull', 'Inverse Gaussian', 'Exponential'};
    gof_fam = {'', 'Lognormal', 'Weibull', 'InverseGaussian', 'Exponential'};
    gof_col = {[213 94 0]/255, [0 158 115]/255, [204 121 167]/255, [86 180 233]/255, [0.5 0.5 0.5]};
    gof_sty = {'-', '--', '-.', ':', ':'};
    for k = 1:numel(Gf.gof)
        g  = Gf.gof(k);
        xs = double(g.x(:));
        pd = cell(1, numel(gof_lab));
        pd{1} = makedist('Gamma', 'a', g.K_nlos, 'b', g.b);
        for f = 2:numel(gof_lab), pd{f} = fitdist(xs, gof_fam{f}); end
        [~, ord] = sort(g.KS);

        figure('Color','w','Units','inches','Position',[1 1 6 4.2]);
        ax = gca; hold(ax,'on'); box(ax,'on'); grid(ax,'on');
        xg = linspace(0, quantile(xs, 0.995), 400);
        histogram(ax, xs(xs <= xg(end)), 80, 'Normalization','pdf', ...
            'FaceColor',[0.85 0.85 0.85], 'EdgeColor','none', 'DisplayName','Simulated');
        for f = flip(ord)                                   % best fit drawn last
            plot(ax, xg, pdf(pd{f}, xg), gof_sty{f}, 'Color', gof_col{f}, 'LineWidth', 2, ...
                'DisplayName', gof_lab{f});
        end
        xlabel(ax, '$P_{\rm nlos}/\bar P_{\rm nlos}$');
        ylabel(ax, 'PDF');
        legend(ax, 'Location','northeast','Box','on','FontSize',11);
        fn = fullfile(fig_dir, sprintf('fig_gof_nbit%d', g.N_bit));
        if numel(unique([Gf.gof.UE])) > 1
            fn = fullfile(fig_dir, sprintf('fig_gof_ue%d_nbit%d', g.UE, g.N_bit));
        end
        savefig(gcf, [fn '.fig']);
        exportgraphics(gcf, [fn '.pdf'], 'ContentType','vector');
    end
    fprintf('STEP 6d: %d goodness-of-fit figure(s) saved to %s\n', numel(Gf.gof), fig_dir);
else
    warning('main8_main:skipGof', ...
        'STEP 6d skipped: %s not found (run check_gamma_gof first).', gof_mat);
end

% ---------------------------------------------------------------------
%  STEP 6e — Detection vs. SNR for each N_bit^prbs (from main5b)  (Sec. IV)
%  Reads exported_statistics/statistics_detection_vs_nbit_ml.csv (main5b with
%  USE_DENOISER = true) if it exists, else statistics_detection_vs_nbit.csv.
%  (a) fig_pfa_vs_nbit  P_fa vs. SNR_Tx
%  (b) fig_pd_vs_nbit   P_d  vs. SNR_Tx
%      Chains 'data' and 'nodata' only, nominal CA-CFAR threshold. Same data as the
%      main5b figures with the axes swapped: with data = one solid curve + markers
%      per N_bit^prbs; without data = ONE dashed black curve pooled over N_bit^prbs
%      (the data-free map does not depend on N_bit^prbs). Each figure has a zoom
%      inset (window and position set in DET_ZOOM below). main5b writes figures
%      of the same names with a '_ml' tag when USE_DENOISER = true; its untagged
%      figures are overwritten here.
%  (c) fig_pd_matched_pfa: P_d vs. SNR_Tx at the same EMPIRICAL P_fa = PFA_MATCH_6E for
%      every chain of CH_6E (Conventional = CA-CFAR on the RD map, and RDPDNet, each with
%      the inputs without data, with data + LS, with data), from the ROC file
%      <det_csv>_roc.csv (main5b with ROC = true): counts pooled over N_bit^prbs at each
%      design P_fa, then P_d interpolated linearly in log10(P_fa), as pd_at_pfa of main5b.
%      At the nominal threshold the denoised maps have a much larger P_fa, so only this
%      matched comparison is fair. Zoom inset over ZOOM_6E.snr.
%  (d) fig_roc: ROC (P_d vs. empirical P_fa, log x-axis) of the same chains, pooled over
%      N_bit^prbs AND over the SNRs in ROC_SNR_POOL_6E (threshold averaging: the counts of
%      every (SNR, N_bit) are summed at the same design P_fa, so each SNR has equal weight);
%      the large marker on each curve is its operating point at the nominal threshold
%      (design P_fa = NOM_PFA_6E), and the dotted vertical line is PFA_MATCH_6E of (c).
%  (c)-(d) use the style of STEP 6f/6g: colour = method, line style / marker = input
%  (INP_6E, as INP_BCRB), legend below the axes.
% ---------------------------------------------------------------------
det_csv = fullfile(script_dir, 'exported_statistics', 'statistics_detection_vs_nbit.csv');
det_csv_ml = fullfile(script_dir, 'exported_statistics', 'statistics_detection_vs_nbit_ml.csv');
if isfile(det_csv_ml), det_csv = det_csv_ml; end
% zoom insets: SNR window [dB] and inset position [left bottom width height] (figure units)
DET_ZOOM.pd  = struct('snr', [10 20],  'pos', [0.17 0.40 0.32 0.27]);
DET_ZOOM.pfa = struct('snr', [-15 5],  'pos', [0.17 0.30 0.32 0.25]);
if isfile(det_csv)
    Dt = readtable(det_csv);
    Dd = Dt(strcmp(Dt.Chain, 'data'),   :);
    Dn = Dt(strcmp(Dt.Chain, 'nodata'), :);
    nbit_list = unique(Dd.N_bit);
    snr_list  = unique(Dt.SNR_dB);
    pfa_nom   = Dt.Pfa_nominal(1);
    nbitCol = {col.blue, col.orange, col.green, col.vermil, col.purple, [86 180 233]/255};
    nbitMk  = {'o','s','^','d','v','>'};

    % pooled no-data reference per SNR (as in main5b)
    Pd_nd = nan(size(snr_list));  Pfa_nd = nan(size(snr_list));
    for is = 1:numel(snr_list)
        r = Dn.SNR_dB == snr_list(is);
        Pd_nd(is)  = sum(Dn.n_hit(r)) / sum(Dn.n_tar(r));
        Pfa_nd(is) = sum(Dn.n_fa(r))  / sum(Dn.n_cell(r));
    end

    for kind = ["pfa", "pd"]
        fh = figure('Color','w','Units','inches','Position',FIG_WIDE);
        ax = gca; hold(ax,'on'); box(ax,'on'); grid(ax,'on');
        h_curves = gobjects(0);
        Y = nan(numel(snr_list), numel(nbit_list));
        for n = 1:numel(nbit_list)
            c = nbitCol{mod(n-1, numel(nbitCol)) + 1};
            S = sortrows(Dd(Dd.N_bit == nbit_list(n), :), 'SNR_dB');
            if kind == "pfa", y = S.Pfa; else, y = S.Pd; end
            Y(:, n) = y;
            h_curves(end+1) = plot(ax, S.SNR_dB, y, '-', 'Color', c, 'LineWidth', 1.6, ...
                'Marker', nbitMk{mod(n-1, numel(nbitMk)) + 1}, 'MarkerFaceColor', c, ...
                'MarkerSize', 6, ...
                'DisplayName', sprintf('$N_{\\mathrm{bit}}^{\\mathrm{prbs}} = %d$', nbit_list(n))); %#ok<SAGROW>
        end
        if kind == "pfa", y_nd = Pfa_nd; else, y_nd = Pd_nd; end
        h_curves(end+1) = plot(ax, snr_list, y_nd, '--', 'Color', col.black, 'LineWidth', 1.6, ...
            'DisplayName', 'Without data');
        xlabel(ax, '$\mathrm{SNR}_{\mathrm{Tx}}$ (dB)');
        xticks(ax, snr_list);  xlim(ax, [min(snr_list) max(snr_list)]);
        ymax = max([Y(:); y_nd(:)]);
        if kind == "pfa"
            % linear axis in units of 1e-4: the values span less than one decade
            yline(ax, pfa_nom, ':', 'Color', col.gray, 'LineWidth', 1.6, ...
                'DisplayName', 'Nominal $P_{\mathrm{fa}}$');
            ylim(ax, [0.8*pfa_nom, ceil(ymax/pfa_nom) * pfa_nom]);
            ax.YAxis.Exponent = -4;
            ylabel(ax, 'False-alarm probability $P_{\mathrm{fa}}$');
            title(ax, '$P_{\mathrm{fa}}$ vs. SNR for Different $N_{\mathrm{bit}}^{\mathrm{prbs}}$');
            Z = DET_ZOOM.pfa;  fn = fullfile(fig_dir, 'fig_pfa_vs_nbit');
        else
            ylim(ax, [0, min(1, ceil(10*ymax)/10)]);
            ylabel(ax, 'Detection probability $P_{\mathrm{d}}$');
            title(ax, '$P_{\mathrm{d}}$ vs. SNR for Different $N_{\mathrm{bit}}^{\mathrm{prbs}}$');
            Z = DET_ZOOM.pd;   fn = fullfile(fig_dir, 'fig_pd_vs_nbit');
        end
        legend(ax, 'Location', 'northwest', 'Box', 'on', 'FontSize', 10, 'NumColumns', 2);

        % ---- zoom inset: y-range of the with-data curves inside the SNR window ----
        in_win = snr_list >= Z.snr(1) & snr_list <= Z.snr(2);
        yw = Y(in_win, :);  yw = yw(:);
        pad = 0.08 * (max(yw) - min(yw) + eps);
        ylo = min(yw) - pad;  yhi = max(yw) + pad;
        rectangle(ax, 'Position', [Z.snr(1), ylo, diff(Z.snr), yhi - ylo], ...
            'LineStyle', ':', 'EdgeColor', col.gray, 'LineWidth', 1.0);
        axi = axes('Position', Z.pos);
        hold(axi, 'on'); box(axi, 'on'); grid(axi, 'on');
        copyobj(h_curves, axi);
        if kind == "pfa"
            yline(axi, pfa_nom, ':', 'Color', col.gray, 'LineWidth', 1.2);
            axi.YAxis.Exponent = -4;
        end
        xlim(axi, Z.snr);  ylim(axi, [ylo yhi]);
        xticks(axi, snr_list(in_win));
        set(axi, 'FontSize', 10);

        drawnow;                    % render fh completely before saving (exportgraphics otherwise
                                % may write the previous figure, seen on macOS)
        savefig(fh, [fn '.fig']);
        exportgraphics(fh, [fn '.pdf'], 'ContentType', 'vector');
    end
    fprintf('STEP 6e: fig_pfa_vs_nbit and fig_pd_vs_nbit saved to %s\n', fig_dir);
else
    warning('main8_main:skipDetection', ...
        'STEP 6e skipped: %s not found (run main5b_detection_vs_nbit first).', det_csv);
end

% ---- STEP 6e (c)-(d): style as STEP 6f/6g: colour = method, line style / marker = input ----
PFA_MATCH_6E = 1e-3;           % empirical P_fa at which the chains are compared in (c)
PDNET_6E     = 'pdnet';     % chain prefix of RDPDNet in the main5b CSV
% input: {label, chain suffix, line style, marker, open marker}, as INP_BCRB of STEP 6f/6g
INP_6E = {'No data',   'nodata',  ':',  's', true;
          'Data + LS', 'data_ls', '-.', '^', false;
          'Data',      'data',    '--', 'o', false};
% method: {label, chain prefix, colour}; Conventional = CA-CFAR on the RD map (blue, as 6f/6g)
EST_6E = {'Conventional', '',             col.blue;
          'RDPDNet',      [PDNET_6E '_'], col.green};
% chains: {legend label, chain name in the CSV, colour, line style, marker, open marker}
CH_6E = cell(0, 6);
for e = 1:size(EST_6E, 1)
    for k = 1:size(INP_6E, 1)
        CH_6E(end+1, :) = {sprintf('%s, %s', EST_6E{e,1}, INP_6E{k,1}), [EST_6E{e,2} INP_6E{k,2}], ...
            EST_6E{e,3}, INP_6E{k,3}, INP_6E{k,4}, INP_6E{k,5}}; %#ok<SAGROW>
    end
end
LW_6E  = LW_NUM;  MS_6E = 7;   % as LW_NUM / MS_BCRB of 6f/6g
FIG_6E = [1 1 7 6];            % as 6f/6g (legend below the axes)
% (c) zoom inset: SNR window [dB] and position [left bottom width height] as fractions of the axes
ZOOM_6E = struct('snr', [10 20], 'pos', [0.05 0.50 0.42 0.45]);
roc_csv = strrep(det_csv, '.csv', '_roc.csv');
if isfile(roc_csv)
    Rc = readtable(roc_csv);
    if ismember('log10_Pfa_design', Rc.Properties.VariableNames)
        Rc.lpd = Rc.log10_Pfa_design;          % log10 design P_fa (Pfa_design itself underflows to 0)
    else
        Rc.lpd = log10(Rc.Pfa_design);         % ROC CSV without the log10_Pfa_design column
    end
end

% ---- STEP 6e (c): P_d at the same empirical P_fa for every chain (ROC of main5b) ----
if isfile(roc_csv)
    snr6e = unique(Rc.SNR_dB);
    fh = figure('Color','w','Units','inches','Position',FIG_6E);
    ax = gca; hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    hh6e = gobjects(0);  Y6e = nan(numel(snr6e), 0);
    for c = 1:size(CH_6E, 1)
        Rk = Rc(strcmp(Rc.Chain, CH_6E{c,2}), :);
        if isempty(Rk)
            warning('main8_main:step6e', 'STEP 6e (c): chain %s not in %s, curve skipped.', CH_6E{c,2}, roc_csv);
            continue
        end
        y6 = nan(numel(snr6e), 1);
        for is = 1:numel(snr6e)
            Rs = Rk(Rk.SNR_dB == snr6e(is), :);
            [~, ~, g] = unique(Rs.lpd);                                      % one ROC point per design P_fa
            pd_roc  = accumarray(g, Rs.n_hit) ./ accumarray(g, Rs.n_tar);    % pooled over N_bit^prbs
            pfa_roc = accumarray(g, Rs.n_fa)  ./ accumarray(g, Rs.n_cell);
            y6(is) = pd_at_pfa_6e(pd_roc, pfa_roc, PFA_MATCH_6E);
        end
        Y6e(:, end+1) = y6; %#ok<SAGROW>
        if CH_6E{c,6}, mfc = 'w'; else, mfc = CH_6E{c,3}; end
        hh6e(end+1) = plot(ax, snr6e, y6, 'LineStyle', CH_6E{c,4}, 'LineWidth', LW_6E, ...
            'Color', CH_6E{c,3}, 'Marker', CH_6E{c,5}, 'MarkerFaceColor', mfc, 'MarkerSize', MS_6E, ...
            'DisplayName', CH_6E{c,1}); %#ok<SAGROW>
    end
    xlabel(ax, '$\mathrm{SNR}_{\mathrm{Tx}}$ (dB)');
    ylabel(ax, 'Detection probability $P_{\mathrm{d}}$');
    title(ax, sprintf('$P_{\\mathrm{d}}$ at the Empirical $P_{\\mathrm{fa}} = 10^{%d}$', round(log10(PFA_MATCH_6E))));
    xticks(ax, snr6e);  xlim(ax, [min(snr6e) max(snr6e)]);
    ylim(ax, [0, min(1, ceil(10*max(Y6e(:)))/10)]);
    legend(hh6e, 'Location', 'southoutside', 'NumColumns', 3, 'Box', 'on', 'FontSize', 10);
    % zoom inset over the high-SNR window, where the chains separate (placed after the legend)
    Z = ZOOM_6E;
    in_win = snr6e >= Z.snr(1) & snr6e <= Z.snr(2);
    yw = Y6e(in_win, :);  yw = yw(isfinite(yw));
    pad = 0.08 * (max(yw) - min(yw) + eps);
    ylo = min(yw) - pad;  yhi = max(yw) + pad;
    rectangle(ax, 'Position', [Z.snr(1), ylo, diff(Z.snr), yhi - ylo], ...
        'LineStyle', ':', 'EdgeColor', col.gray, 'LineWidth', 1.0);
    drawnow;                    % axes position after the outside legend is placed
    ax.Units = 'normalized';  pax = ax.Position;
    axi = axes(fh, 'Position', [pax(1) + Z.pos(1)*pax(3), pax(2) + Z.pos(2)*pax(4), Z.pos(3)*pax(3), Z.pos(4)*pax(4)]);
    hold(axi, 'on'); box(axi, 'on'); grid(axi, 'on');
    copyobj(hh6e, axi);
    xlim(axi, Z.snr);  ylim(axi, [ylo yhi]);  xticks(axi, snr6e(in_win));
    set(axi, 'FontSize', 10);
    fn = fullfile(fig_dir, 'fig_pd_matched_pfa');
    drawnow;                    % render fh completely before saving (exportgraphics otherwise
                                % may write the previous figure, seen on macOS)
    savefig(fh, [fn '.fig']);
    exportgraphics(fh, [fn '.pdf'], 'ContentType', 'vector');
    fprintf('STEP 6e (c): fig_pd_matched_pfa (P_fa = %.0e) saved to %s\n', PFA_MATCH_6E, fig_dir);
else
    warning('main8_main:skipDetectionRoc', ...
        'STEP 6e (c) skipped: %s not found (run main5b_detection_vs_nbit with ROC = true).', roc_csv);
end

% ---- STEP 6e (d): one ROC of every chain, pooled over N_bit^prbs and over an SNR window ----
ROC_SNR_POOL_6E = [10 20];     % SNR_Tx window [dB] pooled into the ROC (equal weight per SNR)
NOM_PFA_6E      = 1e-4;        % nominal design P_fa of main5b (P_fa), marked on each curve
ROC_XLIM_6E     = [1e-5 1e-1]; % empirical P_fa axis
ROC_NMK_6E      = 8;           % approximate number of markers along each curve
if isfile(roc_csv)
    in_pool = Rc.SNR_dB >= ROC_SNR_POOL_6E(1) & Rc.SNR_dB <= ROC_SNR_POOL_6E(2);
    fh = figure('Color','w','Units','inches','Position',FIG_6E);
    ax = gca; hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    hroc = gobjects(0);  ymax = 0;
    for c = 1:size(CH_6E, 1)
        Rs = Rc(strcmp(Rc.Chain, CH_6E{c,2}) & in_pool, :);
        if isempty(Rs)
            warning('main8_main:step6e', 'STEP 6e (d): chain %s not in %s, curve skipped.', CH_6E{c,2}, roc_csv);
            continue
        end
        [lpd, ~, g] = unique(Rs.lpd);                                    % one ROC point per design P_fa
        pd_roc  = accumarray(g, Rs.n_hit) ./ accumarray(g, Rs.n_tar);    % pooled over N_bit^prbs and SNR
        pfa_roc = accumarray(g, Rs.n_fa)  ./ accumarray(g, Rs.n_cell);
        k = pfa_roc > 0;                                                 % log axis: P_fa = 0 dropped
        xy = sortrows([pfa_roc(k) pd_roc(k)]);  x = xy(:,1);  y = xy(:,2);
        % markers spread evenly in log10(P_fa) over the visible range
        xv = x >= ROC_XLIM_6E(1) & x <= ROC_XLIM_6E(2);
        if ~any(xv)
            warning('main8_main:step6e', 'STEP 6e (d): chain %s has no point in ROC_XLIM_6E.', CH_6E{c,2});
            continue
        end
        lx = log10(x);  tg = linspace(min(lx(xv)), max(lx(xv)), ROC_NMK_6E);
        mi = unique(arrayfun(@(t) find(xv & abs(lx - t) == min(abs(lx(xv) - t)), 1), tg));
        if CH_6E{c,6}, mfc = 'w'; else, mfc = CH_6E{c,3}; end
        hroc(end+1) = plot(ax, x, y, 'LineStyle', CH_6E{c,4}, 'LineWidth', LW_6E, ...
            'Color', CH_6E{c,3}, 'Marker', CH_6E{c,5}, 'MarkerFaceColor', mfc, 'MarkerSize', MS_6E, ...
            'MarkerIndices', mi, 'DisplayName', CH_6E{c,1}); %#ok<SAGROW>
        [~, i0] = min(abs(lpd - log10(NOM_PFA_6E)));                    % nominal operating point
        plot(ax, pfa_roc(i0), pd_roc(i0), 'LineStyle', 'none', 'Color', col.black, ...
            'Marker', CH_6E{c,5}, 'MarkerFaceColor', CH_6E{c,3}, 'MarkerSize', 12, 'LineWidth', 1.2, ...
            'HandleVisibility', 'off');
        ymax = max([ymax; y(xv)]);
    end
    hroc(end+1) = xline(ax, PFA_MATCH_6E, ':', 'Color', col.gray, 'LineWidth', 1.6, ...
        'DisplayName', sprintf('$P_{\\mathrm{fa}} = 10^{%d}$', round(log10(PFA_MATCH_6E))));
    hroc(end+1) = plot(ax, nan, nan, 'LineStyle', 'none', 'Marker', 'o', 'MarkerSize', 12, ...
        'Color', col.black, 'MarkerFaceColor', col.gray, 'LineWidth', 1.2, 'DisplayName', 'Nominal threshold');
    set(ax, 'XScale', 'log');  xlim(ax, ROC_XLIM_6E);  ylim(ax, [0, min(1, ceil(10*ymax)/10)]);
    xlabel(ax, 'Empirical false-alarm probability $P_{\mathrm{fa}}$');
    ylabel(ax, 'Detection probability $P_{\mathrm{d}}$');
    title(ax, sprintf('ROC, $%d \\le \\mathrm{SNR}_{\\mathrm{Tx}} \\le %d$ dB', ROC_SNR_POOL_6E));
    legend(hroc, 'Location', 'southoutside', 'NumColumns', 3, 'Box', 'on', 'FontSize', 10);
    fn = fullfile(fig_dir, 'fig_roc');
    drawnow;                    % render fh completely before saving (exportgraphics otherwise
                                % may write the previous figure, seen on macOS)
    savefig(fh, [fn '.fig']);
    exportgraphics(fh, [fn '.pdf'], 'ContentType', 'vector');
    fprintf('STEP 6e (d): fig_roc (SNR %d to %d dB pooled) saved to %s\n', ROC_SNR_POOL_6E, fig_dir);
else
    warning('main8_main:skipDetectionRoc', ...
        'STEP 6e (d) skipped: %s not found (run main5b_detection_vs_nbit with ROC = true).', roc_csv);
end

% ---------------------------------------------------------------------
%  STEP 6f — RMSE vs. BCRB, conventional chains (from main5c_bcrb_check)   (Sec. IV-D2)
%  Reads two main5c runs with clutter, the LS chain and no denoiser
%  (NO_CLUTTER = false, LS_CANCEL = true, USE_DENOISER = false):
%    BCRB_TAG_SINC = '_ls'            READOUT = "sinc"       -> statistics_bcrb_check_ls.csv
%    BCRB_TAG_DIR  = '_dirichlet_ls'  READOUT = "dirichlet"  -> statistics_bcrb_check_dirichlet_ls.csv
%  and draws, vs. SNR_Tx,
%    fig_bcrb_range_vs_snr  sqrt(E[RMSE_r^2]),  fig_bcrb_vel_vs_snr  sqrt(E[RMSE_v^2]):
%    theory (BCRB of Proposition 1 for the map with data, solid black, read from the sinc run;
%    the bound does not depend on the read-out) and, for the Dirichlet and the sinc read-out,
%    the inputs without data, with data + LS, with data.
%  Style shared with STEP 6g (INP_BCRB, MS_BCRB below): colour = estimator (CA-CFAR Dirichlet blue,
%  as in 6g; CA-CFAR sinc purple, 6f only); line style / marker = input (data '--' o,
%  data + LS '-.' ^, no data ':' open s). Sinc markers are smaller and drawn on top, since the
%  with-data curves of the two read-outs nearly coincide. Zoom inset over BCRB_ZOOM.snr (as 6e).
% ---------------------------------------------------------------------
% Style shared by STEP 6f and 6g: {input label, column suffix, line style, marker, open marker};
% rows in legend order (no data, data + LS, data); colour encodes the estimator in both figures
INP_BCRB = {'No data',   'nodata',  ':',  's', true;
            'Data + LS', 'data_ls', '-.', '^', false;
            'Data',      'data',    '--', 'o', false};
MS_BCRB  = 7;
% zoom inset (as STEP 6e): SNR window [dB] and inset position [left bottom width height] as fractions of
% the main axes (bottom-left, below the BCRB at low SNR; the legend is placed below the axes)
BCRB_ZOOM = struct('snr', [5 15], 'pos', [0.05 0.10 0.36 0.32]);
BCRB_TAG_SINC = '_ls';
BCRB_TAG_DIR  = '_dirichlet_ls';
bcrb_csv_s = fullfile(script_dir, 'exported_statistics', ['statistics_bcrb_check' BCRB_TAG_SINC '.csv']);
bcrb_csv_d = fullfile(script_dir, 'exported_statistics', ['statistics_bcrb_check' BCRB_TAG_DIR '.csv']);
if isfile(bcrb_csv_s) && isfile(bcrb_csv_d)
    Ts = sortrows(readtable(bcrb_csv_s), 'SNR_dB');
    Td = sortrows(readtable(bcrb_csv_d), 'SNR_dB');
    assert(isequal(Ts.SNR_dB, Td.SNR_dB), 'STEP 6f: the two main5c runs have different SNR grids');
    xb = Ts.SNR_dB;
    % bound column: 'BCRB_<r|v>_data' (written by main5c), else 'BCRB_<r|v>_unknown_data' if present
    bcol = @(T, q) T.(pick_col(T, sprintf('BCRB_%s_data', q), sprintf('BCRB_%s_unknown_data', q)));
    % {label, table, colour, marker size}; Dirichlet first, sinc (smaller markers) on top
    ro6f = {'Dirichlet', Td, col.blue,   MS_BCRB;
            'Sinc',      Ts, col.purple, MS_BCRB - 3};
    for p = 1:2
        if p == 1, q = 'r'; ylab = 'Range RMSE (m)';      fname = 'fig_bcrb_range_vs_snr';
        else,      q = 'v'; ylab = 'Velocity RMSE (m/s)'; fname = 'fig_bcrb_vel_vs_snr';
        end
        figure('Color','w','Units','inches','Position',[1 1 7 6]);
        hold on; box on; grid on;
        hh = plot(xb, bcol(Ts, q), LS_THEO, 'Color', col.black, 'LineWidth', LW_THEO, ...
            'DisplayName', 'Theory (BCRB)');
        for e = 1:size(ro6f, 1)
            for k = 1:size(INP_BCRB, 1)
                if INP_BCRB{k,5}, mfc = 'w'; else, mfc = ro6f{e,3}; end
                hh(end+1) = plot(xb, ro6f{e,2}.(sprintf('RMSE_%s_%s', q, INP_BCRB{k,2})), ...
                    'LineStyle', INP_BCRB{k,3}, 'LineWidth', LW_NUM, 'Color', ro6f{e,3}, ...
                    'Marker', INP_BCRB{k,4}, 'MarkerFaceColor', mfc, 'MarkerSize', ro6f{e,4}, ...
                    'DisplayName', sprintf('%s, %s', ro6f{e,1}, INP_BCRB{k,1})); %#ok<SAGROW>
            end
        end
        set(gca, 'YScale', 'log');
        xlim([min(xb) max(xb)]);  xticks(xb);
        xlabel('$\mathrm{SNR}_{\mathrm{Tx}}$ (dB)');  ylabel(ylab);
        legend(hh, 'Location', 'southoutside', 'NumColumns', 3, 'Box', 'on', 'FontSize', 10);
        add_rmse_zoom(gca, hh, xb, BCRB_ZOOM.snr, BCRB_ZOOM.pos, col.gray);
        savefig(gcf, fullfile(fig_dir, [fname '.fig']));
        exportgraphics(gcf, fullfile(fig_dir, [fname '.pdf']), 'ContentType', 'vector');
    end
    fprintf('STEP 6f: fig_bcrb_range_vs_snr and fig_bcrb_vel_vs_snr saved to %s\n', fig_dir);
else
    warning('main8_main:skipBcrb', ...
        'STEP 6f skipped: %s or %s not found (run main5c_bcrb_check, sinc and Dirichlet, LS_CANCEL = true).', ...
        bcrb_csv_s, bcrb_csv_d);
end

% ---------------------------------------------------------------------
%  STEP 6g — RMSE vs. BCRB with the learned denoisers (from main5c_bcrb_check)
%  Reads one main5c run, READOUT = "dirichlet", LS_CANCEL = true, USE_DENOISER = true
%  (DENOISER_MODELS includes PDNET_6G, DNCNN_6G and the models of NOAFM_6G;
%  DENOISER_LS_MODELS includes PDNET_6G and DNCNN_6G):
%    BCRB_TAG_ML = '_dirichlet_ls_ml'  ->  statistics_bcrb_check_dirichlet_ls_ml.csv
%  and draws, vs. SNR_Tx,
%    fig_bcrb_ml_range_vs_snr  sqrt(E[RMSE_r^2]),  fig_bcrb_ml_vel_vs_snr  sqrt(E[RMSE_v^2]):
%    theory (BCRB of Proposition 1 for the map with data, solid black) and, for Conventional
%    (CA-CFAR, no denoiser), RDPDNet and DnCNN, the inputs with data, with data + LS, without data.
%  Style shared with STEP 6f (INP_BCRB, MS_BCRB): colour = estimator (Conventional blue, as the
%  Dirichlet curves of 6f; RDPDNet green; DnCNN vermilion); line style / marker = input (data '--' o,
%  data + LS '-.' ^, no data ':' open s). Zoom inset over BCRB_ZOOM.snr (as 6e).
%  AFM ablation (NOAFM_6G): RDPDNet and DnCNN trained without AFM, data input, same colour as the
%  model, solid line with x markers. A column missing from the CSV is skipped with a warning.
% ---------------------------------------------------------------------
BCRB_TAG_ML = '_dirichlet_ls_ml';
PDNET_6G    = 'pdnet';      % column prefix of the RDPDNet model in the main5c CSV
DNCNN_6G    = 'dncnn';
% AFM ablation: {label, column prefix of the model trained without AFM, colour}; data input only
NOAFM_6G    = {'RDPDNet w/o AFM', 'pdnet_noafm', col.green;
               'DnCNN w/o AFM',   'dncnn_noafm',    col.vermil};
bcrb_csv_ml = fullfile(script_dir, 'exported_statistics', ['statistics_bcrb_check' BCRB_TAG_ML '.csv']);
if isfile(bcrb_csv_ml)
    Tm = sortrows(readtable(bcrb_csv_ml), 'SNR_dB');
    xb = Tm.SNR_dB;
    % {estimator label, column infix ('' = CA-CFAR on the map itself), colour}
    est6g = {'Conventional', '',                  col.blue;
             'RDPDNet',   [PDNET_6G '_'],      col.green;
             'DnCNN',   [DNCNN_6G '_'],      col.vermil};
    inp6g = INP_BCRB;   % shared with STEP 6f
    inp_order = {[1 2 3], [3 2 1], [3 2 1]};   % Conventional: no data, data + LS, data; RDPDNet/DnCNN: data, data + LS, no data
    for p = 1:2
        if p == 1, q = 'r'; ylab = 'Range RMSE (m)';      fname = 'fig_bcrb_ml_range_vs_snr';
        else,      q = 'v'; ylab = 'Velocity RMSE (m/s)'; fname = 'fig_bcrb_ml_vel_vs_snr';
        end
        figure('Color','w','Units','inches','Position',[1 1 7 6]);
        hold on; box on; grid on;
        hb = plot(xb, Tm.(sprintf('BCRB_%s_data', q)), LS_THEO, 'Color', col.black, ...
            'LineWidth', LW_THEO, 'DisplayName', 'Theory (BCRB)');
        hh = hb;
        for e = 1:size(est6g, 1)
            for k = inp_order{e}
                cname = sprintf('RMSE_%s_%s%s', q, est6g{e,2}, inp6g{k,2});
                if ~ismember(cname, Tm.Properties.VariableNames)
                    warning('main8_main:step6g', 'STEP 6g: column %s not in %s, curve skipped.', cname, bcrb_csv_ml);
                    continue
                end
                if inp6g{k,5}, mfc = 'w'; else, mfc = est6g{e,3}; end
                hh(end+1) = plot(xb, Tm.(cname), 'LineStyle', inp6g{k,3}, 'LineWidth', LW_NUM, ...
                    'Color', est6g{e,3}, 'Marker', inp6g{k,4}, 'MarkerFaceColor', mfc, 'MarkerSize', MS_BCRB, ...
                    'DisplayName', sprintf('%s, %s', est6g{e,1}, inp6g{k,1})); %#ok<SAGROW>
            end
        end
        for e = 1:size(NOAFM_6G, 1)                   % AFM ablation: data input, solid, x markers
            cname = sprintf('RMSE_%s_%s_data', q, NOAFM_6G{e,2});
            if ~ismember(cname, Tm.Properties.VariableNames)
                warning('main8_main:step6g', 'STEP 6g: column %s not in %s, curve skipped.', cname, bcrb_csv_ml);
                continue
            end
            hh(end+1) = plot(xb, Tm.(cname), 'LineStyle', '-', 'LineWidth', LW_NUM, ...
                'Color', NOAFM_6G{e,3}, 'Marker', 'x', 'MarkerSize', MS_BCRB + 1, ...
                'DisplayName', sprintf('%s, Data', NOAFM_6G{e,1})); %#ok<SAGROW>
        end
        set(gca, 'YScale', 'log');
        xlim([min(xb) max(xb)]);  xticks(xb);
        xlabel('$\mathrm{SNR}_{\mathrm{Tx}}$ (dB)');  ylabel(ylab);
        legend(hh, 'Location', 'southoutside', 'NumColumns', 3, 'Box', 'on', 'FontSize', 10);
        add_rmse_zoom(gca, hh, xb, BCRB_ZOOM.snr, BCRB_ZOOM.pos, col.gray);
        savefig(gcf, fullfile(fig_dir, [fname '.fig']));
        exportgraphics(gcf, fullfile(fig_dir, [fname '.pdf']), 'ContentType', 'vector');
    end
    fprintf('STEP 6g: fig_bcrb_ml_range_vs_snr and fig_bcrb_ml_vel_vs_snr saved to %s\n', fig_dir);
else
    warning('main8_main:skipBcrbMl', ...
        'STEP 6g skipped: %s not found (run main5c_bcrb_check with READOUT = "dirichlet", LS_CANCEL and USE_DENOISER = true).', ...
        bcrb_csv_ml);
end

% ---------------------------------------------------------------------
%  STEP 6h — Table of RMSE sweeps (RMSE vs N_chip, N_prbs, N_bit^prbs)  (Sec. IV)
%  N_chip / N_prbs: conventional chain of main5a_sensing_sweep, with and
%  without data, root mean square of the RMSE over SNR_WIN_SWEEP.
%  N_bit^prbs: CA-CFAR chain with Dirichlet interpolation of main5c_bcrb_check
%  (runs NBIT_EXT = NBIT_TAB with READOUT = "dirichlet", LS_CANCEL = true ->
%  statistics_bcrb_check_dirichlet_ls_nbit<N>.csv), with and without data, and
%  the BCRB, root mean square over SNR_WIN_NBIT. Printed to the console and
%  written to exported_statistics/table_rmse_sweep.csv.
% ---------------------------------------------------------------------
SNR_WIN_SWEEP = [-10 30];          % SNR_Tx window of the N_chip / N_prbs blocks [dB]
SNR_WIN_NBIT  = [20 30];           % SNR_Tx window of the N_bit^prbs blocks [dB]
NBIT_TAB      = [4 16 64 128];     % N_bit^prbs of the main5c runs
rms_win  = @(v, s, w) sqrt(mean(v(s >= w(1) & s <= w(2)).^2));
nbit_csv = @(n) fullfile(script_dir, 'exported_statistics', ...
                         sprintf('statistics_bcrb_check_dirichlet_ls_nbit%d.csv', n));
tab_rows = cell(0, 4);             % {block, method, parameter value, RMSE}

if isfile(sweep_nchip_csv) && isfile(sweep_nprbs_csv)
    % {block label, CSV, parameter column, RMSE column prefix}
    blk6h = {'Range RMSE (m) vs N_chip',      sweep_nchip_csv, 'N_chip', 'RMSE_Range_m';
             'Velocity RMSE (m/s) vs N_prbs', sweep_nprbs_csv, 'N_prbs', 'RMSE_Vel_mps'};
    for b = 1:size(blk6h, 1)
        T6h = readtable(blk6h{b,2});
        for n = unique(T6h.(blk6h{b,3}))'
            Tn = T6h(T6h.(blk6h{b,3}) == n, :);
            tab_rows(end+1, :) = {blk6h{b,1}, 'Conv., data', n, ...
                rms_win(Tn.([blk6h{b,4} '_mid']), Tn.SNR_dB, SNR_WIN_SWEEP)}; %#ok<AGROW>
            tab_rows(end+1, :) = {blk6h{b,1}, 'Conv., no data', n, ...
                rms_win(Tn.([blk6h{b,4} '_nodata_mid']), Tn.SNR_dB, SNR_WIN_SWEEP)}; %#ok<AGROW>
        end
    end
else
    warning('main8_main:skipTableSweep', ...
        'STEP 6h: %s or %s not found (run main5a_sensing_sweep first).', sweep_nchip_csv, sweep_nprbs_csv);
end

if all(arrayfun(@(n) isfile(nbit_csv(n)), NBIT_TAB))
    qty6h = {'r', 'Range RMSE (m) vs N_bit^prbs';
             'v', 'Velocity RMSE (m/s) vs N_bit^prbs'};
    row6h = {'Conv., data', 'RMSE_%s_data'; 'Conv., no data', 'RMSE_%s_nodata'; 'BCRB', 'BCRB_%s_data'};
    for iq = 1:size(qty6h, 1)
        for n = NBIT_TAB
            Tn = readtable(nbit_csv(n));
            for ir = 1:size(row6h, 1)
                tab_rows(end+1, :) = {qty6h{iq,2}, row6h{ir,1}, n, ...
                    rms_win(Tn.(sprintf(row6h{ir,2}, qty6h{iq,1})), Tn.SNR_dB, SNR_WIN_NBIT)}; %#ok<AGROW>
            end
        end
    end
else
    warning('main8_main:skipTableNbit', ...
        'STEP 6h: %s not found for every N_bit^prbs in [%s] (run main5c_bcrb_check with NBIT_EXT).', ...
        nbit_csv(NBIT_TAB(1)), num2str(NBIT_TAB));
end

if ~isempty(tab_rows)
    T_tab = cell2table(tab_rows, 'VariableNames', {'Block', 'Method', 'Value', 'RMSE'});
    writetable(T_tab, fullfile(script_dir, 'exported_statistics', 'table_rmse_sweep.csv'));
    blocks6h = unique(T_tab.Block, 'stable');
    for ib = 1:numel(blocks6h)
        Tb = T_tab(strcmp(T_tab.Block, blocks6h{ib}), :);
        fprintf('\n%s\n', blocks6h{ib});
        rows6h = unique(Tb.Method, 'stable');
        for ir = 1:numel(rows6h)
            Tr = Tb(strcmp(Tb.Method, rows6h{ir}), :);
            fprintf('  %-16s', rows6h{ir});
            fprintf('  %4d: %-7.3g', [Tr.Value.'; Tr.RMSE.']);
            fprintf('\n');
        end
    end
    fprintf('STEP 6h: table_rmse_sweep.csv written to %s\n', fullfile(script_dir, 'exported_statistics'));
end

end   % NO_RESULT_PLOTS


% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================
function y = arcsine_pdf(x, A)
    % PDF of x = A*sin(theta), theta ~ Uniform(0,2*pi): U-shaped ("arcsine")
    % distribution on [-A, A].
    y = zeros(size(x));
    inside = abs(x) < A;
    y(inside) = 1 ./ (pi * sqrt(A.^2 - x(inside).^2));
end

function z = per_trial_unit_parts(h, n_per)
    % Pooled real and imaginary parts of h, each trial (n_per consecutive
    % samples) divided by its own rms and scaled by sqrt(2) to unit variance.
    h  = h(:);
    nt = floor(numel(h) / n_per);
    H  = reshape(h(1:nt*n_per), n_per, nt);
    rms_t = sqrt(mean(abs(H).^2, 1));
    keep  = rms_t > 0 & isfinite(rms_t);
    Hn = sqrt(2) * H(:, keep) ./ rms_t(keep);
    z  = [real(Hn(:)); imag(Hn(:))];
    z  = z(isfinite(z));
end
function y = gaussian_pdf(x, mu, sigma)
    % PDF of N(mu, sigma^2) at x.
    y = 1 ./ (sigma .* sqrt(2*pi)) .* exp(-0.5 * ((x - mu) ./ sigma).^2);
end

function add_rmse_zoom(ax, hh, xb, win, pos, gray)
% Zoom inset (STEP 6f/6g, as STEP 6e) of the curves hh over the SNR window win, log y-axis.
% pos = [left bottom width height] as fractions of the main axes. The y-range is set by the RMSE
% curves hh(2:end) inside the window (hh(1) = theory is drawn but not used for the range) and is
% marked by a dotted box on ax.
    in_win = xb >= win(1) & xb <= win(2);
    yw = [];
    for h = hh(2:end)
        y = h.YData(:);  yw = [yw; y(in_win)]; %#ok<AGROW>
    end
    yw  = yw(isfinite(yw) & yw > 0);
    ylo = min(yw) / 1.2;  yhi = max(yw) * 1.2;
    rectangle(ax, 'Position', [win(1), ylo, diff(win), yhi - ylo], ...
        'LineStyle', ':', 'EdgeColor', gray, 'LineWidth', 1.0);
    drawnow;                                   % axes position after the outside legend is placed
    ax.Units = 'normalized';  P = ax.Position;
    axi = axes('Position', [P(1) + pos(1)*P(3), P(2) + pos(2)*P(4), pos(3)*P(3), pos(4)*P(4)]);
    hold(axi, 'on'); box(axi, 'on'); grid(axi, 'on');
    copyobj(hh, axi);
    set(axi, 'YScale', 'log', 'FontSize', 10);
    xlim(axi, win);  ylim(axi, [ylo yhi]);  xticks(axi, xb(in_win));
end

function pd = pd_at_pfa_6e(pd_roc, pfa_roc, pfa0)
% P_d at the empirical P_fa = pfa0 by linear interpolation in log10(P_fa) along the ROC
% (as pd_at_pfa of main5b: points with P_fa = 0 dropped; NaN if pfa0 is outside the range)
    k = pfa_roc > 0 & isfinite(pfa_roc);
    x = log10(pfa_roc(k));  y = pd_roc(k);
    [x, iu] = unique(x);  y = y(iu);
    if numel(x) < 2 || log10(pfa0) < x(1) || log10(pfa0) > x(end), pd = NaN; return; end
    pd = interp1(x, y, log10(pfa0));
end

function name = pick_col(T, preferred, fallback)
% First of two column names that exists in table T (STEP 6f: name of the BCRB column).
    if ismember(preferred, T.Properties.VariableNames), name = preferred; else, name = fallback; end
end
