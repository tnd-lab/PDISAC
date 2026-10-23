% MAIN5D_FIG_MAPS — RD maps and AFM masks of one test scene, one PDF per panel
% -------------------------------------------------------------------------
% One random scene with the settings of the test split (dataset_test_cfg of main1),
% drawn as in generate_radar_dataset: 5 targets + the UE, 200 static scatterers
% outside the near-range exclusion of func_generate_static_scatterers. Seeded with
% SEED + 1000*SCENE_IDX (SEED differs from the dataset seeds of generate_radar_dataset),
% so the scene is not in the training set. Every map except Z_rd^prbs contains noise.
%
% Panels (fig_exported/fig_maps/<name>.pdf, suffix _snr<SNR> only if SNR_SHOW is a vector)
%   fig_Z_rd_data      Z_rd, data-carrying waveform (system input)
%   fig_Z_rd_data_ls   Z_rd after LS cancellation of the static clutter (as main5c_bcrb_check)
%   fig_Z_rd_nodata    Z_rd, data-free MLS waveform
%   fig_Z_rd_clean     Z_rd^prbs (noise-free, data-free; training label)
%   fig_Z_rd_pdnet     Z_hat_rd of RDPDNet (PROPOSED = model pdnet of inference.json)
%   fig_Z_rd_pdnet_noafm  RDPDNet without AFM (NOAFM)
%   fig_Z_rd_dncnn     Z_hat_rd of DnCNN;  fig_Z_rd_dncnn_noafm  DnCNN without AFM
%                      (a model whose checkpoint is missing is skipped)
%   fig_Z_rd_alt       Z_rd', second realisation of the same scene (other symbols and noise)
%   fig_Z_rd_mix       Z_rd^mix, AFM mixed map of PROPOSED
%   fig_Z_rd_mix_den   f_den^infer(Z_rd^mix) of PROPOSED
%   fig_M_afm, fig_M_mask, fig_M_tars   AFM masks of PROPOSED on this scene
% Ground truth: o on every map. Colour bars: fig_colorbar_db, fig_colorbar_mask.
% fig_preview.png: all panels on one page (layout check only).
% Maps: dB relative to their own peak (DB_RANGE); masks: 20 log10 M (MASK_DB).
%
% Run standalone: main5d_fig_maps (settings below). Calls main1_matlab_config,
% main3_signal_channel_model, main4_sensing_process and, through
% PDISAC_cfg.inference.python_executable, the Python script
% alg_pdisac/scripts/infer/afm_maps_mat.py (models of configs/inference.json).
% -------------------------------------------------------------------------
clear; clc; close all;
script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(script_dir));
NO_SIGNAL_PLOT = true;  NO_BPSK_PLOT = true;  NO_RADAR_PLOT = true;  NO_CFAR_PLOT = true;

main1_matlab_config;

% =====================================================================
%  Settings
% =====================================================================
SNR_SHOW  = 20;                   % [dB]; a vector exports one set per SNR (suffix _snr<SNR>)
N_SYM     = 8;                    % N_sym_per_block (= N_bit^prbs) of the shown frame
SCENE_IDX = 1;                    % index of the test scene (not selected by outcome)
SEED      = 5150000;             % scene / noise seed (as main5b, main5c); not a dataset seed
PROPOSED  = "pdnet";           % model names of alg_pdisac/configs/inference.json
DNCNN     = "dncnn";              % skipped if its checkpoint is missing
NOAFM     = "pdnet_noafm";     % RDPDNet without AFM (same version as PROPOSED)
DNCNN_NOAFM = "dncnn_noafm";      % DnCNN without AFM
R_SHOW    = "auto";               % displayed range [m]: [r_min r_max], or "auto" = targets + ZOOM_MARGIN
V_SHOW    = "auto";               % displayed velocity [m/s]: [v_min v_max], or "auto"
ZOOM_MARGIN = [10 5];             % "auto" margin around the targets [m, m/s] (rounded outward)
LS_TAPS   = [-64 256];           % static-clutter taps [chips] of the LS canceller (as main5c_bcrb_check)
CMAP      = "viridis";             % colormap of maps and masks: "parula" (MATLAB default) or "viridis"
DB_RANGE  = 40;                   % dynamic range of the RD maps [dB]
MASK_DB   = 40;                   % dynamic range of the masks [dB] (20 log10 M)
PANEL_IN  = [2.4 2.1];            % panel size [in] (about one third of the text width)
FONT      = 10;

fig_dir = fullfile(script_dir, 'fig_exported', 'fig_maps');
if ~isfolder(fig_dir), mkdir(fig_dir); end
set(groot, 'defaultTextInterpreter','latex', 'defaultAxesTickLabelInterpreter','latex', ...
    'defaultLegendInterpreter','latex', 'defaultAxesFontName','Times New Roman', ...
    'defaultTextFontName','Times New Roman', 'defaultAxesFontSize', FONT);

cfg  = dataset_test_cfg;
G_tx = 10^(cfg.system_override.gain_tx_db/10);
G_rx = 10^(cfg.system_override.gain_rx_db/10);
G_ue = 10^(cfg.system_override.gain_ue_db/10);
N_ref_tars = PDISAC_cfg.scene.communication_reflectors;    % needed by main3 (comm part)
python_exe = string(PDISAC_cfg.inference.python_executable);

range_res  = c / (2 * B);
range_axis = (0 : N_chip-1) * range_res;
vel_axis   = -(Lambda/2) * (-N_block/2 : N_block/2-1) * (1/T_pmcw/N_block);
col0       = N_block/2 + 1;                                % removed zero-Doppler column
if CMAP == "parula", cmap_db = parula(256); else, cmap_db = viridis_map(256); end
cmap_mask = cmap_db;

% =====================================================================
%  Scene (as generate_radar_dataset)
% =====================================================================
scene_idx = SCENE_IDX;
rng(SEED + 1000*scene_idx, 'twister');
N_ue        = 1;
UE_position = rand_pos_in_roi(Region_of_interest, N_ue);
UE_vel      = VEL_RANGE_UE * (2*rand(3,1) - 1);   UE_vel(3) = 0;
UE_rcs      = 10^( (10*log10(10) + (10*log10(20)-10*log10(10))*rand()) / 10 );
UE_motion   = phased.Platform('InitialPosition', UE_position, 'Velocity', UE_vel);
N_tars        = cfg.fixed_targets;
Tars_position = rand_pos_in_roi(Region_of_interest, N_tars);
Tars_vel      = VEL_RANGE_TAR * (2*rand(3, N_tars) - 1);   Tars_vel(3,:) = 0;
Tars_rcs      = 10.^( (10*log10(10) + (10*log10(20)-10*log10(10))*rand(1, N_tars)) / 10 );
N_tars        = N_tars + 1;
Tars_position = [Tars_position, UE_position];
Tars_vel      = [Tars_vel,      UE_vel];
Tars_rcs      = [Tars_rcs,      UE_rcs];
Tars_motion   = phased.Platform('InitialPosition', Tars_position, 'Velocity', Tars_vel);
N_scats        = cfg.scatterers;
Scats_position = func_generate_static_scatterers(N_scats, Region_of_interest);
Scats_vel      = zeros(size(Scats_position));
Scats_rcs      = 10.^( (10*log10(5) + (10*log10(10)-10*log10(5))*rand(1, N_scats)) / 10 );
Scats_motion   = phased.Platform('InitialPosition', Scats_position, 'Velocity', Scats_vel);
r_clutter      = min(sqrt(sum((Scats_position - Tx_position).^2, 1)));   % nearest scatterer [m]

rel        = Tars_position - Tx_position;
truth_r    = sqrt(sum(rel.^2, 1));
truth_v    = sum(Tars_vel .* rel ./ (truth_r + eps), 1);
truth      = [truth_r(:) truth_v(:)];
d_true     = zeros(1, N_tars);
for m = 1:N_tars, [~, d_true(m)] = min(abs(vel_axis - truth_v(m))); end
vis        = d_true ~= col0;                               % invisible in the zero-Doppler column
if isstring(R_SHOW) || ischar(R_SHOW)                      % zoom on the targets (all, incl. the UE)
    R_SHOW = [max(0, 10*floor((min(truth(:,1)) - ZOOM_MARGIN(1))/10)), 10*ceil((max(truth(:,1)) + ZOOM_MARGIN(1))/10)];
end
if isstring(V_SHOW) || ischar(V_SHOW)
    V_SHOW = [5*floor((min(truth(:,2)) - ZOOM_MARGIN(2))/5), 5*ceil((max(truth(:,2)) + ZOOM_MARGIN(2))/5)];
end
fprintf(['*** main5d | test scene %d (nearest scatterer %.1f m) | %d targets (%d visible) | ' ...
    'N_sym = %d | SNR = %s dB\n'], scene_idx, r_clutter, N_tars, nnz(vis), N_SYM, mat2str(SNR_SHOW));

N_sym_per_block = N_SYM;
L_slot_per_sym  = N_chip / (2 * N_sym_per_block);
N_slot          = N_chip / L_slot_per_sym;

for snr = SNR_SHOW
    if isscalar(SNR_SHOW), sfx = ''; else, sfx = sprintf('_snr%d', snr); end
    P_tx = Noise_power_sen * 10^(snr/10);
    rng(SEED + 1000*scene_idx + 1, 'twister');            % same symbols and noise at every SNR
    main3_signal_channel_model;  main4_sensing_process;    % second realisation (AFM input)
    Z_alt = RD_map_shifted;
    main3_signal_channel_model;  main4_sensing_process;    % primary realisation
    Z       = RD_map_shifted;                              % with data, with noise
    Z_ls    = rd_map_ls(Data_Rx, Data_Tx, LS_TAPS);        % with data + LS, with noise
    Z_nd    = RD_map_noise_no_data_shifted;                % without data, with noise
    Z_clean = RD_map_no_noise_shifted;                     % Z_rd^prbs

    out = run_afm_maps(python_exe, PDISAC_repo_root, Z, Z_alt, Z_clean, truth, ...
                       [PROPOSED NOAFM DNCNN DNCNN_NOAFM], PROPOSED);
    fprintf('  SNR %+d dB | models %s\n', snr, strjoin(out.model_name, ', '));
    den = @(nm) squeeze(out.den(out.model_name == nm, :, :));

    fdb = @(M) @(ax) draw_db(ax, M, range_axis, vel_axis, R_SHOW, V_SHOW, DB_RANGE, cmap_db, truth);
    fmk = @(M) @(ax) draw_mask(ax, M, range_axis, vel_axis, R_SHOW, V_SHOW, MASK_DB, cmap_mask, truth);
    P = struct('file', {}, 'title', {}, 'draw', {});
    P(end+1) = struct('file', 'fig_Z_rd_data',    'title', '$\mathbf{Z}_{\rm rd}=\mathbf{Z}_{\rm rd}^{\rm prbs}+\mathbf{Z}_{\rm rd}^{\rm ran}+\mathbf{N}_{\rm rd}^{\rm mf}$',          'draw', fdb(Z));
    P(end+1) = struct('file', 'fig_Z_rd_data_ls', 'title', '$\mathbf{Z}_{\rm rd}$, LS',     'draw', fdb(Z_ls));
    P(end+1) = struct('file', 'fig_Z_rd_nodata',  'title', '$\mathbf{Z}_{\rm rd}=\mathbf{Z}_{\rm rd}^{\rm prbs}+\mathbf{N}_{\rm rd}^{\rm mf}$',       'draw', fdb(Z_nd));
    P(end+1) = struct('file', 'fig_Z_rd_clean',   'title', '$\mathbf{Z}_{\rm rd}^{\rm prbs}$',          'draw', fdb(Z_clean));
    P(end+1) = struct('file', 'fig_Z_rd_pdnet',   'title', '$\hat{\mathbf{Z}}_{\rm rd}$, RDPDNet',      'draw', fdb(den(PROPOSED)));
    opt = {NOAFM, 'fig_Z_rd_pdnet_noafm', '$\hat{\mathbf{Z}}_{\rm rd}$, RDPDNet w/o AFM'; ...
           DNCNN, 'fig_Z_rd_dncnn', '$\hat{\mathbf{Z}}_{\rm rd}$, DnCNN'; ...
           DNCNN_NOAFM, 'fig_Z_rd_dncnn_noafm', '$\hat{\mathbf{Z}}_{\rm rd}$, DnCNN w/o AFM'};
    for q = 1:size(opt, 1)                                 % skipped if the checkpoint is missing
        if any(out.model_name == opt{q, 1})
            P(end+1) = struct('file', opt{q, 2}, 'title', opt{q, 3}, 'draw', fdb(den(opt{q, 1}))); %#ok<SAGROW>
        end
    end
    P(end+1) = struct('file', 'fig_Z_rd_alt',     'title', '$\mathbf{Z}_{\rm rd}''$',               'draw', fdb(Z_alt));
    P(end+1) = struct('file', 'fig_Z_rd_mix',     'title', '$\mathbf{Z}_{\rm rd}^{\rm mix}$',       'draw', fdb(out.Z_mix));
    P(end+1) = struct('file', 'fig_Z_rd_mix_den', 'title', '$f_{\rm den}^{\rm infer}(\mathbf{Z}_{\rm rd}^{\rm mix})$', 'draw', fdb(out.den_mix));
    P(end+1) = struct('file', 'fig_M_afm',  'title', '$\mathbf{M}_{\rm afm}$',  'draw', fmk(out.M_afm));
    P(end+1) = struct('file', 'fig_M_mask', 'title', '$\mathbf{M}_{\rm mask}$', 'draw', fmk(out.M_mask));
    P(end+1) = struct('file', 'fig_M_tars', 'title', '$\mathbf{M}_{\rm tars}$', 'draw', fmk(out.M_tars));

    for p = 1:numel(P)
        export_panel(P(p).draw, P(p).title, fullfile(fig_dir, [P(p).file sfx '.pdf']), PANEL_IN);
    end
    export_colorbar(cmap_db,   [-DB_RANGE 0], 'Power (dB)', fullfile(fig_dir, ['fig_colorbar_db' sfx '.pdf']));
    export_colorbar(cmap_mask, [-MASK_DB 0], 'Mask, $20\log_{10} M$ (dB)', fullfile(fig_dir, ['fig_colorbar_mask' sfx '.pdf']));
    export_preview({P.draw}, {P.title}, [4 4], fullfile(fig_dir, ['fig_preview' sfx '.png']), PANEL_IN);
end
fprintf('Panels written to %s\n', fig_dir);


% =====================================================================
%  LOCAL FUNCTIONS
% =====================================================================
function out = run_afm_maps(python_exe, repo_root, Z, Z_alt, Z_clean, truth, models, mask_model)
% Denoised maps and AFM quantities from alg_pdisac/scripts/infer/afm_maps_mat.py.
    if ~isfile(python_exe), error('main5d:noPython', 'Python executable not found: %s', python_exe); end
    script = fullfile(repo_root, "alg_pdisac", "scripts", "infer", "afm_maps_mat.py");
    tmp = string(tempname);  f_in = tmp + "_in.mat";  f_out = tmp + "_out.mat";
    Z_rd = single(Z);  Z_rd_alt = single(Z_alt);  Z_rd_prbs = single(Z_clean); %#ok<NASGU>
    save(f_in, 'Z_rd', 'Z_rd_alt', 'Z_rd_prbs', 'truth', '-v7');
    cleanup = onCleanup(@() delete_if(f_in, f_out)); %#ok<NASGU>
    cmd = sprintf('"%s" "%s" --input "%s" --output "%s" --models %s --mask-model %s', ...
        python_exe, script, f_in, f_out, strjoin(models, ','), mask_model);
    [st, msg] = system(cmd);
    if st ~= 0, error('main5d:python', 'afm_maps_mat.py failed (status %d):\n%s', st, msg); end
    out = load(f_out);
    out.model_name = string(out.model_name);
    out.coverage = double(out.coverage);
end

function delete_if(varargin)
    for i = 1:nargin, if isfile(varargin{i}), delete(varargin{i}); end, end
end

function draw_db(ax, Z, range_axis, vel_axis, R_SHOW, V_SHOW, DB_RANGE, cmap, truth)
    P = abs(Z).^2;  P = 10*log10(P / max(P(:)) + 1e-30);
    imagesc(ax, vel_axis, range_axis, P, [-DB_RANGE 0]);  colormap(ax, cmap);
    finish_map(ax, R_SHOW, V_SHOW, truth);
end

function draw_mask(ax, M, range_axis, vel_axis, R_SHOW, V_SHOW, MASK_DB, cmap, truth)
% masks in [0, 1] shown as 20 log10 M
    imagesc(ax, vel_axis, range_axis, 20*log10(max(double(M), 1e-12)), [-MASK_DB 0]);  colormap(ax, cmap);
    finish_map(ax, R_SHOW, V_SHOW, truth);
end

function finish_map(ax, R_SHOW, V_SHOW, truth)
% ground truth only (o), no detections
    MARKER_SIZE = 11;
    axis(ax, 'xy');  hold(ax, 'on');  box(ax, 'on');
    % open rings larger than a target cell, so the peak stays visible inside
    plot(ax, truth(:,2), truth(:,1), 'o', 'Color', [1 1 1], 'MarkerSize', MARKER_SIZE, 'LineWidth', 0.8);
    plot(ax, truth(:,2), truth(:,1), 'o', 'Color', [0 0 0], 'MarkerSize', MARKER_SIZE + 1.5, 'LineWidth', 0.4);
    xlim(ax, V_SHOW);  ylim(ax, R_SHOW);
    xlabel(ax, 'Velocity (m/s)');  ylabel(ax, 'Range (m)');
end

function RD = rd_map_ls(Data_Rx, Data_Tx, taps)
% LS cancellation of the static clutter, then the RD map of main4 (as main5c_bcrb_check):
% y[m] = sum_k h_k x[m-k] + (moving targets + noise), k = taps(1)..taps(2), over the whole
% frame; h by the normal equations R h = b (FFT), y_ls = y - x * h_hat.
    [N_block, N_chip, ~] = size(Data_Rx);
    x = reshape(Data_Tx(:, :, 1).', [], 1);
    y = reshape(Data_Rx(:, :, 1).', [], 1);
    N_fr = numel(x);
    lags = (taps(1):taps(2)).';
    Nfft = 2^nextpow2(2*N_fr + numel(lags));
    X = fft(x, Nfft);  Yf = fft(y, Nfft);
    cc = ifft(Yf .* conj(X));
    ac = real(ifft(abs(X).^2));
    b  = cc(mod(lags, Nfft) + 1);
    R  = toeplitz(ac(1:numel(lags)));
    h  = R \ b;
    hp = zeros(Nfft, 1);  hp(mod(lags, Nfft) + 1) = h;
    yc = ifft(X .* fft(hp));
    y_ls = y - yc(1:N_fr);
    Rx_mat = reshape(y_ls, N_chip, N_block);
    Tx_mat = reshape(x,    N_chip, N_block);
    mf = fft(Rx_mat) .* conj(fft(Tx_mat));
    mf_fft = fft(mf, [], 2);
    mf_fft(:, 1) = 0;
    RD = fftshift(ifft(mf_fft, [], 1), 2);
end

function export_panel(drawfun, ttl, file, panel_in)
    f = figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 panel_in], 'Visible', 'off');
    ax = axes(f);  drawfun(ax);
    title(ax, ttl, 'FontWeight', 'normal');
    exportgraphics(f, file, 'ContentType', 'vector');
    close(f);
end

function export_preview(draws, ttls, grid_rc, file, panel_in)
% All panels of one figure on a grid, for checking the layout only.
    f = figure('Color', 'w', 'Units', 'inches', 'Visible', 'off', ...
        'Position', [1 1 panel_in(1)*grid_rc(2) panel_in(2)*grid_rc(1)]);
    tl = tiledlayout(f, grid_rc(1), grid_rc(2), 'TileSpacing', 'compact', 'Padding', 'compact');
    for p = 1:numel(draws)
        ax = nexttile(tl);  draws{p}(ax);
        title(ax, sprintf('(%s) %s', char('a' + p - 1), ttls{p}), 'FontWeight', 'normal');
    end
    exportgraphics(f, file, 'Resolution', 150);
    close(f);
end

function export_colorbar(cmap, lims, label, file)
    f = figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 3.4 0.75], 'Visible', 'off');
    ax = axes(f, 'Visible', 'off');  colormap(ax, cmap);  caxis(ax, lims);
    cb = colorbar(ax, 'southoutside');
    cb.Units = 'normalized';  cb.Position = [0.06 0.55 0.88 0.22];
    cb.Label.String = label;  cb.Label.Interpreter = 'latex';  cb.TickLabelInterpreter = 'latex';
    exportgraphics(f, file, 'ContentType', 'vector');
    close(f);
end

function cmap = viridis_map(n)
% viridis colormap of matplotlib, 9 anchors interpolated
    a = [0.267 0.005 0.329; 0.279 0.175 0.483; 0.230 0.322 0.546; 0.173 0.449 0.558; ...
         0.128 0.567 0.551; 0.158 0.684 0.502; 0.369 0.789 0.383; 0.678 0.864 0.190; 0.993 0.906 0.144];
    cmap = interp1(linspace(0, 1, size(a, 1)), a, linspace(0, 1, n));
end

function pos = rand_pos_in_roi(roi, N)
% as generate_radar_dataset
    pos = zeros(3, N);
    for dim = 1:3
        lo = roi(dim,1);  hi = roi(dim,2);
        if lo == hi, pos(dim,:) = lo; else, pos(dim,:) = lo + (hi - lo) .* rand(1, N); end
    end
end
