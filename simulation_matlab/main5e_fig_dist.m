% MAIN5E_FIG_DIST — distributions of RDPDNet on one test scene, one 3-D surface PDF per panel
% -------------------------------------------------------------------------
% Same scene, SNR, symbols and noise as main5d (SEED, SCENE_IDX, SNR_SHOW, N_SYM), so the panels
% match the RD maps of main5d_fig_maps.  Hierarchy of Sec. IV-B (L levels, L = 3):
%   encoder (training)  p(Z_rd^prbs) -> q(z_1|Z_rd^prbs) -> q(z_2|z_1) -> ... -> q(z_L|z_{L-1})
%   inference           p(Z_rd) -> p(z_L|Z_rd) (extractor) -> p(z_{L-1}|z_L) -> ... -> p(z_1|z_2)
%                       -> p(Zhat_rd|z_1) = f_den^infer(Z_rd)
% Every latent is its mean (as f_den^infer); the inference chain is the extractor + decoder path,
% not the teacher-forced decoder of model.distribution().
% Surface height: dB relative to the panel's own peak, clipped at -DB_RANGE;
%   RD maps |Z|, latents RMS over the latent channels.
%
% Panels (fig_exported/fig_dist/<name>.pdf), titled in the notation of Sec. IV-B (dist_label):
%   p(Z_rd^prbs), q(z_1|Z_rd^prbs), q(z_l|z_{l-1}), p(Z_rd), p(z_L|Z_rd), p(z_l|z_{l+1}), p(Zhat_rd|z_1),
%   with z = \vec{\mathfrak{z}} (falls back to \vec{z} if MATLAB's LaTeX cannot render \mathfrak).
%   fig_dist_Z_prbs, fig_dist_q1 .. fig_dist_qL      encoder row
%   fig_dist_Z_rd, fig_dist_pL .. fig_dist_p1, fig_dist_Zhat   inference row
% fig_dist_colorbar.pdf: shared colour bar; fig_dist_preview.png: layout check only.
%
% Run standalone: main5e_fig_dist (settings below). Calls main1_matlab_config,
% main3_signal_channel_model, main4_sensing_process and, through
% PDISAC_cfg.inference.python_executable, the Python script
% alg_pdisac/scripts/infer/distribution_mat.py (MODEL of configs/inference.json, RDPDNet).
% -------------------------------------------------------------------------
clear; clc; close all;
script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(script_dir));
NO_SIGNAL_PLOT = true;  NO_BPSK_PLOT = true;  NO_RADAR_PLOT = true;  NO_CFAR_PLOT = true;

main1_matlab_config;

% =====================================================================
%  Settings (scene settings as main5d)
% =====================================================================
SNR_SHOW  = 20;                   % [dB], scalar
N_SYM     = 8;                  % N_sym_per_block (= N_bit^prbs) of the shown frame
SCENE_IDX = 1;                    % index of the test scene (as main5d)
SEED      = 5150000;             % scene / noise seed (as main5d)
MODEL     = "pdnet";           % name in alg_pdisac/configs/inference.json (RDPDNet)
R_SHOW    = "auto";               % displayed range [m]: [r_min r_max], "auto" = targets + ZOOM_MARGIN, "full"
V_SHOW    = "auto";               % displayed velocity [m/s], same options
ZOOM_MARGIN = [10 5];             % "auto" margin around the targets [m, m/s] (as main5d)
DB_RANGE  = 40;                   % surface height range [dB] below each panel's peak
VIEW      = [-37.5 30];           % [azimuth elevation] of the 3-D view
CMAP      = "viridis";             % "parula" or "viridis" (as main5d)
PANEL_IN  = [2.4 2.1];            % panel size [in]
FONT      = 9;
Z_SYM     = '\vec{\mathfrak{z}}';  % latent symbol of the manuscript (fallback '\vec{z}', see latex_ok)

fig_dir = fullfile(script_dir, 'fig_exported', 'fig_dist');
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
if CMAP == "parula", cmap = parula(256); else, cmap = viridis_map(256); end

% =====================================================================
%  Scene (as main5d / generate_radar_dataset)
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

rel     = Tars_position - Tx_position;
truth_r = sqrt(sum(rel.^2, 1));
truth_v = sum(Tars_vel .* rel ./ (truth_r + eps), 1);
R_SHOW  = show_window(R_SHOW, truth_r, ZOOM_MARGIN(1), 10, [0 range_axis(end)], true);
V_SHOW  = show_window(V_SHOW, truth_v, ZOOM_MARGIN(2), 5,  [min(vel_axis) max(vel_axis)], false);
fprintf('*** main5e | test scene %d | %d targets | N_sym = %d | SNR = %d dB | %s\n', ...
    scene_idx, N_tars, N_SYM, SNR_SHOW, MODEL);

N_sym_per_block = N_SYM;
L_slot_per_sym  = N_chip / (2 * N_sym_per_block);
N_slot          = N_chip / L_slot_per_sym;
P_tx = Noise_power_sen * 10^(SNR_SHOW/10);
rng(SEED + 1000*scene_idx + 1, 'twister');                % as main5d: same symbols and noise
main3_signal_channel_model;  main4_sensing_process;        % second realisation of main5d (not used)
main3_signal_channel_model;  main4_sensing_process;        % primary realisation (as main5d)
Z       = RD_map_shifted;                                  % with data, with noise
Z_clean = RD_map_no_noise_shifted;                         % Z_rd^prbs

% =====================================================================
%  Distributions (Python) and panels
% =====================================================================
out  = run_distribution(python_exe, PDISAC_repo_root, Z, Z_clean, MODEL);
keys = string(out.keys);
L    = double(out.level);
fprintf('  %s (%s, L = %d): %s\n', MODEL, string(out.version), L, strjoin(keys, ', '));
if ~latex_ok(['$' Z_SYM '_1$'])
    fprintf('  MATLAB LaTeX cannot render %s; latent symbol shown as \\vec{z}\n', Z_SYM);
    Z_SYM = '\vec{z}';
end
lbl = strings(size(keys));
for k = 1:numel(keys), lbl(k) = dist_label(keys(k), L, string(out.version), Z_SYM); end
ir = range_axis >= R_SHOW(1) & range_axis <= R_SHOW(2);
iv = vel_axis   >= V_SHOW(1) & vel_axis   <= V_SHOW(2);
for k = 1:numel(keys)
    M = squeeze(out.maps(k, :, :));
    export_surface(M(ir, iv), range_axis(ir), vel_axis(iv), DB_RANGE, VIEW, cmap, ...
        lbl(k), fullfile(fig_dir, "fig_dist_" + keys(k) + ".pdf"), PANEL_IN);
end
export_colorbar(cmap, [-DB_RANGE 0], 'Magnitude (dB)', fullfile(fig_dir, 'fig_dist_colorbar.pdf'));

% preview: encoder row on top, inference row below (same order as the hierarchy)
top = ["Z_prbs", "q" + (1:L)];
bot = ["Z_rd", "p" + (L:-1:1), "Zhat"];
f = figure('Color', 'w', 'Units', 'inches', 'Visible', 'off', ...
    'Position', [1 1 PANEL_IN(1)*numel(bot) PANEL_IN(2)*2]);
tl = tiledlayout(f, 2, numel(bot), 'TileSpacing', 'loose', 'Padding', 'compact');
rows = {top, bot};
for r = 1:2
    for j = 1:numel(rows{r})
        k = find(keys == rows{r}(j), 1);
        ax = nexttile(tl, (r-1)*numel(bot) + j);
        draw_surface(ax, squeeze(out.maps(k, ir, iv)), range_axis(ir), vel_axis(iv), DB_RANGE, VIEW, cmap);
        title(ax, lbl(k), 'Interpreter', 'latex', 'FontWeight', 'normal');
    end
end
exportgraphics(f, fullfile(fig_dir, 'fig_dist_preview.png'), 'Resolution', 150);
close(f);
fprintf('Panels written to %s\n', fig_dir);


% =====================================================================
%  LOCAL FUNCTIONS
% =====================================================================
function out = run_distribution(python_exe, repo_root, Z, Z_clean, model)
% Encoder / inference distributions from alg_pdisac/scripts/infer/distribution_mat.py.
    if ~isfile(python_exe), error('main5e:noPython', 'Python executable not found: %s', python_exe); end
    script = fullfile(repo_root, "alg_pdisac", "scripts", "infer", "distribution_mat.py");
    tmp = string(tempname);  f_in = tmp + "_in.mat";  f_out = tmp + "_out.mat";
    Z_rd = single(Z);  Z_rd_prbs = single(Z_clean); %#ok<NASGU>
    save(f_in, 'Z_rd', 'Z_rd_prbs', '-v7');
    cleanup = onCleanup(@() delete_if(f_in, f_out)); %#ok<NASGU>
    cmd = sprintf('"%s" "%s" --input "%s" --output "%s" --model %s', python_exe, script, f_in, f_out, model);
    [st, msg] = system(cmd);
    if st ~= 0, error('main5e:python', 'distribution_mat.py failed (status %d):\n%s', st, msg); end
    out = load(f_out);
end

function delete_if(varargin)
    for i = 1:nargin, if isfile(varargin{i}), delete(varargin{i}); end, end
end

function w = show_window(w, truth, margin, step, full, nonneg)
% "auto": targets +- margin, rounded outward to step (as main5d); "full": whole axis; else as given.
    if isstring(w) || ischar(w)
        if string(w) == "full"
            w = full;
        else
            w = [step*floor((min(truth) - margin)/step), step*ceil((max(truth) + margin)/step)];
            if nonneg, w(1) = max(0, w(1)); end
        end
    end
end

function draw_surface(ax, M, r, v, db_range, vw, cmap)
% dB relative to the panel's own peak, clipped at -db_range
    P = 20*log10(double(M) / max(M(:)) + 1e-30);
    P = max(P, -db_range);
    surf(ax, v, r, P, 'EdgeColor', 'none');
    colormap(ax, cmap);  caxis(ax, [-db_range 0]);
    xlim(ax, [min(v) max(v)]);  ylim(ax, [min(r) max(r)]);  zlim(ax, [-db_range 0]);   % vel_axis is decreasing
    view(ax, vw(1), vw(2));  grid(ax, 'on');  box(ax, 'off');
    % about N_TICK ticks per horizontal axis (1-2-5 steps) and short labels, so they do not overlap
    N_TICK = 3;
    xticks(ax, nice_ticks(v, N_TICK));  yticks(ax, nice_ticks(r, N_TICK));  zticks(ax, [-db_range, -db_range/2, 0]);
    xlabel(ax, '$v$ (m/s)');  ylabel(ax, '$r$ (m)');  zlabel(ax, 'dB');
end

function t = nice_ticks(x, n)
% about n ticks over [min(x) max(x)] with a 1-2-5 step
    lo = min(x);  hi = max(x);
    raw = (hi - lo) / n;  e = 10^floor(log10(raw));
    steps = [1 2 5 10] * e;  step = steps(find(steps >= raw, 1));
    t = ceil(lo/step)*step : step : floor(hi/step)*step;
end

function export_surface(M, r, v, db_range, vw, cmap, ttl, file, panel_in)
    f = figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 panel_in], 'Visible', 'off');
    ax = axes(f);
    draw_surface(ax, M, r, v, db_range, vw, cmap);
    title(ax, ttl, 'Interpreter', 'latex', 'FontWeight', 'normal');
    if numel(M) <= 5e4                                   % large surfaces as a 300-dpi image
        exportgraphics(f, file, 'ContentType', 'vector');
    else
        exportgraphics(f, file, 'ContentType', 'image', 'Resolution', 300);
    end
    close(f);
end

function s = dist_label(key, L, version, z)
% Panel label in the notation of Sec. IV-B (encoder q, inference p); z = latent symbol.
    zl = @(l) sprintf('%s_{%d}', z, l);
    z0 = '\mathbf{Z}_{\rm rd}^{\rm prbs}';
    switch key
        case "Z_prbs", s = '$p(\mathbf{Z}_{\rm rd}^{\rm prbs})$';
        case "Z_rd",   s = '$p(\mathbf{Z}_{\rm rd})$';
        case "Zhat",   s = sprintf('$p(\\hat{\\mathbf{Z}}_{\\rm rd}|%s)$', zl(1));
        otherwise
            l = str2double(extractAfter(key, 1));
            if startsWith(key, "q")
                if l == 1, s = sprintf('$q(%s|%s)$', zl(1), z0);
                else,      s = sprintf('$q(%s|%s)$', zl(l), zl(l-1)); end
            else
                if l == L, s = sprintf('$p(%s|\\mathbf{Z}_{\\rm rd})$', zl(L));
                else,      s = sprintf('$p(%s|%s)$', zl(l), zl(l+1)); end
            end
    end
    s = string(s);
end

function ok = latex_ok(s)
% true if MATLAB's LaTeX interpreter renders s without a syntax warning
    f = figure('Visible', 'off');  ax = axes(f);
    lastwarn('');
    text(ax, 0.5, 0.5, s, 'Interpreter', 'latex');  drawnow;
    ok = isempty(lastwarn);
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
% viridis (as main5d), 9 anchors interpolated
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
