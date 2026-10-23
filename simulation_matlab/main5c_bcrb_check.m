% MAIN5C_BCRB_CHECK  BCRB of Proposition 1 (Sec. IV-D) vs. the evaluated estimation chains
% -------------------------------------------------------------------------
% Computes the Bayesian CRB of the range and radial velocity of N targets and compares
% it with the RMSE of the conventional chain of Sec. III (CA-CFAR + association +
% sub-bin read-out) on the data-bearing map, the data-free map, the data map after LS
% clutter cancellation and, optionally, the maps denoised by RDPDNet / DnCNN.
% Runs main2_topology (which runs main1_matlab_config), then main3 + main4 per scene and SNR.
%
% Evaluation scenes are drawn from the prior of Proposition 1:
%   r_m ~ p_rc(r; r_bar_m, L_r),  v_m ~ p_rc(v; v_bar_m, L_v)   (raised-cosine prior),
% with disjoint range windows W_m; azimuth is drawn independently of (r_m, v_m), and
% the gain magnitude |alpha_m| = sqrt(P_tx G_tx A_e RCS_m) follows the raised cosine on
% [|alpha|_min, |alpha|_max] (from target_rcs_range), so p(Theta) = p(Psi) p(|alpha|) (2 pi)^-N.
% Adjacent prior supports are separated by more than the CA-CFAR window (N_g + N_t bins)
% in range or Doppler, so the local-maximum selection cannot mask a target (assert).
% The last target is the UE (main3 needs a UE). Velocities are radial.
%
% Chains (same scenes, common random numbers over SNR):
%   data   : CA-CFAR on RD_map_shifted               (data-carrying map)
%   nodata : CA-CFAR on RD_map_noise_no_data_shifted (data-free MLS map)
%   data_ls: CA-CFAR on the data map after LS clutter cancellation (LS_CANCEL)
%   association of Sec. IV-D, with the window taken as the prior support in
%   range AND velocity: the strongest detection whose (range, velocity) cell lies
%   in [r_bar_m +- L_r/2] x [v_bar_m +- L_v/2] (widened by half a bin) is target m;
%   if none, the fallback (r_bar_m, v_bar_m). Then sub-bin read-out (READOUT).
%   Reported: sqrt(E[RMSE^2]) (what Proposition 1 bounds).
%
% Bound, per scene, i.e. per (alpha, D) with D the transmitted chip matrix:
%   G from the frame-level band-limited delayed chip stream and its tau-derivative,
%   projector Pi_A per fast-time frequency bin (general form, covers d_hat = 0),
%   Theta = (r_m, v_m, |alpha_m|, phi_m)_m, phi_m = arg alpha_m;
%   F = (2/sigma^2) Re{G^H Pi_A G} (columns: d/dr, d/dv, echo/|alpha|, j*echo),
%   J_B = E_Theta[F] + blkdiag(4 pi^2/L^2 for Psi, 4 pi^2/L_a^2 for |alpha|, 0 for phi),
%   E_Psi by K_PRIOR prior samples of (Psi, phi); E over |alpha| in closed form (each column of G is
%   |alpha| times a unit-gain column, except the |alpha| column); bound = Psi block of
%   J_B^{-1} (Schur complement over the unknown nuisance), averaged over the scenes (D draws).
%   Variants (flags; Proposition 1 is KNOW_MAG = KNOW_PHASE = false):
%     KNOW_MAG   true : |alpha_m| known (this scene's value), no |alpha| column
%     KNOW_PHASE true : phi_m known, no phase column
%   F is linear in SNR_Tx, so it is computed once per scene and scaled.
%   Raw-data bound (chain data+LS): same with F = (2/sigma^2) Re{G^H G} (no Pi_A), since the
%   LS canceller acts on Y_sen and not on z; it is <= the Pi_A bound (check (ii)).
%
% Checks printed:
%   (i)   echo model: alpha*S_m vs. h_sen_tars * func_get_x_delay_frac (relative error)
%   (ii)  F with Pi_A never exceeds F without it (min eigenvalue of the difference)
%   (iii) BCRB vs. its per-target closed form (unknown phase; known-phase 2x2 form
%         if KNOW_PHASE) -- ratio should be >= ~1
%   (iv)  low SNR: bound -> L/(2 pi);  fallback (r_bar, v_bar) RMSE -> L*sqrt(1/12-1/(2 pi^2))
%   (v)   every chain >= its bound (data, nodata: Pi_A bound; data+LS: raw-data bound)
%
% Flags (Settings section below):
%   SMOKE      true        : 2 scenes, SNR [0 20] dB, K_PRIOR = 2 (quick check)
%   SNR_DIAG   [..]        : run only these SNRs [dB] (diagnosis)
%   READOUT    "sinc"      : func_sinc_interpolation (sinc kernel, +-16-bin window)
%              "dirichlet" : exact trigonometric (Dirichlet-kernel) interpolation of the
%                            finite DFTs, alternating range / Doppler maximisation; for one
%                            target in white noise this is the periodogram maximiser (ML)
%   NO_CLUTTER true        : scatterer RCS set to 0 (tests clutter leakage; bound unchanged)
%   KNOW_MAG   true        : gain magnitude known to the Rx (default false: nuisance)
%   KNOW_PHASE true        : gain phase known to the Rx (default false: nuisance)
%   LS_CANCEL  true        : add a third chain, "data + LS": the static clutter channel
%                            h (taps LS_TAPS, chip spacing) is estimated by least squares
%                            from the whole received frame, y = x * h + (targets + noise),
%                            with x the known transmitted chip stream (D_isac); x * h_hat is
%                            subtracted before the MF / Doppler FFT / zero-Doppler removal of
%                            main4. The clutter is NOT assumed known: only that it is static
%                            and the transmitted stream are known.
%   USE_DENOISER true      : also denoise the data and data-free maps with every model of
%                            alg_pdisac/configs/inference.json (inference_run_pdnet, one
%                            Python call per scene) and run the same CA-CFAR / association /
%                            read-out: chains <model>_data and <model>_nodata; tag '_ml'.
%                            The baseline chains are unchanged (Python uses no MATLAB RNG).
%   DENOISER_LS_MODELS     : models also applied to the data + LS map (needs LS_CANCEL):
%                            chains <model>_data_ls, a function of vec(Y_sen) like data+LS,
%                            hence checked against the raw-data bound. The models are trained
%                            on maps without LS, so this chain is a mismatched input for them.
%   PLOT_RD    true        : RD maps of scene 1 at SNR = PLOT_RD_SNR for the three chains
%                            (with data, with data + LS, without data), common dB scale,
%                            true targets (o) and prior windows (dashed)
%   NBIT_EXT (workspace variable, optional): runs the conventional chains only (data,
%                            no data, and data + LS with LS_CANCEL) and the bound with
%                            N_bit^prbs = NBIT_EXT instead of main1; USE_DENOISER and
%                            PLOT_RD are forced off; tag '_nbit<NBIT_EXT>'. Sweep from
%                            the Command Window:
%                                for NBIT_EXT = [4 16 64 128], main5c_bcrb_check; end
%   Settings of the evaluation scenes: N_SCENE, K_PRIOR, R_BAR / V_BAR / L_R / L_V (prior
%   windows), AZ_DEG, SEED; system: sensing_statistics_cfg of main1 (as main5a).
%
% Outputs (in simulation_matlab):
%   exported_statistics/statistics_bcrb_check<tag>.csv  RMSE, BCRB, closed form and
%                                                        fallback fraction per SNR and chain
%   exported_statistics/estimates_bcrb_check<tag>.mat   per-target estimates and true values
%   fig_exported/fig_bcrb_check<tag>.(fig|pdf)          (a) range, (b) velocity RMSE vs. SNR_Tx
%   fig_exported/fig_bcrb_rd_maps<tag>.(fig|pdf)        RD maps of scene 1 (PLOT_RD)
%   <tag>: '' for READOUT = "sinc", with clutter, no LS chain, no denoisers; the flags
%   append, in this order, '_dirichlet', '_noclutter', '_ls', '_knowmag', '_knowphase',
%   '_smoke', '_diag', '_ml', '_nbit<NBIT_EXT>'.
% -------------------------------------------------------------------------
clearvars -except NBIT_EXT; clc; close all;     % NBIT_EXT: optional N_bit^prbs sweep (header)

fig_dir  = fullfile(fileparts(mfilename('fullpath')), 'fig_exported');
stat_dir = fullfile(fileparts(mfilename('fullpath')), 'exported_statistics');
if ~exist(fig_dir,'dir'),  mkdir(fig_dir);  end
if ~exist(stat_dir,'dir'), mkdir(stat_dir); end

NO_SIGNAL_PLOT = true;  NO_RADAR_PLOT = true;  NO_CFAR_PLOT = true;
rng("default");

% =====================================================================
%  Settings
% =====================================================================
SMOKE    = false;                 % true: 2 scenes, SNR [0 20], K_PRIOR = 2
SNR_DIAG = [];                    % e.g. [20 30]: run only these SNRs (diagnosis, tag '_diag')
READOUT  = "dirichlet";           % "sinc" or "dirichlet" (see header)
NO_CLUTTER = false;               % true: no stationary scatterers (RCS = 0)
KNOW_MAG   = false;               % true: |alpha_m| known to the Rx (false: nuisance, as Proposition 1)
KNOW_PHASE = false;               % true: arg alpha_m known to the Rx (false: nuisance, as Proposition 1)
LS_CANCEL  = true;                % true: add the chain "data + LS clutter cancellation"
PLOT_RD    = true;                % true: RD maps of scene 1 (data, data + LS, no data)
PLOT_RD_SNR = 20;                 % SNR [dB] of those maps (must be in SNR_db)
LS_TAPS    = [-64 256];           % static-clutter taps [chips]: clutter delays <= ~141 chips; the
                                  % extra taps hold the 1/k sinc tails of fractional delays
                                  % (clutter residual about -36 dB, vs. -31 dB with [-16 160])
USE_DENOISER   = true;           % true: add the denoiser chains (Python, configs/inference.json)
DENOISER_MODELS = "pdnet,pdnet_noafm,dncnn,dncnn_noafm";             % "" = every model of inference.json with a checkpoint, or "pdnet,dncnn"
DENOISER_LS_MODELS = "pdnet,dncnn";  % subset of DENOISER_MODELS also fed the data + LS map ("" = none)
SEED     = 5150000;
N_SCENE  = 150;                    % scenes (alpha, D, clutter, Psi) per SNR
K_PRIOR  = 16;                    % prior samples of (Psi, phi) per scene for E[F]
R_BAR    = [25 40 55 70 85];      % window centres [m]   (disjoint, L_R = 10 m)
V_BAR    = [-15 15 -35 35 -55];   % window centres [m/s]: away from zero Doppler; range-adjacent
                                  % windows >= 20 m/s (9 Doppler bins) apart (no CFAR masking);
                                  % ALL windows disjoint in Doppler (gap >= 10 m/s): the augmented
                                  % MLS (511 + 1 chip) has periodic sidelobes up to -24 dB, which
                                  % spread along the Doppler column of a strong target and would
                                  % hit the window of a weak target at the same velocity
L_R      = 10 * ones(1, numel(R_BAR));
L_V      = 10 * ones(1, numel(V_BAR));
AZ_DEG   = [-60 60];              % azimuth drawn uniformly, independent of r
N_PAD    = 2;                     % zero padding of the frame, in blocks, each side

% =====================================================================
%  System (sensing_statistics_cfg overrides of main1, as main5a)
% =====================================================================
TOPO_NO_PLOT = true; main2_topology;          % config, scatterers, N_ref_tars, ...
if exist('NBIT_EXT', 'var')                   % N_bit^prbs sweep: conventional chains only
    assert(mod(N_chip, 2*NBIT_EXT) == 0, 'N_chip must be a multiple of 2*NBIT_EXT');
    N_sym_per_block = NBIT_EXT;                % overrides main1 (used by main3)
    L_slot_per_sym  = N_chip / (2 * N_sym_per_block);
    N_slot          = N_chip / L_slot_per_sym;
    USE_DENOISER = false;  PLOT_RD = false;
end
G_tx = 10^(sensing_statistics_cfg.system_override.gain_tx_db/10);
G_rx = 10^(sensing_statistics_cfg.system_override.gain_rx_db/10);
G_ue = 10^(sensing_statistics_cfg.system_override.gain_ue_db/10);
N_F_com = 10^(sensing_statistics_cfg.system_override.noise_figure_com_db/10);
Noise_power_com = physconst('Boltzmann')*T_ref*N_F_com*B;
SNR_db = sensing_statistics_cfg.snr_db(:).';
if SMOKE, N_SCENE = 2; K_PRIOR = 2; SNR_db = [0 20]; end
if ~isempty(SNR_DIAG), SNR_db = SNR_DIAG; end
SNR_lin = 10.^(SNR_db/10);
n_snr = numel(SNR_db);
tag = '';                                         % output-file suffix from the flags
if READOUT == "dirichlet", tag = [tag '_dirichlet']; end
if NO_CLUTTER,             tag = [tag '_noclutter']; end
if LS_CANCEL,              tag = [tag '_ls'];        end
if KNOW_MAG,               tag = [tag '_knowmag'];   end
if KNOW_PHASE,             tag = [tag '_knowphase']; end
if SMOKE,                  tag = [tag '_smoke'];     end
if ~isempty(SNR_DIAG),     tag = [tag '_diag'];      end
if USE_DENOISER,           tag = [tag '_ml'];        end
if exist('NBIT_EXT', 'var'), tag = [tag sprintf('_nbit%d', NBIT_EXT)]; end
den_ls = USE_DENOISER && LS_CANCEL && strlength(DENOISER_LS_MODELS) > 0;   % denoisers on the data + LS map
n_grp  = 2 + den_ls;                              % input maps per SNR: data, data-free (, data + LS)

N_TARS = numel(R_BAR);
assert(all(diff(R_BAR) > (L_R(1:end-1) + L_R(2:end))/2), 'range windows must be disjoint');
assert(all(abs(V_BAR) > L_V/2), 'velocity windows must exclude zero Doppler');
eta   = 4*pi/Lambda;
A_e   = G_rx * Lambda^2 / (4*pi);
t_i   = (1:N_block) * T_pmcw;                 % slow-time instants (h_sen_tars: T_step = i*T_pmcw)
N_fr  = N_chip * N_block;
range_res  = c / (2*B);
range_axis = (0:N_chip-1) * range_res;
PRF        = 1 / T_pmcw;
fd_axis    = (-N_block/2 : N_block/2-1) * (PRF/N_block);
vel_axis   = -(Lambda/2) * fd_axis;
JP = reshape([4*pi^2./L_R.^2; 4*pi^2./L_V.^2], [], 1);   % prior information, order (r1,v1,r2,v2,...)

% supports must not fall inside each other's CA-CFAR window in BOTH axes (peak selection)
dv_bin = abs(vel_axis(2) - vel_axis(1));
for m = 1:N_TARS
    for n = m+1:N_TARS
        gap_r = (abs(R_BAR(m) - R_BAR(n)) - (L_R(m) + L_R(n))/2) / range_res;
        gap_v = (abs(V_BAR(m) - V_BAR(n)) - (L_V(m) + L_V(n))/2) / dv_bin;
        assert(gap_r > N_guard_range + N_train_range || gap_v > N_guard_doppler + N_train_doppler, ...
            'prior supports %d and %d can mask each other (CA-CFAR peak selection)', m, n);
        assert(gap_v > 2, 'Doppler windows %d and %d must be disjoint (MLS sidelobes along Doppler columns)', m, n);
    end
end

idx_r   = 4*(1:N_TARS) - 3;  idx_v = 4*(1:N_TARS) - 2;
idx_a   = 4*(1:N_TARS) - 1;  idx_phi = 4*(1:N_TARS);
idx_psi = sort([idx_r idx_v]);                            % (r1,v1,r2,v2,...) inside the 4N ordering
idx_keep = idx_psi;                                       % rows/columns of the bound's Theta
if ~KNOW_MAG,   idx_keep = [idx_keep idx_a];   end
if ~KNOW_PHASE, idx_keep = [idx_keep idx_phi]; end
idx_keep = sort(idx_keep);
pos_psi  = ismember(idx_keep, idx_psi);                   % Psi entries inside idx_keep (same order)

% gain prior: a_m = |alpha_m|/sqrt(P_tx) = sqrt(G_tx A_e RCS_m) ~ raised cosine on [a_min, a_max]
A_LIM = sqrt(G_tx * A_e * PDISAC_cfg.scene.target_rcs_range);
A_BAR = mean(A_LIM);  L_A = diff(A_LIM);                  % window of a (normalised |alpha|)
JA    = 4*pi^2 / L_A^2;                                   % prior information of a_m
uq  = linspace(-pi/2, pi/2, 4001);  wq = (2/pi) * cos(uq).^2;
aq  = A_BAR + L_A * uq / pi;
Ea  = trapz(uq, wq .* aq);  Ea2 = trapz(uq, wq .* aq.^2);
% E[c_i c_j]: column c of target m is a_m for (r, v, phi) and 1 for |alpha| (d(a S)/da = S)
Ec  = [Ea Ea 1 Ea];
Ms  = [Ea2 Ea2 Ea Ea2; Ea2 Ea2 Ea Ea2; Ea Ea 1 Ea; Ea2 Ea2 Ea Ea2];
Mw  = kron(ones(N_TARS), Ec' * Ec) + kron(eye(N_TARS), Ms - Ec' * Ec);

scat_rcs_db   = 10*log10(PDISAC_cfg.scene.scatterer_rcs_range);

% =====================================================================
%  Storage
% =====================================================================
chains  = ["data", "nodata", "data_ls"];   % chain 3 only if LS_CANCEL
se_r    = nan(N_SCENE, n_snr, 3);      % per-scene RMSE_r^2 (mean over targets)
se_v    = nan(N_SCENE, n_snr, 3);
n_fb    = zeros(n_snr, 3);             % number of fallbacks (no detection in W_m)
bu_r    = nan(N_SCENE, n_snr, 3);  bu_v = bu_r;   % bound (per scene, mean over m): data, nodata, raw data
bc_r    = nan(N_SCENE, n_snr);     bc_v = bc_r;    % closed form (cyclic)
min_eig_check = inf;
est_r = nan(N_SCENE, n_snr, 3, N_TARS);  est_v = est_r;   % per-target estimates (diagnosis)
tru_r = nan(N_SCENE, N_TARS);  tru_v = tru_r;  tru_a = tru_r;
cfar  = struct('Ng_r', N_guard_range, 'Ng_d', N_guard_doppler, 'Nt_r', N_train_range, ...
               'Nt_d', N_train_doppler, 'P_fa', P_fa, 'peak_select', peak_select);
ml_names = strings(1, 0);  ml_meta = struct();            % denoisers (USE_DENOISER)

fprintf('*** main5c | %d scenes x %d SNR, K_PRIOR = %d, %d targets | READOUT = %s, NO_CLUTTER = %d\n', ...
    N_SCENE, n_snr, K_PRIOR, N_TARS, READOUT, NO_CLUTTER);

for s = 1:N_SCENE
    % ---------------- scene draw (fixed over SNR) ----------------
    rng(SEED + s, 'twister');
    r_true = R_BAR + L_R .* rc_sample(N_TARS) / pi;
    v_true = V_BAR + L_V .* rc_sample(N_TARS) / pi;
    az     = deg2rad(AZ_DEG(1) + diff(AZ_DEG) * rand(1, N_TARS));
    u_hat  = [cos(az); sin(az); zeros(1, N_TARS)];
    Tars_position = Tx_position + u_hat .* r_true;
    Tars_vel      = u_hat .* v_true;                  % radial only: v_m = <vel, u_hat>
    Tars_rcs      = (A_BAR + L_A * rc_sample(N_TARS) / pi).^2 / (G_tx * A_e);   % prior of |alpha|
    N_tars = N_TARS;
    UE_position = Tars_position(:, end);   UE_vel = Tars_vel(:, end);
    UE_motion   = phased.Platform('InitialPosition', UE_position, 'Velocity', UE_vel);
    Tars_motion = phased.Platform('InitialPosition', Tars_position, 'Velocity', Tars_vel);
    Scats_position = func_generate_static_scatterers(N_scats, Region_of_interest);
    Scats_vel = zeros(size(Scats_position));
    Scats_rcs = 10.^((scat_rcs_db(1) + diff(scat_rcs_db) * rand(1, N_scats)) / 10);
    if NO_CLUTTER, Scats_rcs(:) = 0; end            % same random stream, no clutter echo
    w_amp = sqrt(Tars_rcs * G_tx * A_e);              % |alpha_m| / sqrt(P_tx)
    tru_r(s, :) = r_true;  tru_v(s, :) = v_true;  tru_a(s, :) = w_amp;
    if USE_DENOISER, RD_in = complex(zeros(n_grp*n_snr, N_chip, N_block, 'single')); end

    for is = 1:n_snr
        P_tx = Noise_power_sen * SNR_lin(is);
        rng(SEED + 1e5 + s, 'twister');               % same data and noise at every SNR
        main3_signal_channel_model;
        main4_sensing_process;

        % ---------------- bound, once per scene ----------------
        if is == 1
            D_data   = reshape(Data_Tx(:, :, 1), N_block, N_chip);
            D_nodata = reshape(Data_Tx_no_data(:, :, 1), N_block, N_chip);
            if s == 1
                check_echo(D_data, Data_Tx, r_true, v_true, Tars_position, Tars_rcs, Tars_vel, ...
                    w_amp * sqrt(P_tx), Tx_position, N_tx_ant, Rx_position, N_rx_ant, P_tx, ...
                    G_tx, G_rx, Lambda, fc, T_chip, T_pmcw, eta, c, N_PAD);
            end
            Fbar = zeros(4*N_TARS, 4*N_TARS, 3);     % data, nodata, raw data (no Pi_A)
            for k = 1:K_PRIOR
                r_k = R_BAR + L_R .* rc_sample(N_TARS) / pi;
                v_k = V_BAR + L_V .* rc_sample(N_TARS) / pi;
                p_k = 2*pi * rand(1, N_TARS);            % phi ~ U[0, 2pi)
                for ch = 1:2
                    if ch == 1, D = D_data; else, D = D_nodata; end
                    [Fk, F_noPi] = fim_unit(D, r_k, v_k, p_k, T_chip, t_i, eta, c, N_PAD);
                    Fbar(:, :, ch) = Fbar(:, :, ch) + Fk / K_PRIOR;
                    if ch == 1, Fbar(:, :, 3) = Fbar(:, :, 3) + F_noPi / K_PRIOR; end
                    min_eig_check = min(min_eig_check, min(eig((F_noPi - Fk + (F_noPi - Fk)')/2)) ...
                                                       / max(eig(F_noPi)));
                end
            end
            for js = 1:n_snr
                for ch = 1:3
                    if KNOW_MAG                       % this scene's |alpha| (a column unused)
                        W = kron(diag(w_amp), diag([1 1 0 1]));
                        J = SNR_lin(js) * (W * Fbar(:, :, ch) * W);
                    else                              % E over the prior of |alpha|
                        J = SNR_lin(js) * (Mw .* Fbar(:, :, ch));
                        J(idx_a, idx_a) = J(idx_a, idx_a) + JA * eye(N_TARS);
                    end
                    J(idx_psi, idx_psi) = J(idx_psi, idx_psi) + diag(JP);
                    Ji = spd_inv(J(idx_keep, idx_keep));
                    du = diag(Ji(pos_psi, pos_psi));      % Psi block = Schur over the nuisance
                    bu_r(s, js, ch) = mean(du(1:2:end));  bu_v(s, js, ch) = mean(du(2:2:end));
                end
                if KNOW_MAG, g = SNR_lin(js) * w_amp.^2; else, g = SNR_lin(js) * Ea2 * ones(1, N_TARS); end
                [cr, cv] = closed_form(g, R_BAR, L_R, L_V, N_fr, eta, B, c, t_i, KNOW_PHASE);
                bc_r(s, js) = mean(cr);  bc_v(s, js) = mean(cv);
            end
        end

        % ---------------- conventional chains ----------------
        maps = {RD_map_shifted, RD_map_noise_no_data_shifted};
        if LS_CANCEL
            maps{3} = rd_map_ls(Data_Rx, Data_Tx, LS_TAPS);
        end
        for ch = 1:numel(maps)
            [r_hat, v_hat, nf] = estimate_targets(maps{ch}, cfar, R_BAR, V_BAR, L_R, L_V, ...
                range_axis, vel_axis, range_res, READOUT);
            n_fb(is, ch) = n_fb(is, ch) + nf;
            se_r(s, is, ch) = mean((r_hat - r_true).^2);
            se_v(s, is, ch) = mean((v_hat - v_true).^2);
            est_r(s, is, ch, :) = r_hat;  est_v(s, is, ch, :) = v_hat;
        end
        if USE_DENOISER                               % kept for one Python call per scene
            RD_in(is, :, :)         = reshape(RD_map_shifted, [1 N_chip N_block]);
            RD_in(n_snr + is, :, :) = reshape(RD_map_noise_no_data_shifted, [1 N_chip N_block]);
            if den_ls, RD_in(2*n_snr + is, :, :) = reshape(maps{3}, [1 N_chip N_block]); end
        end
        % RD maps of scene 1 at PLOT_RD_SNR (the LS map is computed here if LS_CANCEL = false)
        if PLOT_RD && s == 1 && SNR_db(is) == PLOT_RD_SNR
            if numel(maps) >= 3, RD_ls = maps{3}; else, RD_ls = rd_map_ls(Data_Rx, Data_Tx, LS_TAPS); end
            plot_rd_maps({RD_map_shifted, RD_ls, RD_map_noise_no_data_shifted}, ...
                {'With data', 'With data + LS clutter cancellation', 'Without data'}, ...
                range_axis, vel_axis, r_true, v_true, R_BAR, L_R, V_BAR, L_V, PLOT_RD_SNR, ...
                fullfile(fig_dir, ['fig_bcrb_rd_maps' tag]));
        end
        % data / data+LS / nodata = this scene's RMSE (mean over targets); BCRB = this scene's
        % bound (data map), i.e. E over the prior for its D; data+LS is NaN when LS_CANCEL = false
        fprintf(['  scene %3d/%d | SNR %+3d dB | RMSE_r data %.4f / data+LS %.4f / nodata %.4f / BCRB %.4f m', ...
                 ' | RMSE_v data %.4f / data+LS %.4f / nodata %.4f / BCRB %.4f m/s\n'], ...
            s, N_SCENE, SNR_db(is), sqrt(se_r(s,is,1)), sqrt(se_r(s,is,3)), sqrt(se_r(s,is,2)), ...
            sqrt(bu_r(s,is,1)), sqrt(se_v(s,is,1)), sqrt(se_v(s,is,3)), sqrt(se_v(s,is,2)), sqrt(bu_v(s,is,1)));
    end

    % ---------------- denoisers: data and data-free maps of all SNRs, one Python call ----------------
    if USE_DENOISER
        [RD_den, meta] = inference_run_pdnet(RD_in, Models = DENOISER_MODELS);   % K x n_grp n_snr x H x W
        if isempty(ml_names)                          % first scene: two chains per model (+ LS chains)
            ml_names = meta.name;  ml_meta = meta;
            ls_idx = [];
            if den_ls
                ls_req = strtrim(split(DENOISER_LS_MODELS, ",")).';
                [ok, ls_idx] = ismember(ls_req, ml_names);
                if ~all(ok), error('main5c:LSModel', 'DENOISER_LS_MODELS not in DENOISER_MODELS: %s', strjoin(ls_req(~ok), ', ')); end
            end
            n_ml = 2 * numel(ml_names) + numel(ls_idx);
            chains = [chains, reshape([ml_names + "_data"; ml_names + "_nodata"], 1, []), ml_names(ls_idx) + "_data_ls"];
            se_r  = cat(3, se_r,  nan(N_SCENE, n_snr, n_ml));  se_v = cat(3, se_v, nan(N_SCENE, n_snr, n_ml));
            est_r = cat(3, est_r, nan(N_SCENE, n_snr, n_ml, N_TARS));
            est_v = cat(3, est_v, nan(N_SCENE, n_snr, n_ml, N_TARS));
            n_fb  = [n_fb, zeros(n_snr, n_ml)];
            fprintf('  denoisers: %s\n', strjoin(compose("%s (%s, epoch %d)", ml_names(:), ...
                meta.version(:), meta.epoch(:)), ', '));
        elseif ~isequal(meta.name, ml_names) || ~isequal(meta.checkpoint, ml_meta.checkpoint)
            error('main5c:DenoiserChanged', 'alg_pdisac/configs/inference.json changed during the run.');
        end
        for k = 1:numel(ml_names)
            for h = 1:2                               % 1: data map, 2: data-free map
                ch = 3 + 2*(k-1) + h;
                for is = 1:n_snr
                    RD = double(reshape(RD_den(k, (h-1)*n_snr + is, :, :), N_chip, N_block));
                    [r_hat, v_hat, nf] = estimate_targets(RD, cfar, R_BAR, V_BAR, L_R, L_V, ...
                        range_axis, vel_axis, range_res, READOUT);
                    n_fb(is, ch) = n_fb(is, ch) + nf;
                    se_r(s, is, ch) = mean((r_hat - r_true).^2);
                    se_v(s, is, ch) = mean((v_hat - v_true).^2);
                    est_r(s, is, ch, :) = r_hat;  est_v(s, is, ch, :) = v_hat;
                end
            end
        end
        for j = 1:numel(ls_idx)                       % data + LS map through the selected denoisers
            k = ls_idx(j);  ch = 3 + 2*numel(ml_names) + j;
            for is = 1:n_snr
                RD = double(reshape(RD_den(k, 2*n_snr + is, :, :), N_chip, N_block));
                [r_hat, v_hat, nf] = estimate_targets(RD, cfar, R_BAR, V_BAR, L_R, L_V, ...
                    range_axis, vel_axis, range_res, READOUT);
                n_fb(is, ch) = n_fb(is, ch) + nf;
                se_r(s, is, ch) = mean((r_hat - r_true).^2);
                se_v(s, is, ch) = mean((v_hat - v_true).^2);
                est_r(s, is, ch, :) = r_hat;  est_v(s, is, ch, :) = v_hat;
            end
        end
        for is = 1:n_snr                              % every SNR was denoised; one line per SNR
            fprintf('  scene %3d/%d | SNR %+3d dB | denoised RMSE_r / RMSE_v: %s\n', s, N_SCENE, SNR_db(is), ...
                strjoin(compose("%s %.4f / %.4f", chains(4:end).', sqrt(squeeze(se_r(s, is, 4:end))), ...
                sqrt(squeeze(se_v(s, is, 4:end)))), ' | '));
        end
    end
end

% =====================================================================
%  Aggregate:  sqrt(E[RMSE^2])  and  sqrt(mean over scenes of the bound)
% =====================================================================
rm_r = sqrt(squeeze(mean(se_r, 1)));     rm_v = sqrt(squeeze(mean(se_v, 1)));    % n_snr x 3 (data, nodata, data+LS)
Bu_r = sqrt(squeeze(mean(bu_r, 1)));     Bu_v = sqrt(squeeze(mean(bu_v, 1)));
Bc_r = sqrt(mean(bc_r, 1)).';            Bc_v = sqrt(mean(bc_v, 1)).';
fb_frac = n_fb / (N_SCENE * N_TARS);
prior_r = mean(L_R) / (2*pi);            prior_v = mean(L_V) / (2*pi);           % bound at SNR -> 0
fallb_r = mean(L_R) * sqrt(1/12 - 1/(2*pi^2));  fallb_v = mean(L_V) * sqrt(1/12 - 1/(2*pi^2));

fprintf('\n===== Checks =====\n');
fprintf('(ii)  min eig(F_noPi - F)/max eig(F_noPi) = %.2e  (>= -1e-10 expected)\n', min_eig_check);
fprintf('(iii) BCRB / closed form (data map), range: %s\n', mat2str(Bu_r(:,1).' ./ Bc_r.', 3));
fprintf('      BCRB / closed form (data map), vel.:  %s\n', mat2str(Bu_v(:,1).' ./ Bc_v.', 3));
fprintf('(iv)  prior level L/(2pi): r %.3f m, v %.3f m/s | fallback RMSE: r %.3f m, v %.3f m/s\n', ...
    prior_r, prior_v, fallb_r, fallb_v);
fprintf('      BCRB at lowest SNR (data): r %.3f m, v %.3f m/s  [KNOW_MAG = %d, KNOW_PHASE = %d]\n', ...
    Bu_r(1,1), Bu_v(1,1), KNOW_MAG, KNOW_PHASE);
fprintf('      fallback fraction (data | nodata | data+LS): %s | %s | %s\n', mat2str(fb_frac(:,1).',2), ...
    mat2str(fb_frac(:,2).',2), mat2str(fb_frac(:,3).',2));
fprintf('      raw-data bound / Pi_A bound (data), range: %s\n', mat2str(Bu_r(:,3).' ./ Bu_r(:,1).', 3));
viol = [rm_r(:,1) < Bu_r(:,1); rm_r(:,2) < Bu_r(:,2); rm_v(:,1) < Bu_v(:,1); rm_v(:,2) < Bu_v(:,2)];
if LS_CANCEL, viol = [viol; rm_r(:,3) < Bu_r(:,3); rm_v(:,3) < Bu_v(:,3)]; end   % LS acts on Y_sen
for ch = 4:numel(chains)                      % bound of the input map: data / data-free (z), data + LS (raw data)
    b = 1 + endsWith(chains(ch), "_nodata") + 2*endsWith(chains(ch), "_data_ls");
    viol = [viol; rm_r(:,ch) < Bu_r(:,b); rm_v(:,ch) < Bu_v(:,b)]; %#ok<AGROW>
end
fprintf('      chain below the bound at %d of %d points (0 expected)\n', nnz(viol), numel(viol));
if USE_DENOISER
    fprintf('\n===== Denoisers vs. CA-CFAR: RMSE_r (m) / RMSE_v (m/s) =====\n%-24s', 'chain \ SNR (dB)');
    fprintf('%17d', SNR_db);  fprintf('\n');
    for ch = [1 2 3*ones(1, LS_CANCEL) 4:numel(chains)]
        fprintf('%-24s', chains(ch));  fprintf('  %7.4f/%7.4f', [rm_r(:,ch) rm_v(:,ch)].');  fprintf('\n');
    end
    fprintf('%-24s', 'BCRB (data map)');  fprintf('  %7.4f/%7.4f', [Bu_r(:,1) Bu_v(:,1)].');  fprintf('\n');
    if LS_CANCEL, fprintf('%-24s', 'BCRB (raw data)');  fprintf('  %7.4f/%7.4f', [Bu_r(:,3) Bu_v(:,3)].');  fprintf('\n'); end
end

% =====================================================================
%  CSV
% =====================================================================
T = table(SNR_db(:), rm_r(:,1), rm_r(:,2), rm_v(:,1), rm_v(:,2), ...
    Bu_r(:,1), Bu_v(:,1), Bu_r(:,2), Bu_v(:,2), ...
    Bc_r, Bc_v, fb_frac(:,1), fb_frac(:,2), rm_r(:,3), rm_v(:,3), fb_frac(:,3), Bu_r(:,3), Bu_v(:,3), ...
    'VariableNames', {'SNR_dB', 'RMSE_r_data', 'RMSE_r_nodata', 'RMSE_v_data', 'RMSE_v_nodata', ...
    'BCRB_r_data', 'BCRB_v_data', 'BCRB_r_nodata', 'BCRB_v_nodata', ...
    'Closed_r', 'Closed_v', 'Fallback_frac_data', 'Fallback_frac_nodata', ...
    'RMSE_r_data_ls', 'RMSE_v_data_ls', 'Fallback_frac_data_ls', 'BCRB_r_raw', 'BCRB_v_raw'});
for ch = 4:numel(chains)                      % denoiser chains
    T.(char("RMSE_r_" + chains(ch))) = rm_r(:, ch);
    T.(char("RMSE_v_" + chains(ch))) = rm_v(:, ch);
    T.(char("Fallback_frac_" + chains(ch))) = fb_frac(:, ch);
end
writetable(T, fullfile(stat_dir, ['statistics_bcrb_check' tag '.csv']));
save(fullfile(stat_dir, ['estimates_bcrb_check' tag '.mat']), 'est_r', 'est_v', 'tru_r', 'tru_v', ...
    'tru_a', 'SNR_db', 'chains', 'R_BAR', 'V_BAR', 'L_R', 'L_V', 'ml_meta');

% outliers at SNR >= 20 dB: |error| > 1 range bin or 1 Doppler bin
vel_res = abs(vel_axis(2) - vel_axis(1));
amp_rx  = tru_a ./ tru_r.^2;                      % received amplitude ~ |alpha| / r^2
fprintf('\n===== Outliers (SNR >= 20 dB, |dr| > %.2f m or |dv| > %.2f m/s) =====\n', range_res, vel_res);
fprintf('scene SNR  chain    m   r_true   r_hat    v_true   v_hat   rx power rel. strongest (dB)\n');
for is = find(SNR_db >= 20)
    for ch = 1:numel(chains)
        for s = 1:N_SCENE
            for m = 1:N_TARS
                dr = est_r(s, is, ch, m) - tru_r(s, m);  dv = est_v(s, is, ch, m) - tru_v(s, m);
                if abs(dr) > range_res || abs(dv) > vel_res
                    fprintf('%5d %3d  %-7s %2d  %7.2f  %7.2f  %7.2f  %7.2f   %6.1f\n', s, SNR_db(is), ...
                        chains(ch), m, tru_r(s, m), est_r(s, is, ch, m), tru_v(s, m), est_v(s, is, ch, m), ...
                        20*log10(amp_rx(s, m) / max(amp_rx(s, :))));
                end
            end
        end
    end
end

% =====================================================================
%  Figure: (a) range, (b) velocity
% =====================================================================
set(groot, 'defaultTextInterpreter','latex', 'defaultAxesTickLabelInterpreter','latex', ...
    'defaultLegendInterpreter','latex', 'defaultAxesFontName','Times New Roman', ...
    'defaultTextFontName','Times New Roman', 'defaultAxesFontSize',14);
cb = [0 114 178]/255;  co = [230 159 0]/255;  ck = [0 0 0];  cg = [0.5 0.5 0.5];
known = ["", "magnitude", "phase", "gain"];  kk = known(1 + KNOW_MAG + 2*KNOW_PHASE);
if kk == "", blabel = 'BCRB'; else, blabel = sprintf('BCRB, known %s', kk); end
n_den = numel(ml_names);                       % denoisers (USE_DENOISER): data, nodata (+ data_ls)
% colours distinct from the baseline (blue / orange / green / black / gray)
cml = [117 112 179; 231 41 138; 166 118 29; 165 0 38; 102 102 0; 0 0 128] / 255;
mk  = 'dv<>ph';                               % one marker per model
fig = figure('Color','w','Units','inches','Position',[1 1 12 6.2]);
tl  = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for p = 1:2
    ax = nexttile(tl);  hold(ax, 'on');  box(ax, 'on');  grid(ax, 'on');
    if p == 1
        Y = rm_r; Bu = Bu_r; pl = prior_r; fl = fallb_r; yl = 'Range RMSE (m)';  ax1 = ax;
    else
        Y = rm_v; Bu = Bu_v; pl = prior_v; fl = fallb_v; yl = 'Velocity RMSE (m/s)';
    end
    plot(ax, SNR_db, Y(:,1), '-o', 'Color', cb, 'MarkerFaceColor', cb, 'LineWidth', 1.6, ...
        'DisplayName', 'CA-CFAR, with data');
    plot(ax, SNR_db, Y(:,2), '-s', 'Color', co, 'MarkerFaceColor', co, 'LineWidth', 1.6, ...
        'DisplayName', 'CA-CFAR, without data');
    if LS_CANCEL
        cgr = [0 158 115]/255;
        plot(ax, SNR_db, Y(:,3), '-^', 'Color', cgr, 'MarkerFaceColor', cgr, 'LineWidth', 1.6, ...
            'DisplayName', 'CA-CFAR, with data + LS');
        plot(ax, SNR_db, Bu(:,3), '--', 'Color', cgr, 'LineWidth', 1.2, 'DisplayName', [blabel ', raw data']);
    end
    for k = 1:n_den                            % denoisers: with data solid, without data dotted
        ck_ = cml(1 + mod(k-1, size(cml, 1)), :);  mk_ = mk(1 + mod(k-1, numel(mk)));
        nm  = strrep(char(ml_names(k)), '_', ' ');
        plot(ax, SNR_db, Y(:, 3 + 2*k - 1), ['-' mk_], 'Color', ck_, 'MarkerFaceColor', ck_, ...
            'MarkerSize', 6, 'LineWidth', 1.4, 'DisplayName', sprintf('%s, with data', nm));
        plot(ax, SNR_db, Y(:, 3 + 2*k), [':' mk_], 'Color', ck_, 'MarkerFaceColor', 'w', ...
            'MarkerSize', 6, 'LineWidth', 1.4, 'DisplayName', sprintf('%s, without data', nm));
        j = find(chains == ml_names(k) + "_data_ls");      % data + LS input: dash-dot
        if ~isempty(j)
            plot(ax, SNR_db, Y(:, j), ['-.' mk_], 'Color', ck_, 'MarkerFaceColor', ck_, ...
                'MarkerSize', 6, 'LineWidth', 1.4, 'DisplayName', sprintf('%s, with data + LS', nm));
        end
    end
    plot(ax, SNR_db, Bu(:,1), '--', 'Color', ck, 'LineWidth', 1.8, 'DisplayName', blabel);
    yline(ax, fl, '-.', 'Color', cg, 'LineWidth', 1.1, 'DisplayName', 'Fallback $(\bar r_m,\bar v_m)$');
    yline(ax, pl, ':',  'Color', cg, 'LineWidth', 1.1, 'DisplayName', 'Prior, $L/(2\pi)$');
    set(ax, 'YScale', 'log');  xlim(ax, [min(SNR_db) max(SNR_db)]);  xticks(ax, SNR_db);
    xlabel(ax, '$\mathrm{SNR}_{\mathrm{Tx}}$ (dB)');  ylabel(ax, yl);
    title(ax, sprintf('(%s)', char('a' + p - 1)), 'FontWeight', 'normal');
end
lg = legend(ax1, 'NumColumns', 3 + (n_den > 0), 'FontSize', 10, 'Box', 'on');   % one legend, below
lg.Layout.Tile = 'south';
savefig(gcf, fullfile(fig_dir, ['fig_bcrb_check' tag '.fig']));
exportgraphics(gcf, fullfile(fig_dir, ['fig_bcrb_check' tag '.pdf']), 'ContentType', 'vector');
fprintf('Wrote %s and fig_bcrb_check%s\n', fullfile(stat_dir, ['statistics_bcrb_check' tag '.csv']), tag);
clear NBIT_EXT                                 % a later plain run uses main1 again


% =====================================================================
%  LOCAL FUNCTIONS
% =====================================================================
function [r_hat, v_hat, n_fb] = estimate_targets(RD, cfar, R_BAR, V_BAR, L_R, L_V, ...
                                             range_axis, vel_axis, range_res, READOUT)
% CA-CFAR on RD, association of target m to the strongest detection whose CELL intersects
% its prior window W_m (cell centre within half a bin of W_m: a target near the edge of W_m
% can have its peak cell centred just outside it), read-out (sinc or Dirichlet); without
% such a detection: fallback (r_bar, v_bar).
    [~, ~, det_pos, Pm] = func_ca_cfar_adaptive_threshold(RD, cfar.Ng_r, cfar.Ng_d, ...
        cfar.Nt_r, cfar.Nt_d, cfar.P_fa, cfar.peak_select);
    r_hat = R_BAR;  v_hat = V_BAR;  n_fb = 0;
    dv = abs(vel_axis(2) - vel_axis(1));
    for m = 1:numel(R_BAR)
        if isempty(det_pos), n_fb = n_fb + 1; continue; end
        rk = range_axis(det_pos(:, 1));  vk = vel_axis(det_pos(:, 2));
        in_w = abs(rk(:) - R_BAR(m)) <= (L_R(m) + range_res)/2 & abs(vk(:) - V_BAR(m)) <= (L_V(m) + dv)/2;
        if ~any(in_w), n_fb = n_fb + 1; continue; end
        cand = find(in_w);
        pw = Pm(sub2ind(size(Pm), det_pos(cand, 1), det_pos(cand, 2)));
        [~, j] = max(pw);
        if READOUT == "dirichlet"
            [r_hat(m), v_hat(m)] = dirichlet_readout(RD, det_pos(cand(j), :), range_res, vel_axis);
        else
            [r_hat(m), v_hat(m)] = func_sinc_interpolation(RD, det_pos(cand(j), :), range_axis, vel_axis);
        end
    end
end

function u = rc_sample(n)
% n draws of u in [-pi/2, pi/2] with density (2/pi) cos^2(u); x = x_bar + L*u/pi ~ p_rc.
    u = zeros(1, n);
    for j = 1:n
        while true
            cand = pi * (rand - 0.5);
            if rand <= cos(cand)^2, u(j) = cand; break; end
        end
    end
end

function [Q, Qd] = frame_delay(D, delay, T_chip, N_pad)
% Frame-level band-limited delay of the chip stream of D (N_block x N_chip), Q (N_chip x N_block),
% and its derivative w.r.t. the delay, Qd [1/s].
% Same convention as func_get_x_delay_frac: integer part by indexing, fractional part
% by a DFT phase ramp with a real Nyquist bin; zero signal before and after the frame.
    [N_block, N_chip] = size(D);
    x  = reshape(D.', [], 1);                       % chips of step 1, then step 2, ...
    P  = N_pad * N_chip;
    xp = [zeros(P, 1); x; zeros(P, 1)];
    L  = numel(xp);
    Xf = fft(xp);
    kvec = [0:ceil(L/2)-1, -floor(L/2):-1].';
    dt = delay / T_chip;  d_int = floor(dt);  d_frac = dt - d_int;
    ramp  = exp(-1j*2*pi*d_frac*kvec/L);
    dramp = (-1j*2*pi*kvec/L) .* ramp;              % d/d(d_frac)
    if mod(L, 2) == 0
        ramp(L/2+1)  = cos(pi*d_frac);
        dramp(L/2+1) = -pi*sin(pi*d_frac);
    end
    y  = real(ifft(Xf .* ramp));
    yd = real(ifft(Xf .* dramp));
    idx = P + (1:N_block*N_chip).' - d_int;
    Q  = reshape(y(idx),  N_chip, N_block);
    Qd = reshape(yd(idx), N_chip, N_block) / T_chip;
end

function [F, F_noPi] = fim_unit(D, r, v, phi, T_chip, t_i, eta, c, N_pad)
% FIM of (r_m, v_m, |alpha_m|, phi_m)_m for unit |alpha_m|, phases phi_m, unit noise variance:
%   F = 2 Re{G^H Pi_A G}, columns ordered (r1, v1, a1, phi1, r2, ...);
%   echo alpha S_0 with S = e^{j phi} S_0:  d/d|alpha| = S,  d/dphi = j S (at unit |alpha|).
    [N_block, N_chip] = size(D);
    N = numel(r);
    Gh = zeros(N_chip, N_block, 4*N);
    ph_v_all = exp(-1j * eta * (v(:) * t_i));       % N x N_block
    for m = 1:N
        [Q, Qd] = frame_delay(D, 2*r(m)/c, T_chip, N_pad);
        a  = 1 / (4*pi*r(m)^2);                     % sqrt(PL)
        pr = exp(1j*(phi(m) - eta*r(m)));           % e^{j phi_m} included
        pv = ph_v_all(m, :);                        % 1 x N_block
        S  = a * pr * Q .* pv;
        dSr = S * (-2/r(m) - 1j*eta) + a * pr * (2/c) * Qd .* pv;
        dSv = S .* (-1j*eta*t_i);
        dSp = 1j * S;
        Gh(:, :, 4*m-3) = fft(dSr) / sqrt(N_chip);  % unitary fast-time DFT per step
        Gh(:, :, 4*m-2) = fft(dSv) / sqrt(N_chip);
        Gh(:, :, 4*m-1) = fft(S)   / sqrt(N_chip);  % |alpha|
        Gh(:, :, 4*m)   = fft(dSp) / sqrt(N_chip);
    end
    ncol = 4*N;
    Gm = reshape(Gh, [], ncol);
    Gram = Gm' * Gm;
    % projector per bin l: c_i = conj(d_hat_i[l])
    cc = conj(fft(D.'));                            % N_chip x N_block
    tol = 1e-9 * max(abs(cc(:)));
    Z  = abs(cc) < tol;
    rowsZ = any(Z, 2);
    % bins without zeros: null space spanned by b = 1./c
    b = 1 ./ cc;  b(rowsZ, :) = 0;
    V = squeeze(sum(conj(b) .* Gh, 2));             % N_chip x ncol
    if ncol == 1, V = V(:); end
    nb = sum(abs(b).^2, 2);  nb(rowsZ) = 1;
    corr1 = V' * (V ./ nb);
    % bins with zeros: null space spanned by e_i with c_i = 0
    Gz = reshape(Gh .* Z, [], ncol);
    corr2 = Gz' * Gz;
    F      = 2 * real(Gram - corr1 - corr2);
    F_noPi = 2 * real(Gram);
end

function [cr, cv] = closed_form(g, R_BAR, L_R, L_V, N_fr, eta, B, c, t_i, know_phase)
% Closed-form Bayesian information and BCRB per target (targets decoupled), returned as
% squared errors cr [m^2], cv [(m/s)^2]; g_m = |alpha_m|^2/sigma^2 (KNOW_MAG) or E[|alpha_m|^2]/sigma^2.
% Unknown phase: the Schur complement over phi_m removes eta^2 from range and the cross term;
% the one over |alpha_m| only lowers the 4 rho6 term, so keeping it is still an upper bound:
%   J = 2 g N_fr diag((4 pi^2 B^2/c^2) rho4 + 4 rho6, eta^2 sigma_t^2 rho4) + prior.
% Known phase: the 2x2 form with eta^2 in range and the cross term eta^2 t_bar.
    tb = mean(t_i);  t2 = mean(t_i.^2);  st2 = t2 - tb^2;
    N = numel(R_BAR);  cr = zeros(1, N);  cv = zeros(1, N);
    for m = 1:N
        u  = linspace(-pi/2, pi/2, 2001);
        rr = R_BAR(m) + L_R(m) * u / pi;
        wgt = (2/pi) * cos(u).^2;
        PL = 1 ./ (4*pi*rr.^2).^2;
        rho4 = trapz(u, wgt .* PL);  rho6 = trapz(u, wgt .* PL ./ rr.^2);
        if know_phase
            Jb = 2 * g(m) * N_fr * [ (eta^2 + 4*pi^2*B^2/c^2)*rho4 + 4*rho6, eta^2*tb*rho4; ...
                                     eta^2*tb*rho4,                           eta^2*t2*rho4 ] ...
                 + diag([4*pi^2/L_R(m)^2, 4*pi^2/L_V(m)^2]);
            Ji = spd_inv(Jb);  cr(m) = Ji(1,1);  cv(m) = Ji(2,2);
        else
            Jb = 2 * g(m) * N_fr * [ (4*pi^2*B^2/c^2)*rho4 + 4*rho6, eta^2*st2*rho4 ] ...
                 + [4*pi^2/L_R(m)^2, 4*pi^2/L_V(m)^2];
            cr(m) = 1 / Jb(1);  cv(m) = 1 / Jb(2);
        end
    end
end

function plot_rd_maps(rd_list, ttl, range_axis, vel_axis, r_true, v_true, R_BAR, L_R, ...
                      V_BAR, L_V, snr_db, fname)
% RD power maps in dB on a COMMON scale (0 dB = peak of the last map, the data-free one),
% with the true targets (o) and the prior windows (dashed) of the evaluation scene.
    Pref = max(abs(rd_list{end}(:)).^2);
    vmax = max(abs(V_BAR) + L_V) + 10;   rmax = max(R_BAR + L_R) + 20;
    figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 15 4.6]);
    for p = 1:numel(rd_list)
        subplot(1, numel(rd_list), p);
        P = 10*log10(abs(rd_list{p}).^2 / Pref + 1e-30);
        imagesc(vel_axis, range_axis, P);  axis xy;  hold on;
        caxis([-70 0]);  colormap(parula);
        plot(v_true, r_true, 'wo', 'MarkerSize', 8, 'LineWidth', 1.5);
        for m = 1:numel(R_BAR)
            rectangle('Position', [V_BAR(m) - L_V(m)/2, R_BAR(m) - L_R(m)/2, L_V(m), L_R(m)], ...
                      'EdgeColor', 'w', 'LineStyle', '--');
        end
        xlim([-vmax vmax]);  ylim([0 rmax]);
        xlabel('Velocity (m/s)');  ylabel('Range (m)');
        title(sprintf('%s, SNR = %d dB', ttl{p}, snr_db));
        if p == numel(rd_list), cb = colorbar; ylabel(cb, 'Power (dB, rel. to data-free peak)'); end
    end
    savefig(gcf, [fname '.fig']);
    exportgraphics(gcf, [fname '.pdf'], 'ContentType', 'vector');
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

function [r_hat, v_hat] = dirichlet_readout(RD, pos, range_res, vel_axis)
% Exact trigonometric interpolation of the RD map around the detected cell pos = [k, d]
% (RD: N_chip x N_block, Doppler fftshifted as in main4). The finite DFTs interpolate with
% the Dirichlet kernel, not with sinc; evaluating the continuous DFT sums directly removes
% the deterministic error of a truncated sinc kernel. Range and Doppler are refined
% alternately (2 sweeps), each on the other coordinate's interpolated value, so the full
% target energy is used. For one target in white noise this maximises the periodogram (ML).
% The only remaining model effect is the removed zero-Doppler bin (part of the map).
    [Nc, Nd] = size(RD);
    Xs = ifft(ifftshift(RD, 2), [], 2);            % slow-time samples n = 0..Nd-1
    W  = fft(Xs, [], 1);                           % fast-time frequency l x slow time n
    lvec = [0:Nc/2-1, -Nc/2:-1].';                 % symmetric fast-time frequency index
    nvec = 0:Nd-1;
    kap = pos(1) - 1;                              % range index, 0-based (range_axis(1) = 0)
    b   = pos(2) - Nd/2 - 1;                       % signed Doppler bin
    for it = 1:2
        cl  = W * exp(-1j*2*pi*b*nvec(:)/Nd);      % range profile at Doppler b (spectrum)
        kap = refine_peak(@(x) abs(exp(1j*2*pi*x(:)*lvec.'/Nc) * cl), kap);
        sn  = (exp(1j*2*pi*kap*lvec.'/Nc) * W) / Nc;   % slow-time samples at range kap
        b   = refine_peak(@(x) abs(exp(-1j*2*pi*x(:)*nvec/Nd) * sn.'), b);
    end
    r_hat = kap * range_res;
    v_hat = interp1(1:Nd, vel_axis, b + Nd/2 + 1, 'linear', 'extrap');
end

function x = refine_peak(fun, x0)
% Maximise fun on [x0 - 0.5, x0 + 0.5]: 257-point grid, then a second 257-point grid
% around the best point (final step about 3e-5 bin).
    g = linspace(x0 - 0.5, x0 + 0.5, 257);
    [~, i] = max(fun(g));
    g = linspace(g(max(i-1, 1)), g(min(i+1, numel(g))), 257);
    [~, i] = max(fun(g));
    x = g(i);
end

function Mi = spd_inv(M)
% Inverse of a symmetric positive-definite information matrix with diagonal
% equilibration. The entries span many orders of magnitude (eta^2 ~ 1e7 for range,
% eta^2 t^2 for velocity, ||S||^2 for phase), and with an unknown phase the range
% and phase columns are nearly collinear (the -j*eta*S part of dS/dr), so a plain
% inv() reports RCOND ~ 1e-16. After scaling to unit diagonal the condition number
% is set only by that collinearity, about (eta*c/(2*pi*B))^2 ~ 1e7, which double
% precision handles.
    M  = (M + M') / 2;
    d  = 1 ./ sqrt(diag(M));
    Ms = (d .* M) .* d.';
    Mi = (d .* (Ms \ eye(size(M)))) .* d.';
end

function check_echo(D, Data_Tx, r, v, pos, rcs, vel, alpha, Tx_position, N_tx_ant, Rx_position, ...
                    N_rx_ant, P_tx, G_tx, G_rx, Lambda, fc, T_chip, T_pmcw, eta, c, N_pad)
% (i) alpha_m * S_m (bound model) vs. the simulator echo h_sen_tars * func_get_x_delay_frac.
    [N_block, ~] = size(D);
    blocks = unique([1, round(N_block/2), N_block]);
    err = zeros(size(blocks));  ref = zeros(size(blocks));
    for m = 1:numel(r)
        Q = frame_delay(D, 2*r(m)/c, T_chip, N_pad);
        for ib = 1:numel(blocks)
            i = blocks(ib);
            S_i = alpha(m) / (4*pi*r(m)^2) * exp(-1j*eta*r(m)) * exp(-1j*eta*v(m)*i*T_pmcw) * Q(:, i);
            H = h_sen_tars(1, pos(:, m), rcs(m), vel(:, m), Tx_position, N_tx_ant, Rx_position, ...
                           N_rx_ant, P_tx, G_tx, G_rx, Lambda, fc, i*T_pmcw);
            X = func_get_x_delay_frac(Data_Tx, i, 2*r(m)/c, T_chip);     % N_chip x N_tx_ant
            Y_sim = (H(:, :, 1) * X.').';
            err(ib) = err(ib) + norm(Y_sim - S_i)^2;  ref(ib) = ref(ib) + norm(Y_sim)^2;
        end
    end
    fprintf('(i)   echo model vs. simulator, relative error per block %s: %s\n', ...
        mat2str(blocks), mat2str(sqrt(err./ref), 3));
    fprintf('      (last block: the simulator segment ends at the frame end and its phase ramp wraps)\n');
end
