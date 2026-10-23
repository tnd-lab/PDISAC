%CHECK_GAMMA_GOF  Goodness of fit of the Gamma model of the aggregate NLOS power.
%
%   Checks whether the moment-matched Gamma model of the despread aggregate
%   NLOS power P_nlos (Sec. V) describes its distribution. P_nlos is drawn as
%   the closed forms of main7_comm_analysis model it:
%       P_nlos = sum_m kappa_m sigma_m |c_m|^2 ,
%   with
%     kappa_m = PL_nlos^(m) P_tx G_tx A_e at the expected reflector geometry
%               (same 400-scene average and seed as main7's
%               func_expected_reflector_geometry),
%     sigma_m   log-uniform RCS from the scene mixture (target with probability
%               N_tars/(N_tars+N_scats) on [10,20] m^2, else [5,10] m^2),
%     c_m     = p_even^H d^{even,(m)} evaluated on the actual MLS, BPSK-modulated
%               and delayed by the path's lag (+- a random jitter of JITTER_CHIP
%               chips) with func_get_x_delay_frac, so cross-slot ISI is included.
%               No Gaussian approximation is used for c_m.
%
%   The simulated samples (normalised by their mean) are compared with
%     - the Gamma model with the closed-form (K_nlos, Omega) (no fitting; the
%       moments of c_m are measured on the same draws, as main7 measures them
%       on the code), and
%     - Lognormal / Weibull / Inverse Gaussian / Exponential laws fitted by MLE.
%   Kolmogorov-Smirnov distances D are reported in the legend, the console and
%   the CSV; p-values are not, since with 1e4-1e5 samples every law is rejected.
%   Two Gamma variants are printed as diagnostics only (not plotted): Gamma at
%   the sample moments, and Gamma with the large-L expressions of the c_m moments.
%
%   Settings: N_UE_POS, N_DRAW, JITTER_CHIP, N_BIT_LIST (N_bit^prbs values),
%   N_block (below); N_tars, N_scats, gains and SNR from
%   communication_statistics_cfg and N_ref_tars from
%   scene.communication_reflectors of main1_matlab_config.
%
%   Outputs:
%     fig_exported/fig_gof_ue<UE>_nbit<N_bit>.png / .fig   one PDF figure per
%                                                         (UE, N_bit^prbs)
%     exported_statistics/statistics_gamma_gof.csv        KS distance per model
%     exported_statistics/statistics_gamma_gof.mat        gof (samples and KS),
%                                                         gof_models; read by
%                                                         main8_main STEP 6d
%
%   Dependencies: main1_matlab_config, func_get_x_delay_frac,
%   func_generate_static_scatterers.
%   Run standalone from simulation_matlab/.

clear; clc;
here    = fileparts(mfilename('fullpath'));
sim_dir = here;                      % this file lives in simulation_matlab/
addpath(sim_dir);
main1_matlab_config;

% ---- settings ----
N_UE_POS   = 1;           % UE positions (statistics.communication: random_scenes = 1)
N_DRAW     = 10000;         % RCS / data / lag-jitter draws (one modulated frame each)
JITTER_CHIP = 2;           % reflector position uncertainty inside the resolution
                           % cell, expressed as a lag jitter of +-JITTER_CHIP chips
                           % (+-2 chips = +-2 m of excess path). It leaves the path
                           % losses unchanged but decorrelates c_m, which is what
                           % the model assumes; without it the few slot positions of
                           % a small N_bit^prbs give only a handful of c_m values.
N_BIT_LIST = [4 32 128];
N_block    = 4;            % blocks per frame; block 1 is skipped (no history)
rng(20260922, 'twister');

stat_cfg = communication_statistics_cfg;
N_ref_tars = PDISAC_cfg.scene.communication_reflectors;    % reflectors kept (closest to the UE)
N_tars     = stat_cfg.fixed_targets;                       % scene size for the reflector
N_scats    = stat_cfg.scatterers;                          % geometry and RCS mixture
G_tx = 10^(stat_cfg.system_override.gain_tx_db/10);
G_ue = 10^(stat_cfg.system_override.gain_ue_db/10);
P_tx = Noise_power_com * 10^(median(stat_cfg.snr_db)/10);   % any value: the normalised shape is scale-free
A_e  = G_ue * Lambda^2 / (4*pi);
cT   = physconst('LightSpeed') * T_chip;
C_w_time = func_generate_waveform_local(N_chip);            % same MLS as main3_signal_channel_model
fig_dir = fullfile(sim_dir, 'fig_exported');   if ~isfolder(fig_dir), mkdir(fig_dir); end
stat_dir = fullfile(sim_dir, 'exported_statistics');  if ~isfolder(stat_dir), mkdir(stat_dir); end
out_csv = fullfile(stat_dir, 'statistics_gamma_gof.csv');
out_mat = fullfile(stat_dir, 'statistics_gamma_gof.mat');   % read by main8_main STEP 6d

fam = {'Lognormal','Weibull','InverseGaussian','Exponential'};
lab = {'Gamma, closed form ($K_{\rm nlos}$, $\Omega$)', 'Lognormal (MLE)', ...
       'Weibull (MLE)', 'Inverse Gaussian (MLE)', 'Exponential (MLE)'};
col = {[213 94 0]/255, [0 158 115]/255, [204 121 167]/255, [86 180 233]/255, [0.5 0.5 0.5]};
sty = {'-', '--', '-.', ':', ':'};

res = cell(0, 7);
gof = struct('UE', {}, 'N_bit', {}, 'x', {}, 'K_nlos', {}, 'b', {}, 'KS', {});   % samples and fit for main8_main
for idx_ue = 1:N_UE_POS
    UE_position = rand_pos_in_roi(Region_of_interest, 1);
    [r_tx, r_ue] = expected_reflector_geometry(UE_position, N_ref_tars, N_tars, N_scats, Region_of_interest);
    kappa = P_tx * G_tx * A_e ./ ((4*pi*r_tx.^2) .* (4*pi*r_ue.^2));
    lag   = (r_tx + r_ue - norm(UE_position)) / cT;                  % Tx at the origin
    t_mix = N_tars / (N_tars + N_scats);
    fprintf('\nUE %d at [%.1f %.1f] m | r_tx = %s m | r_ue = %s m\n', idx_ue, UE_position(1:2), ...
        num2str(r_tx, '%.0f '), num2str(r_ue, '%.0f '));

    for nb = N_BIT_LIST
        Ls = N_chip / (2*nb);

        % ---- draw P_nlos: RCS per draw, c_m from the real code per slot ----
        P = zeros(N_DRAW*(N_block-1)*nb, 1);  ptr = 1;
        S2 = zeros(1, N_ref_tars);  S4 = S2;  n_c = 0;   % measured c_m moments
        S22 = zeros(N_ref_tars);                          % E[|c_m|^2 |c_n|^2], all pairs
        for idx_draw = 1:N_DRAW
            lag_d  = max(lag + JITTER_CHIP*(2*rand(1, N_ref_tars) - 1), 0);
            is_tar = rand(1, N_ref_tars) < t_mix;
            sigma  = is_tar .* (10 * 2.^rand(1, N_ref_tars)) + ...
                     ~is_tar .* ( 5 * 2.^rand(1, N_ref_tars));        % log-uniform mixture
            frame  = zeros(N_block+1, N_chip, 1);                      % BPSK-modulated, as in main3
            for i = 1:N_block
                x = C_w_time(:);  sym = 1 - 2*randi([0 1], nb, 1);
                for j = 1:nb
                    d = (2*j-1)*Ls+1 : 2*j*Ls;   x(d) = x(d) * sym(j);
                end
                frame(i,:,1) = x;
            end
            for i = 2:N_block
                R = reshape(func_get_x_delay_frac(frame, i, lag_d*T_chip, T_chip), [], N_ref_tars);
                for j = 1:nb
                    d  = (2*j-1)*Ls+1 : 2*j*Ls;
                    cm = C_w_time(d).' * R(d, :);                      % c_m, ISI included
                    S2 = S2 + abs(cm).^2;   S4 = S4 + abs(cm).^4;   n_c = n_c + 1;
                    S22 = S22 + (abs(cm).^2).' * abs(cm).^2;
                    P(ptr) = sum(kappa .* sigma .* abs(cm).^2);  ptr = ptr + 1;
                end
            end
        end
        P = P(1:ptr-1);  P = P(P > 0);
        sc = mean(P);  x = P / sc;  n = numel(x);                      % scale-free

        % ---- Gamma model with the closed-form parameters (no fitting) ----
        % E[sigma] and E[sigma^2] of the log-uniform RCS mixture
        Es1 = t_mix*(20-10)/log(2)         + (1-t_mix)*(10-5)/log(2);
        Es2 = t_mix*(20^2-10^2)/(2*log(2)) + (1-t_mix)*(10^2-5^2)/(2*log(2));
        % (a) diagnostic: large-L expressions of E|c_m|^2 and E|c_m|^4, averaged
        % over the same lag jitter as the draws
        jit = linspace(-JITTER_CHIP, JITTER_CHIP, 2001);
        a2  = mean(Ls^2 * sinc(max(lag(:) + jit, 0)).^2, 2).';
        a4  = mean(Ls^4 * sinc(max(lag(:) + jit, 0)).^4, 2).';
        % c_m = cbar_m + ctilde_m with ctilde_m a sum of Ls chip products (+-1):
        % E[ctilde^2] = Ls and E[ctilde^4] = (3 - 2/Ls) Ls^2 (Rademacher kurtosis),
        % which replaces the Gaussian value 3 Ls^2 and matters when Ls is small.
        Ec2 = Ls + a2;   Ec4 = a4 + 6*a2*Ls + (3 - 2/Ls)*Ls^2;
        mu  = sum(kappa .* Es1 .* Ec2);
        v   = sum(kappa.^2 .* (Es2*Ec4 - (Es1*Ec2).^2));
        K_L = mu^2 / v;   Omega_L = v / mu;      % diagnostic: L-based c moments

        % (b) the closed form, as main7_comm_analysis computes it: the even moments
        % of c_m are measured on the code at these lags, not approximated by L.
        % The RCS are independent across paths, but |c_m|^2 are not: paths whose
        % lags are within about a chip share the same code segment and symbols.
        % Var[P] = sum_{m,n} kappa_m kappa_n ( E[s_m s_n] E[|c_m|^2|c_n|^2]
        %                                      - E[s]^2 E|c_m|^2 E|c_n|^2 ),
        % E[s_m s_n] = E[s^2] if m = n, E[s]^2 otherwise.
        Ec2_m  = S2 / n_c;   Ec4_m = S4 / n_c;   M22 = S22 / n_c;   % diag(M22) = Ec4_m
        Esig   = Es1^2 * ones(N_ref_tars) + (Es2 - Es1^2) * eye(N_ref_tars);
        mu_m   = sum(kappa .* Es1 .* Ec2_m);
        v_m    = kappa * (Esig .* M22 - Es1^2 * (Ec2_m.' * Ec2_m)) * kappa.';
        K_nlos = mu_m^2 / v_m;   Omega = v_m / mu_m;
        % diagnostic: the same, with the paths assumed independent (diagonal only)
        v_ind  = sum(kappa.^2 .* (Es2*Ec4_m - (Es1*Ec2_m).^2));
        K_ind  = mu_m^2 / v_ind;
        % correlation coefficient of |c_m|^2 and |c_n|^2 between paths
        C2 = (M22 - Ec2_m.' * Ec2_m) ./ sqrt(max(diag(M22) - Ec2_m.'.^2, eps) * max(diag(M22).' - Ec2_m.^2, eps));
        rho_max = max(C2(~eye(N_ref_tars)));
        pd = cell(1, numel(lab));
        pd{1} = makedist('Gamma', 'a', K_nlos, 'b', Omega/sc);
        pd_L  = makedist('Gamma', 'a', K_L, 'b', Omega_L/sc);
        [~,~,D_L] = kstest(x, 'CDF', pd_L);
        for f = 1:numel(fam), pd{f+1} = fitdist(x, fam{f}); end

        KS = nan(1, numel(lab));
        for f = 1:numel(lab), [~,~,KS(f)] = kstest(x, 'CDF', pd{f}); end
        gof(end+1) = struct('UE', idx_ue, 'N_bit', nb, 'x', single(x), ...
                            'K_nlos', K_nlos, 'b', Omega/sc, 'KS', KS); %#ok<SAGROW>

        % diagnostic, printed but NOT plotted: same Gamma family, parameters from
        % the two sample moments. Comparing it with the closed form separates
        % "is the family right?" from "are K_nlos and Omega right?".
        K_mm  = mean(x)^2 / var(x);
        pd_mm = makedist('Gamma', 'a', K_mm, 'b', var(x)/mean(x));
        [~,~,D_mm] = kstest(x, 'CDF', pd_mm);
        fprintf(['  N_bit = %3d | n = %6d | K_nlos = %5.2f (closed form) vs %5.2f (sample) ' ...
                 '| mean sim/closed = %.2f | D of Gamma at the sample moments = %.4f\n'], ...
            nb, n, K_nlos, K_mm, sc/mu_m, D_mm);
        fprintf('              | lags = %s chips | max corr(|c_m|^2,|c_n|^2) = %.2f | K if paths independent = %5.2f\n', ...
            num2str(lag, '%.1f '), rho_max, K_ind);
        res(end+1,:) = {idx_ue, nb, 'Gamma, sample moments (diagnostic)', D_mm, n, K_mm, sc/mu_m}; %#ok<SAGROW>
        fprintf(['              | E|c|^2 / L per path = %s | with the L-based c moments: K = %5.2f, ' ...
                 'mean sim/that = %.2f, D = %.4f\n'], num2str(Ec2_m/Ls, '%.2f '), K_L, sc/mu, D_L);
        res(end+1,:) = {idx_ue, nb, 'Gamma, L-based c moments (diagnostic)', D_L, n, K_L, sc/mu}; %#ok<SAGROW>
        [~, ord] = sort(KS);
        for f = ord
            fprintf('     %-34s D = %.4f\n', lab{f}, KS(f));
            res(end+1,:) = {idx_ue, nb, lab{f}, KS(f), n, K_nlos, sc/mu_m}; %#ok<SAGROW>
        end

        % ---- one figure: the PDFs ----
        fg = figure('Color','w','Units','inches','Position',[1 1 6 4.2]);
        hold on; box on; grid on;
        xg = linspace(0, quantile(x, 0.995), 400);
        histogram(x(x <= xg(end)), 80, 'Normalization','pdf', 'FaceColor',[0.85 0.85 0.85], ...
            'EdgeColor','none', 'DisplayName','Simulated');
        for f = flip(ord)                                   % best fit drawn last
            plot(xg, pdf(pd{f}, xg), sty{f}, 'Color', col{f}, 'LineWidth', 2, ...
                'DisplayName', sprintf('%s, $D = %.3f$', lab{f}, KS(f)));
        end
        xlabel('$P_{\rm nlos}/\bar P_{\rm nlos}$','Interpreter','latex');
        ylabel('PDF','Interpreter','latex');
        title(sprintf('UE %d, $N_{\\rm bit}^{\\rm prbs} = %d$, $K_{\\rm nlos} = %.2f$', ...
            idx_ue, nb, K_nlos), 'Interpreter','latex');
        legend('Location','northeast','Interpreter','latex','FontSize',9,'Box','on');
        fn = fullfile(fig_dir, sprintf('fig_gof_ue%d_nbit%d', idx_ue, nb));
        exportgraphics(fg, [fn '.png'], 'Resolution', 300);   savefig(fg, [fn '.fig']);
    end
end

writetable(cell2table(res, 'VariableNames', {'UE','N_bit','Model','KS_D','n','K_nlos','MeanRatio'}), out_csv);
fprintf('\nWrote %s\n', out_csv);
gof_models = lab;   % order of gof(k).KS: Gamma (closed form), Lognormal, Weibull, Inverse Gaussian, Exponential
save(out_mat, 'gof', 'gof_models', '-v7.3');
fprintf('Wrote %s\n', out_mat);

% =====================================================================
function [r_tx_k, r_k_ue] = expected_reflector_geometry(UE_position, N_ref_tars, N_tars, N_scats, ROI)
% same construction, seed and sample count as func_expected_reflector_geometry
% of main7_comm_analysis: mean Tx- and UE-side distances of the N_ref_tars
% reflectors closest to the UE over 400 scenes; caller's RNG state restored
    N_DRAW = 400;
    s_rng = rng;  rng(20260920, 'twister');
    R_tx = zeros(N_DRAW, N_ref_tars);  R_ue = R_tx;
    for j = 1:N_DRAW
        pos = [rand_pos_in_roi(ROI, N_tars), func_generate_static_scatterers(N_scats, ROI)];
        [~, idx] = sort(vecnorm(UE_position(:) - pos, 2, 1), 'ascend');
        sel = pos(:, idx(1:N_ref_tars));
        R_tx(j,:) = vecnorm(sel, 2, 1);                      % Tx at the origin
        R_ue(j,:) = vecnorm(UE_position(:) - sel, 2, 1);
    end
    r_tx_k = mean(R_tx, 1);   r_k_ue = mean(R_ue, 1);
    rng(s_rng);
end

function pos = rand_pos_in_roi(roi, N)
    pos = zeros(3, N);
    for dim = 1:3
        lo = roi(dim,1);  hi = roi(dim,2);
        if lo == hi, pos(dim,:) = lo; else, pos(dim,:) = lo + (hi - lo) .* rand(1, N); end
    end
end

function C_w_time = func_generate_waveform_local(N_chip)
% copy of the local func_generate_waveform of main3_signal_channel_model
% (local functions of a script cannot be called from outside it)
    n = log2(N_chip);
    switch n
        case 4,  taps = [4, 3];        case 5,  taps = [5, 2];
        case 6,  taps = [6, 1];        case 7,  taps = [7, 1];
        case 8,  taps = [8, 4, 3, 2];  case 9,  taps = [9, 4];
        case 10, taps = [10, 3];       case 11, taps = [11, 2];
        case 12, taps = [12, 6, 4, 1];
        otherwise, error('unsupported N_chip');
    end
    L = N_chip - 1;  reg = ones(1, n);  m = zeros(1, L);
    for i = 1:L
        m(i) = reg(n);
        fb   = 0;
        for t = 1:numel(taps), fb = xor(fb, reg(taps(t))); end
        reg  = [fb, reg(1:n-1)];
    end
    C_w_time = [1 - 2*m, 1].';          % augmented m-sequence, +-1, length N_chip
end
