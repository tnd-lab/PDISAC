% MAIN7_COMM_ANALYSIS  Communication sweep: BER, net throughput and
% BPSK-input achievable rate, Monte-Carlo (realised channel) against the
% closed forms (Sec. V numerical results).
% -------------------------------------------------------------------------
% Loops over random UE realisations x N_bit^prbs (symbols_per_block) x SNR_Tx x
% Monte-Carlo trials. Each trial re-draws the scene (N_tars moving targets plus
% the UE as an extra target, N_scats static scatterers; the N_ref_tars reflectors
% closest to the UE form the NLOS channel), runs main3_signal_channel_model and
% main6_comm_process, and evaluates:
%
%   Numerical BER      bit-error rate of the four delay/CSI conditions of
%                      main6_comm_process (estimated/true delay x
%                      estimated/true channel).
%   Theoretical BER    func_ana_ber_closeform: E[Q(sqrt(2 C/(P_nlos + N)))]
%                      with the aggregate NLOS power P_nlos ~ Gamma(K, theta),
%                      moment-matched at the expected reflector geometry, the
%                      moments of c_m = p_even^H d^{even,(m)} measured on the
%                      code (func_code_c_moments).
%   Net throughput     Numerical: R_raw (1 - BER) with the true-delay /
%                      true-channel BER and R_raw = N_bit^prbs / T_pmcw.
%                      Theoretical: func_ana_net, semi-analytical from the
%                      realised channel of the trial (despread SINR per data
%                      slot -> Q-function).
%   Achievable rate    BPSK-input rate (func_mi_bpsk) of the despread SINR
%                      gamma_{k,i} of each data slot (the same SINR as the BER
%                      analysis):
%                        C_seq^(i) = sum_{k=1}^{N_bit^prbs} C_BPSK(gamma_{k,i})
%                        Cbar_seq  = (1/N_prbs) sum_i C_seq^(i)   [bits/sequence]
%                        Cbar_net  = Cbar_seq / T_prbs            [bits/s]
%                      Numerical (func_cap_seq): from the realised channel of
%                      the trial. Theoretical (func_ana_cap_closeform):
%                      E[C_BPSK] over the same Gamma model of P_nlos as the
%                      theoretical BER.
%
% Settings: communication_statistics_cfg of main1_matlab_config (random_scenes,
% monte_carlo, symbols_per_block, snr_db, fixed_targets = N_tars, scatterers =
% N_scats; gains in system_override) and scene.communication_reflectors =
% N_ref_tars.
%
% Outputs:
%   exported_statistics/statistics_communication.csv   one row per trial:
%       UE position/velocity, SNR, the four numerical BERs, theoretical BER,
%       numerical/theoretical net throughput and achievable rate, Gamma K and
%       theta; read by main8_main STEP 6a
%   fig_exported/fig_cap_seq_snr_nsym.fig / .png   Cbar_seq vs. SNR_Tx
%   fig_exported/fig_cap_net_snr_nsym.fig / .png   Cbar_net vs. SNR_Tx
%   fig_exported/fig_ber_snr_nsym.fig / .png       BER (true delay and channel)
%                                                  vs. SNR_Tx
%   (per N_bit^prbs: Monte-Carlo as markers, closed form as dashed lines)
%
% Dependencies: main1_matlab_config, main3_signal_channel_model,
% main6_comm_process, func_generate_static_scatterers, func_get_x_delay_frac,
% func_mi_bpsk.
% Long Monte-Carlo sweep; run standalone from simulation_matlab/.

% Run the pipeline headless during the sweep: main3/main6 are called in every
% trial; the main6 BPSK constellation is a slow, blocking System object.
NO_SIGNAL_PLOT = true;  NO_BPSK_PLOT = true;
NO_RADAR_PLOT  = true;  NO_CFAR_PLOT = true;

main1_matlab_config;

% =====================================================================
%  Monte-Carlo BER / throughput / capacity sweep (deterministic-per-path channel)
% =====================================================================
stat_cfg = communication_statistics_cfg;
script_dir = fileparts(mfilename('fullpath'));
stat_dir   = fullfile(script_dir, 'exported_statistics');
if ~exist(stat_dir, 'dir'), mkdir(stat_dir); end
out_csv  = fullfile(stat_dir, 'statistics_communication.csv');

G_tx = 10^(stat_cfg.system_override.gain_tx_db/10);
G_rx = 10^(stat_cfg.system_override.gain_rx_db/10);
G_ue = 10^(stat_cfg.system_override.gain_ue_db/10);
N_ran       = stat_cfg.random_scenes;
Mc          = stat_cfg.monte_carlo;
N_sym_list  = stat_cfg.symbols_per_block(:).';
N_sym_cases = numel(N_sym_list);
SNR_Tx_db   = stat_cfg.snr_db(:).';
SNR_Tx      = 10.^(SNR_Tx_db / 10);

% ---- results buffer ----
n_rows_total = N_ran * N_sym_cases * numel(SNR_Tx) * Mc;
col_names = { 'RAN_Index','N_sym_per_block','UE_X','UE_Y','UE_Z', ...
    'UE_Vx','UE_Vy','UE_Vz','SNR_dB', ...
    'Numerical_BER_Est_Tau_Est_h','Numerical_BER_Est_Tau_Perf_h', ...
    'Numerical_BER_Perf_Tau_Est_h','Numerical_BER_Perf_Tau_Perf_h', ...
    'Theoretical_BER', 'Numerical_Net','Theoretical_Net', ...
    'Cap_Seq_Numerical','Cap_Net_Numerical', ...
    'Cap_Seq_Theoretical','Cap_Net_Theoretical','MC_Index', ...
    'Gamma_K','Gamma_theta'};
results_buf = nan(n_rows_total, numel(col_names));
row_ptr = 1;

% Aggregate arrays kept for the per-SNR console summary and the figures
% [SNR x Mc x N_ran x N_sym]
ber_numerical_all_v4     = zeros(numel(SNR_Tx), Mc, N_ran, N_sym_cases);  % true delay + true channel
ber_theorectical_all     = zeros(numel(SNR_Tx), Mc, N_ran, N_sym_cases);  % func_ana_ber_closeform
net_numerical_all        = zeros(numel(SNR_Tx), Mc, N_ran, N_sym_cases);
net_theorectical_all     = zeros(numel(SNR_Tx), Mc, N_ran, N_sym_cases);
cap_seq_numerical_all    = zeros(numel(SNR_Tx), Mc, N_ran, N_sym_cases);  % Cbar_seq, Numerical
cap_net_numerical_all    = zeros(numel(SNR_Tx), Mc, N_ran, N_sym_cases);  % Cbar_net, Numerical
cap_seq_theorectical_all = zeros(numel(SNR_Tx), Mc, N_ran, N_sym_cases);  % Cbar_seq, Theoretical
cap_net_theorectical_all = zeros(numel(SNR_Tx), Mc, N_ran, N_sym_cases);  % Cbar_net, Theoretical

% ---- LOOP 1: random UE realisation ----
for idx_ran = 1:N_ran
    fprintf('\n========== UE Realisation %d / %d ==========\n', idx_ran, N_ran);

    N_ue        = 1;
    UE_position = rand_pos_in_roi(Region_of_interest, N_ue);
    UE_vel      = VEL_RANGE_UE * (2*rand(3, N_ue) - 1);  UE_vel(3) = 0;
    UE_rcs      = 10.^( (10*log10(10) + (10*log10(20)-10*log10(10))*rand(1)) / 10 );

    Rel_positions = UE_position - Tx_position;
    [~,~,Range_Tx_UE] = cart2sph(Rel_positions(1),Rel_positions(2),Rel_positions(3));
    Radial_UE_vel = dot(UE_vel, Rel_positions / norm(Rel_positions));

    % ---- LOOP 2: N_sym_per_block ----
    for idx_nsym = 1:N_sym_cases
        N_sym_per_block = N_sym_list(idx_nsym);
        L_slot_per_sym  = N_chip / (2 * N_sym_per_block);
        N_slot          = N_chip / L_slot_per_sym;
        R_raw           = N_sym_per_block / T_pmcw;
        fprintf('\n  ----- N_sym_per_block = %d -----\n', N_sym_per_block);

        % ---- LOOP 3: SNR ----
        for idx_snr = 1:numel(SNR_Tx)
            P_tx = Noise_power_com * SNR_Tx(idx_snr);

            % ---- LOOP 4: Monte-Carlo (re-draw scene) ----
            for idx_mc = 1:Mc
                % re-draw moving targets, then append UE as last target
                N_tars        = stat_cfg.fixed_targets;
                Tars_position = rand_pos_in_roi(Region_of_interest, N_tars);
                Tars_vel      = VEL_RANGE_TAR * (2*rand(3, N_tars) - 1);  Tars_vel(3,:) = 0;
                Tars_rcs      = 10.^( (10*log10(10) + (10*log10(20)-10*log10(10))*rand(1, N_tars)) / 10 );
                N_tars        = N_tars + 1;
                Tars_position = [Tars_position, UE_position];
                Tars_vel      = [Tars_vel,      UE_vel];
                Tars_rcs      = [Tars_rcs,      UE_rcs];
                Tars_motion   = phased.Platform('InitialPosition',Tars_position,'Velocity',Tars_vel);
                UE_motion     = phased.Platform('InitialPosition',UE_position,'Velocity',UE_vel);

                % re-draw static scatterers
                N_scats        = stat_cfg.scatterers;
                Scats_position = func_generate_static_scatterers(N_scats, Region_of_interest);
                Scats_vel      = zeros(size(Scats_position));
                Scats_rcs      = 10.^( (10*log10(5) + (10*log10(10)-10*log10(5))*rand(1, N_scats)) / 10 );
                Scats_motion   = phased.Platform('InitialPosition',Scats_position,'Velocity',Scats_vel);

                N_ref_tars = PDISAC_cfg.scene.communication_reflectors;   % reflectors closest to the UE

                % --- signal pipeline ---
                main3_signal_channel_model;
                main6_comm_process;

                % --- numerical BER (4 CSI/delay conditions) ---
                ber_v1 = mean(Data_UE_est_tau_est_h_bin(:)  ~= Data_Tx_bin(:));
                ber_v2 = mean(Data_UE_est_tau_perf_h_bin(:) ~= Data_Tx_bin(:));
                ber_v3 = mean(Data_UE_perf_tau_est_h_bin(:) ~= Data_Tx_bin(:));
                ber_v4 = mean(Data_UE_perf_tau_perf_h_bin(:)~= Data_Tx_bin(:));
                ber_numerical_all_v4(idx_snr, idx_mc, idx_ran, idx_nsym) = ber_v4;

                % --- closed-form BER (Gamma model of P_nlos) ---
                [ber_theo, K_g, theta_g] = func_ana_ber_closeform(C_w_time, P_tx, G_tx, G_ue, L_slot_per_sym, N_ref_tars, ...
                                                Noise_power_com, UE_position, Lambda, ...
                                                N_tars-1, N_scats, Region_of_interest, T_chip);   % N_tars-1: moving targets without the UE
                ber_theorectical_all(idx_snr, idx_mc, idx_ran, idx_nsym) = ber_theo;

                % --- net throughput (uncoded, r_c = 1) ---
                r_c     = 1;
                net_num = R_raw * (1 - ber_v4);
                net_numerical_all(idx_snr, idx_mc, idx_ran, idx_nsym) = net_num;
                net_theo = func_ana_net(H_com_los_all, H_com_nlos_all, C_w_time, Data_Tx_pad, ...
                    Noise_power_com, N_block, N_ref_tars, L_slot_per_sym, N_sym_per_block, R_raw, r_c, ...
                    Delay_los_all, Delay_nlos_all, T_chip);
                net_theorectical_all(idx_snr, idx_mc, idx_ran, idx_nsym) = net_theo;

                % --- BPSK-input achievable rate (Cbar_seq, Cbar_net) ---
                % Numerical: from the realised channel of this trial.
                [Cseq_num, Cnet_num] = func_cap_seq(H_com_los_all, H_com_nlos_all, C_w_time, Data_Tx_pad, ...
                    Noise_power_com, N_block, N_ref_tars, L_slot_per_sym, N_sym_per_block, T_pmcw, ...
                    Delay_los_all, Delay_nlos_all, T_chip);
                cap_seq_numerical_all(idx_snr, idx_mc, idx_ran, idx_nsym) = Cseq_num;
                cap_net_numerical_all(idx_snr, idx_mc, idx_ran, idx_nsym) = Cnet_num;

                % Theoretical: E[C_BPSK(gamma)] over Gamma-distributed P_nlos.
                [Cseq_theo, Cnet_theo] = func_ana_cap_closeform(C_w_time, P_tx, G_tx, G_ue, L_slot_per_sym, ...
                    N_sym_per_block, N_ref_tars, Noise_power_com, UE_position, Lambda, ...
                    N_tars-1, N_scats, Region_of_interest, T_pmcw, T_chip);
                cap_seq_theorectical_all(idx_snr, idx_mc, idx_ran, idx_nsym) = Cseq_theo;
                cap_net_theorectical_all(idx_snr, idx_mc, idx_ran, idx_nsym) = Cnet_theo;

                % --- store row ---
                results_buf(row_ptr, :) = [ idx_ran, N_sym_per_block, ...
                    UE_position(1), UE_position(2), UE_position(3), ...
                    UE_vel(1), UE_vel(2), UE_vel(3), SNR_Tx_db(idx_snr), ...
                    ber_v1, ber_v2, ber_v3, ber_v4, ber_theo, ...
                    net_num, net_theo, ...
                    Cseq_num, Cnet_num, Cseq_theo, Cnet_theo, idx_mc, K_g, theta_g ];
                row_ptr = row_ptr + 1;
            end % idx_mc
            % Gamma parameters of the last trial (K*theta = mean NLOS power)
            fprintf('    [Gamma] K = %.3f | mean K*theta = %.4e\n', K_g, K_g*theta_g);

            fprintf(['    Nsym=%3d | r_ue=%5.1f m | v_ue=%+5.1f m/s | SNR=%+3d dB | ' ...
                     'BER(num)=%.4e | BER(theo)=%.4e | ' ...
                     'Net(num)=%.4e | Net(theo)=%.4e | ' ...
                     'Cseq(num)=%.4e | Cseq(theo)=%.4e | ' ...
                     'Cnet(num)=%.4e | Cnet(theo)=%.4e\n'], ...
                N_sym_per_block, Range_Tx_UE, Radial_UE_vel, SNR_Tx_db(idx_snr), ...
                mean(ber_numerical_all_v4(idx_snr,:,idx_ran,idx_nsym), 'omitnan'), ...
                mean(ber_theorectical_all(idx_snr,:,idx_ran,idx_nsym), 'omitnan'), ...
                mean(net_numerical_all   (idx_snr,:,idx_ran,idx_nsym), 'omitnan'), ...
                mean(net_theorectical_all(idx_snr,:,idx_ran,idx_nsym), 'omitnan'), ...
                mean(cap_seq_numerical_all   (idx_snr,:,idx_ran,idx_nsym), 'omitnan'), ...
                mean(cap_seq_theorectical_all(idx_snr,:,idx_ran,idx_nsym), 'omitnan'), ...
                mean(cap_net_numerical_all   (idx_snr,:,idx_ran,idx_nsym), 'omitnan'), ...
                mean(cap_net_theorectical_all(idx_snr,:,idx_ran,idx_nsym), 'omitnan'));
        end % idx_snr
    end % idx_nsym
end % idx_ran

% ---- write the CSV once, after the full sweep ----
results_table = array2table(results_buf, 'VariableNames', col_names);
writetable(results_table, out_csv);   % overwritten on every run
fprintf('\nResults saved to %s  (%d rows x %d cols)\n', ...
    out_csv, height(results_table), width(results_table));

% =====================================================================
%  Figures: Cbar_seq [bits/sequence], Cbar_net [bits/s] and BER vs. SNR_Tx,
%  per N_bit^prbs, Numerical (markers) + Theoretical (dashed).
% =====================================================================
fig_dir = fullfile(script_dir, 'fig_exported');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

set(groot, 'defaultTextInterpreter',          'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter',        'latex');

% Okabe-Ito colour-blind-safe palette (same as main8_main STEP 6a)
col.black  = [0 0 0];       col.orange = [230 159   0]/255;
col.green  = [  0 158 115]/255; col.yellow = [240 228  66]/255;
col.blue   = [  0 114 178]/255; col.vermil = [213  94   0]/255;
col.purple = [204 121 167]/255; col.gray   = [0.5 0.5 0.5];
N_sym_col  = {col.blue, col.orange, col.green, col.vermil, col.purple, col.yellow};

% Average over UE realisations (dim 3) and Monte-Carlo trials (dim 2) -> [SNR x N_sym_cases]
Cseq_num_avg  = squeeze(mean(mean(cap_seq_numerical_all,    2, 'omitnan'), 3, 'omitnan'));
Cseq_theo_avg = squeeze(mean(mean(cap_seq_theorectical_all, 2, 'omitnan'), 3, 'omitnan'));
Cnet_num_avg  = squeeze(mean(mean(cap_net_numerical_all,    2, 'omitnan'), 3, 'omitnan'));
Cnet_theo_avg = squeeze(mean(mean(cap_net_theorectical_all, 2, 'omitnan'), 3, 'omitnan'));

% ---- Figure 1: Cbar_seq vs. SNR ---------------------------------------------
figure('Color','w','Units','inches','Position',[1 1 7 5]);
hold on; box on; grid on;
for n = 1:N_sym_cases
    c = N_sym_col{mod(n-1, numel(N_sym_col)) + 1};
    plot(SNR_Tx_db, Cseq_num_avg(:,n), 'o', 'Color', c, 'MarkerFaceColor', c, ...
        'MarkerSize', 6, 'DisplayName', sprintf('$N_{\\mathrm{bit}}^{\\mathrm{prbs}} = %d$', N_sym_list(n)));
    plot(SNR_Tx_db, Cseq_theo_avg(:,n), '--', 'Color', c, 'LineWidth', 1.5, 'HandleVisibility', 'off');
end
plot(nan, nan, 'o', 'Color', col.black, 'MarkerFaceColor', col.black, ...
    'DisplayName', 'Num.: Monte-Carlo (realized channel)');
plot(nan, nan, '--', 'Color', col.black, 'LineWidth', 1.5, ...
    'DisplayName', 'Theo.: analytical (moment-matched Gamma)');
xlabel('$\mathrm{SNR}_{\mathrm{Tx}}$ (dB)');
ylabel('$\bar{\mathcal{C}}_{\rm seq}$ (bits/sequence)');
title('BPSK-Constrained Capacity per Sequence vs. SNR');
legend('Location','northwest','Box','on','FontSize',10,'NumColumns',2);
savefig(gcf, fullfile(fig_dir,'fig_cap_seq_snr_nsym.fig'));
exportgraphics(gcf, fullfile(fig_dir,'fig_cap_seq_snr_nsym.png'), 'Resolution',300);

% ---- Figure 2: Cbar_net vs. SNR ---------------------------------------------
figure('Color','w','Units','inches','Position',[1 1 7 5]);
hold on; box on; grid on;
for n = 1:N_sym_cases
    c = N_sym_col{mod(n-1, numel(N_sym_col)) + 1};
    plot(SNR_Tx_db, Cnet_num_avg(:,n), 'o', 'Color', c, 'MarkerFaceColor', c, ...
        'MarkerSize', 6, 'DisplayName', sprintf('$N_{\\mathrm{bit}}^{\\mathrm{prbs}} = %d$', N_sym_list(n)));
    plot(SNR_Tx_db, Cnet_theo_avg(:,n), '--', 'Color', c, 'LineWidth', 1.5, 'HandleVisibility', 'off');
end
plot(nan, nan, 'o', 'Color', col.black, 'MarkerFaceColor', col.black, ...
    'DisplayName', 'Num.: Monte-Carlo (realized channel)');
plot(nan, nan, '--', 'Color', col.black, 'LineWidth', 1.5, ...
    'DisplayName', 'Theo.: analytical (moment-matched Gamma)');
xlabel('$\mathrm{SNR}_{\mathrm{Tx}}$ (dB)');
ylabel('$\bar{\mathcal{C}}_{\rm net}$ (bits/s)');
title('BPSK-Constrained Net Capacity vs. SNR');
legend('Location','northwest','Box','on','FontSize',10,'NumColumns',2);
savefig(gcf, fullfile(fig_dir,'fig_cap_net_snr_nsym.fig'));
exportgraphics(gcf, fullfile(fig_dir,'fig_cap_net_snr_nsym.png'), 'Resolution',300);

fprintf('\nCapacity figures saved: fig_cap_seq_snr_nsym.png, fig_cap_net_snr_nsym.png (in %s)\n', fig_dir);


% ---- Figure 3: BER vs. SNR, numerical (perfect tau, h) vs. theoretical ----
BER_num_avg  = squeeze(mean(mean(ber_numerical_all_v4, 2, 'omitnan'), 3, 'omitnan'));
BER_theo_avg = squeeze(mean(mean(ber_theorectical_all, 2, 'omitnan'), 3, 'omitnan'));
figure('Color','w','Units','inches','Position',[1 1 7 5]);
set(gca, 'YScale', 'log'); hold on; box on; grid on;
for n = 1:N_sym_cases
    c = N_sym_col{mod(n-1, numel(N_sym_col)) + 1};
    plot(SNR_Tx_db, BER_num_avg(:,n), 'o', 'Color', c, 'MarkerFaceColor', c, ...
        'MarkerSize', 6, 'DisplayName', sprintf('$N_{\\mathrm{bit}}^{\\mathrm{prbs}} = %d$', N_sym_list(n)));
    plot(SNR_Tx_db, BER_theo_avg(:,n), '--', 'Color', c, 'LineWidth', 1.5, 'HandleVisibility', 'off');
end
plot(nan, nan, 'o', 'Color', col.black, 'MarkerFaceColor', col.black, ...
    'DisplayName', 'Num.: Monte-Carlo (perfect $\tau$, $h$)');
plot(nan, nan, '--', 'Color', col.black, 'LineWidth', 1.5, ...
    'DisplayName', 'Theo.: analytical (moment-matched Gamma)');
xlabel('$\mathrm{SNR}_{\mathrm{Tx}}$ (dB)');
ylabel('Average BER');
title('BER vs. SNR');
legend('Location','southwest','Box','on','FontSize',10,'NumColumns',2);
savefig(gcf, fullfile(fig_dir,'fig_ber_snr_nsym.fig'));
exportgraphics(gcf, fullfile(fig_dir,'fig_ber_snr_nsym.png'), 'Resolution',300);
fprintf('BER figure saved: fig_ber_snr_nsym.png (in %s)\n', fig_dir);

% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function pos = rand_pos_in_roi(roi, N)
    pos = zeros(3, N);
    for dim = 1:3
        lo = roi(dim,1);  hi = roi(dim,2);
        if lo == hi
            pos(dim,:) = lo;
        else
            pos(dim,:) = lo + (hi - lo) .* rand(1, N);
        end
    end
end

function [Net_throughput] = func_ana_net(H_com_los_all, H_com_nlos_all, C_w_time, Data_Tx_pad, Noise_power_com, N_block, N_ref_tars, L_slot_per_sym, N_sym_per_block, R_b, r_c, Delay_los_all, Delay_nlos_all, T_chip)
    % FUNC_ANA_NET  Semi-analytical net throughput from the realised channel.
    % Per-slot BER = Q(sqrt(2 gamma_{k,i})) from the despread SINR of each slot, then
    % Net_throughput = mean over blocks and slots of r_c * R_b * (1 - BER) [bits/s].

    BER = zeros(N_block, N_sym_per_block);
    for i = 1:N_block
        h_com_los_i  = H_com_los_all(i, :, :);
        h_com_nlos_i = H_com_nlos_all(i, :, :, :);
        % LOS-synchronised NLOS replicas d_{k,i}^{even,(m)}: the transmitted
        % (data-modulated) signal delayed by the lag relative to the LOS path, so
        % chips spilling in from neighbouring slots/blocks carry their own symbols.
        lag_i = max((Delay_nlos_all(i, :) - Delay_los_all(i)) / T_chip, 0);        % 1 x N_ref_tars
        D_i   = reshape(func_get_x_delay_frac(Data_Tx_pad, i, lag_i * T_chip, T_chip), ...
                        [], N_ref_tars);                                           % N_chip x N_ref_tars
        for j = 1:N_sym_per_block
            idx_d = ((j-1)*2 + 1)*L_slot_per_sym + 1 : ((j-1)*2 + 2)*L_slot_per_sym;
            p = C_w_time(idx_d);          % p_prbs_even_j (column vector)
            ph_p = p' * p;                % scalar: p^H * p = ||p||^2  (real, positive)

            % Numerator:  |h_los|^2 * L^2   (LOS replica aligned after synchronisation)
            signal_component = abs(h_com_los_i)^2 * abs(ph_p)^2;

            % Denominator (interference): sum_k |h_nlos^(k)|^2 |c_k|^2, with c_k
            % evaluated at the true relative delay of path k
            interference_component = 0;
            for k = 1:N_ref_tars
                h_nlos_k = h_com_nlos_i(:, :, k);
                c_k      = p' * D_i(idx_d, k);   % c_m = p_even^H d^{even,(m)}
                interference_component = interference_component + abs(h_nlos_k)^2 * abs(c_k)^2;
            end

            % Denominator (noise):  sigma_n^2 * ||p||^2
            noise_component = Noise_power_com * real(ph_p);

            SINR_i_j = signal_component / (interference_component + noise_component);
            BER(i, j) = qfunc(sqrt(2 * SINR_i_j));
        end
    end
    R_sym = r_c * R_b * (1 - BER);
    Net_throughput = mean(R_sym(:));  % [bits/s]
end

function [Cbar_seq, Cbar_net] = func_cap_seq(H_com_los_all, H_com_nlos_all, C_w_time, Data_Tx_pad, ...
        Noise_power_com, N_block, N_ref_tars, L_slot_per_sym, N_sym_per_block, T_prbs, Delay_los_all, Delay_nlos_all, T_chip)
    % FUNC_CAP_SEQ  Numerical BPSK-input achievable rate from the realised channel.
    % Evaluated directly from the geometry-realised channel of this Monte-Carlo
    % trial, with the same despread SINR gamma_{k,i} as func_ana_net (signal
    % gain L^2, noise gain L):
    %   C_seq^(i) = sum_k C_BPSK(gamma_{k,i}),  Cbar_seq = mean_i C_seq^(i),
    %   Cbar_net  = Cbar_seq / T_prbs.
    % i (1..N_block) indexes the N_prbs MLS sequences of the frame;
    % j (1..N_sym_per_block) indexes the N_bit^prbs data slots of a sequence
    % (index k of C_seq).
    C_seq_per_block = zeros(N_block, 1);   % C_seq^{(i)}
    for i = 1:N_block
        h_com_los_i  = H_com_los_all(i, :, :);
        h_com_nlos_i = H_com_nlos_all(i, :, :, :);
        % LOS-synchronised NLOS replicas d_{k,i}^{even,(m)}: the transmitted
        % (data-modulated) signal delayed by the lag relative to the LOS path, so
        % chips spilling in from neighbouring slots/blocks carry their own symbols.
        lag_i = max((Delay_nlos_all(i, :) - Delay_los_all(i)) / T_chip, 0);        % 1 x N_ref_tars
        D_i   = reshape(func_get_x_delay_frac(Data_Tx_pad, i, lag_i * T_chip, T_chip), ...
                        [], N_ref_tars);                                           % N_chip x N_ref_tars
        C_sum = 0;
        for j = 1:N_sym_per_block
            idx_d = ((j-1)*2 + 1)*L_slot_per_sym + 1 : ((j-1)*2 + 2)*L_slot_per_sym;
            p = C_w_time(idx_d);          % p_prbs_even_j (column vector)
            ph_p = p' * p;                % scalar: p^H * p = ||p||^2  (real, positive)

            % Numerator:  |h_los|^2 * L^2   (LOS replica aligned after synchronisation)
            signal_component = abs(h_com_los_i)^2 * abs(ph_p)^2;

            % Denominator (interference): sum_k |h_nlos^(k)|^2 |c_k|^2, with c_k
            % evaluated at the true relative delay of path k
            interference_component = 0;
            for k = 1:N_ref_tars
                h_nlos_k = h_com_nlos_i(:, :, k);
                c_k      = p' * D_i(idx_d, k);   % c_m = p_even^H d^{even,(m)}
                interference_component = interference_component + abs(h_nlos_k)^2 * abs(c_k)^2;
            end

            % Noise: sigma_com^2 * L  (despreading gain L, not L^2)
            noise_component = Noise_power_com * real(ph_p);

            gamma_kj = signal_component / (interference_component + noise_component);
            C_sum = C_sum + func_mi_bpsk(gamma_kj);   % one term of C_seq^(i)
        end
        C_seq_per_block(i) = C_sum;                % C_seq^{(i)}
    end
    Cbar_seq = mean(C_seq_per_block);  % [bits/sequence]
    Cbar_net = Cbar_seq / T_prbs;       % [bits/s]
end

function [Cbar_seq, Cbar_net] = func_ana_cap_closeform(C_w_time, P_tx, G_tx, G_ue, L_slot_per_sym, N_bit_prbs, N_ref_tars, ...
        Noise_power_com, UE_position, Lambda, N_tars, N_scats, Region_of_interest, T_prbs, T_chip)
    % FUNC_ANA_CAP_CLOSEFORM  Theoretical BPSK-input achievable rate.
    % E[C_BPSK(gamma)] with gamma = C / (P_nlos + N) and the aggregate NLOS
    % power P_nlos ~ Gamma(K, theta), then scaled by the N_bit^prbs data slots
    % per sequence and normalised by T_prbs:
    %   Cbar_seq = N_bit^prbs * E[C_BPSK(gamma)],  Cbar_net = Cbar_seq / T_prbs.
    % Steps 1-2 (LOS power, noise, expected reflector geometry, Gamma
    % parameters) are identical to func_ana_ber_closeform.

    % ---- 1. System Parameters & Constants (identical to func_ana_ber_closeform) ----
    L = L_slot_per_sym;
    L_sq = L^2;
    A_e = G_ue * (Lambda^2) / (4 * pi);

    r_ue = norm(UE_position);
    PL_los = 1 / (4 * pi * r_ue^2);
    C = PL_los * P_tx * G_tx * A_e * L_sq;   % P_los * L^2
    N = Noise_power_com * L;                  % sigma_com^2 * L

    % ---- 2. Expected reflector geometry (identical to func_ana_ber_closeform) ----
    % Averaged over scenes drawn from the generators the simulation uses (see
    % func_expected_reflector_geometry).
    [r_tx_k_virtual, r_k_ue_virtual] = func_expected_reflector_geometry( ...
        UE_position, N_ref_tars, N_tars, N_scats, Region_of_interest);

    E_sigma_tar  = 10 / log(2);
    E_sigma_scat = 5 / log(2);
    E_sigma_rcs  = (N_tars / (N_tars + N_scats)) * E_sigma_tar + ...
                   (N_scats / (N_tars + N_scats)) * E_sigma_scat;

    PL_nlos_virtual = 1 ./ ((4 * pi * r_tx_k_virtual.^2) .* (4 * pi * r_k_ue_virtual.^2));
    % Despreading factor of each virtual reflector at its LOS-referred
    % excess-delay lag (r_tx,m + r_m,ue - r_ue)/(c*T_chip), which is set by the
    % geometry and is generally not an integer. The moments E|c_m|^2 and
    % E[|c_m|^2 |c_n|^2] are measured on the code itself at these lags
    % (func_code_c_moments). The L-based expressions E|c_m|^2 = L + L^2 sinc^2(lag)
    % (partial-period MLS correlation plus band-limited mainlobe tail; Sarwate &
    % Pursley, Proc. IEEE 1980) and E|c_m|^4 ~ (3-2/L) L^2 are only their large-L
    % limit, and a two-valued {L^2, L} selector is wrong at non-integer lags.
    c_light   = physconst('LightSpeed');
    lag_virt  = (r_tx_k_virtual + r_k_ue_virtual - r_ue) / (c_light * T_chip);
    [c2_virt, ~, m22_virt] = func_code_c_moments(C_w_time, lag_virt, L, N_chip_local(C_w_time), T_chip);
    nlos_powers_virtual = E_sigma_rcs * PL_nlos_virtual * P_tx * G_tx * A_e .* c2_virt;

    % moment-matched Gamma(K, theta) of P_nlos at the expected geometry
    kappa = PL_nlos_virtual * P_tx * G_tx * A_e;
    [K, theta] = func_gamma_moment_match(kappa, c2_virt, m22_virt, N_tars, N_scats);
    assert(abs(K*theta - sum(nlos_powers_virtual)) <= 1e-9 * sum(nlos_powers_virtual), ...
        'moment-matched mean differs from sum(nlos_powers_virtual)');

    % ---- 3. E[C_BPSK(gamma)] over P_nlos ~ Gamma(K, theta) ----
    E_C_bpsk = func_gamma_expect(@(x) func_mi_bpsk(C ./ (x + N)), K, theta);

    % ---- 4. Scale by N_bit_prbs data slots, normalize by sequence duration ----
    Cbar_seq = N_bit_prbs * E_C_bpsk;         % [bits/sequence]
    Cbar_net = Cbar_seq / T_prbs;               % [bits/s]
end

function [avg_ber, K, theta] = func_ana_ber_closeform(C_w_time, P_tx, G_tx, G_ue, L_slot_per_sym, N_ref_tars, ...
                                                Noise_power_com, UE_position, Lambda, ...
                                                N_tars, N_scats, Region_of_interest, T_chip)
    % FUNC_ANA_BER_CLOSEFORM  Closed-form average BER under the Gamma model.
    % avg_ber = E[Q(sqrt(2 C/(P_nlos + N)))], P_nlos ~ Gamma(K, theta), with
    % C = P_los L^2 the despread LOS power, N = sigma_com^2 L the despread noise
    % power, and (K, theta) moment-matched to P_nlos = sum_m kappa_m sigma_m |c_m|^2
    % at the expected reflector geometry. Also returns K and theta.
    % =====================================================================
    % 1. System Parameters & Constants
    % =====================================================================
    L = L_slot_per_sym;
    L_sq = L^2;
    A_e = G_ue * (Lambda^2) / (4 * pi);

    % LOS Power (C)
    r_ue = norm(UE_position);
    PL_los = 1 / (4 * pi * r_ue^2);
    C = PL_los * P_tx * G_tx * A_e * L_sq;

    % Noise Power (N)
    N = Noise_power_com * L;

    % =====================================================================
    % 2. Expected reflector geometry
    % =====================================================================
    % The N_ref_tars reflectors that h_com_nlos retains are the ones closest to
    % the UE; their expected Tx- and UE-side distances are averaged over scenes
    % drawn from the generators the simulation uses (uniform moving targets +
    % func_generate_static_scatterers). A uniform density over the ROI is not
    % used because the scatterer generator is clustered (cluster centres not
    % confined to the ROI), which a uniform surrogate would place far too close
    % to the UE.
    [r_tx_k_virtual, r_k_ue_virtual] = func_expected_reflector_geometry( ...
        UE_position, N_ref_tars, N_tars, N_scats, Region_of_interest);

    % Expected RCS E[sigma_rcs] of the log-uniform mixture (targets [10,20] m^2,
    % scatterers [5,10] m^2)
    E_sigma_tar  = 10 / log(2);
    E_sigma_scat = 5 / log(2);
    E_sigma_rcs  = (N_tars / (N_tars + N_scats)) * E_sigma_tar + ...
                   (N_scats / (N_tars + N_scats)) * E_sigma_scat;

    % NLOS path loss of the N_ref_tars closest reflectors (Tx -> reflector -> UE)
    PL_nlos_virtual = 1 ./ ((4 * pi * r_tx_k_virtual.^2) .* (4 * pi * r_k_ue_virtual.^2));
    % Despreading factor of each virtual reflector at its LOS-referred
    % excess-delay lag (r_tx,m + r_m,ue - r_ue)/(c*T_chip), which is set by the
    % geometry and is generally not an integer. The moments E|c_m|^2 and
    % E[|c_m|^2 |c_n|^2] are measured on the code itself at these lags
    % (func_code_c_moments). The L-based expressions E|c_m|^2 = L + L^2 sinc^2(lag)
    % (partial-period MLS correlation plus band-limited mainlobe tail; Sarwate &
    % Pursley, Proc. IEEE 1980) and E|c_m|^4 ~ (3-2/L) L^2 are only their large-L
    % limit, and a two-valued {L^2, L} selector is wrong at non-integer lags.
    c_light   = physconst('LightSpeed');
    lag_virt  = (r_tx_k_virtual + r_k_ue_virtual - r_ue) / (c_light * T_chip);
    [c2_virt, ~, m22_virt] = func_code_c_moments(C_w_time, lag_virt, L, N_chip_local(C_w_time), T_chip);
    nlos_powers_virtual = E_sigma_rcs * PL_nlos_virtual * P_tx * G_tx * A_e .* c2_virt;

    % moment-matched Gamma(K, theta) of P_nlos at the expected geometry
    kappa = PL_nlos_virtual * P_tx * G_tx * A_e;
    [K, theta] = func_gamma_moment_match(kappa, c2_virt, m22_virt, N_tars, N_scats);
    assert(abs(K*theta - sum(nlos_powers_virtual)) <= 1e-9 * sum(nlos_powers_virtual), ...
        'moment-matched mean differs from sum(nlos_powers_virtual)');

    p_min = qfunc(sqrt(2 * C / N));               % interference-free floor
    if p_min >= 0.5
        avg_ber = 0.5;
        return;
    end

    % E[Q(sqrt(2C/(P_nlos+N)))] over Gamma(K, theta), the closed-form average
    % BER of Sec. V, evaluated over the full support.
    avg_ber = func_gamma_expect(@(x) qfunc(sqrt(2 * C ./ (x + N))), K, theta);
end

function [r_tx_k, r_k_ue] = func_expected_reflector_geometry(UE_position, N_ref_tars, N_tars, N_scats, Region_of_interest)
    % FUNC_EXPECTED_REFLECTOR_GEOMETRY  Expected geometry of the N_ref_tars
    % reflectors closest to the UE (mean Tx-side distance r_tx_k and UE-side
    % distance r_k_ue, Tx at the origin), averaged over 400 scenes drawn from the
    % generators the simulation uses (uniform moving targets +
    % func_generate_static_scatterers). The k-th nearest distance has finite
    % moments -- the heavy tail is in the power, not in the distance -- so the
    % sample mean is the right statistic. Cached per UE position; fixed seed,
    % caller's RNG state restored, so the Monte-Carlo stream is untouched.
    persistent cache
    if isempty(cache), cache = containers.Map('KeyType','char','ValueType','any'); end
    key = sprintf('%.6g_%.6g_%d_%d_%d', UE_position(1), UE_position(2), N_ref_tars, N_tars, N_scats);
    if isKey(cache, key)
        g = cache(key);  r_tx_k = g.r_tx;  r_k_ue = g.r_ue;  return
    end

    N_DRAW = 400;
    s_rng  = rng;  rng(20260920, 'twister');
    R_tx = zeros(N_DRAW, N_ref_tars);
    R_ue = zeros(N_DRAW, N_ref_tars);
    for j = 1 : N_DRAW
        pos = [rand_pos_in_roi(Region_of_interest, N_tars), ...
               func_generate_static_scatterers(N_scats, Region_of_interest)];
        [~, idx] = sort(vecnorm(UE_position(:) - pos, 2, 1), 'ascend');
        sel = pos(:, idx(1:N_ref_tars));
        R_tx(j, :) = vecnorm(sel, 2, 1);                  % Tx at the origin
        R_ue(j, :) = vecnorm(UE_position(:) - sel, 2, 1);
    end
    r_tx_k = mean(R_tx, 1);
    r_k_ue = mean(R_ue, 1);
    rng(s_rng);
    cache(key) = struct('r_tx', r_tx_k, 'r_ue', r_k_ue);
end

function [K, theta] = func_gamma_moment_match(kappa, Ec2, M22, N_tars, N_scats)
    % FUNC_GAMMA_MOMENT_MATCH  Gamma shape K and scale theta matching the mean
    % mu and variance v of P_nlos = sum_m kappa_m sigma_m |c_m|^2
    % (K = mu^2/v, theta = v/mu). The RCS sigma_m (log-uniform mixture,
    % targets [10,20], scatterers [5,10] m^2) are independent across paths; the
    % |c_m|^2 are not (paths within about a chip share code segment and symbols):
    %   Var = sum_{m,n} kappa_m kappa_n ( E[s_m s_n] M22(m,n) - E[s]^2 Ec2_m Ec2_n ),
    % E[s_m s_n] = E[s^2] if m = n, E[s]^2 otherwise; M22(m,n) = E[|c_m|^2 |c_n|^2].
    t   = N_tars / (N_tars + N_scats);
    Es1 = t * (20 - 10) / log(2)          + (1 - t) * (10 - 5) / log(2);          % E[sigma]
    Es2 = t * (20^2 - 10^2) / (2*log(2))  + (1 - t) * (10^2 - 5^2) / (2*log(2));  % E[sigma^2]
    kappa = kappa(:).';  Ec2 = Ec2(:).';  N = numel(kappa);
    Esig  = Es1^2 * ones(N) + (Es2 - Es1^2) * eye(N);
    mu  = sum(kappa .* Es1 .* Ec2);
    v   = kappa * (Esig .* M22 - Es1^2 * (Ec2.' * Ec2)) * kappa.';
    K     = mu^2 / v;
    theta = v / mu;
end

function [Ec2, Ec4, M22] = func_code_c_moments(C_w_time, lag, L, N_chip, T_chip)
    % FUNC_CODE_C_MOMENTS  E|c_m|^2, E|c_m|^4 and M22(m,n) = E[|c_m|^2 |c_n|^2]
    % of c_m = p_even^H d^{even,(m)}, measured on the code at the given lags
    % (in chips): N_FRAME modulated frames (even slots carry random +-1 symbols,
    % as in main3), delayed by func_get_x_delay_frac, correlated over
    % every data slot. Deterministic (fixed seed, caller's RNG state restored) and
    % cached per (lag, L); no dependence on the Monte-Carlo scene.
    persistent cache
    if isempty(cache), cache = containers.Map('KeyType','char','ValueType','any'); end
    key = sprintf('%.6g_', lag(:).', L);
    if isKey(cache, key), g = cache(key);  Ec2 = g.Ec2;  Ec4 = g.Ec4;  M22 = g.M22;  return, end

    N_FRAME = 32;  N_BLK = 3;  nb = N_chip / (2*L);  N_ref = numel(lag);
    s_rng = rng;  rng(20260923, 'twister');
    S2 = zeros(1, N_ref);  S4 = S2;  S22 = zeros(N_ref);  cnt = 0;
    lag_c = max(lag, 0);
    for f = 1 : N_FRAME
        frame = zeros(N_BLK+1, N_chip, 1);
        for i = 1 : N_BLK
            x = C_w_time(:);  sym = 1 - 2*randi([0 1], nb, 1);
            for j = 1 : nb
                d = (2*j-1)*L+1 : 2*j*L;   x(d) = x(d) * sym(j);
            end
            frame(i,:,1) = x;
        end
        for i = 2 : N_BLK                       % block 1 has no history before it
            R = reshape(func_get_x_delay_frac(frame, i, lag_c*T_chip, T_chip), [], N_ref);
            for j = 1 : nb
                d  = (2*j-1)*L+1 : 2*j*L;
                cm = C_w_time(d).' * R(d, :);
                S2 = S2 + abs(cm).^2;   S4 = S4 + abs(cm).^4;   cnt = cnt + 1;
                S22 = S22 + (abs(cm).^2).' * abs(cm).^2;     % E[|c_m|^2 |c_n|^2]
            end
        end
    end
    rng(s_rng);
    Ec2 = S2 / cnt;   Ec4 = S4 / cnt;   M22 = S22 / cnt;   % diag(M22) = Ec4
    cache(key) = struct('Ec2', Ec2, 'Ec4', Ec4, 'M22', M22);
end

function N = N_chip_local(C_w_time)
    % code length N_chip (number of chips of the MLS)
    N = numel(C_w_time);
end

function E = func_gamma_expect(g, K, theta)
    % FUNC_GAMMA_EXPECT  E[g(X)], X ~ Gamma(K, theta), as int_0^1 g(F^{-1}(u)) du split at the
    % median; the upper half uses the complementary inverse so the tail is exact.
    opt = {'AbsTol', 1e-16, 'RelTol', 1e-6};
    E = integral(@(u) g(theta * gammaincinv(u, K)),          0, 0.5, opt{:}) ...
      + integral(@(s) g(theta * gammaincinv(s, K, 'upper')), 0, 0.5, opt{:});
end
