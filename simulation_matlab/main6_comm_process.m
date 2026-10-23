% MAIN6_COMM_PROCESS  UE-side communication processing: delay estimation,
% pilot-slot channel estimation and BPSK detection (Sec. V).
% -------------------------------------------------------------------------
% Script (run after main3_signal_channel_model, in the same workspace).
% Each MLS block carries N_sym_per_block (N_bit^prbs) slot pairs: the odd slot
% k is a pilot (plain code p_odd), the even slot carries one BPSK symbol
% (code p_even times s_{k,i}).
%
%   1. Delay estimation: circular-correlation (ACF) search of the MLS over a
%      two-block buffer (func_get_delay_index_acf), combining the coherent
%      pilot-slot correlation and the non-coherent data-slot correlation; the
%      integer peak is refined to a sub-chip offset by sinc interpolation of
%      the pilot correlation.
%   2. Synchronisation: the sub-chip part of the delay is removed from the
%      received signal by a band-limited shift (func_frac_shift) over a
%      3-block context [i-1, i, i+1], then the N_chip window at the integer
%      index is extracted, so every chip sits in its own slot and despreading
%      uses the plain code. (Shifting the reference instead is not equivalent:
%      despreading is restricted to one slot, which does not commute with the
%      shift, so chips of neighbouring pilot slots would leak in as cross-slot
%      ISI.)
%   3. Channel estimation, scalar LS per slot pair:
%          hat_h_com,i^(k) = (p_odd^H y_p) / (p_odd^H p_odd)
%   4. Detection: matched-filter statistic and equalisation,
%          z_com,i^(k) = p_even^H y_d,
%          hat_s_k,i   = z_com,i^(k) / ( hat_h_com,i^(k) * (p_even^H p_even) ),
%      followed by a hard BPSK decision (func_bpsk_demod).
%
% Four delay/CSI conditions are evaluated:
%   estimated delay + estimated channel   -> Data_UE_est_tau_est_h_bin
%   estimated delay + true channel        -> Data_UE_est_tau_perf_h_bin
%   true delay      + estimated channel   -> Data_UE_perf_tau_est_h_bin
%   true delay      + true channel        -> Data_UE_perf_tau_perf_h_bin
% (and the soft symbols *_bpsk, [N_block x N_sym_per_block]).
% "True delay" uses the exact LOS delay Delay_los_all (nearest chip + sub-chip
% residual); "estimated delay" uses the ACF peak and its sub-chip refinement.
% "True channel" is the noise-free LS pilot estimate on the LOS-only clean
% signal Data_UE_los_clean (effective LOS channel, including any residual
% sub-chip loss).
%
% Requires in the workspace (from main1_matlab_config / main3): Data_UE,
% Data_UE_los_clean, Data_Tx_bin, Delay_los_all, H_com_los_all, H_com_nlos_all,
% C_w_time, N_block, N_chip, N_ref_tars, N_slot, N_sym_per_block,
% L_slot_per_sym, T_chip.
% Also sets Rho_subchip (sub-chip loss factor h_eff / h_geo per slot pair).
%
% Flags:
%   NO_BPSK_PLOT = true       skip the constellation diagram of the realistic
%                             case (estimated delay and channel).
%   NO_SUBCHIP_REPORT = true  suppress the one-line sub-chip synchronisation
%                             check; the script sets it to true after printing
%                             it once, so repeated calls in a sweep stay quiet.
% Called by main7_comm_analysis and main8_main.

%% Detect --------------------------------------------------------------------

% Construct a buffer from two consecutive blocks to handle propagation delay
Data_UE_buffer = [Data_UE(1:N_block, :), Data_UE(2:(N_block+1), :)];

% Estimate the starting index of each block from the reference MLS by circular
% correlation (ACF), and the sub-chip offset of the correlation peak.
[Index_delay, Frac_delay] = func_get_delay_index_acf(Data_UE_buffer(:, 1:N_chip), C_w_time, N_slot, L_slot_per_sym);

% True timing: nearest chip + exact sub-chip residual.
% Nearest, not floor(): a correlator peaks on the nearest chip, which is what
% func_get_delay_index_acf returns. Index and residual must use the same rounding.
tau_chips        = Delay_los_all(1:N_block) / T_chip;
true_delay_index = round(tau_chips) + 1;
true_delay_frac  = tau_chips - round(tau_chips);        % in [-0.5, 0.5]

% Remove the delay and extract the synchronised chips: 3-block context
% [i-1, i, i+1] (zeros before the frame start), shifted back by the sub-chip
% delay, then the window at the integer index. Sample d of block i sits at
% N_chip + d in the context.
Data_UE_no_delay_perf  = zeros(N_block, N_chip);
Data_UE_no_delay_est   = zeros(N_block, N_chip);
Data_UE_los_clean_sync = zeros(N_block, N_chip);   % LOS only, no noise, true timing
for i = 1:N_block
    if i == 1
        prev = zeros(1, N_chip);  prev_c = prev;
    else
        prev = Data_UE(i-1, :);   prev_c = Data_UE_los_clean(i-1, :);
    end
    ctx   = [prev,   Data_UE(i, :),           Data_UE(i+1, :)].';
    ctx_c = [prev_c, Data_UE_los_clean(i, :), Data_UE_los_clean(i+1, :)].';

    % estimated timing
    d = Index_delay(i);
    y = func_frac_shift(ctx, -Frac_delay(i));
    Data_UE_no_delay_est(i, :) = y(N_chip + d - 1 + (1:N_chip));

    % true timing
    d  = true_delay_index(i);
    y  = func_frac_shift(ctx,   -true_delay_frac(i));
    yc = func_frac_shift(ctx_c, -true_delay_frac(i));
    Data_UE_no_delay_perf(i, :)  = y (N_chip + d - 1 + (1:N_chip));
    Data_UE_los_clean_sync(i, :) = yc(N_chip + d - 1 + (1:N_chip));
end

% Geometric (sub-chip-free) LOS channel aligned to the same window, used as the
% reference for the sub-chip loss factor Rho_subchip.
[H_com_los_all_perf, ~] = func_get_perfect_csi(H_com_los_all, H_com_nlos_all, N_block, N_chip, N_ref_tars, true_delay_index);


% Transpose to [N_chip x N_block] for column-wise processing
Data_UE_no_delay_est = Data_UE_no_delay_est.';
Data_UE_no_delay_perf = Data_UE_no_delay_perf.';
Data_UE_los_clean_sync = Data_UE_los_clean_sync.';


Data_UE_est_tau_est_h_bpsk = zeros(N_block, N_sym_per_block);
Data_UE_est_tau_perf_h_bpsk = zeros(N_block, N_sym_per_block);
Data_UE_perf_tau_est_h_bpsk = zeros(N_block, N_sym_per_block);
Data_UE_perf_tau_perf_h_bpsk = zeros(N_block, N_sym_per_block);
% sub-chip loss factor rho = h_eff / h_geo per slot pair (synchronisation check)
Rho_subchip = zeros(N_block, N_sym_per_block);

for b_idx = 1:N_block
    Data_UE_no_delay_est_i = Data_UE_no_delay_est(:, b_idx);
    Data_UE_no_delay_perf_i = Data_UE_no_delay_perf(:, b_idx);
    Data_UE_los_clean_i = Data_UE_los_clean_sync(:, b_idx);
    H_com_los_all_perf_i = H_com_los_all_perf(b_idx, :).';


    for s_idx = 1:N_sym_per_block

        % -----------------------------------
        % Estimated Delay + Estimated Channel
        % -----------------------------------

        % Pilot index (odd slot k) -> p_{prbs,k}^{odd}, y_{com,i}^{odd,(k)}
        p_idx_num = (s_idx-1)*2 + 1;
        idx_p = (p_idx_num-1)*L_slot_per_sym + 1 : p_idx_num*L_slot_per_sym;
        x_p = C_w_time(idx_p);                     % p_{prbs,k}^{odd}  [L_slot_per_sym x 1]
        y_p = Data_UE_no_delay_est_i(idx_p);       % y_{com,i}^{odd,(k)}

        % Scalar LS channel estimate:
        %   hat_h_{com,i}^{(k)} = (p_odd^H y_p) / (p_odd^H p_odd)
        h_hat = (x_p' * y_p) / (x_p' * x_p);       % scalar

        % Effective LOS channel for the two "true channel" branches: the
        % noise-free limit of the same LS estimate, pilot slots only, so no
        % data symbol enters the "known" channel.
        x_p_perf = C_w_time(idx_p);
        h_eff = (x_p_perf' * Data_UE_los_clean_i(idx_p)) / (x_p_perf' * x_p_perf);
        Rho_subchip(b_idx, s_idx) = h_eff / mean(H_com_los_all_perf_i(idx_p));

        % Data index (even slot k) -> p_{prbs,k}^{even}, y_{com,i}^{even,(k)}
        d_idx_num = p_idx_num + 1;
        idx_d = (d_idx_num-1)*L_slot_per_sym + 1 : d_idx_num*L_slot_per_sym;
        x_d = C_w_time(idx_d);                      % p_{prbs,k}^{even}  [L_slot_per_sym x 1]
        y_d = Data_UE_no_delay_est_i(idx_d);        % y_{com,i}^{even,(k)}

        % Matched-filter statistic z_{com,i}^{(k)} = p_even^H y_d
        z_com = x_d' * y_d;

        % Equalization:
        %   hat_s_{k,i} = z_{com,i}^{(k)} / ( hat_h_{com,i}^{(k)} * (p_even^H p_even) )
        s_est = z_com / (h_hat * (x_d' * x_d));

        % Store BPSK symbol
        Data_UE_est_tau_est_h_bpsk(b_idx, s_idx) = s_est;

        % -----------------------------
        % Estimate Delay + True Channel
        % -----------------------------

        % Data index
        d_idx_num = p_idx_num + 1;
        idx_d = (d_idx_num-1)*L_slot_per_sym + 1 : d_idx_num*L_slot_per_sym;
        x_d = C_w_time(idx_d);
        y_d = Data_UE_no_delay_est_i(idx_d);

        h_true = h_eff;   % effective LOS channel

        % Matched-filter statistic
        z_com = x_d' * y_d;

        % Equalization with the known scalar channel:
        s_perf = z_com / (h_true * (x_d' * x_d));

        % Store BPSK symbol
        Data_UE_est_tau_perf_h_bpsk(b_idx, s_idx) = s_perf;


        % -------------------------------
        % True Delay + Estimated Channel
        % -------------------------------

        % Pilot index (odd slot k)
        p_idx_num = (s_idx-1)*2 + 1;
        idx_p = (p_idx_num-1)*L_slot_per_sym + 1 : p_idx_num*L_slot_per_sym;
        x_p = C_w_time(idx_p);
        y_p = Data_UE_no_delay_perf_i(idx_p);

        % Scalar LS channel estimate
        h_hat = (x_p' * y_p) / (x_p' * x_p);

        % Data index (even slot k)
        d_idx_num = p_idx_num + 1;
        idx_d = (d_idx_num-1)*L_slot_per_sym + 1 : d_idx_num*L_slot_per_sym;
        x_d = C_w_time(idx_d);
        y_d = Data_UE_no_delay_perf_i(idx_d);

        % Matched-filter statistic
        z_com = x_d' * y_d;

        % Equalization
        s_est = z_com / (h_hat * (x_d' * x_d));

        % Store BPSK symbol
        Data_UE_perf_tau_est_h_bpsk(b_idx, s_idx) = s_est;



        % --------------------------
        % True Delay + True Channel
        % --------------------------

        % Data index
        d_idx_num = p_idx_num + 1;
        idx_d = (d_idx_num-1)*L_slot_per_sym + 1 : d_idx_num*L_slot_per_sym;
        x_d = C_w_time(idx_d);
        y_d = Data_UE_no_delay_perf_i(idx_d);
        h_true = h_eff;   % effective LOS channel

        % Matched-filter statistic
        z_com = x_d' * y_d;

        % Equalization with the known scalar channel
        s_perf = z_com / (h_true * (x_d' * x_d));

        % Store BPSK symbol
        Data_UE_perf_tau_perf_h_bpsk(b_idx, s_idx) = s_perf;
    end
end


% Sub-chip synchronisation check: after the fractional-delay compensation |rho|
% should be ~1 (without it, ~sinc(d_frac)). Printed once per workspace, since
% this script runs inside the Monte-Carlo sweep of main7_comm_analysis; set
% NO_SUBCHIP_REPORT = false to print it again, true to suppress it.
if ~exist('NO_SUBCHIP_REPORT','var') || ~NO_SUBCHIP_REPORT
    frac_off = Delay_los_all(1:N_block)/T_chip - round(Delay_los_all(1:N_block)/T_chip);
    d_bar = mean(frac_off);
    sinc_pred = 1; if d_bar ~= 0, sinc_pred = sin(pi*d_bar)/(pi*d_bar); end
    fprintf(['d_frac: mean %+.3f chip | uncompensated would be %.4f | ' ...
             '|rho|: mean %.4f (%.2f dB), min %.4f | arg(rho): %.2f deg\n'], ...
            d_bar, sinc_pred, mean(abs(Rho_subchip(:))), ...
            20*log10(mean(abs(Rho_subchip(:)))), min(abs(Rho_subchip(:))), ...
            rad2deg(mean(angle(Rho_subchip(:)))));
    NO_SUBCHIP_REPORT = true;   % print once per workspace
end

Data_UE_est_tau_est_h_bin = func_bpsk_demod(Data_UE_est_tau_est_h_bpsk(:));
Data_UE_est_tau_perf_h_bin = func_bpsk_demod(Data_UE_est_tau_perf_h_bpsk(:));
Data_UE_perf_tau_est_h_bin = func_bpsk_demod(Data_UE_perf_tau_est_h_bpsk(:));
Data_UE_perf_tau_perf_h_bin = func_bpsk_demod(Data_UE_perf_tau_perf_h_bpsk(:));



%% BPSK constellation figure (optional; skip with NO_BPSK_PLOT = true) -------
if ~exist('NO_BPSK_PLOT','var') || ~NO_BPSK_PLOT

% Constellation of the realistic case: estimated delay + estimated channel
Data_UE_no_delay_bpsk = Data_UE_est_tau_est_h_bpsk;

refconst = func_bpsk_mod([0 1]);
constellationDiagram = comm.ConstellationDiagram('ReferenceConstellation', refconst, 'Title', 'Slot-Pairing BPSK Constellation (Circular-ACF Delay Sync + Scalar LS Channel Est.)');
constellationDiagram(Data_UE_no_delay_bpsk(:));

end

% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function [delay_idx, delay_frac] = func_get_delay_index_acf(Data_UE, C_w_time, N_slot, L_slot)
    % FUNC_GET_DELAY_INDEX_ACF  Circular-correlation (ACF) delay search.
    %   [delay_idx, delay_frac] = func_get_delay_index_acf(Data_UE, C_w_time, N_slot, L_slot)
    % Kernel R[k] = ifft( fft(y) .* conj(fft(x)) )(k) (circular correlation of
    % the UE delay estimation, Sec. V), with x the MLS masked to the odd (pilot)
    % or even (data) slots; the two correlations are combined by magnitude.
    % Inputs : Data_UE [N_block x N_chip] received chips (one row per block),
    %          C_w_time [N_chip x 1] MLS, N_slot slots of L_slot chips.
    % Outputs: delay_idx  [1 x N_block] integer peak index (1 = no delay),
    %          delay_frac [1 x N_block] sub-chip offset in [-0.5, 0.5] chips,
    %          from sinc interpolation of the coherent pilot correlation R_odd,
    %          so the UE can remove the fractional delay from the received signal.
    N_block = size(Data_UE, 1);
    N_chip  = size(Data_UE, 2);
    delay_idx  = zeros(1, N_block);
    delay_frac = zeros(1, N_block);
    N_sinc_ref = 16;

    % Build odd/even masks
    mask_odd  = zeros(N_chip, 1);
    mask_even = zeros(N_chip, 1);
    for slot = 1:N_slot
        slot_idx = (slot-1)*L_slot + 1 : slot*L_slot;
        if mod(slot, 2) ~= 0  % odd
            mask_odd(slot_idx) = 1;
        else                   % even
            mask_even(slot_idx) = 1;
        end
    end

    C_odd  = C_w_time .* mask_odd;   % reference for odd slots
    C_even = C_w_time .* mask_even;  % reference for even slots

    % Frequency-domain references, computed once and reused for every block
    C_odd_freq  = conj(fft(C_odd));
    C_even_freq = conj(fft(C_even));

    for i = 1:N_block
        y_i    = Data_UE(i, :).';     % N_chip x 1
        Y_freq = fft(y_i);

        % Odd slots: coherent (known, no data) -- circular correlation
        R_odd  = ifft(Y_freq .* C_odd_freq);    % N_chip x 1, lag k = 1..N_chip

        % Even slots: non-coherent (unknown because of data) -- circular correlation
        R_even = ifft(Y_freq .* C_even_freq);

        % Combine: coherent odd + non-coherent even
        R_total = abs(R_odd) + abs(R_even);

        [~, peak] = max(R_total);
        delay_idx(i) = peak;

        % Sub-chip refinement: Whittaker-Shannon reconstruction of the
        % COHERENT pilot correlation R_odd (complex, band-limited, no data signs)
        % over the half-chip interval around the peak. R_total is a sum of
        % magnitudes, which is not band-limited, so it is used for the integer
        % peak only.
        kk  = (peak - N_sinc_ref : peak + N_sinc_ref).';
        z   = R_odd(mod(kk - 1, N_chip) + 1);
        u   = linspace(-0.5, 0.5, 257).';
        sx  = u + peak - kk.';
        ker = ones(size(sx));  nz = (sx ~= 0);
        ker(nz) = sin(pi*sx(nz)) ./ (pi*sx(nz));
        [~, q] = max(abs(ker * z));
        delay_frac(i) = u(q);
    end
end

function [H_com_los_all_perf, H_com_nlos_all_perf] = func_get_perfect_csi(H_com_los_all, H_com_nlos_all, N_block, N_chip, N_ref_tars, delay_index)
    % FUNC_GET_PERFECT_CSI  Geometric LOS / NLOS channel aligned to the UE window.
    % The per-block channel coefficients are expanded to N_chip chips, buffered
    % over two consecutive blocks and cut at the same integer window as the
    % received signal. Outputs: H_com_los_all_perf [N_block x N_chip],
    % H_com_nlos_all_perf [N_block x N_ref_tars x N_chip].
    H_com_los_all_buffer  = repmat(squeeze(H_com_los_all),  1, N_chip);
    H_com_los_all_buffer = [H_com_los_all_buffer(1:N_block, :), H_com_los_all_buffer(2:(N_block+1), :)];
    H_com_nlos_all_buffer = repmat(squeeze(H_com_nlos_all), 1, 1, N_chip);
    H_com_nlos_all_buffer = cat(3, H_com_nlos_all_buffer(1:N_block, :, :), H_com_nlos_all_buffer(2:(N_block+1), :, :));
    H_com_los_all_perf = zeros(N_block, N_chip);
    H_com_nlos_all_perf = zeros(N_block, N_ref_tars, N_chip);
    % delay_index is per block; a scalar is applied to every block.
    if isscalar(delay_index), delay_index = repmat(delay_index, N_block, 1); end
    for i = 1:N_block
        d = delay_index(i);
        H_com_los_all_perf(i, :) = H_com_los_all_buffer(i, d:(d+N_chip-1));
        H_com_nlos_all_perf(i, :, :) = H_com_nlos_all_buffer(i, :, d:(d+N_chip-1));
    end
end
