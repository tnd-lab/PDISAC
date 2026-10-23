function X_t_delay = func_get_x_delay_frac(Data_Tx, Index_block, Delay_tars, T_chip, N_margin_blocks)
%FUNC_GET_X_DELAY_FRAC  Delayed transmit block with an exact (fractional-chip) delay.
%
%   X_t_delay = FUNC_GET_X_DELAY_FRAC(Data_Tx, Index_block, Delay_tars, T_chip)
%   X_t_delay = FUNC_GET_X_DELAY_FRAC(..., N_margin_blocks)
%
%   Returns block Index_block of the transmit stream delayed by each entry of
%   Delay_tars. The delay is applied exactly instead of being truncated to a
%   whole chip. With whole-chip delays every echo would sit exactly on a range
%   bin, leaving a deterministic, SNR-independent range error
%       b_r(r) = floor(r/dr)*dr - r ,   dr = c*T_chip/2 ,
%   that is a simulator artefact. With the exact delay the RD-map peak falls
%   between bins and the sinc read-out of func_sinc_interpolation is the
%   matching interpolation kernel. Used for all sensing and communication
%   paths in main3_signal_channel_model (and by main5c_bcrb_check,
%   main7_comm_analysis and check_gamma_gof).
%
%   Method: tau/T_chip = d_int + d_frac. The integer part is applied by
%   indexing; the fractional part by a DFT phase ramp,
%   X(f) -> X(f)*exp(-j*2*pi*f*d_frac), i.e. an exact band-limited
%   sub-sample shift. The phase ramp is circular, so the segment carries
%   N_margin_blocks of context on each side; the wrap-around residual decays
%   as 1/(pi*D) with the distance D to the segment edge. The Nyquist bin is
%   forced real so a real Data_Tx stays real. A truncated FIR
%   fractional-delay filter is not used: the chip sequence is full-band and
%   the passband droop would add its own delay-dependent error.
%
% Inputs
%   Data_Tx         : (N_block, N_sample, N_ant) transmitted blocks
%   Index_block     : scalar, 1-based block index to evaluate at
%   Delay_tars      : (1,N_tars) or (N_tars,1) delays [s], must be >= 0
%   T_chip          : chip duration [s]
%   N_margin_blocks : (optional, default 6) blocks of extra context each side
%
% Output
%   X_t_delay       : (N_sample, N_ant, N_tars) delayed signal at Index_block;
%                     samples that would come from before the first block are zero

    if nargin < 5 || isempty(N_margin_blocks), N_margin_blocks = 6; end

    [N_block_tot, N_sample, N_ant] = size(Data_Tx);
    Delay_tars = Delay_tars(:);
    N_tars     = numel(Delay_tars);

    dt_chip = Delay_tars / T_chip;              % total delay in chips (real)
    assert(all(dt_chip >= 0), 'Delay must be non-negative (causal system).');
    d_int  = floor(dt_chip);
    d_frac = dt_chip - d_int;                   % in [0,1)

    is_cplx   = ~isreal(Data_Tx);
    X_t_delay = zeros(N_sample, N_ant, N_tars);
    if is_cplx, X_t_delay = complex(X_t_delay); end

    % ---- one shared segment, sized by the largest delay ------------------
    n_back = ceil(max(d_int) / N_sample) + 1 + N_margin_blocks;
    b_lo_w = Index_block - n_back + 1;                        % wanted first block
    b_lo   = max(1, b_lo_w);
    b_hi   = min(N_block_tot, Index_block + N_margin_blocks); % look-ahead context

    X_hist  = reshape(permute(Data_Tx(b_lo:b_hi, :, :), [2 1 3]), [], N_ant);

    % explicit zero history for the part of the frame that does not exist yet
    pad_pre = (b_lo - b_lo_w) * N_sample;
    if pad_pre > 0
        X_seg = [zeros(pad_pre, N_ant, 'like', X_hist); X_hist];
    else
        X_seg = X_hist;
    end

    % position, inside X_seg, of the LAST sample of block Index_block
    tail = pad_pre + (Index_block - b_lo + 1) * N_sample;
    L    = size(X_seg, 1);

    Xf   = fft(X_seg, [], 1);
    kvec = [0:ceil(L/2)-1, -floor(L/2):-1].';   % DFT bin indices, zero-centred

    for j = 1 : N_tars
        % ---- band-limited fractional shift -------------------------------
        if d_frac(j) == 0
            X_del = X_seg;
        else
            ramp = exp(-1j * 2*pi * d_frac(j) * kvec / L);
            if mod(L, 2) == 0
                ramp(L/2 + 1) = cos(pi * d_frac(j));   % real Nyquist bin
            end
            X_del = ifft(Xf .* ramp, [], 1);
            if ~is_cplx, X_del = real(X_del); end
        end

        % ---- integer shift by indexing
        idx = (tail - N_sample - d_int(j)) + (1:N_sample).';
        ok  = (idx >= 1) & (idx <= L);
        if any(ok)
            X_t_delay(ok, :, j) = X_del(idx(ok), :);
        end
        % samples whose source predates the frame stay zero
    end
end
