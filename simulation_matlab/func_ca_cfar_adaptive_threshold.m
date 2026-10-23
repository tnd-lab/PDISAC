function [threshold_map, noise_avg, detected_positions, RD_power_map] = ...
        func_ca_cfar_adaptive_threshold(RD_map, N_guard_range, N_guard_doppler, ...
                                        N_train_range, N_train_doppler, P_fa, peak_select)
%FUNC_CA_CFAR_ADAPTIVE_THRESHOLD  2-D cell-averaging CFAR (CA-CFAR) detector on an RD map.
%   [threshold_map, noise_avg, detected_positions, RD_power_map] = ...
%       func_ca_cfar_adaptive_threshold(RD_map, N_guard_range, N_guard_doppler, ...
%                                       N_train_range, N_train_doppler, P_fa, peak_select)
%
%   CA-CFAR of Sec. III on the cell powers |RD_map|^2. No Image Processing
%   Toolbox required.
%
%   CFAR window (per dimension, each side of the cell under test, CUT):
%
%     |<- N_train ->|<- N_guard ->| CUT |<- N_guard ->|<- N_train ->|
%
%     Training cells: estimate the local noise/clutter floor.
%     Guard cells:    keep the target energy out of that estimate.
%
%   Threshold: with N_eff(r,d) training cells available at each CUT (fewer at
%   the map edges, where the window is clipped),
%     alpha(r,d)     = N_eff(r,d) * ( P_fa^(-1/N_eff(r,d)) - 1 )
%     threshold(r,d) = alpha(r,d) * noise_avg(r,d)
%   alpha is computed for each N_eff, so the design P_fa holds at the edge
%   cells as well (alpha is non-linear in N_eff). The formula assumes
%   exponentially distributed cell powers (complex Gaussian noise).
%
%   Inputs
%     RD_map           (N_range x N_doppler) complex RD map
%     N_guard_range    guard cells each side, range dimension
%     N_guard_doppler  guard cells each side, Doppler dimension
%     N_train_range    training cells each side, range dimension
%     N_train_doppler  training cells each side, Doppler dimension
%     P_fa             design probability of false alarm per cell
%     peak_select      (optional, default true) keep only the detections that are
%                      the maximum of their (2*(N_guard+N_train)+1)-cell window
%                      in both dimensions, so one target gives one detection
%
%   Outputs
%     threshold_map       (N_range x N_doppler) adaptive threshold [power]
%     noise_avg           (N_range x N_doppler) mean training-cell power per CUT
%     detected_positions  (K x 2) [range_bin, doppler_bin], descending power
%     RD_power_map        (N_range x N_doppler) |RD_map|^2

    %% ── Defaults & validation ───────────────────────────────────────────────
    if nargin < 7 || isempty(peak_select), peak_select = true; end

    validateattributes(RD_map,          {'numeric'}, {'2d'});
    validateattributes(N_guard_range,   {'numeric'}, {'scalar','nonnegative','integer'});
    validateattributes(N_guard_doppler, {'numeric'}, {'scalar','nonnegative','integer'});
    validateattributes(N_train_range,   {'numeric'}, {'scalar','positive','integer'});
    validateattributes(N_train_doppler, {'numeric'}, {'scalar','positive','integer'});
    validateattributes(P_fa,            {'numeric'}, {'scalar','positive','<',1});

    [N_range, N_doppler] = size(RD_map);

    %% ── Power map ───────────────────────────────────────────────────────────
    RD_power_map = abs(RD_map).^2;

    %% ── CFAR mask ───────────────────────────────────────────────────────────
    %   Binary mask: 1 = training cell, 0 = guard cell or CUT.
    %   Kernel size is (2*R_out+1) × (2*D_out+1); center = (R_out+1, D_out+1).
    R_out = N_guard_range   + N_train_range;
    D_out = N_guard_doppler + N_train_doppler;
    R_grd = N_guard_range;
    D_grd = N_guard_doppler;

    mask = ones(2*R_out + 1, 2*D_out + 1);
    % Zero-out guard window + CUT (rows/cols centred on kernel centre)
    mask(R_out + 1 - R_grd : R_out + 1 + R_grd, ...
         D_out + 1 - D_grd : D_out + 1 + D_grd) = 0;

    N_train_nom = sum(mask(:));   % nominal (interior) training cell count

    %% ── Local noise sum and actual training-cell count per CUT ──────────────
    %   conv2 with 'same' handles boundaries: at edge cells the kernel is
    %   clipped, so count_map < N_train_nom there.
    noise_sum  = conv2(RD_power_map, mask, 'same');   % power sum over training cells
    count_map  = conv2(ones(N_range, N_doppler), mask, 'same');  % N_eff per CUT

    %% ── Per-cell noise average ───────────────────────────────────────────────
    noise_avg = noise_sum ./ max(count_map, 1);   % guard against divide-by-zero

    %% ── Per-cell alpha ───────────────────────────────────────────────────────
    %   alpha(N) = N*(P_fa^(-1/N) - 1) is computed for each distinct N_eff
    %   (non-linear in N, so a single nominal alpha cannot be rescaled).
    %   count_map takes only a few distinct values (interior = N_train_nom,
    %   smaller counts at the edges), so this loop is fast.
    alpha_map   = zeros(N_range, N_doppler);
    unique_N    = unique(count_map(:));
    for n = unique_N.'
        if n < 1, continue; end
        a = n * (P_fa^(-1/n) - 1);
        alpha_map(count_map == n) = a;
    end

    %% ── Adaptive threshold ───────────────────────────────────────────────────
    threshold_map = alpha_map .* noise_avg;

    %% ── Detection ───────────────────────────────────────────────────────────
    [row_idx, col_idx] = find(RD_power_map > threshold_map);

    %% ── Local-maxima peak selection (no toolbox required) ────────────────────
    %   Discard a detected cell if any cell of its CFAR window (guard + training
    %   cells, clipped at the map edges) has strictly higher power, so one peak
    %   is kept per target blob.
    if peak_select && ~isempty(row_idx)
        keep = true(numel(row_idx), 1);
        for k = 1:numel(row_idx)
            r = row_idx(k);  d = col_idx(k);
            r1 = max(1, r - R_out);  r2 = min(N_range,   r + R_out);
            d1 = max(1, d - D_out);  d2 = min(N_doppler, d + D_out);
            if RD_power_map(r, d) < max(max(RD_power_map(r1:r2, d1:d2)))
                keep(k) = false;
            end
        end
        row_idx = row_idx(keep);
        col_idx = col_idx(keep);
    end

    %% ── Sort detections by descending power ─────────────────────────────────
    if ~isempty(row_idx)
        pwr = RD_power_map(sub2ind([N_range, N_doppler], row_idx, col_idx));
        [~, s] = sort(pwr, 'descend');
        row_idx = row_idx(s);
        col_idx = col_idx(s);
    end

    detected_positions = [row_idx, col_idx];   % (K × 2)

end