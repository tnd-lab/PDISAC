function [det_range_interp, det_vel_interp] = func_sinc_interpolation( ...
        RD_map_complex, detected_positions, range_axis, vel_axis, N_sinc, N_fine)
%FUNC_SINC_INTERPOLATION  Sub-bin range/velocity refinement of CA-CFAR detections.
%
%   [det_range_interp, det_vel_interp] = FUNC_SINC_INTERPOLATION( ...
%       RD_map_complex, detected_positions, range_axis, vel_axis)
%   [...] = FUNC_SINC_INTERPOLATION(..., N_sinc, N_fine)
%
%   For each detection, the complex RD map is reconstructed on a fine grid
%   within +-0.5 bin of the detected cell by Whittaker-Shannon (sinc)
%   interpolation over a window of +-N_sinc bins, separately along range and
%   Doppler; the refined position is the maximum of the reconstructed
%   magnitude, mapped to physical units through range_axis / vel_axis.
%
% Inputs
%  RD_map_complex      (N_range x N_doppler)  complex RD map (not magnitude),
%                                             Doppler axis fftshift-ed
%  detected_positions  (K x 2)                [range_bin, doppler_bin]
%  range_axis          (1 x N_range)          physical range    [m]
%  vel_axis            (1 x N_doppler)        physical velocity [m/s]
%  N_sinc              half-width of the sinc window in bins (default 16)
%  N_fine              fine-grid points per half-bin (default 128)
%
% Outputs
%  det_range_interp    (K x 1)  refined range    [m]
%  det_vel_interp      (K x 1)  refined velocity [m/s]

    if nargin < 5 || isempty(N_sinc), N_sinc = 16;  end
    if nargin < 6 || isempty(N_fine), N_fine = 128; end

    [N_range, N_doppler] = size(RD_map_complex);
    K = size(detected_positions, 1);

    det_range_interp = zeros(K, 1);
    det_vel_interp   = zeros(K, 1);

    for k = 1 : K
        r = detected_positions(k, 1);
        d = detected_positions(k, 2);

        %% Range dimension ────────────────────────────────────────────────
        r_lo  = max(1,       r - N_sinc);
        r_hi  = min(N_range, r + N_sinc);
        r_win = r_lo : r_hi;

        % Use complex samples, then take magnitude after reconstruction.
        Z_range = RD_map_complex(r_win, d);         % complex (W x 1)

        % Search the local sub-bin interval around the detected peak.
        f_range    = linspace(r - 0.5, r + 0.5, 2*N_fine + 1);

        % sinc matrix: W x (2*N_fine+1); sinc is even, so the sign of the argument does not matter
        sinc_mat_r = sinc(r_win(:) - f_range);

        % Reconstruct complex signal at fine grid, then take magnitude
        X_fine_r   = abs(Z_range.' * sinc_mat_r);

        [~, idx_max_r] = max(X_fine_r);
        r_refined      = f_range(idx_max_r);

        det_range_interp(k) = interp1(1:N_range, range_axis, ...
                                      r_refined, 'linear', 'extrap');

        %% Doppler dimension ──────────────────────────────────────────────
        d_lo  = max(1,         d - N_sinc);
        d_hi  = min(N_doppler, d + N_sinc);
        d_win = d_lo : d_hi;

        Z_dop      = RD_map_complex(r, d_win) .* ...
                     exp(1j*pi*(N_doppler-1)*(d_win - N_doppler/2 - 1)/N_doppler);   % remove the linear phase of the slow-time DFT kernel

        f_dop      = linspace(d - 0.5, d + 0.5, 2*N_fine + 1);

        sinc_mat_d = sinc(d_win(:) - f_dop);        % (W x (2*N_fine+1))

        X_fine_d   = abs(Z_dop * sinc_mat_d);

        [~, idx_max_d] = max(X_fine_d);
        d_refined      = f_dop(idx_max_d);

        det_vel_interp(k) = interp1(1:N_doppler, vel_axis, ...
                                    d_refined, 'linear', 'extrap');
    end
end
