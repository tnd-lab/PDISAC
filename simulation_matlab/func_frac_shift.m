function y = func_frac_shift(x, d)
%FUNC_FRAC_SHIFT  Band-limited fractional shift, y(n) = x(n - d).
%
%   y = FUNC_FRAC_SHIFT(x, d)
%
%   d is in samples, may be negative and need not be an integer. Applied as a
%   DFT phase ramp, so the shift is circular: exact for a periodic sequence
%   (e.g. the length-N_chip MLS), approximate otherwise. The Nyquist bin is
%   forced real so a real x stays real. Used by main6_comm_process for UE
%   timing alignment.
%
% Inputs
%   x : (L x 1) real or complex sequence (a row vector is treated as a column)
%   d : scalar shift in samples (positive = delay)
%
% Output
%   y : (L x 1) shifted sequence, real if x is real

    L    = numel(x);
    k    = [0:ceil(L/2)-1, -floor(L/2):-1].';
    ramp = exp(-1j * 2*pi * d * k / L);
    if mod(L, 2) == 0
        ramp(L/2 + 1) = cos(pi * d);
    end
    y = ifft(fft(x(:)) .* ramp);
    if isreal(x), y = real(y); end
end
