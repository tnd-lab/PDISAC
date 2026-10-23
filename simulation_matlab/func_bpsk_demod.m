function b = func_bpsk_demod(y)
% FUNC_BPSK_DEMOD  Binary-PSK hard-decision demodulation (drop-in for pskdemod(y,2)).
%   b = FUNC_BPSK_DEMOD(y) decides on the sign of the real part (decision
%   boundary = imaginary axis): Re{y} >= 0 -> 0, Re{y} < 0 -> 1. The output is
%   double and has the shape of y. Inverse of func_bpsk_mod; matches
%   pskdemod(y,2) with zero phase offset. BPSK detection of Sec. V.
    b = double(real(y) < 0);
end
