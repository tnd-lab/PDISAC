function s = func_bpsk_mod(b)
% FUNC_BPSK_MOD  Binary-PSK modulation (drop-in for pskmod(b,2)).
%   s = FUNC_BPSK_MOD(b) maps bits b in {0,1} to BPSK symbols s in {+1,-1}
%   using s = 1 - 2*b  (0 -> +1, 1 -> -1), i.e. exp(1j*pi*b) with zero phase
%   offset, the same convention as pskmod(b,2). The output is complex and has
%   the shape of b. This is the data-slot mapping s_{k,i} = 1 - 2 b_{k,i}
%   of Sec. V.
    s = complex(1 - 2 * double(b));
end
