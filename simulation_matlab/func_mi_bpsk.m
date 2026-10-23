function C = func_mi_bpsk(g_in)
%FUNC_MI_BPSK  BPSK-input achievable rate of the despread data-slot channel.
%   C = FUNC_MI_BPSK(g_in) returns, element-wise,
%
%       C = 1 - E_u[ log2(1 + exp(-4*g - 2*sqrt(2*g)*u)) ],   u ~ N(0,1),
%
%   the mutual information (bits per data slot) of a BPSK symbol over a
%   real-valued AWGN channel with SINR g.
%
%   Input:
%     g_in  post-despreading SINR gamma_{k,i} of a data slot (Sec. V), linear,
%           any array size; the paper's convention is BER = Q(sqrt(2*g)).
%           Negative or complex values are clipped to max(real(g), 0).
%   Output:
%     C     BPSK-input rate in bits per data slot, same size as g_in,
%           0 <= C <= 1. C -> g*log2(e) as g -> 0 (same as log2(1+g)) and
%           C -> 1 as g -> inf. log2(1+g) is the Gaussian-input value, not
%           attainable with BPSK signalling.
%
%   The expectation is computed by 40-point Gauss-Hermite quadrature
%   (probabilists' Hermite nodes, cached in a persistent variable).
%
%   Refs: G. Ungerboeck, IEEE Trans. Inf. Theory, 1982;
%         G. Caire, G. Taricco, E. Biglieri, IEEE Trans. Inf. Theory, 1998.

    persistent u w
    if isempty(u)
        [u, w] = local_gauss_hermite_prob(40);
    end

    g   = max(real(g_in(:)), 0);
    arg = -4*g - 2*sqrt(2*g) .* u.';                         % numel(g) x 40
    % log2(1 + e^arg), numerically stable for large |arg|
    l2  = (max(arg, 0) + log1p(exp(-abs(arg)))) / log(2);
    C   = reshape(1 - l2 * w, size(g_in));
end

function [x, w] = local_gauss_hermite_prob(n)
% Nodes/weights for E[f(u)], u ~ N(0,1) (Golub-Welsch, probabilists' Hermite).
    b      = sqrt(1:n-1);
    [V, D] = eig(diag(b, 1) + diag(b, -1));
    [x, i] = sort(diag(D));
    w      = (V(1, i).^2).';                                 % sums to 1
end
