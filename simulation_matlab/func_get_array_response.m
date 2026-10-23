function array = func_get_array_response(N_ant, d, azi, ele, lambda, position, is_bbe_pattern)
%FUNC_GET_ARRAY_RESPONSE  Steering vectors of a y-axis ULA with optional back baffle.
%
%   array = FUNC_GET_ARRAY_RESPONSE(N_ant, d, azi, ele, lambda, position)
%   array = FUNC_GET_ARRAY_RESPONSE(..., is_bbe_pattern)
%
%   a(u) = g(u) * exp(j*k*p_n^T u), k = 2*pi/lambda, with element positions
%   p_n on a ULA along the y axis centred at 'position', and direction
%   u = [sin(ele)cos(azi); sin(ele)sin(azi); cos(ele)].
%
% Inputs
%   N_ant          : number of elements
%   d              : element spacing [m]
%   azi            : 1 x N_azi azimuth angles [deg]
%   ele            : 1 x N_ele polar angles from the z axis [deg]
%                    (callers pass 90 - elevation)
%   lambda         : wavelength [m]
%   position       : 3 x 1 array centre [m]
%   is_bbe_pattern : (optional, default true) back-baffled element pattern,
%                    g = 1 for u_x > 0 (front half-space) and 0 otherwise;
%                    false gives an isotropic element (g = 1)
%
% Output
%   array          : N_ant x (N_azi*N_ele), azimuth index varying fastest

    if nargin < 7
        is_bbe_pattern = true; % default: back-baffled element
    end

    % Constants
    k = 2*pi/lambda;

    % Meshgrid over all (azi, ele) combinations
    [AZI, ELE] = meshgrid(azi, ele); % N_ele x N_azi
    
    phi   = deg2rad(AZI);  
    theta = deg2rad(ELE);

    % Unit vectors
    ux = sin(theta(:))' .* cos(phi(:))';
    uy = sin(theta(:))' .* sin(phi(:))';
    uz = cos(theta(:))';
    U  = [ux; uy; uz];   % 3 x N_points

    % Array Geometry (Y-axis ULA, centered)
    n_indices = ((0:N_ant-1)' - (N_ant-1)/2);
    pos = [zeros(N_ant,1), n_indices*d, zeros(N_ant,1)];
    pos = pos + position(:)';

    % Steering Matrix
    AF = exp(1j * k * (pos * U));   % N_ant x N_points

    % Back-Baffled Element Pattern
    if is_bbe_pattern
        % front half-space u_x > 0 (-90 < azi < 90 for 0 < ele < 180)
        g = double(ux > 0);   % 1 x N_points
    else
        g = ones(1, size(U,2)); % no attenuation
    end

    % Apply element pattern
    array = AF .* g;

    % Reshape
    N_azi = numel(azi);
    N_ele = numel(ele);
    array = reshape(array, N_ant, N_ele, N_azi);
    array = permute(array, [1, 3, 2]);  % N_ant x N_azi x N_ele
    array = reshape(array, size(array,1), []);
end