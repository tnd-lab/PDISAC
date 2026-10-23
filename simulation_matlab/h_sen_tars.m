function [H_sen_tars, Range_tars, Radial_tars_vel] = h_sen_tars(N_tars, Tars_position, Tars_rcs, Tars_vel, Tx_position, N_tx_ant, Rx_position, N_rx_ant, P_tx, G_tx, G_rx, Lambda, fc, T_step)
%H_SEN_TARS  Monostatic sensing channel of point reflectors at one time step.
%
%   [H_sen_tars, Range_tars, Radial_tars_vel] = H_SEN_TARS(N_tars, Tars_position, ...
%       Tars_rcs, Tars_vel, Tx_position, N_tx_ant, Rx_position, N_rx_ant, ...
%       P_tx, G_tx, G_rx, Lambda, fc, T_step)
%
%   Radar-equation channel of each reflector i (moving target or static
%   scatterer):
%     H_i = sqrt(rcs_i) * (sqrt(PL_i) a_rx,i .* sqrt(A_e)) (a_tx,i .* sqrt(P_tx/N_tx_ant * G_tx)).'
%           * exp(-j 2 pi fc tau_i) * exp(j 2 pi f_D,i T_step),
%   with PL_i = 1/(4 pi R_i^2)^2, A_e = G_rx lambda^2/(4 pi), tau_i = 2 R_i/c,
%   f_D,i = -2 v_i/lambda. Angles and ranges are taken from Tx_position for
%   both hops (Rx assumed co-located with Tx); the Rx array is placed at
%   Rx_position with element spacing N_tx_ant*Lambda/2.
%
% Inputs
%   N_tars          : number of reflectors
%   Tars_position   : 3 x N_tars positions [m]
%   Tars_rcs        : 1 x N_tars RCS [m^2]
%   Tars_vel        : 3 x N_tars velocity vectors [m/s]
%   Tx_position, Rx_position : 3 x 1 array centres [m]
%   N_tx_ant, N_rx_ant       : number of Tx / Rx elements
%   P_tx            : total transmit power [W]
%   G_tx, G_rx      : Tx / Rx antenna gains (linear)
%   Lambda, fc      : wavelength [m], carrier frequency [Hz]
%   T_step          : slow time of the current block [s]
%
% Outputs
%   H_sen_tars      : N_rx_ant x N_tx_ant x N_tars channel matrices
%   Range_tars      : 1 x N_tars Tx-to-reflector range [m]
%   Radial_tars_vel : 1 x N_tars radial velocity [m/s] (positive = receding)

    c = physconst('LightSpeed');
    % Convert to spherical coordinates angle from Tx to targets
    Rel_positions = Tars_position - Tx_position; 
    [Azi_Tx_tars, Ele_Tx_tars, Range_tars] = cart2sph(Rel_positions(1,:), Rel_positions(2,:), Rel_positions(3,:));
    Azi_Tx_tars = rad2deg(Azi_Tx_tars);
    Ele_Tx_tars = rad2deg(Ele_Tx_tars);
    
    % Convert to spherical coordinates angle from targets to Rx (echo back)
    Azi_tars_Rx = Azi_Tx_tars;
    Ele_tars_Rx = Ele_Tx_tars;

    % Array responses Tx -> targets and targets -> Rx
    Tx_tars_array = cell2mat(arrayfun(@(i) func_get_array_response(N_tx_ant, Lambda/2, Azi_Tx_tars(i), 90-Ele_Tx_tars(i), Lambda, Tx_position), 1:N_tars, 'UniformOutput', false));
    Tars_Rx_array = cell2mat(arrayfun(@(i) func_get_array_response(N_rx_ant, N_tx_ant * Lambda/2, Azi_tars_Rx(i), 90-Ele_tars_Rx(i), Lambda, Rx_position), 1:N_tars, 'UniformOutput', false));
    
    % Compute radial velocities
    Radial_tars_vel = zeros(1, N_tars);
    for i = 1:N_tars
        R_hat = Rel_positions(:,i) / norm(Rel_positions(:,i));  % (3×1)
        Radial_tars_vel(i) = dot(Tars_vel(:,i), R_hat);  
    end

    P_tx_ant = (P_tx / N_tx_ant) * ones(N_tx_ant,1); % Tx power of each antenna P_tx / N_tx_ant with shape N_tx_ant x 1 
    G_tx_ant = G_tx * ones(N_tx_ant,1); % Tx gain: N_tx_ant x 1 
    G_rx_ant = G_rx * ones(N_rx_ant,1); % Rx gain: N_rx_ant x 1
    A_e_rx_ant = G_rx_ant * Lambda^2 / (4 * pi); % Effective antenna aperture 
    PL_moving_tars = 1 ./ ((4 * pi * Range_tars.^2) .* (4 * pi * Range_tars.^2)); % 1 x N_tars
    Delay_tars = 2 * Range_tars / c; % 1 x N_tars
    Doppler_tars = -2 * Radial_tars_vel / Lambda; % 1 x N_tars
    
    
    % Sensing channel of each reflector at slow time T_step
    H_sen_tars = zeros(N_rx_ant, N_tx_ant, N_tars);
    for i = 1:N_tars
        H_sen_tars(:,:,i) = sqrt(Tars_rcs(:, i)) ... % target coefficient reflection
                              * ((sqrt(PL_moving_tars(:,i)) .* Tars_Rx_array(:,i) .* sqrt(A_e_rx_ant)) ... % receiver component: antenna gain, path loss, receiver antenna array 
                              * (Tx_tars_array(:,i) .* sqrt(P_tx_ant) .* sqrt(G_tx_ant)).') ... % transmitter component: antenna power, antenna gain, transmit antenna array 
                              * exp(-1j * 2*pi * fc * Delay_tars(:,i)) ... % delay component
                              * exp( 1j * 2*pi * Doppler_tars(:,i) * T_step); % doppler component
    end
end