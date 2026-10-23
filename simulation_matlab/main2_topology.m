% MAIN2_TOPOLOGY  Scene geometry of the PDISAC simulator (Sec. II)
% -------------------------------------------------------------------------
% Runs main1_matlab_config, then builds the scene: one UE, the moving targets
% of the configuration, and N_scats random clustered static scatterers
% (func_generate_static_scatterers). RCS values are drawn uniformly in dB over
% the configured ranges. The UE is appended as the last entry of the target
% set because it also reflects the sensing signal. Optionally prints the
% target/scatterer/UE geometry and draws the 2-D plan view (figure Name
% 'fig_topology').
%
% Run as a script in the base workspace:
%     main2_topology;
%
% Flags (set before running):
%     TOPO_NO_PLOT = true   build the scene without console report or figure
%
% Populates: N_ue, UE_position, UE_vel, UE_rcs, UE_motion,
%   N_tars, Tars_position, Tars_vel, Tars_rcs, Tars_motion (targets + UE),
%   N_scats, Scats_position, Scats_vel, Scats_rcs, Scats_motion,
%   plus all variables of main1_matlab_config.

%% Build scene ----------------------------------------------------------------
main1_matlab_config;

% ---- User equipment (UE) ----
N_ue = 1;
UE_position = PDISAC_cfg.scene.ue_position_m(:);
UE_vel = PDISAC_cfg.scene.ue_velocity_mps(:);
UE_motion = phased.Platform('InitialPosition', UE_position, 'Velocity', UE_vel);

target_rcs_min_db = 10 * log10(PDISAC_cfg.scene.target_rcs_range(1));
target_rcs_max_db = 10 * log10(PDISAC_cfg.scene.target_rcs_range(2));
UE_rcs_db = target_rcs_min_db + ...
    (target_rcs_max_db - target_rcs_min_db) * rand(1, N_ue);
UE_rcs = 10.^(UE_rcs_db / 10);

% ---- Moving targets ----
Tars_position = PDISAC_cfg.scene.target_positions_m;
Tars_vel = PDISAC_cfg.scene.target_velocities_mps;
N_tars = size(Tars_position, 2);
Tars_rcs_db = target_rcs_min_db + ...
    (target_rcs_max_db - target_rcs_min_db) * rand(1, N_tars);
Tars_rcs = 10.^(Tars_rcs_db / 10);

% Append the UE to the target set (UE also acts as a reflector for sensing).
Tars_position = [Tars_position, UE_position];
Tars_vel = [Tars_vel, UE_vel];
Tars_rcs = [Tars_rcs, UE_rcs];
N_tars = size(Tars_position, 2);
Tars_motion = phased.Platform( ...
    'InitialPosition', Tars_position, 'Velocity', Tars_vel);

% ---- Static scatterers ----
N_scats = PDISAC_cfg.scene.scatterers;
Scats_position = func_generate_static_scatterers( ...
    N_scats, Region_of_interest);
Scats_vel = zeros(size(Scats_position));
scatter_rcs_min_db = 10 * log10(PDISAC_cfg.scene.scatterer_rcs_range(1));
scatter_rcs_max_db = 10 * log10(PDISAC_cfg.scene.scatterer_rcs_range(2));
Scats_rcs_db = scatter_rcs_min_db + ...
    (scatter_rcs_max_db - scatter_rcs_min_db) * rand(1, N_scats);
Scats_rcs = 10.^(Scats_rcs_db / 10);
Scats_motion = phased.Platform( ...
    'InitialPosition', Scats_position, 'Velocity', Scats_vel);

%% Visualize ------------------------------------------------------------------
if ~exist('TOPO_NO_PLOT', 'var') || ~TOPO_NO_PLOT
    show_topology(Tx_position, Rx_position, UE_position, UE_vel, ...
        Tars_position, Tars_vel, Scats_position, ...
        N_tx_ant, N_rx_ant, N_ue_ant, N_tars, N_scats, N_ue, Lambda, N_ref_tars, Region_of_interest);
end


% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================
function show_topology(Tx_position, Rx_position, UE_position, UE_vel, ...
        Tars_position, Tars_vel, Scats_position, ...
        N_tx_ant, N_rx_ant, N_ue_ant, N_tars, N_scats, N_ue, Lambda, N_ref_tars, Region_of_interest)
% Angle bookkeeping, console report, and 2-D plan-view rendering.

    % ---- Moving targets (Tx -> target -> Rx) --------------------------------
    Rel_positions = Tars_position - Tx_position;
    [Azi_Tx_tars, Ele_Tx_tars, Range_tars] = ...
        cart2sph(Rel_positions(1,:), Rel_positions(2,:), Rel_positions(3,:));
    Azi_Tx_tars = rad2deg(Azi_Tx_tars);
    Ele_Tx_tars = rad2deg(Ele_Tx_tars);
    Azi_tars_Rx = Azi_Tx_tars;
    Ele_tars_Rx = Ele_Tx_tars;

    for i = 1:N_tars
        fprintf('Moving Target %d: Range = %.2f, Azimuth = %.2f deg, Elevation = %.2f deg\n', ...
            i, Range_tars(i), Azi_Tx_tars(i), Ele_Tx_tars(i));
    end

    % ---- Static scatterers --------------------------------------------------
    Rel_positions = Scats_position - Tx_position;
    [Azi_Tx_scats, Ele_Tx_scats, Range_scats] = ...
        cart2sph(Rel_positions(1,:), Rel_positions(2,:), Rel_positions(3,:));
    Azi_Tx_scats = rad2deg(Azi_Tx_scats);
    Ele_Tx_scats = rad2deg(Ele_Tx_scats);

    for i = 1:N_scats
        fprintf('Static Target %d: Range = %.2f m, Azimuth = %.2f deg, Elevation = %.2f deg\n', ...
            i, Range_scats(i), Azi_Tx_scats(i), Ele_Tx_scats(i));
        if i == 10, break; end
    end

    % ---- UE -----------------------------------------------------------------
    Rel_positions = UE_position - Tx_position;
    [Azi_Tx_UE, Ele_Tx_UE, Range_Tx_UE] = ...
        cart2sph(Rel_positions(1,:), Rel_positions(2,:), Rel_positions(3,:));
    Azi_Tx_UE = rad2deg(Azi_Tx_UE);
    Ele_Tx_UE = rad2deg(Ele_Tx_UE);

    Rel_positions = UE_position - Tx_position;
    R_hat = Rel_positions / norm(Rel_positions);
    Radial_UE_vel = dot(UE_vel, R_hat);

    for i = 1:N_ue
        fprintf('UE %d: Range = %.2f, Azimuth = %.2f deg, Elevation = %.2f deg, Radial Velocity = %.2f m/s\n', ...
            i, Range_Tx_UE(i), Azi_Tx_UE(i), Ele_Tx_UE(i), Radial_UE_vel(i));
    end

    plot_fov(Tx_position, Rx_position, UE_position, UE_vel, ...
        Tars_position, Tars_vel, Scats_position, ...
        N_tx_ant, N_rx_ant, N_ue_ant, Lambda, ...
        Azi_Tx_tars, 90-Ele_Tx_tars, ...
        Azi_tars_Rx, 90-Ele_tars_Rx, ...
        Azi_Tx_UE,   90-Ele_Tx_UE, ...
        Azi_Tx_UE,   90-Ele_Tx_UE, ...
        N_ref_tars, Region_of_interest);
    view(2)   % top-down plan view
end

function plot_fov(Tx_position, Rx_position, UE_position, UE_vel, ...
    Tars_position, Tars_vel, Scats_position, ...
    N_tx_ant, N_rx_ant, N_ue_ant, Lambda, ...
    Azi_Tx_tars, Ele_Tx_tars, ...
    Azi_tars_Rx, Ele_tars_Rx, ...
    Azi_Tx_UE,   Ele_Tx_UE, ...
    Azi_UE,      Ele_UE, ...
    N_ref_tars, Region_of_interest)
%PLOT_FOV  2-D plan view of the ISAC scene, drawn at its print size
%   (single IEEE column, 3.5 in wide) so fonts keep their nominal size when
%   the exported PDF is included at full column width.
%   Velocity arrows are the displacement over 1 s (arrow length = speed in m).

    if nargin < 20 || isempty(N_ref_tars)
        N_ref_tars = 6;   % draw the closest NLoS reflectors by default
    end

    % -- Print size and fonts (pt, at the final printed size) -----------
    FIG_IN    = [3.5 2.45];   % [width height] in inches
    FS_AX     = 8;           % tick labels
    FS_LBL    = 9;           % axis labels
    FS_TXT    = 8;           % UE / target labels
    FS_RING   = 7;           % range-ring labels
    FS_LGD    = 7;           % legend
    fontName  = 'Times New Roman';

    % -- Palette --------------------------------------------------------
    c_bs   = [1.00 0.85 0.00];
    c_ue   = [0.00 0.45 0.74];
    c_tar  = [0.85 0.20 0.20];
    c_scat = [0.55 0.55 0.55];
    c_nlos = [0.20 0.62 0.30];
    c_sen  = [0.93 0.55 0.13];

    % -- Drop the UE copy appended as the last "target" in the build cell
    is_ue = vecnorm(Tars_position - UE_position) < 1e-6;
    Tp = Tars_position(:, ~is_ue);
    Tv = Tars_vel(:,      ~is_ue);
    nT = size(Tp, 2);

    bs = Tx_position(:);
    ue = UE_position(:);

    % -- Figure (Name is used by main8_main STEP 5 as the exported file name)
    fig = figure('Color','w', 'Name','fig_topology', 'Units','inches', ...
           'Position',[1 1 FIG_IN]);
    ax = axes(fig); hold(ax,'on'); box(ax,'on'); grid(ax,'off');
    set(ax,'Layer','top','FontSize',FS_AX,'FontName',fontName, ...
           'TickLabelInterpreter','latex','LineWidth',0.6);

    % -- Region of interest (from the configuration) --------------------
    roi_x = Region_of_interest(1,:);  roi_y = Region_of_interest(2,:);
    plot([roi_y(1) roi_y(1) roi_y(2) roi_y(2) roi_y(1)], ...
         [roi_x(1) roi_x(2) roi_x(2) roi_x(1) roi_x(1)], ...
         ':', 'Color',[0.6 0.6 0.6], 'LineWidth',0.6, 'HandleVisibility','off');

    % -- Range rings (half circles, x >= 0) ------------------------------
    ring_step = 25;
    ring_r = ring_step:ring_step:max([roi_x(2), abs(roi_y)]);
    th = linspace(0, pi, 181);
    for k = 1:numel(ring_r)
        r = ring_r(k);
        plot(bs(2)+r*cos(th), bs(1)+r*sin(th), '-', ...
             'Color',[0.85 0.85 0.85], 'LineWidth',0.5, 'HandleVisibility','off');
        text(bs(2)+r*cosd(4), bs(1)+r*sind(4), sprintf('%d m',r), ...
             'Color',[0.5 0.5 0.5], 'FontSize',FS_RING, 'FontName',fontName, ...
             'Interpreter','latex', 'Clipping','on', ...
             'HorizontalAlignment','right', 'VerticalAlignment','bottom');
    end

    % -- Static scatterers ----------------------------------------------
    h_scat = scatter(Scats_position(2,:), Scats_position(1,:), 6, c_scat, 'o', ...
            'filled', 'MarkerFaceAlpha',0.5, 'DisplayName','Scatterers');

    % -- NLoS reflectors: the N_ref_tars targets/scatterers closest to the UE
    %    (same rule as h_com_nlos in main3_signal_channel_model) ----------
    All_tars_position = [Tp, Scats_position];
    Range_All_tars    = vecnorm(UE_position - All_tars_position, 2, 1);
    [~, idx]          = sort(Range_All_tars, 'ascend');
    Ref_tars_position = All_tars_position(:, idx(1:min(N_ref_tars, numel(idx))));

    % -- Sensing links: BS <-> target -----------------------------------
    for i = 1:nT
        h = plot([bs(2) Tp(2,i)], [bs(1) Tp(1,i)], '-', ...
             'Color',[c_sen 0.8], 'LineWidth',0.9, 'DisplayName','Sensing link');
        if i == 1, h_sen = h; else, h.HandleVisibility = 'off'; end
    end

    % -- Communication NLoS links: BS -> reflector -> UE ------------------
    for i = 1:size(Ref_tars_position, 2)
        refl = Ref_tars_position(:,i);
        h = plot([bs(2) refl(2) ue(2)], [bs(1) refl(1) ue(1)], '--', ...
             'Color',c_nlos, 'LineWidth',0.5, 'DisplayName','NLoS link');
        if i == 1, h_nlos = h; else, h.HandleVisibility = 'off'; end
        scatter(refl(2), refl(1), 16, c_nlos, 'o', 'LineWidth',0.7, ...
                'HandleVisibility','off');
    end

    % -- Communication LoS link: BS -> UE (on top of the NLoS paths) -----
    h_los = plot([bs(2) ue(2)], [bs(1) ue(1)], '-', ...
         'Color',c_ue, 'LineWidth',1.4, 'DisplayName','LoS link');

    % -- Radar targets + velocity vectors (displacement over 1 s) --------
    for i = 1:nT
        quiver(Tp(2,i), Tp(1,i), Tv(2,i), Tv(1,i), 0, ...
               'Color',c_tar, 'LineWidth',0.9, 'MaxHeadSize',0.6, ...
               'HandleVisibility','off');
        text(Tp(2,i)+3, Tp(1,i)+4, sprintf('Tar %d', i), ...
             'Interpreter','latex', 'FontSize',FS_TXT, 'FontName',fontName, ...
             'Color',c_tar);
    end
    h_tar = scatter(Tp(2,:), Tp(1,:), 30, c_tar, '^', 'filled', ...
            'MarkerEdgeColor','k', 'LineWidth',0.5, 'DisplayName','Targets');

    % -- UE + velocity vector -------------------------------------------
    quiver(ue(2), ue(1), UE_vel(2), UE_vel(1), 0, ...
           'Color',c_ue, 'LineWidth',0.9, 'MaxHeadSize',0.8, ...
           'HandleVisibility','off');
    h_ue = scatter(ue(2), ue(1), 36, c_ue, 's', 'filled', ...
            'MarkerEdgeColor','k', 'LineWidth',0.5, 'DisplayName','UE');
    text(ue(2)-5, ue(1)+1, 'UE', 'Interpreter','latex', 'FontSize',FS_TXT, ...
         'FontName',fontName, 'Color',c_ue, 'HorizontalAlignment','right', ...
         'BackgroundColor','w', 'Margin',0.5);

    % -- Base station (co-located Tx/Rx) ---------------------------------
    h_bs = scatter(bs(2), bs(1), 70, c_bs, 'p', 'filled', ...
            'MarkerEdgeColor','k', 'LineWidth',0.5, 'DisplayName','Tx/Rx');

    % -- Axes, labels, legend --------------------------------------------
    pad = 8;
    xl = [roi_y(1)-pad, roi_y(2)+pad];
    yl = [roi_x(1)-pad, roi_x(2)+pad];
    xlim(ax, xl); ylim(ax, yl); daspect(ax, [1 1 1]);
    xlabel('$y$ (m)', 'Interpreter','latex', 'FontSize',FS_LBL, 'FontName',fontName);
    ylabel('$x$ (m)', 'Interpreter','latex', 'FontSize',FS_LBL, 'FontName',fontName);

    lgd = legend([h_bs h_ue h_tar h_scat h_sen h_los h_nlos], ...
                 'NumColumns',4, 'Box','off', ...
                 'Interpreter','latex', 'FontSize',FS_LGD);
    lgd.FontName = fontName;
    lgd.ItemTokenSize = [10 6];

    % -- Explicit layout in inches: the axes height follows the data aspect
    %    ratio and the legend sits right above it, so no empty band is left.
    M_L = 0.42; M_R = 0.08; M_B = 0.40;   % room for y/x tick labels and labels
    GAP = 0.02; M_T = 0.01;               % axes-legend gap, top margin
    pw  = FIG_IN(1) - M_L - M_R;
    ph  = pw * diff(yl) / diff(xl);
    lgd.Units = 'inches';  lp = lgd.Position;
    fig.Position(4) = M_B + ph + GAP + lp(4) + M_T;
    ax.Units = 'inches';   ax.Position = [M_L M_B pw ph];
    lgd.Position = [(FIG_IN(1) - lp(3))/2, M_B + ph + GAP, lp(3), lp(4)];
    hold(ax,'off');
end
