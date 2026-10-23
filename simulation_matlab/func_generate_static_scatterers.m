function [positions] = func_generate_static_scatterers(num_scatterers, Region_of_interest)
%FUNC_GENERATE_STATIC_SCATTERERS  Random clustered static scatterers in the ROI.
%
%   positions = FUNC_GENERATE_STATIC_SCATTERERS(num_scatterers, Region_of_interest)
%
%   Draws 40 cluster centres uniformly over the region of interest and
%   ceil(num_scatterers/40) scatterers per cluster, Gaussian around the centre
%   with a per-axis standard deviation drawn uniformly in [0.5, 3.5] % of the
%   ROI size along that axis. Any cluster with a member closer than R_MIN = 10 m
%   to the origin (the co-located Tx/Rx in the default configuration) is redrawn
%   (near-range exclusion). The set is then trimmed to num_scatterers.
%
% Inputs
%   num_scatterers     : number of scatterers to return
%   Region_of_interest : 3 x 2 [x_min x_max; y_min y_max; z_min z_max] [m]
%
% Output
%   positions          : 3 x num_scatterers scatterer positions [m]

    R_MIN = 10;  % near-range exclusion radius [m]

    numClusters = 40;                  % number of clusters
    N = ceil(num_scatterers/numClusters); % scatterers per cluster
    
    % Size of the region
    A = Region_of_interest(1,2) - Region_of_interest(1,1);
    B = Region_of_interest(2,2) - Region_of_interest(2,1);
    C = Region_of_interest(3,2) - Region_of_interest(3,1);
    
    % Random cluster centers in 3D, uniform over the region of interest
    clusterCenters = [rand(1,numClusters) * A + Region_of_interest(1,1); 
                      rand(1,numClusters) * B + Region_of_interest(2,1); 
                      rand(1,numClusters) * C + Region_of_interest(3,1)];
    
    positions = zeros(3, N*numClusters);
    
    for i = 1:numClusters
        % Gaussian spread around the cluster center; per-axis std uniform in
        % [0.5, 3.5] % of the ROI size
        a = A * (rand()*0.03 + 0.005); % x-spread
        b = B * (rand()*0.03 + 0.005); % y-spread
        c = C * (rand()*0.03 + 0.005); % z-spread
    
        positions(:, (i-1)*N+1 : i*N) = [randn(1,N)*a; randn(1,N)*b; randn(1,N)*c] + clusterCenters(:,i);
    end

    % Near-range exclusion: redraw every cluster with a member inside R_MIN
    for i = 1:numClusters
        idx = (i-1)*N+1 : i*N;
        while any(vecnorm(positions(:, idx), 2, 1) < R_MIN)
            ctr = [rand() * A + Region_of_interest(1,1);
                   rand() * B + Region_of_interest(2,1);
                   rand() * C + Region_of_interest(3,1)];
            a = A * (rand()*0.03 + 0.005);
            b = B * (rand()*0.03 + 0.005);
            c = C * (rand()*0.03 + 0.005);
            positions(:, idx) = [randn(1,N)*a; randn(1,N)*b; randn(1,N)*c] + ctr;
        end
    end
    
    % Trim to exact number of scatterers
    positions = positions(:, 1:num_scatterers);
end
