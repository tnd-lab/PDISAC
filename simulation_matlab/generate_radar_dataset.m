% GENERATE_RADAR_DATASET  RD-map dataset for training and testing RDPDNet / DnCNN
% -------------------------------------------------------------------------
% Simulates random scenes with the signal model of Sec. II (main3) and the RD-map
% formation of Sec. III (main4: MF, Doppler FFT, zero-Doppler removal) and writes
% one SQLite database per split. Each trial stores
%   RD_map_compressed           noisy data-bearing RD map Z_rd           (network input)
%   RD_map_no_noise_compressed  noise-free, data-free (MLS-only) RD map  (label)
%   RD_map_alt_compressed       a second, independent realisation of the SAME
%                               scene (other data symbols and other noise)
% One database serves all four Python training scripts (alg_pdisac/scripts/train/),
% so they train on identical data: the AFM scripts (train_pdnet_afm, train_dncnn_afm)
% read all three maps, the scripts without AFM read the first two.
% No detection, estimation or bound is computed here: evaluation belongs to the
% main scripts. The ground truth of every target is stored for them.
%
% WHY THE SECOND REALISATION (AFM)
%   AFM mixes two realisations of the same scene outside the target support,
%   Z_rd = C + D (symbols b, noise N) and Z_rd_alt = C + D' (symbols b', noise N'),
%   Z_mix = C + sqrt(1-M).*D + sqrt(M).*D', where D - D' comes from other data
%   symbols and other noise. Both realisations are physically realisable maps with
%   identical targets and the same label, so the mixed map keeps the label and
%   the second-order statistics of a nominal map for any mask M.
%
% SCENES
%   Per scene (RAN index): cfg.fixed_targets moving targets + the UE (appended as
%   the last target), uniform in the ROI, and cfg.scatterers static scatterers
%   (func_generate_static_scatterers). The scene is fixed over N_bit^prbs
%   (cfg.symbols_per_block), SNR_Tx (cfg.snr_db) and the cfg.monte_carlo runs;
%   data symbols and noise are redrawn for every trial.
%
% SPLITS AND HOW TO RUN
%   Set DATASET = 'train' or 'test' (or the env var PDISAC_DATASET) and run the
%   script. Profiles dataset_train / dataset_test / dataset_smoke are defined in
%   main1_matlab_config.m. Each split has its own seed, so the two sets are
%   independent and each is reproducible. SMOKE = true (or the env var
%   PDISAC_SMOKE) runs the small smoke profile instead.
%   Output: <repo root>/<cfg.output_db>, i.e. alg_pdisac/data/dataset_train.db
%   (training, read by alg_pdisac/configs/pdisac_system.json), dataset_test.db
%   (held-out split, not read by the current Python scripts) or dataset_smoke.db.
%   An existing database of the same name is deleted first.
%   Rows are appended scene by scene, so memory stays bounded by one scene.
%   Requires the Database Toolbox (sqlite) and the Phased Array System Toolbox.
%
% DATABASE SCHEMA  (lossless: every map is float32, GZIP, stored as a BLOB)
%   scalar_results  one row per trial: Trial_ID, RAN_Index, N_sym_per_block,
%                   SNR_dB, MC_Index   (RAN_Index = scene, used for GroupKFold)
%   targets         one row per (scene, target): RAN_Index, Target_Index, Is_UE,
%                   Tar_X/Y/Z, Tar_Vx/Vy/Vz, Tar_RCS, True_Range_m, True_Vel_mps
%   trial_maps      one row per trial: Trial_ID, RAN_Index, SNR_dB,
%                   RD_map_compressed, RD_map_alt_compressed
%   labels          one row per (scene, SNR): RAN_Index, SNR_dB, RD_map_no_noise_compressed.
%                   The label (noise-free, data-free) depends only on the scene and
%                   P_tx, not on N_bit^prbs or the Monte-Carlo draw, so it is stored
%                   once; every later trial is checked against it.
%   matrix_data     VIEW joining trial_maps and labels, with the column names the
%                   Python loaders read (Trial_ID + the three maps).
% -------------------------------------------------------------------------

clear; clc; close all;
NO_SIGNAL_PLOT = true;  NO_BPSK_PLOT = true;      % headless: main3/main4 plot
NO_RADAR_PLOT  = true;  NO_CFAR_PLOT = true;      % nothing inside the sweep

addpath(fileparts(mfilename('fullpath')));
main1_matlab_config;

% =====================================================================
%  SPLIT AND SEED
% =====================================================================
DATASET = 'train';                                   % 'train' | 'test'
if ~isempty(getenv('PDISAC_DATASET')), DATASET = lower(getenv('PDISAC_DATASET')); end
SEED = struct('train', 20260925, 'test', 20260926, 'smoke', 20260927);
SMOKE = false;                                        % true: smoke test -> dataset_smoke.db
smoke_mode = SMOKE || ~isempty(getenv('PDISAC_SMOKE'));
if smoke_mode
    cfg = dataset_smoke_cfg;   split = 'smoke';
else
    switch DATASET
        case 'train', cfg = dataset_train_cfg;
        case 'test',  cfg = dataset_test_cfg;
        otherwise,    error('DATASET must be ''train'' or ''test''.');
    end
    split = DATASET;
end
rng(SEED.(split), 'twister');

G_tx = 10^(cfg.system_override.gain_tx_db/10);
G_rx = 10^(cfg.system_override.gain_rx_db/10);
G_ue = 10^(cfg.system_override.gain_ue_db/10);

N_ran       = cfg.random_scenes;
Mc          = cfg.monte_carlo;
N_sym_list  = cfg.symbols_per_block(:).';
SNR_Tx_db   = cfg.snr_db(:).';
SNR_Tx      = 10.^(SNR_Tx_db/10);
N_tars_fixed = cfg.fixed_targets;
N_scats_cfg  = cfg.scatterers;
N_ref_tars   = PDISAC_cfg.scene.communication_reflectors;   % needed by main3 (comm part)

db_filename = fullfile(PDISAC_repo_root, cfg.output_db);
db_dir      = fileparts(db_filename);
fprintf('*** split = %s | seed = %d | N_ran = %d | N_sym = [%s] | SNR = [%s] dB | Mc = %d\n', ...
    split, SEED.(split), N_ran, num2str(N_sym_list), num2str(SNR_Tx_db), Mc);
n_trials_all = N_ran * numel(N_sym_list) * numel(SNR_Tx) * Mc;
n_labels_all = N_ran * numel(SNR_Tx);
fprintf('*** output: %s  (%d trials, about %.1f GB)\n', db_filename, n_trials_all, ...
    (2 * n_trials_all + n_labels_all) * 0.97e6 / 1e9);   % ~0.97 MB per gzip'd float32 map

% =====================================================================
%  DATABASE (rows appended after every scene)
% =====================================================================
if ~isfolder(db_dir), mkdir(db_dir); end
if isfile(db_filename), delete(db_filename); end
conn = sqlite(db_filename, 'create');
execute(conn, ['CREATE TABLE trial_maps (Trial_ID INTEGER PRIMARY KEY, RAN_Index INTEGER, ' ...
               'SNR_dB REAL, RD_map_compressed BLOB, RD_map_alt_compressed BLOB)']);
execute(conn, ['CREATE TABLE labels (RAN_Index INTEGER, SNR_dB REAL, ' ...
               'RD_map_no_noise_compressed BLOB, PRIMARY KEY (RAN_Index, SNR_dB))']);
execute(conn, ['CREATE VIEW matrix_data AS SELECT t.Trial_ID AS Trial_ID, ' ...
               't.RD_map_compressed AS RD_map_compressed, ' ...
               'l.RD_map_no_noise_compressed AS RD_map_no_noise_compressed, ' ...
               't.RD_map_alt_compressed AS RD_map_alt_compressed ' ...
               'FROM trial_maps t JOIN labels l ' ...
               'ON l.RAN_Index = t.RAN_Index AND l.SNR_dB = t.SNR_dB']);
n_per_scene = numel(N_sym_list) * numel(SNR_Tx) * Mc;
trial_id = 0;

% =====================================================================
%  LOOP 1 — scene (UE, targets, scatterers drawn once per scene)
% =====================================================================
for idx_ran = 1:N_ran
    fprintf('\n===== %s: scene %d / %d =====\n', split, idx_ran, N_ran);
    trial_buf  = nan(n_per_scene, 5);   % Trial_ID, RAN_Index, N_sym_per_block, SNR_dB, MC_Index
    target_buf = nan(N_tars_fixed + 1, 12);
    label_ref  = cell(1, numel(SNR_Tx)); % label of this scene at each SNR (stored once)
    k = 0;

    % ---- UE ----
    N_ue        = 1;
    UE_position = rand_pos_in_roi(Region_of_interest, N_ue);
    UE_vel      = VEL_RANGE_UE * (2*rand(3,1) - 1);   UE_vel(3) = 0;
    UE_rcs      = 10^( (10*log10(10) + (10*log10(20)-10*log10(10))*rand()) / 10 );
    UE_motion   = phased.Platform('InitialPosition', UE_position, 'Velocity', UE_vel);

    % ---- moving targets, UE appended as the last one ----
    N_tars        = N_tars_fixed;
    Tars_position = rand_pos_in_roi(Region_of_interest, N_tars);
    Tars_vel      = VEL_RANGE_TAR * (2*rand(3, N_tars) - 1);   Tars_vel(3,:) = 0;
    Tars_rcs      = 10.^( (10*log10(10) + (10*log10(20)-10*log10(10))*rand(1, N_tars)) / 10 );
    N_tars        = N_tars + 1;
    Tars_position = [Tars_position, UE_position];
    Tars_vel      = [Tars_vel,      UE_vel];
    Tars_rcs      = [Tars_rcs,      UE_rcs];
    Tars_motion   = phased.Platform('InitialPosition', Tars_position, 'Velocity', Tars_vel);

    % ---- static scatterers ----
    N_scats        = N_scats_cfg;
    Scats_position = func_generate_static_scatterers(N_scats, Region_of_interest);
    Scats_vel      = zeros(size(Scats_position));
    Scats_rcs      = 10.^( (10*log10(5) + (10*log10(10)-10*log10(5))*rand(1, N_scats)) / 10 );
    Scats_motion   = phased.Platform('InitialPosition', Scats_position, 'Velocity', Scats_vel);

    % ---- ground truth of every target (for the evaluation in the main scripts) ----
    rel         = Tars_position - Tx_position;
    truth_range = sqrt(sum(rel.^2, 1));
    truth_vel   = sum(Tars_vel .* rel ./ (truth_range + eps), 1);
    for j = 1:N_tars
        target_buf(j, :) = [idx_ran, j, double(j == N_tars), ...
            Tars_position(:,j).', Tars_vel(:,j).', Tars_rcs(j), truth_range(j), truth_vel(j)];
    end

    % ==================================================================
    %  LOOP 2/3/4 — N_sym_per_block, SNR, Monte-Carlo (scene fixed)
    % ==================================================================
    for N_sym_per_block = N_sym_list
        L_slot_per_sym = N_chip / (2 * N_sym_per_block);
        N_slot         = N_chip / L_slot_per_sym;
        for idx_snr = 1:numel(SNR_Tx)
            P_tx = Noise_power_sen * SNR_Tx(idx_snr);
            for idx_mc = 1:Mc
                % second realisation of this scene (independent symbols and noise)
                main3_signal_channel_model;
                main4_sensing_process;
                RD_map_alt_shifted = RD_map_shifted;

                % primary realisation
                main3_signal_channel_model;
                main4_sensing_process;

                trial_id = trial_id + 1;  k = k + 1;
                trial_buf(k, :) = [trial_id, idx_ran, N_sym_per_block, SNR_Tx_db(idx_snr), idx_mc];
                execute(conn, sprintf(['INSERT INTO trial_maps VALUES (%d, %d, %.10g, ' ...
                    'X''%s'', X''%s'')'], trial_id, idx_ran, SNR_Tx_db(idx_snr), ...
                    blobHex(RD_map_shifted), blobHex(RD_map_alt_shifted)));

                % label: stored at the first trial of (scene, SNR), checked afterwards
                L_single = single(RD_map_no_noise_shifted);
                if isempty(label_ref{idx_snr})
                    label_ref{idx_snr} = L_single;
                    execute(conn, sprintf('INSERT INTO labels VALUES (%d, %.10g, X''%s'')', ...
                        idx_ran, SNR_Tx_db(idx_snr), blobHex(RD_map_no_noise_shifted)));
                elseif max(abs(L_single(:) - label_ref{idx_snr}(:))) > ...
                        1e-5 * max(abs(label_ref{idx_snr}(:)))
                    error('Label of scene %d at %g dB differs between trials; it cannot be shared.', ...
                        idx_ran, SNR_Tx_db(idx_snr));
                end
            end
        end
        fprintf('  N_sym = %3d done (%d trials so far)\n', N_sym_per_block, trial_id);
    end

    % ---- append this scene's trial and target rows (the maps are already inserted) ----
    sqlwrite(conn, 'scalar_results', array2table(trial_buf(1:k, :), 'VariableNames', ...
        {'Trial_ID','RAN_Index','N_sym_per_block','SNR_dB','MC_Index'}));
    sqlwrite(conn, 'targets', array2table(target_buf, 'VariableNames', ...
        {'RAN_Index','Target_Index','Is_UE','Tar_X','Tar_Y','Tar_Z', ...
         'Tar_Vx','Tar_Vy','Tar_Vz','Tar_RCS','True_Range_m','True_Vel_mps'}));
end
close(conn);
fprintf('\n%s: %d trials, %d scenes written to %s\n', split, trial_id, N_ran, db_filename);


% =====================================================================
%  LOCAL FUNCTIONS
% =====================================================================
function pos = rand_pos_in_roi(roi, N)
    pos = zeros(3, N);
    for dim = 1:3
        lo = roi(dim,1);  hi = roi(dim,2);
        if lo == hi, pos(dim,:) = lo; else, pos(dim,:) = lo + (hi - lo) .* rand(1, N); end
    end
end

function h = blobHex(M)
    % float32 [real; imag] (column-major), GZIP, as a hex string for an SQL X'..'
    % BLOB literal (lossless for the float32 maps).
    rawBytes = [typecast(single(real(M(:))), 'uint8'); typecast(single(imag(M(:))), 'uint8')];
    baos = java.io.ByteArrayOutputStream();
    gzos = java.util.zip.GZIPOutputStream(baos);
    gzos.write(rawBytes);
    gzos.close();
    b = typecast(baos.toByteArray(), 'uint8');
    b = b(:).';                                   % row, so [hi; lo] interleaves per byte
    hexdig = '0123456789ABCDEF';
    h = [hexdig(bitshift(b, -4) + 1); hexdig(bitand(b, 15) + 1)];
    h = reshape(h, 1, []);
end
