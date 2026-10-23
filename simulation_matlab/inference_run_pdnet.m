function [RD_den, meta] = inference_run_pdnet(RD_in, options)
%INFERENCE_RUN_PDNET  Denoise RD maps with the models of alg_pdisac/configs/inference.json.
%
%   [RD_den, meta] = inference_run_pdnet(RD_in)
%   [RD_den, meta] = inference_run_pdnet(RD_in, Models = "pdnet,dncnn")
%
%   RD_in   complex B x H x W (or H x W), H = N_chip, W = N_block (main4 RD_map_shifted)
%   RD_den  complex K x B x H x W (or K x H x W), one slice per model, same order as meta.name
%   meta    .name, .version, .checkpoint (1 x K string), .epoch, .val_nmse (1 x K)
%
%   Name-value options:
%     Models            comma-separated model names; "" (default) = every model of
%                       inference.json with a checkpoint
%     PythonExecutable  "" (default) = PDISAC_cfg.inference.python_executable of
%                       main1_matlab_config; a path not found as given is taken
%                       relative to RepoRoot
%     RepoRoot          "" (default) = parent folder of this file's folder
%     KeepExchangeFiles keep the .mat exchange files (default false)
%     ExchangeDirectory folder of the exchange files (default tempdir)
%
% Which models, their version (pdnet / dncnn) and checkpoints are set in
% alg_pdisac/configs/inference.json; MATLAB only needs the Python interpreter.
% Runs alg_pdisac/scripts/infer/denoise_mat.py through system(), once per call;
% it loads every model once, so pass a batch (e.g. all SNRs of a scene).

arguments
    RD_in {mustBeNumeric, mustBeNonempty}
    options.Models (1,1) string = ""               % "" = every model of inference.json with a checkpoint
    options.PythonExecutable (1,1) string = ""
    options.RepoRoot (1,1) string = ""
    options.KeepExchangeFiles (1,1) logical = false
    options.ExchangeDirectory (1,1) string = string(tempdir)
end

repo_root = options.RepoRoot;
if strlength(repo_root) == 0
    repo_root = string(fileparts(fileparts(mfilename("fullpath"))));
end
python_exe = options.PythonExecutable;
if strlength(python_exe) == 0
    main1_matlab_config;                            % PDISAC_cfg.inference.python_executable
    python_exe = string(PDISAC_cfg.inference.python_executable);
end
if ~isfile(python_exe), python_exe = fullfile(repo_root, python_exe); end
script = fullfile(repo_root, "alg_pdisac", "scripts", "infer", "denoise_mat.py");
must_exist(python_exe, "Python executable");
must_exist(script, "denoise_mat.py");

if ~isfolder(options.ExchangeDirectory), mkdir(options.ExchangeDirectory); end
token = string(char(java.util.UUID.randomUUID));
f_in  = fullfile(options.ExchangeDirectory, "rd_denoise_in_"  + token + ".mat");
f_out = fullfile(options.ExchangeDirectory, "rd_denoise_out_" + token + ".mat");
RD_map_input_inference = single(RD_in); %#ok<NASGU>
save(f_in, "RD_map_input_inference", "-v7");
cleanup = onCleanup(@() cleanup_files(f_in, f_out, options.KeepExchangeFiles)); %#ok<NASGU>

cmd = sprintf('"%s" "%s" --input "%s" --output "%s"', python_exe, script, f_in, f_out);
if strlength(options.Models) > 0, cmd = cmd + " --models " + options.Models; end
[status, msg] = system(cmd);
if status ~= 0
    error("PDISAC:DenoiseFailed", "denoise_mat.py failed (status %d):\n%s", status, msg);
end
if ~isfile(f_out), error("PDISAC:DenoiseNoOutput", "denoise_mat.py produced no %s.", f_out); end

res = load(f_out);
RD_den = res.RD_map_denoised_inference;
meta.name       = string(res.model_name_inference);
meta.version    = string(res.model_version_inference);
meta.checkpoint = string(res.checkpoint_inference);
meta.epoch      = res.epoch_inference;
meta.val_nmse   = res.val_nmse_inference;
K = numel(meta.name);
if ~isequal(size(RD_den), [K size(RD_in)]) && ~(K == 1 && isequal(size(RD_den), size(RD_in)))
    error("PDISAC:DenoiseShape", "denoised size %s, expected %s.", ...
        mat2str(size(RD_den)), mat2str([K size(RD_in)]));
end
RD_den = reshape(RD_den, [K size(RD_in)]);
end


function must_exist(p, what)
if ~isfile(p), error("PDISAC:DenoiseMissing", "%s not found: %s", what, p); end
end

function cleanup_files(f_in, f_out, keep)
if keep, return; end
if isfile(f_in),  delete(f_in);  end
if isfile(f_out), delete(f_out); end
end
