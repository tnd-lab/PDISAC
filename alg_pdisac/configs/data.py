from configs.config_loader import cfg, resolve_path

output_dir = resolve_path(cfg["output_dir"])
db_path = resolve_path(cfg["db_path"])
n_records = cfg["n_records"]
num_bins = cfg["num_bins"]
