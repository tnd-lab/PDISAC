"""ISAC constants derived from the repository-wide shared configuration."""
import numpy as np
import scipy.constants

from configs.config_loader import system_cfg

fc = float(system_cfg["system"]["carrier_hz"])
c = scipy.constants.speed_of_light
Lambda = c / fc
B = float(system_cfg["system"]["bandwidth_hz"])
fs = B
N_chip = int(system_cfg["waveform"]["chips"])
N_block = int(system_cfg["waveform"]["blocks"])

T_chip = 1 / B
T_block = N_chip * T_chip
range_resolution = c / (2 * B)
max_range = range_resolution * N_chip
range_axis = np.arange(N_chip) * range_resolution
range_max = float(np.max(np.asarray(system_cfg["scene"]["roi_m"])[0]))
prf = 1 / T_block
fd_axis = np.arange(-N_block // 2, N_block // 2) * (prf / N_block)
vel_axis = -(Lambda / 2) * fd_axis
vel_max = float(system_cfg["scene"]["max_target_speed_mps"])
