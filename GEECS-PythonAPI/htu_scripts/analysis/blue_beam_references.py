import os
import time
import numpy as np
import calendar as cal
from pathlib import Path
from typing import Optional, Union, Any
import matplotlib.pyplot as plt
from tkinter import filedialog
from progressbar import ProgressBar
from geecs_api.api_defs import SysPath
from geecs_api.tools.scans.scan_data import ScanData
from geecs_api.tools.interfaces.exports import load_py, save_py
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.tools.interfaces.prompts import text_input
from geecs_api.devices.HTU.diagnostics.cameras.camera import Camera
from geecs_api.tools.images.scan_images import ScanImages
from geecs_api.tools.images.filtering import FiltersParameters
from htu_scripts.analysis.beam_analyses_collector import add_beam_analysis
from screens_scan_analysis import screens_scan_analysis, render_screens_scan_analysis


# base_path = Path(r'C:\Users\GuillaumePlateau\Documents\LBL\Data')
base_path = Path(r'Z:\data')

is_local = (str(base_path)[0] == 'C')
if not is_local:
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

# Parameters
_base_tag = (2023, 6, 9, 1)
_screen_labels = ['A2', 'A3', 'U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8']

_save_dir: Path = base_path/'Undulator'/f'Y{_base_tag[0]}'/f'{_base_tag[1]:02d}-{cal.month_name[_base_tag[1]][:3]}'\
                  / f'{str(_base_tag[0])[-2:]}_{_base_tag[1]:02d}{_base_tag[2]:02d}'/'analysis'\
                  / f'Scan{_base_tag[3]:03d}_Screens_{_screen_labels[0]}_{_screen_labels[-1]}'

_analysis_path: Path = _save_dir / 'beam_analysis.dat'

# Load analysis
_data_dict, _analysis_path = load_py(_analysis_path, as_dict=True)

# Targets
targets = [ScanImages.rotated_to_original_ij(pos, shape, rot90)
           for rot90, shape, pos in zip(_data_dict['beam_analysis']['rot_90'],
                                        _data_dict['beam_analysis']['raw_shape_ij'],
                                        _data_dict['beam_analysis']['target_ij'])]

print('done')
