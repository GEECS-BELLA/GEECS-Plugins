import os
import calendar as cal
from pathlib import Path
from typing import Any
from geecs_api.tools.interfaces.exports import load_py, save_py
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.tools.images.scan_images import ScanImages


# base_path = Path(r'C:\Users\GuillaumePlateau\Documents\LBL\Data')
base_path = Path(r'Z:\data')

is_local = (str(base_path)[0] == 'C')
if not is_local:
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

# Parameters
save: bool = True
base_tag = (2023, 6, 9, 1)
screen_labels = ['A2', 'A3', 'U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8']

save_dir: Path = base_path / 'Undulator' / f'Y{base_tag[0]}' / f'{base_tag[1]:02d}-{cal.month_name[base_tag[1]][:3]}' \
                 / f'{str(base_tag[0])[-2:]}_{base_tag[1]:02d}{base_tag[2]:02d}' / 'analysis' \
                 / f'Scan{base_tag[3]:03d}_Screens_{screen_labels[0]}_{screen_labels[-1]}'

analysis_path: Path = save_dir / 'beam_analysis.dat'

# Load analysis
analysis_dict, analysis_path = load_py(analysis_path, as_dict=True)

# Targets
targets = [ScanImages.rotated_to_original_ij(pos, shape, rot90)
           for rot90, shape, pos in zip(analysis_dict['beam_analysis']['rot_90'],
                                        analysis_dict['beam_analysis']['raw_shape_ij'],
                                        analysis_dict['beam_analysis']['com_deltas_pix_means'])]

# Export
data_dict: dict[str, Any] = {'targets': targets,
                             'base_tag': base_tag,
                             'screen_labels': screen_labels,
                             'analysis_path': analysis_path}
if save:
    if not save_dir.is_dir():
        os.makedirs(save_dir)

    export_file_path: Path = save_dir / 'targets'
    save_py(file_path=export_file_path, data=data_dict)
    print(f'Data exported to:\n\t{export_file_path}.dat')

print('done')
