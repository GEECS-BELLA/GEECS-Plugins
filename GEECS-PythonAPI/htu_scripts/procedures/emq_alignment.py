import numpy as np
from geecs_api.api_defs import ScanTag
import geecs_api.experiment.htu as htu
from geecs_api.tools.scans.scan_images import ScanImages
from geecs_api.tools.interfaces.exports import load_py
from geecs_api.tools.scans.scan_data import ScanData


# initialization
# --------------------------------------------------------------------------
base_path, is_local = htu.initialize()

i1 = np.array([-5., -1.5])
i2 = np.array([-3., -0.5])


# matrices
# --------------------------------------------------------------------------
D1 = np.array([[-123.53, -19.45],
               [-8.23,  -100.67]])

D2 = np.array([[-16.80, -0.11],
               [0.17, -7.17]])

P1 = np.array([[138.39, 16.19],
               [-2.78, -132.25]])

P2 = np.array([[91.41, 13.38],
               [1.95, -61.12]])


# references
# --------------------------------------------------------------------------
ref_files = []
ref_dicts = {}
ref_coms = []

for cam, tag in zip(['DP', 'P1'], [ScanTag(2023, 7, 6, 21),
                                   ScanTag(2023, 7, 6, 22)]):
    # read refs
    ref_folder = ScanData.build_folder_path(tag, base_path)
    scan_data = ScanData(ref_folder, load_scalars=False, ignore_experiment_name=is_local)
    scan_images = ScanImages(scan_data, cam)
    analysis_file = scan_images.save_folder / 'profiles_analysis.dat'
    ref_dict, _ = load_py(analysis_file, as_dict=True)

    # store
    ref_coms.append(ref_dict['average_analysis']['positions']['com_ij'])
    ref_files.append(ref_folder)
    ref_dicts[cam] = ref_dict


# current positions
# --------------------------------------------------------------------------
ini_files = []
ini_dicts = {}
ini_coms = []

for cam, tag in zip(['DP', 'P1'], [ScanTag(2023, 8, 9, 14),
                                   ScanTag(2023, 8, 9, 15)]):
    # read positions
    ini_folder = ScanData.build_folder_path(tag, base_path)
    scan_data = ScanData(ini_folder, load_scalars=False, ignore_experiment_name=is_local)
    scan_images = ScanImages(scan_data, cam)
    analysis_file = scan_images.save_folder / 'profiles_analysis.dat'
    ini_dict, _ = load_py(analysis_file, as_dict=True)

    # store
    ini_coms.append(ini_dict['average_analysis']['positions']['com_ij'])
    ini_files.append(ini_folder)
    ini_dicts[cam] = ini_dict


# corrections
# --------------------------------------------------------------------------
ref_coms = np.array(ref_coms)
ini_coms = np.array(ini_coms)

diff_pix = ref_coms - ini_coms
dDP = diff_pix[0]
dP1 = diff_pix[1]
print('corrections [pix]:')
print(f'DP:\n{dDP}')
print(f'P1:\n{dP1}')

iD1 = np.linalg.inv(D1)
iP2 = np.linalg.inv(P2)

di1 = np.dot(iD1, dDP)
di2 = np.dot(iP2, dP1 - np.dot(P1, di1))

print('corrections [A]:')
print(f'S1:\n{di1}')
print(f'S2:\n{di2}')

print('new currents [A]:')
print(f'S1:\n{i1 + di1}')
print(f'S2:\n{i2 + di2}')

print('done')
