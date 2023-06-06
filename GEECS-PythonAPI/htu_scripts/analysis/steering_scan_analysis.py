import os
import inspect
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from typing import Union, NamedTuple
from geecs_api.interface import GeecsDatabase, api_error
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.diagnostics.cameras import Camera
from geecs_api.tools.distributions.binning import unsupervised_binning, BinningResults
from geecs_api.tools.images.scan_images import ScanImages
from geecs_api.tools.scans.scan_data import ScanData
from geecs_api.tools.images.filtering import FiltersParameters
from geecs_api.tools.interfaces.exports import load_py, save_py
from geecs_api.tools.interfaces.prompts import text_input
from geecs_api.tools.distributions.fit_utility import fit_distribution
from htu_scripts.analysis.beam_analyses_collector import add_beam_analysis


def steering_scan_analysis(scan_data: ScanData, device: Union[GeecsDevice, str], camera: Union[int, Camera, str]):
    scan_images = ScanImages(scan_data, camera)
    device_name: str = device.get_name() if isinstance(device, GeecsDevice) else device
    # key_data = scan_data.data_dict[device_name]

    # scan parameters & binning
    measured: BinningResults = unsupervised_binning(_key_data['Current'], _key_data['shot #'])

    Expected = NamedTuple('Expected', start=float, end=float, steps=int, setpoints=np.ndarray, indexes=list(np.ndarray))
    steps: int = \
        round((float(_scan.scan_info['End']) - float(_scan.scan_info['Start'])) / (float(_scan.scan_info['Step']) - 1))
    expected = Expected(start=float(_scan.scan_info['Start']),
                        end=float(_scan.scan_info['End']),
                        steps=steps,
                        setpoints=np.linspace(float(_scan.scan_info['Start']), float(_scan.scan_info['End']), steps),
                        indexes=[np.arange(p * steps, (p+1) * steps - 1) for p in range(steps)])

    matching = all([inds.size == expected.steps for inds in measured.indexes])
    if not matching:
        api_error.warning(f'Observed data binning does not match expected scan parameters (.ini)',
                          f'Function "{inspect.stack()[0][3]}"')

    # list images for each step
    def build_file_name(shot: int):
        return scan_images.image_folder / f'Scan{scan_data.get_tag().number:03d}_{device_name}_{shot:03d}.png'

    if matching:
        indexes = expected.indexes
        setpoints = expected.setpoints
    else:
        indexes = measured.indexes
        setpoints = measured.avg_x

    paths: list[list[Path]] = \
        [[build_file_name(ind+1) for ind in np.sort(inds) if build_file_name(ind+1).is_file()] for inds in indexes]

    # run image analyses
    analysis_files: list[Path] = []

    with ProgressBar(max_value=len(paths)) as pb:
        for it, step_paths in enumerate(paths):
            save_dir: Path = scan_data.get_analysis_folder() / f'Step_{it+1}'
            if not save_dir.is_dir():
                os.makedirs(save_dir)

            scan_images.set_save_folder(save_dir)
            analysis_file, analysis = \
                scan_images.run_analysis_with_checks(images=step_paths, plots=True, save=True,
                                                     initial_filtering=FiltersParameters(com_threshold=0.66))
            keep = text_input(f'Add this analysis to the overall screen scan analysis? : ',
                              accepted_answers=['y', 'yes', 'n', 'no'])
            if keep.lower()[0] == 'n':
                continue

            if not analysis:
                print('Loading analysis...')
                analysis, analysis_file = load_py(analysis_file, as_dict=True)

            if not analysis:
                continue  # skip

            if not analysis_file:
                analysis_file = ''
            analysis_files.append(analysis_file)

            print('Collecting analysis summary...')
            beam_analysis, pos_short_names, pos_long_names = \
                add_beam_analysis(beam_analysis, analysis, pos_short_names, pos_long_names, it, len(paths))

            pb.increment()

    # linear fits
    pos = 'max'  # tmp
    x_opt, x_fit = fit_distribution(setpoints, beam_analysis[f'{pos}_mean_pos_pix'][:, 0], fit_type='linear')
    y_opt, y_fit = fit_distribution(setpoints, beam_analysis[f'{pos}_mean_pos_pix'][:, 1], fit_type='linear')

    # export to .dat
    export_file_path = scan_data.get_analysis_folder() / f'steering_analysis_{device_name}'
    save_py(file_path=export_file_path,
            data={'indexes': indexes,
                  'setpoints': setpoints,
                  'analysis_files': analysis_files,
                  'beam_analysis': beam_analysis,
                  'device_name': device_name,
                  'scan_folder': scan_images.scan.get_folder(),
                  'camera_name': scan_images.camera_name,
                  'pos_short_names': pos_short_names,
                  'pos_long_names': pos_long_names,
                  'x_opt': x_opt,
                  'x_fit': x_fit,
                  'y_opt': y_opt,
                  'y_fit': y_fit})
    print(f'Data exported to:\n\t{export_file_path}.dat')


if __name__ == '__main__':
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

    _base = Path(r'C:\Users\GuillaumePlateau\Documents\LBL\Data')
    # _base: Path = Path(r'Z:\data')
    _base_tag = (2023, 4, 18, 27)
    _camera_tag = 'A1'

    _key_device = 'U_S4H'

    _scan = ScanData(tag=_base_tag, experiment_base_path=_base / 'Undulator')
    _key_data = _scan.data_dict[_key_device]

    _bins: BinningResults = unsupervised_binning(_key_data['Current'], _key_data['shot #'])

    plt.figure()
    plt.plot(_key_data['shot #'], _key_data['Current'], '.b', alpha=0.3)
    plt.xlabel('Shot #')
    plt.ylabel('Current [A]')
    plt.show(block=False)

    plt.figure()
    for x, _ind in zip(_bins.avg_x, _bins.indexes):
        plt.plot(x * np.ones(_ind.shape), _ind, '.', alpha=0.3)
    plt.xlabel('Current [A]')
    plt.ylabel('Indexes')
    plt.show(block=True)

    print('done')
