import os
import time
import inspect
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from typing import Union, NamedTuple, Any, Optional
from geecs_api.interface import GeecsDatabase, api_error
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.diagnostics.cameras import Camera
from geecs_api.tools.distributions.binning import unsupervised_binning, BinningResults
from geecs_api.tools.images.scan_images import ScanImages
from geecs_api.tools.scans.scan_data import ScanData
from geecs_api.tools.images.filtering import FiltersParameters
from geecs_api.tools.interfaces.exports import load_py, save_py
from geecs_api.tools.interfaces.prompts import text_input
# from geecs_api.tools.distributions.fit_utility import fit_distribution


def scan_analysis(scan_data: ScanData, device: Union[GeecsDevice, str], variable: str, camera: Union[int, Camera, str],
                  com_threshold: float = 0.5, blind_loads: bool = False, store_images: bool = True, save: bool = False)\
        -> tuple[Optional[Path], dict[str, Any]]:
    analyses: list[dict[str, Any]] = []

    scan_images = ScanImages(scan_data, camera)
    device_name: str = device.get_name() if isinstance(device, GeecsDevice) else device
    key_data = scan_data.data_dict[device_name]

    # scan parameters & binning
    measured: BinningResults = unsupervised_binning(key_data[variable], key_data['shot #'])

    Expected = NamedTuple('Expected', start=float, end=float, steps=int, shots=int, setpoints=np.ndarray, indexes=list)
    steps: int = 1 + round((float(scan_data.scan_info['End']) - float(scan_data.scan_info['Start']))
                           / float(scan_data.scan_info['Step size']))
    expected = Expected(start=float(scan_data.scan_info['Start']),
                        end=float(scan_data.scan_info['End']),
                        steps=steps,
                        shots=int(scan_data.scan_info['Shots per step']),
                        setpoints=np.linspace(float(scan_data.scan_info['Start']),
                                              float(scan_data.scan_info['End']),
                                              steps),
                        indexes=[np.arange(p * steps, (p+1) * steps) for p in range(steps)])

    matching = \
        all([inds.size == expected.shots for inds in measured.indexes]) and (len(measured.indexes) == expected.steps)
    if not matching:
        api_error.warning(f'Observed data binning does not match expected scan parameters (.ini)',
                          f'Function "{inspect.stack()[0][3]}"')

    # list images for each step
    def build_file_name(shot: int):
        return scan_images.image_folder / \
            f'Scan{scan_data.get_tag().number:03d}_{scan_images.camera_name}_{shot:03d}.png'

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
        for it, (step_paths, step_val) in enumerate(zip(paths, setpoints)):
            # check if analysis exists
            keep: Union[str, bool] = False
            save_dir: Path = scan_data.get_analysis_folder() / f'Step_{it+1}'
            analysis_file: Union[Path, str] = save_dir / 'profiles_analysis.dat'

            analyze: str = 'y'
            if analysis_file.is_file():
                analyze = text_input(f'\nRe-run the analysis (step "{step_val}")? : ',
                                     accepted_answers=['y', 'yes', 'n', 'no'])

            # run/load analysis
            analysis: dict[str, Any] = {}
            if (analyze.lower()[0] == 'y') or (not analysis_file.is_file()):
                print(f'\nAnalyzing step "{step_val}"...')
                if not save_dir.is_dir():
                    os.makedirs(save_dir)

                scan_images.set_save_folder(save_dir)
                analysis_file, analysis = scan_images.run_analysis_with_checks(
                    images=step_paths,
                    initial_filtering=FiltersParameters(com_threshold=com_threshold),
                    plots=True, store_images=store_images, save=save)

            if not analysis:
                print('Loading analysis...')
                analysis, analysis_file = load_py(analysis_file)
                keep = blind_loads
                if not blind_loads:
                    ScanImages.render_image_analysis(analysis['average_analysis'], tag='average_image', block=True)

            if not analysis:
                continue  # skip

            if not analysis_file:
                analysis_file = ''
            analysis_files.append(analysis_file)

            if keep:
                keep = 'y'
            else:
                keep = text_input(f'Add this analysis to the overall screen scan analysis? : ',
                                  accepted_answers=['y', 'yes', 'n', 'no'])
            if keep.lower()[0] == 'n':
                continue

            print('Collecting analysis summary...')
            analyses.append(analysis)

            pb.increment()
            time.sleep(0.01)

    # export to .dat
    data_dict: dict[str, Any] = {
        'indexes': indexes,
        'setpoints': setpoints,
        'analysis_files': analysis_files,
        'analyses': analyses,
        'device_name': device_name,
        'scan_folder': scan_images.scan.get_folder(),
        'camera_name': scan_images.camera_name}

    if save:
        export_file_path = scan_data.get_analysis_folder() / f'scan_analysis_{device_name}'
        save_py(file_path=export_file_path, data=data_dict, as_bulk=False)
        print(f'Data exported to:\n\t{export_file_path}.dat')
    else:
        export_file_path = None

    return export_file_path, data_dict

# data_dict: dict[str, Any] = {
#     'indexes': indexes,
#     'setpoints': setpoints,
#     'analysis_files': analysis_files,
#     'analyses': analyses,
#     'device_name': device_name,
#     'scan_folder': scan_images.scan.get_folder(),
#     'camera_name': scan_images.camera_name}

# self.summary = {'positions_roi': {'data': {}, 'fit': {}},
#                 'positions_raw': {'data': {}, 'fit': {}},
#                 'deltas': {'data': {}, 'fit': {}},
#                 'fwhms': {'pix_ij': {}, 'um_xy': {}}}


def scan_metrics(analyses: list[dict[str, Any]], metric: str = 'mean', values: str = 'raw', ptype: str = 'com',
                 dtype: str = 'data', um_per_pix: float = 1.) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    metric:     'mean', 'median'
    values:     'raw' (positions), 'roi' (positions), 'fwhms', 'deltas'
    ptype:      'max', 'com', 'box', 'ellipse'
    dtype:      'data', 'fit', 'pix_ij' (values == "fwhms")
    """

    # for each step analysis
    data_val = np.empty((0, 2))
    data_err_low = np.empty((0, 2))
    data_err_high = np.empty((0, 2))
    for analysis in analyses:
        ij: np.ndarray
        if values.lower() in ['raw', 'roi']:
            ij = analysis[f'positions_{values}'][dtype][f'{ptype}_ij'] * um_per_pix
        else:
            ij = analysis[values][dtype][f'{ptype}_ij'] * um_per_pix

        if metric == 'mean':
            pos_ij = np.mean(ij, axis=0)
            err_ij = np.std(ij, axis=0)
            data_val = np.concatenate([data_val, [pos_ij]])
            data_err_low = np.concatenate([data_err_low, [err_ij]])
            data_err_high = np.concatenate([data_err_high, [err_ij]])

        if metric == 'median':
            pos_ij = np.median(ij, axis=0)
            err_low_ij = np.array([np.std(ij[ij[:, 0] < pos_ij[0], 0]),
                                   np.std(ij[ij[:, 1] < pos_ij[1], 1])])
            err_high_ij = np.array([np.std(ij[ij[:, 0] >= pos_ij[0], 0]),
                                    np.std(ij[ij[:, 1] >= pos_ij[1], 1])])
            data_val = np.concatenate([data_val, [pos_ij]])
            data_err_low = np.concatenate([data_err_low, [err_low_ij]])
            data_err_high = np.concatenate([data_err_high, [err_high_ij]])

    return data_val, data_err_low, data_err_high


def render_scan_analysis(data_dict: dict[str, Any], physical_units: bool = True, x_label: str = 'scan variable [a.u.]',
                         show_xy: bool = True, show_fwhms: bool = True, show_deltas: bool = True,
                         xy_metric: str = 'mean', fwhms_metric: str = 'mean', deltas_metric: str = 'mean',
                         xy_fit: int = 1, fwhms_fit: int = 1, deltas_fit: int = 1,
                         save_dir: Optional[Path] = None):
    """
    metric:     'mean', 'median'
    """
    scan_axis: np.ndarray = data_dict['setpoints']
    analyses: list[dict[str, Any]] = data_dict['analyses']

    shows: list[bool] = [show_xy, show_fwhms, show_deltas]
    metrics: list[str] = [xy_metric, fwhms_metric, deltas_metric]
    fits: list[int] = [xy_fit, fwhms_fit, deltas_fit]

    n_rows: int = sum(shows)
    um_per_pix: float = analyses[0]['summary']['um_per_pix'] if physical_units else 1.
    units_factor: float
    units_label: str

    for pos, title in zip(analyses[0]['positions']['short_names'], analyses[0]['positions']['long_names']):
        fig, axs = plt.subplots(ncols=1, nrows=n_rows,
                                figsize=(ScanImages.fig_size[0], ScanImages.fig_size[1] * 1.5),
                                sharex='col')
        i_ax = 0
        for show, metric, val, plot_labels, y_label, fit in zip(shows, metrics, ['raw', 'fwhms', 'deltas'],
                                                                [['X', 'Y'], ['FWHM$_x$', 'FWHM$_y$'], ['Dx', 'Dy']],
                                                                ['Positions', 'FWHMs', 'Deltas'], fits):
            units_label: str = 'a.u.'

            if show:
                dtype = 'pix_ij' if val == 'fwhms' else 'data'
                data_val, data_err_low, data_err_high = scan_metrics(analyses, metric, val, pos, dtype, um_per_pix)

                if physical_units:
                    if (np.max(data_val) - np.min(data_val)) > 1000:
                        units_factor = 0.001
                        units_label = 'mm'
                    else:
                        units_factor = 1
                        units_label = r'$\mu$m'
                else:
                    units_factor = 1
                    units_label = 'pix'

                for i_xy, var, c in enumerate(zip([1, 0], plot_labels, ['m', 'y'])):
                    axs[i_ax].fill_between(scan_axis,
                                           units_factor * (data_val[:, i_xy] - data_err_low[:, i_xy]),
                                           units_factor * (data_val[:, i_xy] + data_err_high[:, i_xy]),
                                           color=c, alpha=0.33)
                    axs[i_ax].plot(scan_axis, units_factor * data_val[:, i_xy], 'ob-',
                                   label=rf'{var} ({xy_metric}) [{units_label}]', linewidth=1, markersize=3)

                    if xy_fit > 0:
                        fit_pars = np.polyfit(scan_axis, units_factor * data_val[:, i_xy], xy_fit)
                        fit_vals = np.polyval(fit_pars, scan_axis)
                        axs[i_ax].plot(scan_axis, fit_vals, 'gray', label='fit')

                axs[i_ax].legend(loc='best', prop={'size': 8})
                axs[i_ax].set_ylabel(f'{y_label} [{units_label}]')
                if i_ax == 0:
                    axs[i_ax].set_title(title)
                i_ax += 1

            axs[i_ax].set_ylabel(f'{y_label} [{units_label}]')
        axs[i_ax].set_xlabel(x_label)

        if save_dir:
            save_path = save_dir / f'scan_analysis_{pos}.png'
            plt.savefig(save_path, dpi=300)

    plt.show(block=True)


if __name__ == '__main__':
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

    # _base = Path(r'C:\Users\GuillaumePlateau\Documents\LBL\Data')
    _base: Path = Path(r'Z:\data')

    _base_tag = (2023, 4, 13, 26)
    _key_device = 'U_S4H'
    _camera_tag = 'A2'

    _scan_data = ScanData(tag=_base_tag, experiment_base_path=_base / 'Undulator')

    # data preview
    # _key_data = _scan_data.data_dict[_key_device]
    # _bins: BinningResults = unsupervised_binning(_key_data['Current'], _key_data['shot #'])
    #
    # plt.figure()
    # for x, _ind in zip(_bins.avg_x, _bins.indexes):
    #     plt.plot(x * np.ones(_ind.shape), _ind, '.', alpha=0.3)
    # plt.xlabel('Current [A]')
    # plt.ylabel('Indexes')
    # plt.show(block=True)

    # run analysis
    # _export_file_path, _data_dict = scan_analysis(_scan_data, _key_device, _camera_tag)

    # open analysis
    # _analysis_file = Path(r'Z:\data\Undulator\Y2023\04-Apr\23_0413\analysis\Scan026\steering_analysis_U_S4H.dat')
    # _data_dict, _ = load_py(_analysis_file, as_dict=True)

    print('done')
