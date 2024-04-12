import os
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from typing import Union, Any, Optional
from geecs_python_api.controls.api_defs import ScanTag
from geecs_python_api.controls.experiment.htu import HtuExp
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.controls.devices.HTU.diagnostics.cameras import Camera
from geecs_python_api.analysis.scans.scan_images import ScanImages
from geecs_python_api.analysis.scans.scan_data import ScanData
from geecs_python_api.tools.images.batches import average_images
from geecs_python_api.tools.images.filtering import FiltersParameters
from geecs_python_api.tools.images.displays import polyfit_label
from geecs_python_api.tools.interfaces.exports import load_py, save_py
from geecs_python_api.tools.interfaces.prompts import text_input


class ScanAnalysis:
    def __init__(self, scan_data: ScanData, scan_images: ScanImages, key_device: Union[GeecsDevice, str]):
        self.scan_data: ScanData = scan_data
        self.scan_images: ScanImages = scan_images

        self.device_name: str = key_device.get_name() if isinstance(key_device, GeecsDevice) else key_device
        self.data_dict: dict[str, Any] = {}
        # data_dict = {
        #     'indexes': indexes,
        #     'setpoints': setpoints,
        #     'analysis_files': analysis_files,
        #     'analyses': analyses,
        #     'device_name': device_name,
        #     'scan_folder': get_folder(),
        #     'scan_scalars': scan_scalars,
        #     'camera_name': camera_name}

    def analyze(self, variable: str, initial_filtering=FiltersParameters(), ask_rerun: bool = True,
                blind_loads: bool = False, store_images: bool = True, store_scalars: bool = True,
                save_plots: bool = False, save: bool = False) -> Optional[Path]:
        analyses: list[dict[str, Any]] = []
        scan_scalars: dict[str, Any] = self.scan_data.data_dict

        # scan parameters & binning
        indexes, setpoints, matching = self.scan_data.group_shots_by_step(self.device_name, variable)

        if not store_scalars:
            if hasattr(self.scan_data, 'data_dict'):
                del self.scan_data.data_dict
            if hasattr(self.scan_data, 'data_frame'):
                del self.scan_data.data_frame

        # list images for each step
        def build_file_name(shot: int):
            return self.scan_images.image_folder / \
                f'Scan{self.scan_data.get_tag().number:03d}_{self.scan_images.camera_name}_{shot:03d}.png'

        paths: list[list[Path]] = \
            [[build_file_name(ind+1) for ind in np.sort(inds) if build_file_name(ind+1).is_file()] for inds in indexes]

        # run image analyses
        analysis_files: list[Path] = []

        with ProgressBar(max_value=len(paths)) as pb:
            for it, (step_paths, step_val) in enumerate(zip(paths, setpoints)):
                # check if analysis exists
                keep: Union[str, bool] = False
                save_dir: Path = self.scan_data.get_analysis_folder() / f'Step_{it+1}'
                analysis_file: Union[Path, str] = save_dir / 'profiles_analysis.dat'

                analyze: str = 'y'
                if analysis_file.is_file():
                    if ask_rerun:
                        analyze = text_input(f'\nRe-run the analysis (step "{step_val}")? : ',
                                             accepted_answers=['y', 'yes', 'n', 'no'])
                    else:
                        analyze = 'no'

                # run/load analysis
                analysis: dict[str, Any] = {}
                if (analyze.lower()[0] == 'y') or (not analysis_file.is_file()):
                    print(f'\nAnalyzing step "{step_val}"...')
                    if not save_dir.is_dir():
                        os.makedirs(save_dir)

                    self.scan_images.set_save_folder(save_dir)
                    analysis_file, analysis = self.scan_images.run_analysis_with_checks(
                        images=step_paths,
                        initial_filtering=initial_filtering,
                        profiles=('com', 'max',), plots=True, store_images=store_images,
                        save_plots=save_plots, save=save)

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
        self.data_dict = {
            'indexes': indexes,
            'setpoints': setpoints,
            'analysis_files': analysis_files,
            'analyses': analyses,
            'device_name': self.device_name,
            'scan_folder': self.scan_images.scan_data_folder,
            'scan_scalars': scan_scalars,
            'camera_name': self.scan_images.camera_name}

        if save:
            export_file_path = self.scan_data.get_analysis_folder() / f'scan_analysis_{self.device_name}'
            save_py(file_path=export_file_path, data=self.data_dict)
            export_file_path = Path(str(export_file_path) + '.dat')
            file_size = export_file_path.stat().st_size
            while True:
                if export_file_path.stat().st_size > file_size:
                    file_size = export_file_path.stat().st_size
                    time.sleep(1.)
                else:
                    break
            print(f'Data exported to:\n\t{export_file_path}.dat')
        else:
            export_file_path = None

        return export_file_path

    def render(self, physical_units: bool = True, x_label: str = 'variable [a.u.]',
               show_xy: bool = True, show_fwhms: bool = True, show_deltas: bool = True,
               xy_metric: str = 'mean', fwhms_metric: str = 'mean', deltas_metric: str = 'mean',
               xy_fit: Union[int, tuple[np.ndarray, str]] = 0,
               fwhms_fit: Union[int, tuple[np.ndarray, str]] = 0,
               deltas_fit: Union[int, tuple[np.ndarray, str]] = 0,
               show_figs: bool = True, save_dir: Optional[Path] = None, sync: bool = True) \
            -> list[tuple[Any, Any]]:
        """
        metric:     'mean', 'median'
        """
        scan_axis: np.ndarray = self.data_dict['setpoints']
        analyses: list[dict[str, Any]] = self.data_dict['analyses']

        shows: list[bool] = [show_xy, show_fwhms, show_deltas]
        metrics: list[str] = [xy_metric, fwhms_metric, deltas_metric]
        fits: list[Union[int, tuple[np.ndarray, str]]] = [xy_fit, fwhms_fit, deltas_fit]

        # summary = {'positions_roi': {'data': {}, 'fit': {}},
        #            'positions_raw': {'data': {}, 'fit': {}},
        #            'deltas': {'data': {}, 'fit': {}},
        #            'fwhms': {'pix_ij': {}, 'um_xy': {}}}

        n_rows: int = sum(shows)
        sample_analysis = analyses[0]
        um_per_pix: float = sample_analysis['summary']['um_per_pix'] if physical_units else 1.
        positions: {} = sample_analysis['image_analyses'][0]['positions']
        units_factor: float
        units_label: str

        figs = []

        for it, (pos, title) in enumerate(zip(positions['short_names'], positions['long_names'])):
            fig, axs = plt.subplots(ncols=1, nrows=n_rows, sharex='col',
                                    figsize=(ScanImages.fig_size[0], ScanImages.fig_size[1] * 1.5))
            i_ax = 0
            for show, metric, val, plot_labels, y_label, fit \
                    in zip(shows, metrics, ['raw', 'fwhms', 'deltas'],
                           [['X', 'Y'], ['FWHM$_x$', 'FWHM$_y$'], ['Dx', 'Dy']],
                           ['Positions', 'FWHMs', 'Deltas'], fits):
                if show:
                    dtype = 'pix_ij' if val == 'fwhms' else 'data'
                    data_val, data_err_low, data_err_high = \
                        ScanAnalysis.fetch_metrics(analyses, metric, val, pos, dtype, um_per_pix)

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

                    for i_xy, var, c_fill, c_val in zip([1, 0], plot_labels, ['m', 'y'], ['b', 'g']):
                        axs[i_ax].fill_between(scan_axis,
                                               units_factor * (data_val[:, i_xy] - data_err_low[:, i_xy]),
                                               units_factor * (data_val[:, i_xy] + data_err_high[:, i_xy]),
                                               color=c_fill, alpha=0.33)
                        axs[i_ax].plot(scan_axis, units_factor * data_val[:, i_xy], f'o{c_val}-',
                                       label=rf'{var} ({xy_metric}) [{units_label}]', linewidth=1, markersize=3)

                        if isinstance(fit, int) and (fit > 0) and not np.isnan(data_val).any():
                            fit_pars = np.polyfit(scan_axis, units_factor * data_val[:, i_xy], fit)
                            fit_vals = np.polyval(fit_pars, scan_axis)
                            axs[i_ax].plot(scan_axis, fit_vals, 'k', linestyle='--', linewidth=0.66,
                                           label='fit: ' + polyfit_label(list(fit_pars), res=2, latex=True))

                        if isinstance(fit, tuple):
                            label = fit[1] if fit[1] else 'fit'
                            axs[i_ax].plot(scan_axis, fit[0], 'k', linestyle='--', linewidth=0.66, label=label)

                    axs[i_ax].legend(loc='best', prop={'size': 8})
                    axs[i_ax].set_ylabel(f'{y_label} [{units_label}]')
                    if i_ax == 0:
                        axs[i_ax].set_title(rf'{title} ({um_per_pix:.2f} $\mu$m/pix)')
                    i_ax += 1

            axs[i_ax - 1].set_xlabel(x_label)
            figs.append((fig, axs))

            if save_dir:
                save_path = save_dir / f'scan_analysis_{pos}_{xy_metric}.png'
                if save_path.is_file():
                    os.remove(save_path)
                plt.savefig(save_path, dpi=300)
                while not save_path.is_file():
                    time.sleep(0.1)

        if show_figs:
            if sync:
                plt.show(block=True)
            else:
                plt.pause(.05)

        return figs

    @staticmethod
    def fetch_metrics(analyses: list[dict[str, Any]], metric: str = 'mean', values: str = 'raw', ptype: str = 'com',
                      dtype: str = 'data', um_per_pix: float = 1.) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        metric:     'mean', 'median'
        values:     'raw' (positions), 'roi' (positions), 'fwhms', 'deltas'
        ptype:      'max', 'com', 'box', 'ellipse', 'pix_ij' (values == "deltas")
        dtype:      'data', 'fit', 'pix_ij' (values == "fwhms")
        """

        # for each step analysis
        data_val = np.empty((0, 2))
        data_err_low = np.empty((0, 2))
        data_err_high = np.empty((0, 2))
        for analysis in analyses:
            ij: np.ndarray
            if values in ['raw', 'roi']:
                ij = analysis['summary'][f'positions_{values}'][dtype][f'{ptype}_ij'] * um_per_pix
            elif values == 'fwhms':
                ij = analysis['summary'][values][dtype][ptype] * um_per_pix
            elif values == 'deltas':
                ij = analysis['summary'][values][dtype]['pix_ij'] * um_per_pix
            else:
                continue

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


if __name__ == '__main__':
    # initialization
    # --------------------------------------------------------------------------
    htu = HtuExp(get_info=True)
    _base_tag = ScanTag(2023, 8, 1, 29)
    _bkg_tag = ScanTag(2023, 8, 3, 18)

    _device = 'U_S2V'
    _variable = 'Current'
    _camera = 'DP'
    _metric = 'median'
    # _metric = 'mean'

    _folder = ScanData.build_folder_path(_base_tag, htu.base_path)
    _scan_data = ScanData(_folder, ignore_experiment_name=htu.is_offline)
    _scan_images = ScanImages(_scan_data, _camera)
    _scan_analysis = ScanAnalysis(_scan_data, _scan_images, _device)

    _filters = FiltersParameters(contrast=1.333, hp_median=3, hp_threshold=3., denoise_cycles=0, gauss_filter=5.,
                                 com_threshold=0.75, bkg_image=None, box=True, ellipse=False)

    # background
    # --------------------------------------------------------------------------
    camera_name = Camera.name_from_label(_camera)
    # _bkg_folder = ScanData.build_folder_path(_bkg_tag, _base_path) / camera_name
    _bkg_folder = ''

    avg_image, _ = average_images(_bkg_folder)
    if avg_image is not None:
        if isinstance(_scan_images.camera_roi, np.ndarray) and (_scan_images.camera_roi.size >= 4):
            avg_image = avg_image[_scan_images.camera_roi[-2]:_scan_images.camera_roi[-1] + 1,
                                  _scan_images.camera_roi[0]:_scan_images.camera_roi[1] + 1]
        avg_image = np.rot90(avg_image, _scan_images.camera_r90 // 90)
        _filters.bkg_image = avg_image

    # scan analysis
    # --------------------------------------------------------------------------
    _path = _scan_analysis.analyze(_variable, initial_filtering=_filters, ask_rerun=True, blind_loads=True,
                                   store_images=False, store_scalars=False, save_plots=False, save=False)

    _scan_analysis.render(physical_units=False, x_label='Current [A]',
                          show_xy=True, show_fwhms=True, show_deltas=True,
                          xy_metric=_metric, fwhms_metric=_metric, deltas_metric=_metric,
                          xy_fit=1, fwhms_fit=1, deltas_fit=1,
                          save_dir=_scan_data.get_analysis_folder(), sync=True)

    print('done')
