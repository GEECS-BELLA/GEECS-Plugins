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
from geecs_api.tools.distributions.binning import unsupervised_binning, BinningResults
from geecs_api.tools.scans.scan_images import ScanImages
from geecs_api.tools.scans.scan_data import ScanData
from geecs_api.tools.images.filtering import FiltersParameters
from geecs_api.tools.images.displays import polyfit_label
from geecs_api.tools.interfaces.exports import load_py, save_py
from geecs_api.tools.interfaces.prompts import text_input
from geecs_api.api_defs import ScanTag
# from geecs_api.devices.HTU.laser import LaserCompressor


class ScanAnalysis:
    def __init__(self, scan_data: ScanData, scan_images: ScanImages, key_device: Union[GeecsDevice, str]):
        self.scan_data: ScanData = scan_data
        self.scan_images: ScanImages = scan_images

        self.device_name: str = key_device.get_name() if isinstance(key_device, GeecsDevice) else key_device
        self.key_data = self.scan_data.data_dict[self.device_name]

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

    def analyze(self, variable: str, com_threshold: float = 0.5,
                bkg_image: Optional[Union[Path, np.ndarray]] = None, ask_rerun: bool = True,
                blind_loads: bool = False, store_images: bool = True, store_scalars: bool = True,
                save_plots: bool = False, save: bool = False) -> Optional[Path]:
        analyses: list[dict[str, Any]] = []

        scan_scalars: dict[str, Any] = self.scan_data.data_dict
        if not store_scalars:
            if hasattr(self.scan_data, 'data_dict'):
                del self.scan_data.data_dict
            if hasattr(self.scan_data, 'data_frame'):
                del self.scan_data.data_frame

        # scan parameters & binning
        measured: BinningResults = unsupervised_binning(self.key_data[variable], self.key_data['shot #'])

        Expected = NamedTuple('Expected',
                              start=float,
                              end=float,
                              steps=int,
                              shots=int,
                              setpoints=np.ndarray,
                              indexes=list)
        start = float(self.scan_data.scan_info['Start'])
        end = float(self.scan_data.scan_info['End'])
        steps: int = 1 + round(np.abs(end - start) / float(self.scan_data.scan_info['Step size']))
        shots: int = int(self.scan_data.scan_info['Shots per step'])
        expected = Expected(start=start, end=end, steps=steps, shots=shots,
                            setpoints=np.linspace(float(self.scan_data.scan_info['Start']),
                                                  float(self.scan_data.scan_info['End']), steps),
                            indexes=[np.arange(p * shots, (p+1) * shots) for p in range(steps)])

        matching = all([inds.size == expected.shots for inds in measured.indexes])
        matching = matching and (len(measured.indexes) == expected.steps)
        if not matching:
            api_error.warning(f'Observed data binning does not match expected scan parameters (.ini)',
                              f'Function "{inspect.stack()[0][3]}"')

        # list images for each step
        def build_file_name(shot: int):
            return self.scan_images.image_folder / \
                f'Scan{self.scan_data.get_tag().number:03d}_{self.scan_images.camera_name}_{shot:03d}.png'

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
                        initial_filtering=FiltersParameters(com_threshold=com_threshold, bkg_image=bkg_image),
                        plots=True, store_images=store_images, save_plots=save_plots, save=save)

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
            for show, metric, val, plot_labels, y_label, fit in zip(shows, metrics, ['raw', 'fwhms', 'deltas'],
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
                plt.savefig(save_path, dpi=300)

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
    # database
    # --------------------------------------------------------------------------
    # base_path = Path(r'C:\Users\GuillaumePlateau\Documents\LBL\Data')
    base_path: Path = Path(r'Z:\data')

    is_local = (str(base_path)[0] == 'C')
    if not is_local:
        GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

    _base_tag = ScanTag(2023, 4, 13, 21)
    # _device = LaserCompressor()
    # _variable = _key_device.var_separation
    # _camera = Camera('UC_TopView')
    _device = 'U_S2V'
    _variable = 'Current'
    _camera = 'UC_Phosphor1'

    _folder = ScanData.build_folder_path(_base_tag, base_path)
    _scan_data = ScanData(_folder, ignore_experiment_name=is_local)
    _scan_images = ScanImages(_scan_data, _camera)
    _scan_analysis = ScanAnalysis(_scan_data, _scan_images, _device)

    # scan analysis
    # --------------------------------------------------------------------------
    _path, _dict = _scan_analysis.analyze(_variable, com_threshold=0.5, bkg_image=None, blind_loads=True,
                                          store_images=False, store_scalars=False, save_plots=False, save=True)

    _scan_analysis.render(physical_units=False, x_label='Current [A]',
                          show_xy=True, show_fwhms=True, show_deltas=False,
                          xy_metric='mean', fwhms_metric='mean', deltas_metric='mean',
                          xy_fit=1, fwhms_fit=1, deltas_fit=0,
                          save_dir=_scan_data.get_analysis_folder(), sync=True)

    # _device.close()
    # _camera.close()
    print('done')
