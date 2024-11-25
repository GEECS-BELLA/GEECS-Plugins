import os
import re
import inspect
import numpy as np
import calendar as cal
from pathlib import Path
from scipy.signal import savgol_filter
from datetime import datetime as dtime, date
from typing import Optional, Union, Any, NamedTuple
from configparser import ConfigParser, NoSectionError
from geecs_python_api.controls.api_defs import SysPath, ScanTag
from geecs_python_api.controls.experiment.htu import HtuExp
from geecs_python_api.tools.images.batches import list_files
from geecs_python_api.controls.interface import api_error
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
import geecs_python_api.tools.images.ni_vision as ni
from geecs_python_api.tools.interfaces.tdms import read_geecs_tdms
from geecs_python_api.tools.images.spot import profile_fit, std_to_fwhm
# from image_analysis.analyzers import default_analyzer_generators
from image_analysis.labview_adapters import analyzer_from_device_type
# from image_analysis.analyzers.UC_GenericMagSpecCam import UC_GenericMagSpecCamAnalyzer
from geecs_python_api.analysis.scans.scan_data import ScanData


class ScanDataMagspec(ScanData):
    """ Represents a GEECS experiment scan """

    def __init__(self, folder: Optional[SysPath] = None,
                 tag: Optional[Union[int, ScanTag, tuple]] = None,
                 load_scalars: bool = True,
                 ignore_experiment_name: bool = False,
                 experiment_base_path: Optional[SysPath] = None):
        """
        Parameter(s)
        ----------
        Either parameter can be provided. If both are, "folder" is used.
        Experiment name is retrieved from GeecsDevice static member and must match with folder name.

        folder : Union[str, bytes, PathLike]
            Data folder containing the scan data, e.g. "Z:/data/Undulator/Y2023/05-May/23_0501/scans/Scan002".
        tag : Union[int, tuple[int, int, int, int]]
            Either of:
                - Tuple with the scan identification information, e.g. (year, month, day, scan #) = (2023, 5, 1, 2)
                - scan number only, today's date is used
        ignore_experiment_name : bool
              Allows working offline with local copy of the data, when specifying a folder
        experiment_base_path : SysPath
              Allows working offline with local copy of the data, when specifying a tag
              e.g. experiment_base_path='C:/Users/GuillaumePlateau/Documents/LBL/Data/Undulator'
        """
        if not ignore_experiment_name:
            exp_name = GeecsDevice.exp_info['name']
        else:
            exp_name = None

        if experiment_base_path is None:
            exp_path = Path(GeecsDevice.exp_info['data_path'])
            base_path = exp_path.parent
        else:
            base_path = experiment_base_path

        super().__init__(folder=folder, tag=tag, experiment=exp_name, load_scalars=load_scalars, base_path=base_path)

    @staticmethod
    def build_folder_path(tag: ScanTag, base_directory: Union[Path, str] = r'Z:\data', experiment: str = 'Undulator') \
            -> Path:
        return ScanData.build_scan_folder_path(tag=tag, base_directory=base_directory, experiment=experiment)

    def load_mag_spec_data(self) -> dict[str, Any]:
        magspec_dict = {'full': {}, 'hres': {}}

        magspec_dict['hres']['txt_files'] = False
        path_hres_source: Path = self.get_folder()
        path_found = False

        for use_txt, folder in zip([False] * 3 + [True], ['UC_HiResMagCam',
                                                          'U_HiResMagCam',
                                                          'UC_TestCam',
                                                          'U_HiResMagCam-interpSpec']):
            if (self.get_folder() / folder).exists():
                path_hres_source = self.get_folder() / folder
                magspec_dict['hres']['txt_files'] = use_txt
                path_found = True
                break

        if not path_found:
            return {}

        # tmp
        # path_hres_source = self.__folder / 'U_HiResMagCam-interpSpec'
        # magspec_dict['hres']['txt_files'] = True

        magspec_dict['hres']['paths'] = list_files(path_hres_source, -1,
                                                   '.txt' if magspec_dict['hres']['txt_files'] else '.png')
        shots = []
        specs = []

        image_filename_regex = re.compile(r".*_(\d{3,})\.\w+$")
        for path in magspec_dict['hres']['paths']:
            try:
                shots.append(int(image_filename_regex.match(path.name).group(1)))
                if magspec_dict['hres']['txt_files']:
                    specs.append(np.loadtxt(path, skiprows=1))

            except Exception:
                continue

        magspec_dict['hres']['specs'] = np.array(specs)  # empty for now if loading raw images
        magspec_dict['hres']['shots'] = np.array(shots)

        return magspec_dict

    def analyze_mag_spec(self, indexes: list[np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
        # load raw data
        magspec_data = self.load_mag_spec_data()
        if not magspec_data:
            return {}

        hres_stats = {k: np.zeros((len(indexes),))
                      for s in ['avg', 'med', 'std']
                      for k in
                      [f'{s}_weighted_mean_MeV',
                       f'{s}_weighted_rms_MeV',
                       f'{s}_weighted_dE/E',
                       f'{s}_peak_charge_pC/MeV',
                       f'{s}_peak_charge_MeV',
                       f'{s}_fit_mean_MeV',
                       f'{s}_fit_fwhm_MeV',
                       f'{s}_fit_dE/E',
                       f'{s}_peak_smooth_pC/MeV',
                       f'{s}_peak_smooth_MeV']}

        # text files or images?
        if magspec_data['hres']['txt_files']:
            if not magspec_data['hres']['specs'].size > 0:
                return {}

            axis_MeV: np.ndarray = magspec_data['hres']['specs'][0, :, 0]

            # fit all full-spectra
            if magspec_data['full']:
                n_specs = magspec_data['full']['specs'].shape[0]
                full_specs_fits = {'opt_pars': np.zeros((n_specs, 4)),
                                   'err_pars': np.zeros((n_specs, 4)),
                                   'fits': np.zeros((n_specs, magspec_data['full']['specs'].shape[1]))}
                smooth_full = np.zeros((n_specs, magspec_data['full']['specs'].shape[1]))
                for it, spec in enumerate(magspec_data['full']['specs']):
                    smooth_full[it, :] = savgol_filter(spec[:, 1], 20, 3)
                    opt_pars, err_pars, fit = profile_fit(spec[:, 0] * 1000., spec[:, 1],
                                                          guess_center=spec[np.argmax(smooth_full[it, :]), 0] * 1000.,
                                                          smoothing_window=20,
                                                          crop_sigma_radius=10.)
                    full_specs_fits['opt_pars'][it, :] = opt_pars
                    full_specs_fits['err_pars'][it, :] = err_pars
                    full_specs_fits['fits'][it, :] = fit

            # fit all high-resolution spectra
            n_specs = magspec_data['hres']['specs'].shape[0]
            hres_specs_fits = {'opt_pars': np.zeros((n_specs, 4)),
                               'err_pars': np.zeros((n_specs, 4)),
                               'fits': np.zeros((n_specs, magspec_data['hres']['specs'].shape[1]))}
            smooth_hres = np.zeros((n_specs, magspec_data['hres']['specs'].shape[1]))
            for it, spec in enumerate(magspec_data['hres']['specs']):
                smooth_hres[it, :] = savgol_filter(spec[:, 1], 20, 3)
                opt_pars, err_pars, fit = profile_fit(spec[:, 0], spec[:, 1],
                                                      guess_center=spec[np.argmax(smooth_hres[it, :]), 0],
                                                      crop_sigma_radius=10.)
                hres_specs_fits['opt_pars'][it, :] = opt_pars
                hres_specs_fits['err_pars'][it, :] = err_pars
                hres_specs_fits['fits'][it, :] = fit

            avg_hres_pC = np.zeros((len(indexes), axis_MeV.size))
            med_hres_pC = np.zeros((len(indexes), axis_MeV.size))
            std_hres_pC = np.zeros((len(indexes), axis_MeV.size))

            for it, i_group in enumerate(indexes):
                avg_hres_pC[it, :] = np.mean(magspec_data['hres']['specs'][i_group, :, 1], axis=0)
                med_hres_pC[it, :] = np.median(magspec_data['hres']['specs'][i_group, :, 1], axis=0)
                std_hres_pC[it, :] = np.std(magspec_data['hres']['specs'][i_group, :, 1], axis=0)

                hres_stats['avg_fit_mean_MeV'][it] = np.mean(hres_specs_fits['opt_pars'][i_group, 2], axis=0)
                hres_stats['med_fit_mean_MeV'][it] = np.median(hres_specs_fits['opt_pars'][i_group, 2], axis=0)
                hres_stats['std_fit_mean_MeV'][it] = np.std(hres_specs_fits['opt_pars'][i_group, 2], axis=0)
                hres_stats['avg_fit_fwhm_MeV'][it] = std_to_fwhm(np.mean(hres_specs_fits['opt_pars'][i_group, 3], axis=0))
                hres_stats['med_fit_fwhm_MeV'][it] = std_to_fwhm(np.median(hres_specs_fits['opt_pars'][i_group, 3], axis=0))
                hres_stats['std_fit_fwhm_MeV'][it] = std_to_fwhm(np.std(hres_specs_fits['opt_pars'][i_group, 3], axis=0))
                hres_stats['avg_fit_dE/E'][it] = np.mean(100 * hres_specs_fits['opt_pars'][i_group, 3]
                                                         / hres_specs_fits['opt_pars'][i_group, 2], axis=0)
                hres_stats['med_fit_dE/E'][it] = np.median(100 * hres_specs_fits['opt_pars'][i_group, 3]
                                                           / hres_specs_fits['opt_pars'][i_group, 2], axis=0)
                hres_stats['std_fit_dE/E'][it] = np.std(100 * hres_specs_fits['opt_pars'][i_group, 3]
                                                        / hres_specs_fits['opt_pars'][i_group, 2], axis=0)
                hres_stats['avg_peak_smooth_pC/MeV'][it] = np.mean(np.max(smooth_hres[i_group, :], axis=1))
                hres_stats['med_peak_smooth_pC/MeV'][it] = np.median(np.max(smooth_hres[i_group, :], axis=1))
                hres_stats['std_peak_smooth_pC/MeV'][it] = np.std(np.max(smooth_hres[i_group, :], axis=1))
                hres_stats['avg_peak_smooth_MeV'][it] = np.mean(axis_MeV[np.argmax(smooth_hres[i_group, :], axis=1)])
                hres_stats['med_peak_smooth_MeV'][it] = np.median(axis_MeV[np.argmax(smooth_hres[i_group, :], axis=1)])
                hres_stats['std_peak_smooth_MeV'][it] = np.std(axis_MeV[np.argmax(smooth_hres[i_group, :], axis=1)])

        else:
            # spec_analyzer = default_analyzer_generators.return_default_hi_res_mag_cam_analyzer()
            spec_analyzer = analyzer_from_device_type('UC_HiResMagCam')

            # noinspection PyTypeChecker
            analysis = spec_analyzer.analyze_image(ni.read_imaq_image(magspec_data['hres']['paths'][0]))
            axis_MeV = np.array(analysis['analyzer_return_lineouts'][0, :])
            # stats_keys = list(analysis[1].keys())

            avg_hres_pC = np.zeros((len(indexes), axis_MeV.size))
            med_hres_pC = np.zeros((len(indexes), axis_MeV.size))
            std_hres_pC = np.zeros((len(indexes), axis_MeV.size))
            # charge = []

            for it, i_group in enumerate(indexes):
                specs = []
                # analysis results for the shots in this parameter step group
                # analyzer_returns: list[dict[str, Union[float, int]]] = []
                analyzer_returns: list[Union[float, int, str, np.ndarray]] = []
                smooth_hres = []
                hres_specs_fits = {'opt_pars': [], 'err_pars': [], 'fits': []}
                for path in np.array(magspec_data['hres']['paths'])[i_group]:
                    try:
                        # noinspection PyTypeChecker
                        analysis = spec_analyzer.analyze_image(ni.read_imaq_image(path))
                        specs.append(np.array(analysis['analyzer_return_lineouts']))
                        # charge.append(np.sum(specs[-1][1, :]))

                        analyzer_returns.append(analysis['analyzer_return_dictionary'])
                        smooth_hres.append(savgol_filter(specs[-1][1, :], 20, 3))

                        if np.sum(specs[-1][1, :]) > 40:
                            # noinspection PyTypeChecker
                            opt_pars, err_pars, fit = profile_fit(x_data=specs[-1][0, :],
                                                                  y_data=specs[-1][1, :],
                                                                  guess_center=specs[-1][0, np.argmax(smooth_hres[-1])],
                                                                  crop_sigma_radius=10.)
                            hres_specs_fits['opt_pars'].append(opt_pars)
                            hres_specs_fits['err_pars'].append(err_pars)
                            hres_specs_fits['fits'].append(fit)
                    except Exception:
                        continue

                smooth_hres = np.array(smooth_hres)
                hres_specs_fits['opt_pars'] = np.array(hres_specs_fits['opt_pars'])

                avg_hres_pC[it, :] = np.mean(np.array(specs)[:, 1, :], axis=0)
                med_hres_pC[it, :] = np.median(np.array(specs)[:, 1, :], axis=0)
                std_hres_pC[it, :] = np.std(np.array(specs)[:, 1, :], axis=0)

                for spec_analyzer_metric, hres_stats_metric in [
                     ('weighted_average_energy_MeV', 'weighted_mean_MeV'), 
                     ('energy_spread_weighted_rms_MeV', 'weighted_rms_MeV'), 
                     ('energy_spread_percent', 'weighted_dE/E'), 
                     ('peak_charge_pc/MeV', 'peak_charge_pC/MeV'), 
                     ('peak_charge_energy_MeV', 'peak_charge_MeV')]:

                    analyzer_returns_this_metric = np.array(
                        [analyzer_return_per_shot[spec_analyzer_metric]
                         for analyzer_return_per_shot in analyzer_returns])
                    hres_stats[f"avg_{hres_stats_metric}"][it] = np.mean(analyzer_returns_this_metric)
                    hres_stats[f"med_{hres_stats_metric}"][it] = np.median(analyzer_returns_this_metric)
                    hres_stats[f"std_{hres_stats_metric}"][it] = np.std(analyzer_returns_this_metric)

                if hres_specs_fits['opt_pars'].any():
                    hres_stats['avg_fit_mean_MeV'][it] = np.mean(hres_specs_fits['opt_pars'][:, 2], axis=0)
                    hres_stats['med_fit_mean_MeV'][it] = np.median(hres_specs_fits['opt_pars'][:, 2], axis=0)
                    hres_stats['std_fit_mean_MeV'][it] = np.std(hres_specs_fits['opt_pars'][:, 2], axis=0)
                    hres_stats['avg_fit_fwhm_MeV'][it] = std_to_fwhm(np.mean(hres_specs_fits['opt_pars'][:, 3], axis=0))
                    hres_stats['med_fit_fwhm_MeV'][it] = (
                        std_to_fwhm(np.median(hres_specs_fits['opt_pars'][:, 3], axis=0)))
                    hres_stats['std_fit_fwhm_MeV'][it] = std_to_fwhm(np.std(hres_specs_fits['opt_pars'][:, 3], axis=0))
                    hres_stats['avg_fit_dE/E'][it] = np.mean(100 * hres_specs_fits['opt_pars'][:, 3]
                                                             / hres_specs_fits['opt_pars'][:, 2], axis=0)
                    hres_stats['med_fit_dE/E'][it] = np.median(100 * hres_specs_fits['opt_pars'][:, 3]
                                                               / hres_specs_fits['opt_pars'][:, 2], axis=0)
                    hres_stats['std_fit_dE/E'][it] = np.std(100 * hres_specs_fits['opt_pars'][:, 3]
                                                            / hres_specs_fits['opt_pars'][:, 2], axis=0)
                    hres_stats['avg_peak_smooth_pC/MeV'][it] = np.mean(np.max(smooth_hres, axis=1))
                    hres_stats['med_peak_smooth_pC/MeV'][it] = np.median(np.max(smooth_hres, axis=1))
                    hres_stats['std_peak_smooth_pC/MeV'][it] = np.std(np.max(smooth_hres, axis=1))
                hres_stats['avg_peak_smooth_MeV'][it] = np.mean(axis_MeV[np.argmax(smooth_hres, axis=1)])
                hres_stats['med_peak_smooth_MeV'][it] = np.median(axis_MeV[np.argmax(smooth_hres, axis=1)])
                hres_stats['std_peak_smooth_MeV'][it] = np.std(axis_MeV[np.argmax(smooth_hres, axis=1)])

            # path = magspec_data['hres']['paths'][np.where((100 > np.array(charge)) & (np.array(charge) > 80))[0][0]]
            # noinspection PyTypeChecker
            # analysis = spec_analyzer.analyze_image(ni.read_imaq_image(path))

        spec_hres_pC = {'avg': avg_hres_pC,
                        'med': med_hres_pC,
                        'std': std_hres_pC}

        return {'axis_MeV': axis_MeV,
                'spec_hres_pC/MeV': spec_hres_pC,
                'spec_hres_stats': hres_stats}


if __name__ == '__main__':
    _htu = HtuExp(get_info=True)
    _base_tag = ScanTag(2023, 8, 9, 4)

    _folder = ScanDataMagspec.build_folder_path(_base_tag, _htu.base_path)
    _scan_data = ScanDataMagspec(_folder, ignore_experiment_name=_htu.is_offline)

    _magspec_data = _scan_data.load_mag_spec_data()
    _device, _variable = _scan_data.scan_info['Scan Parameter'].split(' ', maxsplit=1)
    _indexes, _setpoints, _matching = _scan_data.group_shots_by_step(_device, _variable)
    _magspec_analysis = _scan_data.analyze_mag_spec(_indexes)

    # plt.figure()
    # for x, ind in zip(measured.avg_x, measured.indexes):
    #     plt.plot(x * np.ones(ind.shape), ind, '.', alpha=0.3)
    # plt.xlabel('Current [A]')
    # plt.ylabel('Indexes')
    # plt.show(block=True)

    print('Done')
