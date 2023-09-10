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
from geecs_python_api.tools.interfaces.tdms import read_geecs_tdms
from geecs_python_api.tools.images.spot import profile_fit
from geecs_python_api.tools.distributions.binning import unsupervised_binning, BinningResults


class ScanData:
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

        self.identified = False
        self.scan_info: dict[str, str] = {}

        self.__folder: Optional[Path] = None
        self.__tag: Optional[ScanTag] = None
        self.__tag_date: Optional[date] = None
        self.__analysis_folder: Optional[Path] = None

        if folder:
            try:
                folder = Path(folder)

                (exp_name, year_folder_name, month_folder_name, date_folder_name, 
                 scans_literal, scan_folder_name) = folder.parts[-6:]
                
                if (not re.match(r"Y\d{4}", year_folder_name)) or \
                   (not re.match(r"\d{2}-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", month_folder_name)) or \
                   (not re.match(r"\d{2}_\d{4}", date_folder_name)) or \
                   (not scans_literal == 'scans') or \
                   (not re.match(r"Scan\d{3,}", scan_folder_name)):
                    raise ValueError("Folder path does not appear to follow convention")
                elif not folder.exists():
                    raise ValueError("Folder does not exist")

                self.__tag_date = dtime.strptime(date_folder_name, "%y_%m%d").date()
                self.__tag = \
                    ScanTag(self.__tag_date.year, self.__tag_date.month, self.__tag_date.day, int(scan_folder_name[4:]))

                self.identified = ignore_experiment_name or (exp_name == GeecsDevice.exp_info['name'])
                if self.identified:
                    self.__folder = folder

            except Exception:
                raise

        if not self.identified and tag:
            if isinstance(tag, int):
                self.__tag_date = dtime.now().date()
                tag = ScanTag(self.__tag_date.year, self.__tag_date.month, self.__tag_date.day, tag)

            if isinstance(tag, tuple):
                try:
                    if not isinstance(tag, ScanTag):
                        tag = ScanTag(*tag)

                    if experiment_base_path is None:
                        exp_path = Path(GeecsDevice.exp_info['data_path'])
                    else:
                        exp_path = Path(experiment_base_path)

                    if not exp_path.is_dir():
                        raise ValueError("Experiment base folder does not exist")

                    if self.__tag_date is None:
                        self.__tag_date = date(tag.year, tag.month, tag.day)

                    folder = (exp_path /
                              self.__tag_date.strftime("Y%Y") /
                              self.__tag_date.strftime("%m-%b") /
                              self.__tag_date.strftime("%y_%m%d") /
                              'scans'/f'Scan{tag.number:03d}')
                    self.identified = folder.is_dir()
                    if self.identified:
                        self.__tag = tag
                        self.__folder = folder
                    else:
                        raise OSError

                except Exception:
                    raise

        if not self.identified:
            raise ValueError

        # scan info
        self.load_scan_info()

        # folders & files
        top_content = next(os.walk(self.__folder))
        self.files = {'devices': top_content[1], 'files': top_content[2]}

        parts = list(Path(self.__folder).parts)
        parts[-2] = 'analysis'
        self.__analysis_folder = Path(*parts)
        if not self.__analysis_folder.is_dir():
            os.makedirs(self.__analysis_folder)

        # scalar data
        self.data_frame = None  # use tdms.geecs_tdms_dict_to_panda
        if load_scalars:
            self.load_scalar_data()
        else:
            self.data_dict = {}

    @staticmethod
    def build_folder_path(tag: ScanTag, base_directory: Union[Path, str] = r'Z:\data', experiment: str = 'Undulator') \
            -> Path:
        base_directory = Path(base_directory)
        folder: Path = base_directory / experiment
        folder = folder / f'Y{tag[0]}' / f'{tag[1]:02d}-{cal.month_name[tag[1]][:3]}'
        folder = folder / f'{str(tag[0])[-2:]}_{tag[1]:02d}{tag[2]:02d}'
        folder = folder / 'scans' / f'Scan{tag[3]:03d}'

        return folder

    def get_folder(self) -> Optional[Path]:
        return self.__folder

    def get_tag(self) -> Optional[ScanTag]:
        return self.__tag

    def get_analysis_folder(self) -> Optional[Path]:
        return self.__analysis_folder

    def get_device_data(self, device_name: str):
        if device_name in self.data_dict:
            return self.data_dict[device_name]
        else:
            return {}

    def bin_data(self, device: str, variable: str) -> tuple[list[np.ndarray], Optional[np.ndarray], bool]:
        dev_data = self.get_device_data(device)
        if not dev_data:
            return [], None, False

        measured: BinningResults = unsupervised_binning(dev_data[variable], dev_data['shot #'])

        Expected = NamedTuple('Expected',
                              start=float,
                              end=float,
                              steps=int,
                              shots=int,
                              setpoints=np.ndarray,
                              indexes=list)
        start = float(self.scan_info['Start'])
        end = float(self.scan_info['End'])
        steps: int = 1 + round(np.abs(end - start) / float(self.scan_info['Step size']))
        shots: int = int(self.scan_info['Shots per step'])
        expected = Expected(start=start, end=end, steps=steps, shots=shots,
                            setpoints=np.linspace(float(self.scan_info['Start']),
                                                  float(self.scan_info['End']), steps),
                            indexes=[np.arange(p * shots, (p+1) * shots) for p in range(steps)])

        matching = all([inds.size == expected.shots for inds in measured.indexes])
        matching = matching and (len(measured.indexes) == expected.steps)
        if not matching:
            api_error.warning(f'Observed data binning does not match expected scan parameters (.ini)',
                              f'Function "{inspect.stack()[0][3]}"')

        if matching:
            indexes = expected.indexes
            setpoints = expected.setpoints
        else:
            indexes = measured.indexes
            setpoints = measured.avg_x

        return indexes, setpoints, matching

    def load_scan_info(self):
        config_parser = ConfigParser()
        config_parser.optionxform = str

        try:
            config_parser.read(self.__folder / f'ScanInfoScan{self.__tag.number:03d}.ini')
            self.scan_info.update({key: value.strip("'\"")
                                   for key, value in config_parser.items("Scan Info")})
        except NoSectionError:
            api_error.warning(f'ScanInfo file does not have a "Scan Info" section',
                              f'ScanData class, method "{inspect.stack()[0][3]}"')

    def load_scalar_data(self) -> bool:
        tdms_path = self.__folder / f'Scan{self.__tag.number:03d}.tdms'
        if tdms_path.is_file():
            self.data_dict = read_geecs_tdms(tdms_path)

        return tdms_path.is_file()

    def load_mag_spec_data(self) -> dict[str, Any]:
        magspec_dict = {'full': {}, 'hres': {}}
        magspec_dict['full']['paths'] = list_files(self.__folder / 'U_BCaveMagSpec-interpSpec', -1, '.txt')
        magspec_dict['hres']['paths'] = list_files(self.__folder / 'U_HiResMagCam-interpSpec', -1, '.txt')

        for key in ['full', 'hres']:
            specs = []
            shots = []
            for path in magspec_dict[key]['paths']:
                try:
                    specs.append(np.loadtxt(path, skiprows=1))
                    shots.append(int(path.name[-7:-4]))
                except Exception:
                    continue

            magspec_dict[key]['specs'] = np.array(specs)
            magspec_dict[key]['shots'] = np.array(shots)

        return magspec_dict

    def analyze_mag_spec(self, indexes: list[np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
        # load raw data
        magspec_data = self.load_mag_spec_data()
        if not magspec_data['full']['specs'].size > 0:
            return {}

        axis_MeV: dict[str, np.ndarray] = {
            'full': magspec_data['full']['specs'][0, :, 0] * 1000.,
            'hres': magspec_data['hres']['specs'][0, :, 0]}

        # fit all full-spectra
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

        avg_spec_full_pC = np.zeros((len(indexes), axis_MeV['full'].size))
        med_spec_full_pC = np.zeros((len(indexes), axis_MeV['full'].size))
        std_spec_full_pC = np.zeros((len(indexes), axis_MeV['full'].size))
        spec_full_stats = {'avg_MeV': np.zeros((len(indexes),)),
                           'med_MeV': np.zeros((len(indexes),)),
                           'std_MeV': np.zeros((len(indexes),)),
                           'avg_fwhm_MeV': np.zeros((len(indexes),)),
                           'med_fwhm_MeV': np.zeros((len(indexes),)),
                           'std_fwhm_MeV': np.zeros((len(indexes),)),
                           'avg_dE/E': np.zeros((len(indexes),)),
                           'med_dE/E': np.zeros((len(indexes),)),
                           'std_dE/E': np.zeros((len(indexes),)),
                           'avg_peak_fit_pC': np.zeros((len(indexes),)),
                           'med_peak_fit_pC': np.zeros((len(indexes),)),
                           'std_peak_fit_pC': np.zeros((len(indexes),)),
                           'avg_peak_fit_MeV': np.zeros((len(indexes),)),
                           'med_peak_fit_MeV': np.zeros((len(indexes),)),
                           'std_peak_fit_MeV': np.zeros((len(indexes),))}

        avg_spec_hres_pC = np.zeros((len(indexes), axis_MeV['hres'].size))
        med_spec_hres_pC = np.zeros((len(indexes), axis_MeV['hres'].size))
        std_spec_hres_pC = np.zeros((len(indexes), axis_MeV['hres'].size))
        spec_hres_stats = {'avg_MeV': np.zeros((len(indexes),)),
                           'med_MeV': np.zeros((len(indexes),)),
                           'std_MeV': np.zeros((len(indexes),)),
                           'avg_fwhm_MeV': np.zeros((len(indexes),)),
                           'med_fwhm_MeV': np.zeros((len(indexes),)),
                           'std_fwhm_MeV': np.zeros((len(indexes),)),
                           'avg_dE/E': np.zeros((len(indexes),)),
                           'med_dE/E': np.zeros((len(indexes),)),
                           'std_dE/E': np.zeros((len(indexes),)),
                           'avg_peak_fit_pC': np.zeros((len(indexes),)),
                           'med_peak_fit_pC': np.zeros((len(indexes),)),
                           'std_peak_fit_pC': np.zeros((len(indexes),)),
                           'avg_peak_fit_MeV': np.zeros((len(indexes),)),
                           'med_peak_fit_MeV': np.zeros((len(indexes),)),
                           'std_peak_fit_MeV': np.zeros((len(indexes),))}

        for it, i_group in enumerate(indexes):
            avg_spec_full_pC[it, :] = np.mean(magspec_data['full']['specs'][i_group, :, 1], axis=0)
            med_spec_full_pC[it, :] = np.median(magspec_data['full']['specs'][i_group, :, 1], axis=0)
            std_spec_full_pC[it, :] = np.std(magspec_data['full']['specs'][i_group, :, 1], axis=0)
            spec_full_stats['avg_MeV'][it] = np.mean(full_specs_fits['opt_pars'][i_group, 2], axis=0)
            spec_full_stats['med_MeV'][it] = np.median(full_specs_fits['opt_pars'][i_group, 2], axis=0)
            spec_full_stats['std_MeV'][it] = np.std(full_specs_fits['opt_pars'][i_group, 2], axis=0)
            spec_full_stats['avg_fwhm_MeV'][it] = np.mean(full_specs_fits['opt_pars'][i_group, 3], axis=0)
            spec_full_stats['avg_fwhm_MeV'][it] *= 2 * np.sqrt(2 * np.log(2))
            spec_full_stats['med_fwhm_MeV'][it] = np.median(full_specs_fits['opt_pars'][i_group, 3], axis=0)
            spec_full_stats['med_fwhm_MeV'][it] *= 2 * np.sqrt(2 * np.log(2))
            spec_full_stats['std_fwhm_MeV'][it] = np.std(full_specs_fits['opt_pars'][i_group, 3], axis=0)
            spec_full_stats['std_fwhm_MeV'][it] *= 2 * np.sqrt(2 * np.log(2))
            spec_full_stats['avg_dE/E'][it] = \
                np.mean(full_specs_fits['opt_pars'][i_group, 2] / full_specs_fits['opt_pars'][i_group, 3], axis=0)
            spec_full_stats['med_dE/E'][it] = \
                np.median(full_specs_fits['opt_pars'][i_group, 2] / full_specs_fits['opt_pars'][i_group, 3], axis=0)
            spec_full_stats['std_dE/E'][it] = \
                np.std(full_specs_fits['opt_pars'][i_group, 2] / full_specs_fits['opt_pars'][i_group, 3], axis=0)
            spec_full_stats['avg_peak_fit_pC'][it] = np.mean(np.max(smooth_full[i_group, :], axis=1))
            spec_full_stats['med_peak_fit_pC'][it] = np.median(np.max(smooth_full[i_group, :], axis=1))
            spec_full_stats['std_peak_fit_pC'][it] = np.std(np.max(smooth_full[i_group, :], axis=1))
            spec_full_stats['avg_peak_fit_MeV'][it] = \
                np.mean(axis_MeV['full'][np.argmax(smooth_full[i_group, :], axis=1)])
            spec_full_stats['med_peak_fit_MeV'][it] = \
                np.median(axis_MeV['full'][np.argmax(smooth_full[i_group, :], axis=1)])
            spec_full_stats['std_peak_fit_MeV'][it] = \
                np.std(axis_MeV['full'][np.argmax(smooth_full[i_group, :], axis=1)])

            avg_spec_hres_pC[it, :] = np.mean(magspec_data['hres']['specs'][i_group, :, 1], axis=0)
            med_spec_hres_pC[it, :] = np.median(magspec_data['hres']['specs'][i_group, :, 1], axis=0)
            std_spec_hres_pC[it, :] = np.std(magspec_data['hres']['specs'][i_group, :, 1], axis=0)
            spec_hres_stats['avg_MeV'][it] = np.mean(hres_specs_fits['opt_pars'][i_group, 2], axis=0)
            spec_hres_stats['med_MeV'][it] = np.median(hres_specs_fits['opt_pars'][i_group, 2], axis=0)
            spec_hres_stats['std_MeV'][it] = np.std(hres_specs_fits['opt_pars'][i_group, 2], axis=0)
            spec_hres_stats['avg_fwhm_MeV'][it] = np.mean(hres_specs_fits['opt_pars'][i_group, 3], axis=0)
            spec_hres_stats['avg_fwhm_MeV'][it] *= 2 * np.sqrt(2 * np.log(2))
            spec_hres_stats['med_fwhm_MeV'][it] = np.median(hres_specs_fits['opt_pars'][i_group, 3], axis=0)
            spec_hres_stats['med_fwhm_MeV'][it] *= 2 * np.sqrt(2 * np.log(2))
            spec_hres_stats['std_fwhm_MeV'][it] = np.std(hres_specs_fits['opt_pars'][i_group, 3], axis=0)
            spec_hres_stats['std_fwhm_MeV'][it] *= 2 * np.sqrt(2 * np.log(2))
            spec_hres_stats['avg_dE/E'][it] = \
                np.mean(hres_specs_fits['opt_pars'][i_group, 2] / hres_specs_fits['opt_pars'][i_group, 3], axis=0)
            spec_hres_stats['med_dE/E'][it] = \
                np.median(hres_specs_fits['opt_pars'][i_group, 2] / hres_specs_fits['opt_pars'][i_group, 3], axis=0)
            spec_hres_stats['std_dE/E'][it] = \
                np.std(hres_specs_fits['opt_pars'][i_group, 2] / hres_specs_fits['opt_pars'][i_group, 3], axis=0)
            spec_hres_stats['avg_peak_fit_pC'][it] = np.mean(np.max(smooth_hres[i_group, :], axis=1))
            spec_hres_stats['med_peak_fit_pC'][it] = np.median(np.max(smooth_hres[i_group, :], axis=1))
            spec_hres_stats['std_peak_fit_pC'][it] = np.std(np.max(smooth_hres[i_group, :], axis=1))
            spec_hres_stats['avg_peak_fit_MeV'][it] = \
                np.mean(axis_MeV['full'][np.argmax(smooth_full[i_group, :], axis=1)])
            spec_hres_stats['med_peak_fit_MeV'][it] = \
                np.median(axis_MeV['full'][np.argmax(smooth_full[i_group, :], axis=1)])
            spec_hres_stats['std_peak_fit_MeV'][it] = \
                np.std(axis_MeV['full'][np.argmax(smooth_full[i_group, :], axis=1)])

        spec_full_pC = {'avg': avg_spec_full_pC,
                        'med': med_spec_full_pC,
                        'std': std_spec_full_pC}
        spec_hres_pC = {'avg': avg_spec_hres_pC,
                        'med': med_spec_hres_pC,
                        'std': std_spec_hres_pC}

        return {'axis_MeV': axis_MeV,
                'spec_full_pC': spec_full_pC,
                'spec_hres_pC': spec_hres_pC,
                'spec_full_stats': spec_full_stats,
                'spec_hres_stats': spec_hres_stats}


if __name__ == '__main__':
    _htu = HtuExp(get_info=True)
    _base_tag = ScanTag(2023, 7, 6, 4)

    _folder = ScanData.build_folder_path(_base_tag, _htu.base_path)
    _scan_data = ScanData(_folder, ignore_experiment_name=_htu.is_offline)

    print('Loading mag spec data...')
    _magspec_data = _scan_data.load_mag_spec_data()
    # plt.figure()
    # for x, ind in zip(measured.avg_x, measured.indexes):
    #     plt.plot(x * np.ones(ind.shape), ind, '.', alpha=0.3)
    # plt.xlabel('Current [A]')
    # plt.ylabel('Indexes')
    # plt.show(block=True)

    print('Done')
