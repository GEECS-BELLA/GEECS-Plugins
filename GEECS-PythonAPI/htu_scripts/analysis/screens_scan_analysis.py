import os
import time
import numpy as np
import calendar as cal
from pathlib import Path
from typing import Optional, Union, Any
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
import matplotlib.pyplot as plt
from tkinter import filedialog
from progressbar import ProgressBar


def screens_scan_analysis(no_scans: dict[str, tuple[Union[Path, str], Union[Path, str]]], screen_labels: list[str],
                          initial_filtering=FiltersParameters(), save_dir: Optional[SysPath] = None,
                          ignore_experiment_name: bool = False, trim_collection: bool = False,
                          new_targets: bool = False) -> dict[str, Any]:
    """
    no_scans = dict of tuples (analysis file paths, scan data directory paths)
    """
    beam_analysis: dict[str, Any] = {}

    # save analysis
    if not save_dir:
        save_dir = filedialog.askdirectory(title='Save directory:')
        if save_dir:
            save_dir = Path(save_dir)
    elif not isinstance(save_dir, Path):
        save_dir = Path(save_dir)

    analysis_files: list[Path] = []
    scan_paths: list[Path] = []
    pos_short_names: list[str] = []
    pos_long_names: list[str] = []

    with ProgressBar(max_value=len(no_scans)) as pb:
        for it, (lbl, (analysis_file, scan_path)) in enumerate(no_scans.items()):
            analysis: dict[str, Any] = {}
            analyze: str = 'y'
            if analysis_file.is_file():
                analyze = text_input(f'\nRe-run the analysis ({scan_path.name}, {lbl})? : ',
                                     accepted_answers=['y', 'yes', 'n', 'no'])

            if (analyze.lower()[0] == 'y') or (not analysis_file.is_file()):
                print(f'\nAnalyzing {scan_path.name} ("{lbl}")...')
                scan_obj = ScanData(scan_path, ignore_experiment_name=ignore_experiment_name)
                no_scan = ScanImages(scan_obj, lbl)
                analysis_file, analysis = \
                    no_scan.run_analysis_with_checks(images=-1, initial_filtering=initial_filtering,
                                                     trim_collection=trim_collection, new_targets=new_targets,
                                                     plots=True, save=bool(save_dir))

            if not analysis:
                print('Loading analysis...')
                analysis, analysis_file = load_py(analysis_file, as_dict=True)
                scan_obj = ScanData(scan_path, ignore_experiment_name=ignore_experiment_name)
                no_scan = ScanImages(scan_obj, lbl)
                no_scan.render_image_analysis(analysis['average_analysis'], tag='average_image', block=True, save=False)

            if not analysis:
                continue  # skip

            if not analysis_file:
                analysis_file = ''
            analysis_files.append(analysis_file)
            scan_paths.append(scan_path)

            keep = text_input(f'Add this analysis to the overall screen scan analysis? : ',
                              accepted_answers=['y', 'yes', 'n', 'no'])
            if keep.lower()[0] == 'n':
                continue

            # noinspection PyUnboundLocalVariable
            label = Camera.label_from_name(analysis['camera_name'])
            index = screen_labels.index(label)

            print('Collecting analysis summary...')
            beam_analysis, pos_short_names, pos_long_names = \
                add_beam_analysis(beam_analysis, analysis, pos_short_names, pos_long_names, index, len(screen_labels))

            pb.increment()
            time.sleep(0.01)

    data_dict: dict[str, Any] = {'screen_labels': screen_labels,
                                 'analysis_files': analysis_files,
                                 'scan_paths': scan_paths,
                                 'beam_analysis': beam_analysis}
    if save_dir:
        if not save_dir.is_dir():
            os.makedirs(save_dir)

        export_file_path: Path = save_dir / 'beam_analysis'
        save_py(file_path=export_file_path, data=data_dict)
        print(f'Data exported to:\n\t{export_file_path}.dat')

    # plots
    x_axis: np.ndarray = np.arange(1, len(screen_labels) + 1, dtype='int')

    ys_deltas = (np.inf, -np.inf)
    ys_fwhms = (np.inf, -np.inf)
    for pos in pos_short_names:
        ys_deltas = (min(ys_deltas[0], np.min(beam_analysis[f'{pos}_deltas_mm_means'])),
                     max(ys_deltas[1], np.max(beam_analysis[f'{pos}_deltas_mm_means'])))
        ys_fwhms = (min(ys_fwhms[0], np.min(beam_analysis[f'{pos}_fwhm_means'])),
                    max(ys_fwhms[1], np.max(beam_analysis[f'{pos}_fwhm_means'])))

    if (ys_deltas[1] - ys_deltas[0]) > 1.2:
        f_deltas = 1000
        units_deltas = r'$\mu$m'
    else:
        f_deltas = 1
        units_deltas = 'mm'

    if (abs(ys_fwhms[1]) > 1200) or (abs(ys_fwhms[0]) > 1200):
        f_fwhms = 0.001
        units_fwhms = 'mm'
    else:
        f_fwhms = 1
        units_fwhms = r'$\mu$m'

    if pos_short_names:
        render_screens_scan_analysis(pos_short_names, pos_long_names, x_axis, f_deltas, f_fwhms, screen_labels,
                                     units_deltas, units_fwhms, save_dir, beam_analysis)

    return data_dict


def render_screens_scan_analysis(pos_short_names: list[str], pos_long_names: list[str], x_axis: np.ndarray,
                                 f_deltas: Union[int, float], f_fwhms: Union[int, float], screen_labels: list[str],
                                 units_deltas: str, units_fwhms: str, save_dir: Optional[SysPath],
                                 beam_analysis: dict[str, Any]):
    fig, axs = plt.subplots(ncols=len(pos_short_names), nrows=4,
                            figsize=(ScanImages.fig_size[0] * 1.5, ScanImages.fig_size[1] * 1.5),
                            sharex='col', sharey='row')
    for it, pos in enumerate(pos_short_names):
        # Deltas X
        axs[0, it].fill_between(
            x_axis,
            f_deltas * (beam_analysis[f'{pos}_deltas_mm_means'][:, 1] - beam_analysis[f'{pos}_deltas_mm_stds'][:, 1]),
            f_deltas * (beam_analysis[f'{pos}_deltas_mm_means'][:, 1] + beam_analysis[f'{pos}_deltas_mm_stds'][:, 1]),
            label=r'$D_x \pm \sigma$', color='m', alpha=0.33)
        axs[0, it].plot(x_axis, f_deltas * beam_analysis[f'{pos}_deltas_mm_avg_imgs'][:, 1], 'ob-',
                        label=r'$D_x$ $(\mu_{image})$', linewidth=1, markersize=3)
        axs[0, it].legend(loc='best', prop={'size': 8})
        axs[0, it].set_xticks([])
        axs[0, it].set_title(pos_long_names[it])

        # Deltas Y
        axs[1, it].fill_between(
            x_axis,
            f_deltas * (beam_analysis[f'{pos}_deltas_mm_means'][:, 0] - beam_analysis[f'{pos}_deltas_mm_stds'][:, 0]),
            f_deltas * (beam_analysis[f'{pos}_deltas_mm_means'][:, 0] + beam_analysis[f'{pos}_deltas_mm_stds'][:, 0]),
            label=r'$D_y \pm \sigma$', color='m', alpha=0.33)
        axs[1, it].plot(x_axis, f_deltas * beam_analysis[f'{pos}_deltas_mm_avg_imgs'][:, 0], 'ob-',
                        label=r'$D_y$ $(\mu_{image})$', linewidth=1, markersize=3)
        axs[1, it].legend(loc='best', prop={'size': 8})
        axs[1, it].set_xticks([])

        # FWHM X
        axs[2, it].fill_between(
            x_axis,
            f_fwhms * beam_analysis[f'{pos}_fwhm_means'][:, 1] - beam_analysis[f'{pos}_fwhm_stds'][:, 1],
            f_fwhms * beam_analysis[f'{pos}_fwhm_means'][:, 1] + beam_analysis[f'{pos}_fwhm_stds'][:, 1],
            label=r'$FWHM_x \pm \sigma$', color='y', alpha=0.33)
        axs[2, it].plot(x_axis, f_fwhms * beam_analysis[f'{pos}_fwhm_means'][:, 1], 'og-',
                        label=r'$FWHM_x$ $(\mu_{image})$', linewidth=1, markersize=3)
        axs[2, it].legend(loc='best', prop={'size': 8})
        axs[2, it].set_xticks([])

        # FWHM Y
        axs[3, it].fill_between(
            x_axis,
            f_fwhms * beam_analysis[f'{pos}_fwhm_means'][:, 0] - beam_analysis[f'{pos}_fwhm_stds'][:, 0],
            f_fwhms * beam_analysis[f'{pos}_fwhm_means'][:, 0] + beam_analysis[f'{pos}_fwhm_stds'][:, 0],
            label=r'$FWHM_y \pm \sigma$ [$\mu$m]', color='y', alpha=0.33)
        axs[3, it].plot(x_axis, f_fwhms * beam_analysis[f'{pos}_fwhm_means'][:, 0], 'og-',
                        label=r'$FWHM_y$ $(\mu_{image})$', linewidth=1, markersize=3)
        axs[3, it].legend(loc='best', prop={'size': 8})
        axs[3, it].set_xlabel('Screen')
        axs[3, it].set_xticks(x_axis, screen_labels)

    axs[0, 0].set_ylabel(f'X-Offsets [{units_deltas}]')
    axs[1, 0].set_ylabel(f'Y-Offsets [{units_deltas}]')
    axs[2, 0].set_ylabel(f'X-FWHM [{units_fwhms}]')
    axs[3, 0].set_ylabel(f'Y-FWHM [{units_fwhms}]')

    # set matching vertical limits for deltas/FWHMs
    y_lim = (min(axs[0, 0].get_ylim()[0], axs[1, 0].get_ylim()[0]),
             max(axs[0, 0].get_ylim()[1], axs[1, 0].get_ylim()[1]))
    [axs[0, j].set_ylim(y_lim) for j in range(len(pos_short_names))]
    [axs[1, j].set_ylim(y_lim) for j in range(len(pos_short_names))]

    y_lim = (min(axs[2, 0].get_ylim()[0], axs[3, 0].get_ylim()[0]),
             max(axs[2, 0].get_ylim()[1], axs[3, 0].get_ylim()[1]))
    [axs[2, j].set_ylim(y_lim) for j in range(len(pos_short_names))]
    [axs[3, j].set_ylim(y_lim) for j in range(len(pos_short_names))]

    if save_dir:
        save_path = save_dir / 'beam_analysis.png'
        plt.savefig(save_path, dpi=300)

    plt.show(block=True)


if __name__ == '__main__':
    # base_path = Path(r'C:\Users\GuillaumePlateau\Documents\LBL\Data')
    base_path = Path(r'Z:\data')

    is_local = (str(base_path)[0] == 'C')
    if not is_local:
        GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

    # Parameters
    _base_tag = (2023, 6, 9, 0)
    # _first_scan = 1
    # _first_screen = 3
    # _n_screens = 1
    # _scans_screens = [(_first_scan + n, f'U{_first_screen + n}') for n in range(_n_screens)]
    _scans_screens = [(1, 'A2'),
                      (1, 'A3'),
                      (1, 'U1'),
                      (1, 'U2'),
                      (1, 'U3'),
                      (1, 'U4'),
                      (1, 'U5'),
                      (1, 'U6'),
                      (1, 'U7'),
                      (1, 'U8')]

    # Folders/Files
    _analysis_files: list[Path] = \
        [base_path/'Undulator'/f'Y{_base_tag[0]}'/f'{_base_tag[1]:02d}-{cal.month_name[_base_tag[1]][:3]}' /
         f'{str(_base_tag[0])[-2:]}_{_base_tag[1]:02d}{_base_tag[2]:02d}'/'analysis' /
         f'Scan{number[0]:03d}'/Camera.name_from_label(number[1])/'profiles_analysis.dat'
         for number in _scans_screens]

    _scan_paths = [file.parents[3]/'scans'/file.parts[-3] for file in _analysis_files]
    _no_scans: dict[str, tuple[Path, Path]] = \
        {key[1]: (analysis, data) for key, analysis, data in zip(_scans_screens, _analysis_files, _scan_paths)}

    # _save_dir = None
    _save_dir = _analysis_files[0].parents[2] / \
        f'{_scan_paths[0].name}_Screens_{_scans_screens[0][1]}_{_scans_screens[-1][1]}'
    _labels = [label[1] for label in _scans_screens]  # separate from list(_scans_screens.keys()) to define an order

    # Analysis
    _data_dict = screens_scan_analysis(no_scans=_no_scans,
                                       screen_labels=_labels,
                                       initial_filtering=FiltersParameters(com_threshold=0.66,
                                                                           contrast=1.333),
                                       save_dir=_save_dir,
                                       ignore_experiment_name=False,
                                       trim_collection=True,
                                       new_targets=True)

    print('done')
