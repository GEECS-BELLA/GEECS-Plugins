import os
from pathlib import Path
from typing import Optional, Union
from geecs_api.interface import GeecsDatabase
from geecs_api.devices.geecs_device import GeecsDevice
from geecs_api.devices.HTU.diagnostics import EBeamDiagnostics
from htu_scripts.analysis.screens_scan_analysis import screens_scan_analysis
from geecs_api.tools.images.scan_images import ScanImages
from geecs_api.tools.interfaces.prompts import text_input
from geecs_api.tools.images.filtering import FiltersParameters
from geecs_api.tools.scans.scan_data import ScanData


def screens_scan(e_diagnostics: EBeamDiagnostics,
                 first_screen: Optional[str] = 'A1',
                 last_screen: Optional[str] = 'A3',
                 undulator_diagnostic: Optional[str] = 'spectrum',
                 log_comment: str = '',
                 backgrounds: int = 0) -> dict[str, tuple[Union[Path, str], Union[Path, str]]]:
    """
    Outputs:
    ====================
    image_analysis_files:   dict of label (key) and tuple of two Path (value)
                            to the resulting analyses at each screen, and to the no-scan data path
    """

    image_analysis_files: dict[str, tuple[Union[Path, str], Union[Path, str]]] = {}

    if undulator_diagnostic not in e_diagnostics.undulator_stage.diagnostics:
        return image_analysis_files

    # screens
    all_labels: list[str] = list(e_diagnostics.imagers.keys())
    labels_shown: bool = False
    if first_screen is None or first_screen not in all_labels:
        print(f'Screens shorthand labels: {str(all_labels)[1:-1]}')
        labels_shown = True
        first_screen = text_input('First screen: ', accepted_answers=all_labels)

    if last_screen is None or last_screen not in all_labels:
        if not labels_shown:
            print(f'Screens shorthand labels: {str(all_labels)[1:-1]}')
        last_screen = text_input('Last screen: ', accepted_answers=all_labels)

    i1 = all_labels.index(first_screen)
    i2 = all_labels.index(last_screen)
    used_labels: list[str] = all_labels[i1:i2+1] if i2 > i1 else all_labels[i2:i1-1:-1]

    # scan
    success = True
    for label in used_labels:
        screen = e_diagnostics.imagers[label].screen
        camera = e_diagnostics.imagers[label].camera

        # insert screen
        print(f'Inserting {screen.var_alias} ({screen.controller.get_name()})...')
        while True:
            screen.insert()
            if screen.is_inserted():
                break
            else:
                print(f'Failed to insert {screen.var_alias} ({screen.controller.get_name()})')
                repeat = text_input(f'Try again? : ', accepted_answers=['y', 'yes', 'n', 'no'])
                if repeat.lower()[0] == 'n':
                    success = False
                    break

        # set undulator stage
        if label[0] == 'U':
            station = int(label[1])
            print(f'Moving undulator stage to station {station}, "{undulator_diagnostic}"...')
            while True:
                if e_diagnostics.undulator_stage.set_position(station, undulator_diagnostic):
                    break
                else:
                    print(f'Failed to move to station {station}, "{undulator_diagnostic}"')
                    repeat = text_input(f'Try again? : ', accepted_answers=['y', 'yes', 'n', 'no'])
                    if repeat.lower()[0] == 'n':
                        success = False
                        break

        # check
        if not success:
            image_analysis_files[label] = ('', '')
            break  # for loop

        # scan/background
        if backgrounds > 0:
            print(f'Collecting {backgrounds} background images...')
            camera.save_local_background(n_images=backgrounds)
            print(f'Background saved:\n\t')

        print(f'Starting no-scan (camera of interest: {camera.get_name()})...')
        scan_comment: str = f'No-scan with beam on "{label}" ({camera.get_name()})'
        scan_comment: str = f'{log_comment}. {scan_comment}' if log_comment else scan_comment
        scan_path, _, _, _ = \
            GeecsDevice.run_no_scan(monitoring_device=camera, comment=scan_comment, timeout=300.)

        # remove screen
        print(f'Removing {screen.var_alias} ({screen.controller.get_name()})...')
        while True:
            screen.remove()
            if not screen.is_inserted():
                break
            else:
                print(f'Failed to remove {screen.var_alias} ({screen.controller.get_name()})')
                repeat = text_input(f'Try again? : ', accepted_answers=['y', 'yes', 'n', 'no'])
                if repeat.lower()[0] == 'n':
                    success = False
                    break

        # check
        if not success:
            image_analysis_files[label] = ('', '')
            break  # for loop

        # analysis
        scan_path = Path(scan_path)
        no_scan = ScanData(scan_path, ignore_experiment_name=False)
        no_scan_images = ScanImages(no_scan, camera)
        analysis_file: Path = no_scan_images.save_folder.parent / 'profiles_analysis.dat'
        no_scan_images.run_analysis_with_checks(
            initial_filtering=FiltersParameters(com_threshold=0.66, bkg_image=Path(camera.state_background_path())),
            plots=True)
        keep = text_input(f'Add this analysis to the overall screen scan analysis? : ',
                          accepted_answers=['y', 'yes', 'n', 'no'])
        if keep.lower()[0] == 'y':
            image_analysis_files[label] = (analysis_file, scan_path)
        else:
            image_analysis_files[label] = ('', scan_path)

    # analyze scan overall
    print(f'Screen scan done. Processing analyses...')
    if image_analysis_files:
        save_folder = ''
        first_scan: str = 'Scan'
        for label in used_labels:
            if label in image_analysis_files and image_analysis_files[label][0]:
                first_scan = image_analysis_files[label][1].parts[-1]
                save_folder = image_analysis_files[label][1].parents[2]
                break
        save_folder = Path(*save_folder) / 'analysis' / f'{first_scan} - Screens {used_labels[0]}-{used_labels[-1]}'
        if not save_folder.is_dir():
            os.makedirs(save_folder)

        screens_scan_analysis(image_analysis_files, used_labels, save_dir=save_folder)

    return image_analysis_files


if __name__ == '__main__':
    # initialization
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')
    _e_diagnostics = EBeamDiagnostics()

    # scan
    try:
        screens_scan(_e_diagnostics, 'U1', 'U9',
                     undulator_diagnostic='spectrum',
                     log_comment='Screen scan',
                     backgrounds=10)
    except Exception:
        pass
    finally:
        # close connections
        _e_diagnostics.close()
        [controller.close() for controller in _e_diagnostics.controllers]
