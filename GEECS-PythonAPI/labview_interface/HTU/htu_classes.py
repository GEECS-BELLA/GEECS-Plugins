import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional, Any, Union
from pathlib import Path
from labview_interface.lv_interface import Bridge
from geecs_python_api.controls.experiment.experiment import Experiment
from geecs_python_api.controls.experiment.htu import HtuExp
from geecs_python_api.controls.devices.HTU.gas_jet import GasJet
from geecs_python_api.controls.devices.HTU.laser import Laser
# from geecs_python_api.controls.devices.HTU.transport import Steering
from geecs_python_api.analysis.images.scans.scan_data import ScanData


class UserInterface:
    @staticmethod
    def report(message: str):
        Bridge.labview_call('ui', 'report', [], sync=False, message=message)

    @staticmethod
    def add_plots(source: str, plots: list):
        Bridge.labview_call('ui', 'add_plots', [], sync=False, source=source, plots=plots)

    @staticmethod
    def clear_plots(source: str):
        Bridge.labview_call('ui', 'clear_plots', [], sync=False, source=source)


class Handler:
    @staticmethod
    def send_results(source: str, results: list):
        Bridge.labview_call('handler', 'results', [], sync=False, source=source, results=results)

    @staticmethod
    def question(message: str, possible_answers: list) -> Optional[Any]:
        success, answer = Bridge.labview_call('handler', 'question', ['answer'], sync=True, timeout_sec=600.,
                                              message=message, possible_answers=possible_answers)
        if success:
            return answer['answer']
        else:
            return None

    @staticmethod
    def request_values(message: str, values: list[tuple]) -> Optional[Any]:
        """
        Request values from LabVIEW interface

        Args:
            message: string to be presented to the user.
            values: list of tuples describing the values to be collected
            (label: str, type: str = 'bool'/'float'/'int'/'str', min value, max value, initial value);
            use '-inf' for no minimum value; 'inf' for no minimum value

        Returns:
            list of values
        """
        success, answer = Bridge.labview_call('handler', 'values', ['answer'], sync=True, timeout_sec=600.,
                                              message=message, values=values)
        if success:
            return answer['answer']
        else:
            return None


class LPA:
    def __init__(self, is_offline: bool = False):
        if is_offline:
            self.jet = self.laser = None
        else:
            UserInterface.report('Connecting to gas jet elements...')
            self.jet = GasJet()

            UserInterface.report('Connecting to laser elements...')
            self.laser = Laser()

    def close(self):
        UserInterface.report('Disconnecting from gas jet elements...')
        self.jet.close()

        UserInterface.report('Disconnecting from laser elements...')
        self.laser.close()

    def z_scan(self, rough: bool = False) -> tuple[bool, Path]:
        z = round(self.jet.stage.get_position('z'), 2)
        z_alias = self.jet.stage.get_axis_var_alias(2)
        lims = self.jet.stage.var_spans[z_alias]
        values: Union[dict, str]

        while True:
            pars = [6, 11, 0.25, 20] if rough else [z - 1, z + 1, 0.1, 20]
            values = Handler.request_values(f'{"Rough" if rough else "Fine"} Z-scan parameters:',
                                            [('Start [mm]', 'float', lims[0], lims[1], pars[0]),
                                             ('End [mm]', 'float', lims[0], lims[1], pars[1]),
                                             ('Steps [mm]', 'float', 0, 10, pars[2]),
                                             ('Shots/Step', 'int', 1, 'inf', pars[3])])
            if isinstance(values, str) and (values == 'Cancel'):
                cancel = True
                scan_folder = None
                break

            UserInterface.report(f'Starting {"rough" if rough else "fine"} Z-scan...')
            scan_folder, _, _, _ = self.jet.stage.scan(z_alias, values['Start [mm]'], values['End [mm]'],
                                                       values['Steps [mm]'], lims, values['Shots/Step'],
                                                       comment=f'{"rough" if rough else "fine"} Z-scan')

            repeat = Handler.question('Do you want to repeat this Z-scan?', ['Yes', 'No', 'Cancel'])
            cancel = (repeat == 'Cancel')
            if repeat != 'Yes':
                break

        return cancel, scan_folder

    def z_scan_analysis(self, exp: Experiment, scan_path: Path) -> dict[str, np.ndarray]:
        if self.jet is None:
            device = 'U_ESP_JetXYZ'
            variable = 'Position.Axis 3'
        else:
            device = self.jet.stage.get_name()
            variable = self.jet.stage.get_axis_var_name(2)

        scan_data = ScanData(scan_path, ignore_experiment_name=exp.is_offline)
        indexes, setpoints, matching = scan_data.bin_data(device, variable)
        magspec_data = scan_data.analyze_mag_spec(indexes)
        objs = self.objective_analysis(setpoints, magspec_data,
                                       dE_weight=1., pC_weight=2., MeV_weight=1.)

        magspec_data = {
            'setpoints': setpoints,
            'indexes': indexes,
            'axis_MeV': magspec_data['axis_MeV']['hres'],
            **magspec_data['spec_hres_pC/MeV'],
            **magspec_data['spec_hres_stats'],
            **magspec_data['spec_hres_stats'],
            **objs
        }

        return magspec_data

    @staticmethod
    def objective_analysis(setpoints: np.ndarray, magspec_data: dict[str, dict[str, np.ndarray]],
                           dE_weight: float = 1., pC_weight: float = 2., MeV_weight: float = 1.) -> dict[str, Any]:
        objs = {}
        obj_1 = magspec_data['spec_hres_stats']['med_dE/E']
        objs['dE/E obj'] = np.min(obj_1) / obj_1
        obj_2 = magspec_data['spec_hres_stats']['med_peak_smooth_pC/MeV']
        objs['pC/MeV obj'] = obj_2 / np.max(obj_2)
        obj_3 = magspec_data['spec_hres_stats']['med_peak_smooth_MeV']
        objs['100 MeV obj'] = 1 - np.abs(obj_3 / 100. - 1)**2
        g_obj = (dE_weight * objs['dE/E obj'] +
                 pC_weight * objs['pC/MeV obj'] +
                 MeV_weight * objs['100 MeV obj']) / (dE_weight + pC_weight + MeV_weight)
        g_obj_pars = np.polyfit(setpoints, g_obj, round(setpoints.size / 2))
        g_obj_fit_x = np.linspace(setpoints[0], setpoints[-1], 1000)
        g_obj_fit_y = np.polyval(g_obj_pars, g_obj_fit_x)
        g_obj_x = g_obj_fit_x[np.argmax(g_obj_fit_y)]
        objs['global obj'] = {'src_raw': ['med_dE/E', 'med_peak_smooth_pC/MeV', 'med_peak_smooth_MeV'],
                              'src_norm': ['dE/E obj', 'pC/MeV obj', '100 MeV obj'],
                              'weights': [2., 3., 1.],
                              'setpoints': setpoints,
                              'values': g_obj,
                              'fit pars': g_obj_pars,
                              'fit x': g_obj_fit_x,
                              'fit y': g_obj_fit_y,
                              'best': g_obj_x
                              }
        return objs


if __name__ == "__main__":
    _htu = HtuExp(get_info=True)
    _scan_path = Path(_htu.base_path / r'Undulator\Y2023\07-Jul\23_0706\scans\Scan004')

    lpa = LPA(_htu.is_offline)
    _analysis = lpa.z_scan_analysis(_htu, _scan_path)

    for objective in ['dE/E obj', 'pC/MeV obj', '100 MeV obj']:
        plt.figure(figsize=(6.4, 4.8))
        plt.plot(_analysis['setpoints'], _analysis[objective])
        plt.title(objective)

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(_analysis['setpoints'], _analysis['global obj']['values'])
    plt.plot(_analysis['global obj']['fit x'], _analysis['global obj']['fit y'])
    plt.title(f'global objective {_analysis["global obj"]["best"]:.3f}')

    plt.show(block=True)

    # _scan_data = ScanData(_scan_path, ignore_experiment_name=_htu.is_offline)
    # _indexes, _setpoints, _matching = _scan_data.bin_data('U_ESP_JetXYZ', 'Position.Axis 3')
    # _magspec_data = _scan_data.analyze_mag_spec(_indexes)
    #
    # spec = 'hres'
    # plt.figure()
    # ax = plt.subplot(111)
    # im = ax.imshow(_magspec_data['spec_hres_pC']['avg'], aspect='auto', origin='upper')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size=0.2, pad=0.1)
    # plt.colorbar(im, cax=cax)
    # ax.set_title('mean')

    if not _htu.is_offline:
        lpa.close()
    print('done')
