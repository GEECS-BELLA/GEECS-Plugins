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

        magspec_data = {
            'setpoints': setpoints,
            'indexes': indexes,
            'axis_MeV': magspec_data['axis_MeV']['hres'],
            **magspec_data['spec_hres_pC/MeV'],
            **magspec_data['spec_hres_stats']
        }

        # obj_1 = magspec_data['spec_hres_stats']['med_dE/E']
        # analysis['dE/E objective'] = np.min(obj_1) / obj_1
        # obj_2 = magspec_data['spec_hres_stats']['med_peak_fit_pC']
        # analysis['pC/MeV objective'] = obj_2 / np.max(obj_2)
        # obj_3 = magspec_data['spec_hres_stats']['med_peak_fit_MeV']
        # analysis['100 MeV objective'] = 1 - np.abs(obj_3 / 100. - 1)

        return magspec_data


if __name__ == "__main__":
    _htu = HtuExp(get_info=True)
    _scan_path = Path(_htu.base_path / r'Undulator\Y2023\07-Jul\23_0706\scans\Scan004')

    lpa = LPA(_htu.is_offline)
    _analysis = lpa.z_scan_analysis(_htu, _scan_path)

    for objective in ['dE/E objective', 'pC/MeV objective', '100 MeV objective']:
        plt.figure(figsize=(6.4, 4.8))
        plt.plot(_analysis['setpoints'], _analysis[objective])
        plt.title(objective)

    global_objective: np.ndarray = \
        (2 * _analysis['dE/E objective'] +
         3 * _analysis['pC/MeV objective'] +
         1 * _analysis['100 MeV objective']) / 6

    _fit_x = np.linspace(_analysis['setpoints'][0], _analysis['setpoints'][-1], 100)
    _fit_pars = np.polyfit(_analysis['setpoints'], global_objective, round(global_objective.size / 2.))
    _fit_y = np.polyval(_fit_pars, _fit_x)

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(_analysis['setpoints'], global_objective)
    plt.plot(_fit_x, _fit_y)
    plt.title('Global Objective')

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

    lpa.close()
    print('done')
