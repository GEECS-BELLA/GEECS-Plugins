from typing import Optional, Any, Union
from pathlib import Path
from labview_interface.lv_interface import Bridge
from geecs_python_api.controls.experiment.experiment import Experiment
from geecs_python_api.controls.devices.HTU.gas_jet import GasJet
from geecs_python_api.controls.devices.HTU.laser import Laser
from geecs_python_api.controls.devices.HTU.transport import Steering
from geecs_python_api.analysis.images.scans.scan_data import ScanData


class UserInterface:
    @staticmethod
    def report(message: str):
        Bridge.labview_call('ui', 'report', [], sync=False, message=message)


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
    def __init__(self):
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
            pars = [6, 11, 0.25, 10] if rough else [z - 1, z + 1, 0.1, 10]
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

    @staticmethod
    def z_scan_analysis(exp: Experiment, scan_path: Path):
        scan_data = ScanData(scan_path, ignore_experiment_name=exp.is_offline)
        return
