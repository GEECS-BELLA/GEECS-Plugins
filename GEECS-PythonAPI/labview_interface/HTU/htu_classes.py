import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional, Any, Union
from pathlib import Path
from labview_interface.lv_interface import Bridge, flatten_dict
from geecs_python_api.controls.experiment.experiment import Experiment
from geecs_python_api.controls.experiment.htu import HtuExp
from geecs_python_api.controls.devices.HTU.gas_jet import GasJet
from geecs_python_api.controls.devices.HTU.laser import Laser
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
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

    def manage_scan(self, exp: Experiment, device: Union[GeecsDevice, str], var_name: str,
                    min_max_step_steps: tuple, units: str, precision: int, label: str, rough: bool, call: str,
                    dE_weight: float = 1., pC_weight: float = 1., MeV_weight: float = 1.):
        cancel_msg = 'LPA initialization canceled'
        run_scan = Handler.question(f'Next scan: rough {label}-scan. Proceed?', ['Yes', 'Skip', 'Cancel'])

        if run_scan == 'Cancel':
            UserInterface.report(cancel_msg)
            return

        if run_scan == 'Yes':
            if isinstance(device, GeecsDevice):
                cancel, scan_folder = self.dev_scan(device, var_name, min_max_step_steps, units, label, rough)
            else:
                cancel = False
                scan_folder = Path(exp.base_path / r'Undulator\Y2023\07-Jul\23_0706\scans\Scan004')
            if cancel:
                UserInterface.report(cancel_msg)
                return
            else:
                UserInterface.report(rf'Done ({scan_folder.name})')

            UserInterface.report('Running analysis...')
            dev_name = device.get_name() if isinstance(device, GeecsDevice) else device
            results = self.dev_scan_analysis(dev_name, var_name, exp, scan_folder, dE_weight, pC_weight, MeV_weight)
            results['global obj']['precision'] = np.array([precision])
            UserInterface.clear_plots(call)
            Handler.send_results(f'{label.lower()}-scan', flatten_dict(results))

            recommended = results['global obj']['best']
            answer = Handler.question(f'Proceed the recommended {label}-position ({recommended:.{precision}f} mm)?',
                                      ['Yes', 'No'])
            if answer == 'Yes':
                self.jet.stage.set_position('Z', round(recommended, 3))

    @staticmethod
    def dev_scan(device: GeecsDevice, var_name: str, min_max_step_steps: tuple,
                 units: str, label: str, rough: bool) -> tuple[bool, Optional[Path]]:
        var_alias = device.find_alias_by_var(var_name)
        if var_alias in device.var_spans:
            lims = device.var_spans[var_alias]
        else:
            lims = (None, None)
        values: Union[dict, str]

        while True:
            values = Handler.request_values(f'{"Rough" if rough else "Fine"} {label}-scan parameters:',
                                            [(f'Start [{units}]', 'float', lims[0], lims[1], min_max_step_steps[0]),
                                             (f'End [{units}]', 'float', lims[0], lims[1], min_max_step_steps[1]),
                                             (f'Steps [{units}]', 'float', 0, 10, min_max_step_steps[2]),
                                             ('Shots/Step', 'int', 1, 'inf', min_max_step_steps[3])])
            if isinstance(values, str) and (values == 'Cancel'):
                cancel = True
                scan_folder = None
                break

            UserInterface.report(f'Starting {"rough" if rough else "fine"} {label}-scan...')
            scan_folder, _, _, _ = device.scan(var_alias, values[f'Start [{units}]'], values[f'End [{units}]'],
                                               values[f'Steps [{units}]'], lims, values['Shots/Step'],
                                               comment=f'{"rough" if rough else "fine"} Z-scan')

            repeat = Handler.question(f'Do you want to repeat this {label}-scan?', ['Yes', 'No', 'Cancel'])
            cancel = (repeat == 'Cancel')
            if repeat != 'Yes':
                break

        return cancel, scan_folder

    def dev_scan_analysis(self, device: str, variable: str, exp: Experiment, scan_path: Path,
                          dE_weight: float = 1., pC_weight: float = 1., MeV_weight: float = 1.)\
            -> dict[str, np.ndarray]:
        scan_data = ScanData(scan_path, ignore_experiment_name=exp.is_offline)
        indexes, setpoints, matching = scan_data.bin_data(device, variable)
        magspec_data = scan_data.analyze_mag_spec(indexes)
        if magspec_data:
            objs = self.objective_analysis(setpoints, magspec_data, dE_weight, pC_weight, MeV_weight)

            magspec_data = {
                'setpoints': setpoints,
                'indexes': indexes,
                'axis_MeV': magspec_data['axis_MeV'],
                **magspec_data['spec_hres_pC/MeV'],
                **magspec_data['spec_hres_stats'],
                **objs }

        else:
            magspec_data = { 'setpoints': setpoints, 'indexes': indexes }

        return magspec_data

    def jet_scan(self, axis: int, rough: bool = False) -> tuple[bool, Optional[Path]]:
        pos = round(self.jet.stage.get_position(axis), 2)
        axis_alias = self.jet.stage.get_axis_var_alias(axis)
        lims = self.jet.stage.var_spans[axis_alias]
        values: Union[dict, str]

        if axis == 0:  # X-axis
            pars = [5, 8, 0.4, 20] if rough else [5.5, 7.5, 0.1, 20]
            lbl = 'X'
        elif axis == 1:  # Y-axis
            pars = [-8, -8.5, 0.05, 20] if rough else [-8.2, -8.5, 0.015, 20]
            lbl = 'Y'
        elif axis == 2:  # Z-axis
            pars = [6, 11, 0.25, 20] if rough else [pos - 1, pos + 1, 0.1, 20]
            lbl = 'Z'
        else:
            return True, None

        while True:
            values = Handler.request_values(f'{"Rough" if rough else "Fine"} {lbl}-scan parameters:',
                                            [('Start [mm]', 'float', lims[0], lims[1], pars[0]),
                                             ('End [mm]', 'float', lims[0], lims[1], pars[1]),
                                             ('Steps [mm]', 'float', 0, 10, pars[2]),
                                             ('Shots/Step', 'int', 1, 'inf', pars[3])])
            if isinstance(values, str) and (values == 'Cancel'):
                cancel = True
                scan_folder = None
                break

            UserInterface.report(f'Starting {"rough" if rough else "fine"} {lbl}-scan...')
            scan_folder, _, _, _ = self.jet.stage.scan(axis_alias, values['Start [mm]'], values['End [mm]'],
                                                       values['Steps [mm]'], lims, values['Shots/Step'],
                                                       comment=f'{"rough" if rough else "fine"} Z-scan')

            repeat = Handler.question(f'Do you want to repeat this {lbl}-scan?', ['Yes', 'No', 'Cancel'])
            cancel = (repeat == 'Cancel')
            if repeat != 'Yes':
                break

        return cancel, scan_folder

    @staticmethod
    def objective_analysis(setpoints: np.ndarray, magspec_data: dict[str, dict[str, np.ndarray]],
                           dE_weight: float = 1., pC_weight: float = 1., MeV_weight: float = 1.) -> dict[str, Any]:
        def norm(dist: np.ndarray, inv: bool = False) -> np.ndarray:
            if inv:
                dist[np.where(dist == 0)[0]] = np.max(dist)
            dist -= np.min(dist)
            dist /= np.max(dist)
            if inv:
                dist = 1 - dist
            return dist

        objs_types = ['weight-based objs', 'fit-based objs']
        objs = {objs_types[0]: {}, objs_types[1]: {}}
        obj_fit_x = np.linspace(setpoints[0], setpoints[-1], 1000)
        weights = [dE_weight, pC_weight, MeV_weight]

        for obj_type, sources in zip(objs_types,
                                     [['med_weighted_dE/E', 'med_peak_charge_pC/MeV', 'med_peak_charge_MeV'],
                                      ['med_fit_dE/E', 'med_peak_smooth_pC/MeV', 'med_peak_smooth_MeV']]):
            g_obj = np.zeros((setpoints.size,), dtype=float)

            for it, (src, weight) in enumerate(zip(sources, weights)):
                if (magspec_data['spec_hres_stats'][src] != 0).any():
                    obj = magspec_data['spec_hres_stats'][src]
                    if it == 0:
                        obj = norm(obj, inv=True)
                    if it == 1:
                        obj = norm(obj)
                    if it == 2:
                        # obj = norm(np.abs(obj - 100), inv=True)
                        obj = norm(np.sqrt(np.abs(obj - 100)), inv=True)

                    obj_fit = np.polyval(np.polyfit(setpoints, obj, round(setpoints.size / 2)), obj_fit_x)
                    objs[obj_type][f'{src.split("_")[-1]} obj'] = { 'data': np.stack([setpoints, obj]),
                                                                    'fit': np.stack([obj_fit_x, obj_fit]) }
                    g_obj += (weight * obj)

            g_obj /= np.sum(weights)

            try:
                g_obj_fit = np.polyval(np.polyfit(setpoints, g_obj, round(setpoints.size / 2)), obj_fit_x)
                g_obj_best = obj_fit_x[np.argmax(g_obj_fit)]
            except Exception:
                g_obj_fit = np.zeros(obj_fit_x.shape, dtype=float)
                g_obj_best = obj_fit_x[obj_fit_x.size // 2]

            objs[obj_type]['global obj'] = {
                'source raw': sources,
                'source norm': [f'{src.split("_")[-1]} obj' for src in sources],
                'weights': weights,
                'setpoints': setpoints,
                'data': np.stack([setpoints, g_obj]),
                'fit': np.stack([obj_fit_x, g_obj_fit]),
                'best': g_obj_best }

        return objs


if __name__ == "__main__":
    _htu = HtuExp(get_info=True)
    # _scan_path = Path(_htu.base_path / r'Undulator\Y2023\07-Jul\23_0706\scans\Scan004')
    _scan_path = Path(_htu.base_path / r'Undulator\Y2023\07-Jul\23_0706\scans\Scan004')

    lpa = LPA(_htu.is_offline)
    if lpa.jet is None:
        _device = 'U_ESP_JetXYZ'
        _variable = 'Position.Axis 3'
    else:
        _device = lpa.jet.stage.get_name()
        _variable = lpa.jet.stage.get_axis_var_name(2)

    _analysis = lpa.dev_scan_analysis(_device, _variable, _htu, _scan_path)

    for obj_t in ['weight-based objs', 'fit-based objs']:
        if obj_t in _analysis:
            for objective in ['dE/E obj', 'pC/MeV obj', 'MeV obj']:
                plt.figure(figsize=(6.4, 4.8))
                plt.plot(_analysis[obj_t][objective]['data'][0, :],
                         _analysis[obj_t][objective]['data'][1, :])
                plt.plot(_analysis[obj_t][objective]['fit'][0, :],
                         _analysis[obj_t][objective]['fit'][1, :])
                plt.title(objective)

            plt.figure(figsize=(6.4, 4.8))
            plt.plot(_analysis[obj_t]['global obj']['data'][0, :],
                     _analysis[obj_t]['global obj']['data'][1, :])
            plt.plot(_analysis[obj_t]['global obj']['fit'][0, :],
                     _analysis[obj_t]['global obj']['fit'][1, :])
            plt.title(f'global objective ({obj_t.split(" objs")[0]}) {_analysis[obj_t]["global obj"]["best"]:.3f}')

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
