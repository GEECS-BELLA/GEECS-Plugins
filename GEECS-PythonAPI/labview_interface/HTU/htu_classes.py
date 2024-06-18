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
from geecs_python_api.analysis.scans.scan_data import ScanData


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
    def question(message: str, possible_answers: list, modal: bool = True) -> Optional[Any]:
        success, answer = Bridge.labview_call('handler', 'question', ['answer'], sync=True, timeout_sec=900.,
                                              message=message, possible_answers=possible_answers, modal=modal)
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
            self.jet = GasJet()
            self.laser = Laser()
            UserInterface.report('System connected')

    def close(self):
        self.jet.close()
        self.laser.close()
        UserInterface.report('System disconnected')

    def manage_scan(self, exp: Experiment, device: Union[GeecsDevice, str], var_name: str,
                    min_max_step_steps: tuple, units: str, precision: int, label: str, rough: bool, call: str,
                    dE_weight: float = 1., pC_weight: float = 1., MeV_weight: float = 1.) -> Optional[float]:
        if isinstance(device, GeecsDevice):
            cancel, scan_folder = self.dev_scan(device, var_name, min_max_step_steps, units, label, rough)
        else:
            cancel = False
            scan_folder = Path(exp.base_path / r'Undulator\Y2023\07-Jul\23_0706\scans\Scan004')

        if cancel:
            UserInterface.report('LPA initialization canceled by user')
            return None
        else:
            UserInterface.report(rf'Done ({scan_folder.name})')

        UserInterface.report('Running analysis...')
        dev_name = device.get_name() if isinstance(device, GeecsDevice) else device
        results = self.dev_scan_analysis(dev_name, var_name, exp, scan_folder, dE_weight, pC_weight, MeV_weight)
        results['precision'] = np.array([precision])

        recommended: float
        if 'weight-based objs' in results:
            results['weight-based objs']['global obj']['precision'] = np.array([precision])
            recommended = results['weight-based objs']['global obj']['best']
        else:
            results['fit-based objs']['global obj']['precision'] = np.array([precision])
            recommended = results['fit-based objs']['global obj']['best']

        UserInterface.clear_plots(call)
        Handler.send_results(f'{label.lower()}-scan', flatten_dict(results))
        UserInterface.report(f'Done. Recommended: {label} = {recommended:.{precision}f} mm')

        return recommended

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
        indexes, setpoints, matching = scan_data.group_shots_by_step(device, variable)
        magspec_data = scan_data.analyze_mag_spec(indexes)
        if magspec_data:
            objs = self.objective_analysis(setpoints, magspec_data, dE_weight, pC_weight, MeV_weight)

            magspec_data = {
                'setpoints': setpoints,
                'indexes': indexes,
                'axis_MeV': magspec_data['axis_MeV'],
                **magspec_data['spec_hres_pC/MeV'],
                **magspec_data['spec_hres_stats'],
                **objs
            }

        else:
            magspec_data = {'setpoints': setpoints, 'indexes': indexes}

        return magspec_data

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
            obj_vals = np.zeros((3, setpoints.size))

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
                    objs[obj_type][f'{src.split("_")[-1]} obj'] = {'data': np.stack([setpoints, obj]),
                                                                   'fit': np.stack([obj_fit_x, obj_fit])}
                    obj_vals[it] = obj
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
                'source vals': obj_vals,
                'weights': weights,
                'setpoints': setpoints,
                'data': np.stack([setpoints, g_obj]),
                'fit': np.stack([obj_fit_x, g_obj_fit]),
                'best': g_obj_best
            }

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
    # _indexes, _setpoints, _matching = _scan_data.group_shots_by_scan_parameter('U_ESP_JetXYZ', 'Position.Axis 3')
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
