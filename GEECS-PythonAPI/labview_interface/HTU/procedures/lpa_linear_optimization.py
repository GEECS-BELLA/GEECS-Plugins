from typing import Optional
from geecs_python_api.controls.experiment.experiment import Experiment
from labview_interface.HTU.htu_classes import UserInterface, Handler, LPA
from labview_interface.lv_interface import Bridge


def optimize_lpa(exp: Experiment, call: list):
    lpa: Optional[LPA] = None

    try:
        if Handler.question('Are you ready to run an LPA initialization?', ['Yes', 'No']) == 'No':
            return

        lpa = LPA(exp.is_offline)
        min_max_step_steps = ()
        weights = [1., 6., 2.]

        for label, rough in zip(['Z', 'X', 'Compressor', 'Pressure'],
                                [True, False, True, False]):
            run_scan = Handler.question(f'Next scan: rough {label if len(label) == 1 else label.lower()}'
                                        f'-scan. Proceed?', ['Yes', 'Skip', 'Cancel'], modal=False)

            if run_scan == 'Cancel':
                UserInterface.report('LPA initialization canceled by user')
                return

            if label == 'Compressor':
                if exp.is_offline:
                    return

                precision = 1
                # device = lpa.laser.compressor
                device = lpa.laser.compressor.get_name()
                var_name = lpa.laser.compressor.var_separation
                min_max_step_steps = (42500., 43000., 50., 10) if rough else (42700., 42900., 20., 20)

            elif label == 'Pressure':
                if exp.is_offline:
                    return

                precision = 2
                # device = lpa.jet.pressure
                device = lpa.jet.pressure.get_name()
                var_name = lpa.jet.pressure.var_pressure
                min_max_step_steps = (1.8, 2.8, 0.1, 10) if rough else (2., 2.6, 0.05, 20)

            elif label in ['X', 'Y', 'Z']:
                precision = 3
                default_pos = [6.5, 0, 8.5]
                axis = ord(label.upper()) - ord('X')

                if exp.is_offline:
                    device = 'U_ESP_JetXYZ'
                    var_name = f'Position.Axis {axis + 1}'
                    pos = default_pos[axis]

                else:
                    # device = lpa.jet.stage
                    device = lpa.jet.stage.get_name()
                    var_name = lpa.jet.stage.get_axis_var_name(axis)
                    # pos = round(lpa.jet.stage.get_position(label), precision)
                    pos = default_pos[axis]

                if label == 'X':
                    min_max_step_steps = (5, 8, 0.4, 20) if rough else (5.5, 7.5, 0.1, 20)

                if label == 'Y':
                    min_max_step_steps = (-7.5, -8.5, 0.05, 10) if rough else (-7.8, -8.4, 0.025, 20)

                if label == 'Z':
                    min_max_step_steps = (6, 11, 0.25, 20) if rough else (pos - 1, pos + 1, 0.1, 20)

            else:
                return

            if run_scan == 'Yes':
                target: float = lpa.manage_scan(exp, device, var_name, min_max_step_steps,
                                                units='mm', precision=precision, label=label, rough=rough, call=call[0],
                                                dE_weight=weights[0], pC_weight=weights[1], MeV_weight=weights[2])
                if target is None:
                    return

                answer = Handler.question(f'Proceed the recommended {label}-position ({target:.{precision}f} mm)?',
                                          ['Yes', 'No'], modal=False)
                if answer == 'Yes':
                    lpa.jet.stage.set_position(label, round(target, 3))

    except Exception as ex:
        UserInterface.report('LPA initialization failed')
        Bridge.python_error(message=str(ex))

    finally:
        if isinstance(lpa, LPA):
            lpa.close()
