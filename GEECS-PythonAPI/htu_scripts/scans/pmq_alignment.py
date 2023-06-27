import yaml
import pandas as pd
from geecs_api.devices.HTU.transport.transport_hexapod import TransportHexapod


def pmq_alignment(hexa: TransportHexapod, opt_method: str = 'bayes', norm: bool = True):
    # define a dict containing the specific variables and bounds to be used in optimization
    objs = {
        name_alias[0][0]: {
            'Geecs_Object': hexa,
            'variable': name_alias[0],
            'bounds': list(hexa.get_variables()[name_alias[1]])}
        for name_alias in hexa.var_names_by_index.values()}  # 'x', 'y', 'z', 'u', 'v', 'w'

    def normalize_var(obj, val):
        span = objs[obj]['bounds'][1] - objs[obj]['bounds'][0]
        offset = objs[obj]['bounds'][0]
        return 2 * (val - offset) / span - 1.

    # current position as initial values
    initial_point = {name_alias[0][0]: hexa.get_position(it) if it < 3 else hexa.get_angle(it - 3)
                     for it, name_alias in enumerate(hexa.var_names_by_index.values())}

    # define the xopt configuration
    YAML = """
    xopt:
        dump_file: dump.yaml
    generator:
        name:
    evaluator:
        function: __main__.geecs_measurement
    vocs:
        variables: {}
        objectives: {f: "MAXIMIZE"}
    """

    yaml_output = yaml.safe_load(YAML)

    if norm:
        for tag in objs.keys():
            yaml_output['vocs']['variables'][tag] = [-1., 1.]
    else:
        for tag in objs.keys():
            yaml_output['vocs']['variables'][tag] = objs[tag]['bounds']

    # define the generator
    if opt_method == 'bayes':
        yaml_output['generator']['name'] = 'upper_confidence_bound'
        yaml_output['generator']['n_initial'] = 2
        yaml_output['generator']['acq'] = {'beta': 0.1}
        yaml_output['xopt']['dump_file'] = 'bayes.yaml'

    elif opt_method == 'nelder':
        yaml_output['generator']['name'] = 'neldermead'
        yaml_output['generator']['adaptive'] = True
        yaml_output['generator']['xatol'] = 0.01
        yaml_output['generator']['fatol'] = 0.005
        if norm:
            yaml_output['generator']['initial_point'] = {key: normalize_var(key, initial_point[key])
                                                         for key in yaml_output['vocs']['variables'].keys()}
        else:
            yaml_output['generator']['initial_point'] = initial_point
        yaml_output['xopt']['dump_file'] = 'nelder.yaml'

    print(yaml_output)


if __name__ == '__main__':
    hexapod = TransportHexapod()

    pmq_alignment(hexa=hexapod)

    hexapod.close()
