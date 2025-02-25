from typing import Any, Union, NamedTuple


class WaitAction(NamedTuple):
    time: float
    type: str = 'wait'


class GetAction(NamedTuple):
    device: str
    variable: str
    expected_value: Any
    type: str = 'get'


class SetAction(NamedTuple):
    device: str
    variable: str
    value: Any
    type: str = 'set'


class ExecuteAction(NamedTuple):
    name: str
    type: str = 'execute'


ActionStep = Union[WaitAction, GetAction, SetAction, ExecuteAction]


def get_dict_for_saving(action_list: list[ActionStep]) -> dict[str, list]:
    return_list: list[dict] = []
    for step in action_list:
        if step.type == 'wait':
            return_list.append({'wait': step.time})
        elif step.type == 'execute':
            return_list.append({'action': 'execute', 'action_name': step.name})
        elif step.type == 'set':
            return_list.append({'action': 'set', 'device': step.device,
                                'variable': step.variable, 'value': step.value})
        elif step.type == 'get':
            return_list.append({'action': 'get', 'device': step.device,
                                'variable': step.variable, 'expected_value': step.expected_value})
        else:
            raise KeyError(f"Unknown action type '{step.type}'")
    return {'steps': return_list}


def get_action_list_from_steps(step_dict: dict[str, list]) -> list[ActionStep]:
    """
    Used when loading a single list of action steps

    :param step_dict:
    :return:
    """
    action_step_list: list[ActionStep] = []
    for step in step_dict['steps']:
        if 'wait' in step:
            action_step_list.append(WaitAction(time=step['wait']))
        elif step['action'] == 'set':
            action_step_list.append(SetAction(device=step['device'], variable=step['variable'], value=step['value']))
        elif step['action'] == 'get':
            action_step_list.append(GetAction(device=step['device'], variable=step['variable'],
                                              expected_value=step['expected_value']))
        elif step['action'] == 'execute':
            action_step_list.append(ExecuteAction(name=step['action_name']))
        else:
            raise KeyError(f"Unknown action type in step '{step}'")
    return action_step_list


def get_all_action_lists(action_list_dict: dict[str, dict[str, dict]]) -> dict[str, list[ActionStep]]:
    """
    Used when loading an actions.yaml file with multiple named actions

    :param action_list_dict:
    :return:
    """
    action_dict: dict[str, list[ActionStep]] = {}
    for named_action in action_list_dict['actions'].keys():
        action_dict[named_action] = get_action_list_from_steps(action_list_dict['actions'][named_action])
    return action_dict



if __name__ == '__main__':
    """
    test_array: list[ActionStep] = [SetAction(device='device_1', variable='var_1', value='on'),
                                    WaitAction(time=1),
                                    GetAction(device='device_2', variable='var_2', expected_value='on'),
                                    ExecuteAction(name='action_name')]

    for action in test_array:
        print(f"{action.type}: {action}")
    """

    import yaml
    with open("C:\\Users\\loasis.LOASIS\\Documents\\GitHub\\GEECS-Plugins\\GEECS-Scanner-GUI\\scanner_configs\\experiments\\Undulator\\action_library\\actions.yaml") as file:
        data = yaml.safe_load(file)
    print(data)
    print("--")
    print(get_all_action_lists(data))
