from typing import Union
from geecs_scanner.data_acquisition.schemas.actions import SetStep, GetStep, WaitStep, ExecuteStep, RunStep

def get_new_action(action: str) -> Union[None, dict]:
    """
    Generates a default dictionary for a given action keyword using the associated Pydantic model.

    :param action: action keyword
    :return: default dictionary for that action
    """
    if action == 'set':
        return SetStep(action='set', device='', variable='', value='').model_dump()
    elif action == 'get':
        return GetStep(action='get', device='', variable='', expected_value='').model_dump()
    elif action == 'wait':
        return WaitStep(action='wait', wait=1.0).model_dump()  # Default wait duration (you can choose your own)
    elif action == 'execute':
        return ExecuteStep(action='execute', action_name='').model_dump()
    elif action == 'run':
        return RunStep(action='run', file_name='', class_name='').model_dump()
    else:
        return None


# List of available actions, to be used by the completer for the add action line edit
list_of_actions = [
    'set',
    'get',
    'wait',
    'execute',
]


def generate_action_description(action: dict[str, list]) -> str:
    """ For each action type, generate a string that displays all the information for that action step """
    description = "???"
    if action.get("action") == "wait":
        description = f"wait {action.get('wait')}"
    elif action['action'] == 'execute':
        description = f"execute {action['action_name']}"
    elif action['action'] == 'set':
        description = f"{action['action']} {action['device']}:{action['variable']} {action.get('value')}"
    elif action['action'] == 'get':
        description = f"{action['action']} {action['device']}:{action['variable']} {action.get('expected_value')}"
    return description
