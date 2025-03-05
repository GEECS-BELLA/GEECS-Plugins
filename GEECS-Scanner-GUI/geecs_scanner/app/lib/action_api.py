from typing import Union


def get_new_action(action) -> Union[None, dict[str, str]]:
    """
    Translates a given action keyword to a default dictionary for that action keyword

    :param action: action keyword
    :return: default dictionary for the associated action
    """
    default = None
    if action == 'set':
        default = {
            'action': 'set',
            'device': '',
            'variable': '',
            'value': ''
        }
    elif action == 'get':
        default = {
            'action': 'get',
            'device': '',
            'variable': '',
            'expected_value': ''
        }
    elif action == 'wait':
        default = {
            'wait': ''
        }
    elif action == 'execute':
        default = {
            'action': 'execute',
            'action_name': ''
        }
    elif action == 'run':
        default = {
            'action': 'run',
            'file_name': '',
            'class_name': ''
        }
    return default


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
    if action.get("wait") is not None:
        description = f"wait {action['wait']}"
    elif action['action'] == 'execute':
        description = f"execute {action['action_name']}"
    elif action['action'] == 'set':
        description = f"{action['action']} {action['device']}:{action['variable']} {action.get('value')}"
    elif action['action'] == 'get':
        description = f"{action['action']} {action['device']}:{action['variable']} {action.get('expected_value')}"
    return description
