"""
Contains a simpler version of a full "RunControl" that is only capable of performing actions using ActionManager

-Chris
"""
from geecs_scanner.data_acquisition import ActionManager


class ActionControl:
    def __init__(self, experiment_name: str):
        """
        Opens a direct link to the Action Manager within `data_acquisition`

        :param experiment_name: Name of the experiment, used to find the right config files and connect to devices
        """
        self.action_manager = ActionManager(experiment_dir=experiment_name)

    def perform_action(self, action_list: dict):
        """
        Performs the given list of actions using the currently-connected instance of Action Manager

        :param action_list: dict containing steps of the action
        """
        name = 'test_action'
        if 'steps' in action_list and len(action_list['steps']) > 0:
            self.action_manager.add_action({name: action_list})
            self.action_manager.execute_action(name)
            self.action_manager.clear_action(name)
