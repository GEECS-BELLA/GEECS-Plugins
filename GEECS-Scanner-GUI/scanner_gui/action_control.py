"""
Contains a simpler version of a full "RunControl" that is only capable of performing actions using ActionManager

-Chris
"""
from geecs_python_api.controls.data_acquisition.data_acquisition import ActionManager


class ActionControl:
    def __init__(self, experiment_name: str):
        self.action_manager = ActionManager(experiment_dir=experiment_name)

    def perform_action(self, action_list: dict):
        if 'steps' in action_list and len(action_list['steps']) > 0:
            self.action_manager.add_action({'closeout_action': action_list})
            self.action_manager.execute_action('closeout_action')

        # TODO any way to close the TCP connection to the devices we changed?
