"""
While `get_value_demonstration` uses UDP commands to ensure an accurate result, TCP subscriptions are much faster

This is a slimmed-down version that follows the logic contained within DeviceManager of GEECS-Scanner-GUI.  The current
implementation returns a multi-leveled dictionary from `get_all_subscriptions`, which can be seen when printed using the
`print_device_variable_values` method.  Can add more device-variables to the __main__ example below.
"""

from geecs_python_api.controls.devices.scan_device import ScanDevice
from geecs_python_api.controls.interface.geecs_errors import GeecsDeviceInstantiationError

from time import time, sleep
from typing import Any


class TCPSubscriptionHandler:
    """
    Stores the list of currently-subscribed devices and holds methods for subscribing, viewing, and unsubscribing
    """
    def __init__(self):
        self.subscribed_devices: list[ScanDevice] = []

    def subscribe_to_device_variables(self, device_variables: list[str], clear_devices: bool = False):
        """
        Subscribes all given device variables by parsing out and sorting the variables to each device.
        NOTE:  will not add subscription variables to an already-subscribed device.  Will need to clear devices first

        Args:
            device_variables (list[str]): List of all
            clear_devices (bool): Optional, if True will call `unsubscribe_all` before adding new subscriptions
        """

        if clear_devices:
            self.unsubscribe_all()

        device_map = self.preprocess_observables(device_variables)

        for device_name, variable_list in device_map.items():
            if device_name not in self.subscribed_devices:
                self._subscribe_device(device_name, variable_list)

    def _subscribe_device(self, device_name: str, variable_list: list[str]):
        """
        Subscribes a single device to a list of variables

        Args:
            device_name (str or dict): The name of the device to subscribe to, or a dict of info for composite var
            variable_list (list): A list of variables to subscribe to for the device.

        Raises:
            GeecsDeviceInstantiationError:  if an error occurs while setting up TCP subscription
        """

        try:
            device = ScanDevice(device_name)
            device.use_alias_in_TCP_subscription = False
            device.subscribe_var_values(variable_list)
            self.subscribed_devices.append(device)

        except GeecsDeviceInstantiationError as e:
            print(f"Failed to instantiate Geecs device {device_name}: {e}")
            self.unsubscribe_all()
            raise

    def get_all_subscriptions(self) -> dict[str, dict[str, str]]:
        """
        Gets the most-recent TCP message for each device.
        NOTE: May not include subscribed variables if used IMMEDIATELY after the subscription process.  Need to wait 1s

        Returns:
            dict:  all device subscriptions compiled into a multi-layered dict.
        """
        tcp_messages: dict[str, Any] = {}
        for device in self.subscribed_devices:
            tcp_messages[device.get_name()] = device.state
        return tcp_messages

    def print_device_variable_values(self, tcp_messages: dict[str, dict[str, str]] = None):
        """
        Utility function to print the TCP state of each device in an organized way

        Args:
            tcp_messages (dict): Optional: use the output of `get_all_subscriptions()`, will call this if not given
        """
        if tcp_messages is None:
            tcp_messages = self.get_all_subscriptions()

        for device_name, state in tcp_messages.items():
            print()
            print(f"{device_name}: ")
            for var_name, var_value in state.items():
                print(f"  {var_name}: {var_value}")

    def unsubscribe_all(self):
        """
        Unsubscribes and closes the connection to all subscribed devices
        """
        for device in self.subscribed_devices:
            device.unsubscribe_var_values()
            device.close()

    @staticmethod
    def preprocess_observables(observables: list[str]) -> dict[str, list[str]]:
        """
        Preprocess a list of observables by organizing them into device-variable mappings.  This is a copy from
        `DeviceManager` in "geecs_scanner/data_acquisition/device_manager.py",

        Args:
            observables (list): A list of device-variable observables, e.g. [Dev1:Var1, Dev1:var2, Dev2:var1]

        Returns:
            dict: A dictionary mapping device names to a list of their variables.
        """

        device_map = {}
        for observable in observables:
            device_name, var_name = observable.split(':')
            if device_name not in device_map:
                device_map[device_name] = []
            device_map[device_name].append(var_name)
        return device_map


if __name__ == "__main__":
    device_variable_list: list[str] = [
        "U_EMQTripletBipolar:Current_Limit.Ch1",
        "U_EMQTripletBipolar:Current_Limit.Ch2",
        "U_EMQTripletBipolar:Current_Limit.Ch3",
        "U_ChicaneInner:Current",
        "U_ChicaneOuter:Current",
    ]

    print("Initializing all TCP subscriptions:")
    start = time()
    tcp_manager = TCPSubscriptionHandler()
    tcp_manager.subscribe_to_device_variables(device_variables=device_variable_list)
    print(f"... done!  ({time()-start:.3f} s)")
    print()

    print("Checking state of all TCP subscriptions:")
    start = time()
    message = tcp_manager.get_all_subscriptions()
    print(f"... done!  ({time()-start:.3f} s)")
    tcp_manager.print_device_variable_values(message)
    print()

    print("Sleeping 1 second...")
    sleep(1)

    print("Checking state of all TCP subscriptions:")
    start = time()
    message = tcp_manager.get_all_subscriptions()
    print(f"... done!  ({time()-start:.3f} s)")
    tcp_manager.print_device_variable_values(message)
    print()

    print("Closing all TCP subscriptions:")
    start = time()
    tcp_manager.unsubscribe_all()
    print(f"... done!  ({time() - start:.3f} s)")
