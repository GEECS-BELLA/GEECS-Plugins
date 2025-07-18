"""
Uses the logic found within the `return_value` function of `action_manager.py` within
"GEECS-Scanner-GUI/geecs_scanner/data_acquisition/action_manager.py"

This slimmed-down version only imports what is necessary
"""

from geecs_python_api.controls.devices.scan_device import ScanDevice
from time import time

instantiated_devices: dict[str, ScanDevice] = {}  # Some dict to store devices that are already initialized


def return_value(device_name: str, variable: str):
    """
    Get the current value of a device variable

    Args:
        device_name (str): The device to query.
        variable (str): The variable to get the value of.
    """

    if device_name not in instantiated_devices:
        instantiated_devices[device_name] = ScanDevice(device_name)

    device: ScanDevice = instantiated_devices[device_name]
    return device.get(variable)


def show_info(device_name: str, variable: str):
    print(f"Fetching value of {device_name}:{variable}...")
    start = time()
    value = return_value(device_name=device_name, variable=variable)
    print(f"... {value}  ({time()-start:.3f} s)")
    print()


if __name__ == "__main__":
    show_info(device_name="U_EMQTripletBipolar", variable="Current_Limit.Ch1")
    show_info(device_name="U_EMQTripletBipolar", variable="Current_Limit.Ch2")
    show_info(device_name="U_EMQTripletBipolar", variable="Current_Limit.Ch3")
    show_info(device_name="U_ChicaneInner", variable="Current")
    show_info(device_name="U_ChicaneOuter", variable="Current")
