"""
Unit test to check that device manager is functioning in its simplest configuration.  These tests aren't an exhaustive
exploration of all the possible input parameters, might be some blind spots.

Will fail if the two devices below are not connected through MasterControl

-Chris
"""
import pytest
import yaml
from pathlib import Path
from geecs_scanner.data_acquisition import DeviceManager
from copy import deepcopy


TEST_DEVICE_NAME = "UC_TC_Phosphor"  # Update if not valid in your test setup
TEST_VARIABLE_NAME_1 = 'MaxCounts'
TEST_VARIABLE_NAME_2 = 'MeanCounts'

TEST_SCAN_DEVICE = "UC_DiagnosticsPhosphor"
TEST_SCAN_VARIABLE = "exposure"


def test_device_manager_functionality():
    # Initialize a device manager without an experiment name (otherwise tries to load configs from live location)
    device_manager = DeviceManager(experiment_dir=None)

    # Test loading the composite variables in a given yaml file
    config_root_path = Path(__file__).parents[1] / "test_configs" / "Test"
    composite_variable_path = config_root_path / "scan_devices" / "composite_variables.yaml"
    device_manager.load_composite_variables(composite_variable_path)
    assert len(list(device_manager.composite_variables.keys())) == 2

    # Test initializing devices from a yaml file
    test_noscan_path = config_root_path / "aux_configs" / "test_noscan.yaml"
    with open(test_noscan_path, 'r') as file:
        config = yaml.safe_load(file)
    device_manager.reinitialize(config_dictionary=config)  # Need to first load it, otherwise will try the 'live' folder
    assert device_manager.devices.keys() == config['Devices'].keys()

    # Test if device manager can determine no-scans and composite variables
    assert device_manager.is_statistic_noscan("noscan") is True
    assert device_manager.is_statistic_noscan("statistics") is True
    assert device_manager.is_statistic_noscan(TEST_DEVICE_NAME) is False
    assert device_manager.is_composite_variable("test_comp_1") is True
    assert device_manager.is_composite_variable(TEST_DEVICE_NAME) is False

    # Test the parsing function of list of strings into a dictionary
    test_observable = [f"{TEST_DEVICE_NAME}:{TEST_VARIABLE_NAME_1}",
                       f"{TEST_DEVICE_NAME}:{TEST_VARIABLE_NAME_2}"]
    test_device_map = device_manager.preprocess_observables(test_observable)
    assert TEST_DEVICE_NAME in test_device_map
    assert TEST_VARIABLE_NAME_1 in test_device_map[TEST_DEVICE_NAME]
    assert TEST_VARIABLE_NAME_2 in test_device_map[TEST_DEVICE_NAME]

    # Test adding scan variables to the async observables list
    initial_async_observables = deepcopy(device_manager.async_observables)
    device_manager.handle_scan_variables(config)  # Is a noscan so should do nothing
    assert device_manager.async_observables == initial_async_observables
    device_manager.add_scan_device(TEST_SCAN_DEVICE, [TEST_SCAN_VARIABLE])
    assert f"{TEST_SCAN_DEVICE}:{TEST_SCAN_VARIABLE}" in device_manager.async_observables

    # Clears any device subscriptions for a clean python exit
    device_manager.reset()


if __name__ == "__main__":
    pytest.main()
