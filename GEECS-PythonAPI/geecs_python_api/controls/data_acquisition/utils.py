import logging
from pathlib import Path
import shutil
import yaml

def get_full_config_path(experiment: str, config_type:str, config_file: str) -> Path:

    """
    Get the full path to a configuration file within an experiment directory.
    Raise an error if the experiment directory or file does not exist.
    
    Args:
        experiment (str): The name of the experiment subdirectory.
        config_file (str): The name of the configuration file.
    
    Returns:
        Path: Full path to the configuration file.
    """
    
    # Get the path of the current file (where this function is defined)
    current_dir = Path(__file__).parent

    # Set the base directory to be the 'configs' directory relative to the current directory
    base_dir = current_dir / 'configs' / 'experiments'
    
    # Ensure base_dir exists
    if not base_dir.exists():
        raise FileNotFoundError(f"The base config directory {base_dir} does not exist.")
    
    experiment_dir = base_dir / experiment
    
    if not experiment_dir.exists():
        raise FileNotFoundError(f"The experiment directory {experiment_dir} does not exist.")


    config_path = experiment_dir / config_type / config_file
    if not config_path.exists():
        raise FileNotFoundError(f"The config file {config_path} does not exist.")

    return config_path
    
def visa_config_generator(visa_key, diagnostic_type):
    
    # input_filename = '../../geecs_python_api/controls/data_acquisition/configs/HTU/visa_plunger_lookup.yaml'
    
    input_filename = get_full_config_path('Undulator', 'aux_configs', 'visa_plunger_lookup.yaml')
    
    with open(input_filename, 'r') as file:
        visa_lookup = yaml.safe_load(file)
    
    device_info = visa_lookup[visa_key]

    # Define the VisaEBeam camera dynamically based on the visa_key
    visa_ebeam_camera = f"UC_VisaEBeam{visa_key[-1]}"  # Extracts the last number from visa_key (e.g., visa1 -> UC_VisaEBEam1)

    if diagnostic_type == 'energy':
        description = f"collecting data on {visa_key}EBeam and U_FELEnergyMeter"
        setup_steps = [
            {'action': 'execute', 'action_name': 'remove_visa_plungers'},
            {'device': device_info['device'], 'variable': device_info['variable'], 'action': 'set', 'value': 'on'},
            {'device': 'U_Velmex', 'variable': 'Position', 'action': 'set', 'value': device_info['energy_meter_position']}
        ]
        devices = {
            visa_ebeam_camera: {
                'variable_list': ["timestamp"],
                'synchronous': True,
                'save_nonscalar_data': True
            },
            'U_FELEnergyMeter': {
                'variable_list': ["Python Results.ChA", "timestamp"],
                'synchronous': True,
                'save_nonscalar_data': True
            }
        }
    
    elif diagnostic_type == 'spectrometer':
        description = f"collecting data on {visa_key}EBeam and U_Spectrometer"
        setup_steps = [
            {'action': 'execute', 'action_name': 'remove_visa_plungers'},
            {'device': device_info['device'], 'variable': device_info['variable'], 'action': 'set', 'value': 'on'},
            {'device': 'U_Velmex', 'variable': 'Position', 'action': 'set', 'value': device_info['spectrometer_position']}
        ]
        devices = {
            visa_ebeam_camera: {
                'variable_list': ["timestamp"],
                'synchronous': True,
                'save_nonscalar_data': True
            },
            'UC_UndulatorRad2': {
                'variable_list': ["MeanCounts", "timestamp"],
                'synchronous': True,
                'save_nonscalar_data': True
            }
        }

    # Constructing the YAML structure
    output_data = {
        'Devices': devices,
        'scan_info': {
            'description': description
        },
        'setup_action': {
            'steps': setup_steps
        }
    }

    # Writing to a YAML file
    
    output_filename = input_filename.parent.parent / 'save_devices' / f'{visa_key}_{diagnostic_type}_setup.yaml'
    with open(output_filename, 'w') as outfile:
        yaml.dump(output_data, outfile, default_flow_style=False)

    # print(f"YAML file {output_filename} generated successfully!")
    return output_filename    
            
class ConsoleLogger:
    def __init__(self, log_file="system_log.log", level=logging.INFO, console=False):
        self.log_file = log_file
        self.level = level
        self.console = console
        self.logging_active = False

    def setup_logging(self):
        """
        Sets up logging for the module. By default, logs to a file.
        """
        if self.logging_active:
            logging.warning("Logging is already active, cannot start a new session.")
            return

        # Remove any previously configured handlers to prevent duplication
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Configure logging with both file and optional console handlers
        handlers = [logging.FileHandler(self.log_file)]
        if self.console:
            handlers.append(logging.StreamHandler())

        logging.basicConfig(
            level=self.level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=handlers
        )
        self.logging_active = True
        logging.info("Logging session started.")

    def stop_logging(self):
        """
        Stops logging and cleans up handlers.
        """
        if not self.logging_active:
            logging.warning("No active logging session to stop.")
            return

        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)

        self.logging_active = False
        print("Logging has been stopped and handlers have been removed.")

    def move_log_file(self, dest_dir):
        """
        Moves the log file to the destination directory using shutil to handle cross-device issues.
        """
        src_path = Path(self.log_file)
        dest_path = Path(dest_dir) / src_path.name

        print(f"Attempting to move {src_path} to {dest_path}")

        try:
            shutil.move(str(src_path), str(dest_path))
            print(f"Moved log file to {dest_path}")
        except Exception as e:
            print(f"Failed to move {src_path} to {dest_path}: {e}")

    def is_logging_active(self):
        """
        Returns whether logging is active or not.
        """
        return self.logging_active
