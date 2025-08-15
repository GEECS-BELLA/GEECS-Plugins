# Configuration Schemas for Data Acquisition

## Overview

The GEECS Scanner configuration schemas provide a flexible and robust system for defining experimental workflows, device configurations, and action sequences. These schemas leverage Pydantic's powerful validation and type checking to ensure precise and reliable experimental setup.

## Action Schemas

The Action Schemas define a comprehensive system for creating structured, configurable action sequences in experimental control systems.

### Step Types

#### Wait Step
A step that pauses execution for a specified duration.

::: geecs_scanner.data_acquisition.schemas.actions.WaitStep
    options:
      show_source: true
      show_root_heading: false
      show_root_toc_entry: true
      merge_init_into_class: true
      heading_level: 4
      filters: ["!^_"]
      show_if_no_docstring: false

#### Execute Step
A step that invokes another named action, enabling reusable and nested procedural workflows.

::: geecs_scanner.data_acquisition.schemas.actions.ExecuteStep
    options:
      show_source: true
      show_root_heading: false
      show_root_toc_entry: true
      merge_init_into_class: true
      heading_level: 4
      filters: ["!^_"]
      show_if_no_docstring: false

#### Run Step
A step that executes external scripts or classes within the experimental workflow.

::: geecs_scanner.data_acquisition.schemas.actions.RunStep
    options:
      show_source: true
      show_root_heading: false
      show_root_toc_entry: true
      merge_init_into_class: true
      heading_level: 4
      filters: ["!^_"]
      show_if_no_docstring: false

#### Set Step
A step that configures device variables with precise control and optional execution waiting.

::: geecs_scanner.data_acquisition.schemas.actions.SetStep
    options:
      show_source: true
      show_root_heading: false
      show_root_toc_entry: true
      merge_init_into_class: true
      heading_level: 4
      filters: ["!^_"]
      show_if_no_docstring: false

#### Get Step
A step that queries device variables and optionally validates their values.

::: geecs_scanner.data_acquisition.schemas.actions.GetStep
    options:
      show_source: true
      show_root_heading: false
      show_root_toc_entry: true
      merge_init_into_class: true
      heading_level: 4
      filters: ["!^_"]
      show_if_no_docstring: false

### Action Sequence and Library

#### Action Sequence
A comprehensive, ordered sequence of action steps defining an executable experimental procedure.

::: geecs_scanner.data_acquisition.schemas.actions.ActionSequence
    options:
      show_source: true
      show_root_heading: false
      show_root_toc_entry: true
      merge_init_into_class: true
      heading_level: 4
      filters: ["!^_"]
      show_if_no_docstring: false

#### Action Library
A comprehensive library of named action sequences for experimental workflows.

::: geecs_scanner.data_acquisition.schemas.actions.ActionLibrary
    options:
      show_source: true
      show_root_heading: false
      show_root_toc_entry: true
      merge_init_into_class: true
      heading_level: 4
      filters: ["!^_"]
      show_if_no_docstring: false

## Device Configuration Schemas

The Device Configuration Schemas provide a comprehensive framework for defining device configurations, data saving strategies, and pre/post-scan actions.

### Device Configuration

::: geecs_scanner.data_acquisition.schemas.save_devices.DeviceConfig
    options:
      show_source: true
      show_root_heading: false
      show_root_toc_entry: true
      merge_init_into_class: true
      heading_level: 3
      filters: ["!^_"]
      show_if_no_docstring: false

### Save Device Configuration

::: geecs_scanner.data_acquisition.schemas.save_devices.SaveDeviceConfig
    options:
      show_source: true
      show_root_heading: false
      show_root_toc_entry: true
      merge_init_into_class: true
      heading_level: 3
      filters: ["!^_"]
      show_if_no_docstring: false

## Usage Examples

### Creating an Action Sequence

```python
from geecs_scanner.data_acquisition.schemas.actions import ActionSequence, WaitStep, SetStep

# Define a simple action sequence
action_sequence = ActionSequence(steps=[
    WaitStep(action="wait", wait=1.0),
    SetStep(
        action="set",
        device="LaserController",
        variable="power",
        value=5.0,
        wait_for_execution=True
    )
])
```

### Configuring Device Saving

```python
from geecs_scanner.data_acquisition.schemas.save_devices import SaveDeviceConfig, DeviceConfig

# Create a comprehensive device saving configuration
save_config = SaveDeviceConfig(
    Devices={
        'Laser': DeviceConfig(
            synchronous=True,
            save_nonscalar_data=False,
            variable_list=['power', 'wavelength']
        )
    },
    scan_info={'experiment': 'beam_characterization'}
)
```

## Best Practices

1. **Use Explicit Configuration**: Always provide clear, explicit configurations for devices and actions.
2. **Leverage Validation**: Take advantage of Pydantic's built-in validation to catch configuration errors early.
3. **Modularize Actions**: Create reusable action sequences that can be combined and nested.
4. **Handle Optional Parameters**: Utilize optional parameters to create flexible configurations.
5. **Log and Monitor**: Use the built-in logging mechanisms to track configuration and execution.
