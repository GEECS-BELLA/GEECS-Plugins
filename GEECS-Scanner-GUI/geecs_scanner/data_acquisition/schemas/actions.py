"""
Schemas for defining flexible, reusable experimental action sequences.

This module provides Pydantic models for creating structured, configurable
action sequences. These schemas enable dynamic definition of device interactions,
script executions, and complex procedural workflows.

Key Components:
- Individual step types (WaitStep, ExecuteStep, RunStep, SetStep, GetStep)
- Action sequence and library definitions
- Comprehensive validation and type checking

The action system supports:
- Precise device configuration
- Nested and reusable action procedures
- Script execution
- Conditional and sequential workflows

Examples
--------
>>> wait_step = WaitStep(action="wait", wait=2.5)
>>> set_step = SetStep(
...     action="set",
...     device="LaserController",
...     variable="power",
...     value=5.0
... )
>>> action_sequence = ActionSequence(steps=[wait_step, set_step])

See Also
--------
geecs_scanner.data_acquisition.action_manager : Action execution management
"""

from typing import List, Union, Literal, Optional, Annotated
from pydantic import BaseModel, Field, confloat

# -------------------------------------------------------------------
# Individual Step Types
# -------------------------------------------------------------------


class WaitStep(BaseModel):
    """
    A step that pauses execution for a specified duration in experimental workflows.

    This action step provides precise timing control during experimental sequences,
    allowing for controlled delays between device operations, data acquisition
    phases, or system stabilization periods.

    Attributes
    ----------
    action : Literal["wait"]
        The action type identifier for wait operations.
    wait : float
        Duration to pause in seconds. Must be greater than 0.

    Notes
    -----
    Wait steps are commonly used for:
    - Allowing devices to stabilize after configuration changes
    - Introducing delays between measurement phases
    - Synchronizing with external experimental timing

    Examples
    --------
    >>> wait_step = WaitStep(action="wait", wait=2.5)
    >>> # Creates a 2.5 second pause in the action sequence

    See Also
    --------
    ExecuteStep : Execute nested action sequences
    SetStep : Configure device variables
    """

    action: Literal["wait"]
    wait: Annotated[confloat(gt=0), Field(description="Duration to pause in seconds")]


class ExecuteStep(BaseModel):
    """
    A step that invokes another named action, enabling reusable and nested procedural workflows.

    This step allows for complex, modular action sequences by referencing
    pre-defined actions from the action library. It supports hierarchical
    and recursive action composition.

    Attributes
    ----------
    action : Literal["execute"]
        The action type identifier for action execution.
    action_name : str
        Name of the action to execute, as defined in the action library.

    Notes
    -----
    Execute steps enable:
    - Modular action design
    - Reusable procedure templates
    - Complex workflow composition
    - Hierarchical action nesting

    Examples
    --------
    >>> execute_step = ExecuteStep(action="execute", action_name="laser_calibration")
    >>> # Executes a pre-defined "laser_calibration" action sequence

    See Also
    --------
    ActionLibrary : Collection of named action sequences
    WaitStep : Introduce delays in action sequences
    """

    action: Literal["execute"]
    action_name: Annotated[
        str, Field(description="Name of another action defined in the action library")
    ]


class RunStep(BaseModel):
    """
    A step that executes external scripts or classes within the experimental workflow.

    This step provides flexibility in running custom Python scripts or
    instantiating specific classes as part of the action sequence.

    Attributes
    ----------
    action : Literal["run"]
        The action type identifier for script/class execution.
    file_name : str
        Name of the Python file containing the script or class to run.
    class_name : str
        Name of the class to instantiate or method to execute.

    Notes
    -----
    Run steps support:
    - Dynamic script execution
    - Custom class instantiation
    - Flexible integration of external logic
    - Extensible experimental workflows

    Examples
    --------
    >>> run_step = RunStep(
    ...     action="run",
    ...     file_name="beam_analysis.py",
    ...     class_name="BeamProfiler"
    ... )
    >>> # Runs BeamProfiler class from beam_analysis.py

    See Also
    --------
    ExecuteStep : Execute predefined action sequences
    SetStep : Configure device variables
    """

    action: Literal["run"]
    file_name: Annotated[str, Field(description="Name of file to use for run")]
    class_name: Annotated[str, Field(description="Name of class to instantiate")]


class SetStep(BaseModel):
    """
    A step that configures device variables with precise control and optional execution waiting.

    This step enables dynamic device configuration within experimental workflows,
    supporting various data types and optional synchronous execution.

    Attributes
    ----------
    action : Literal["set"]
        The action type identifier for device variable configuration.
    device : str
        Name of the device to configure (must be known to the system).
    variable : str
        Specific variable on the device to set.
    value : Union[str, float, int]
        Value to assign to the device variable.
    wait_for_execution : bool, optional
        Whether to wait for the command to complete before proceeding.
        Defaults to True.

    Notes
    -----
    Set steps provide:
    - Flexible device configuration
    - Support for multiple value types
    - Optional synchronous execution
    - Precise device state management

    Examples
    --------
    >>> set_step = SetStep(
    ...     action="set",
    ...     device="LaserController",
    ...     variable="power",
    ...     value=5.0,
    ...     wait_for_execution=True
    ... )
    >>> # Sets laser power to 5.0, waiting for confirmation

    See Also
    --------
    GetStep : Query device variable values
    WaitStep : Introduce delays in action sequences
    """

    action: Literal["set"]
    device: Annotated[
        str, Field(description="Device name (must be known to the system)")
    ]
    variable: Annotated[str, Field(description="Variable on the device to set")]
    value: Annotated[
        Union[str, float, int], Field(description="Value to assign to the variable")
    ]
    wait_for_execution: Optional[bool] = Field(
        True,
        description="Whether to wait for the command to complete before proceeding.",
    )


class GetStep(BaseModel):
    """
    A step that queries device variables and optionally validates their values.

    This step enables dynamic data retrieval and optional value verification
    within experimental workflows, supporting various data types.

    Attributes
    ----------
    action : Literal["get"]
        The action type identifier for device variable querying.
    device : str
        Name of the device to query (must be known to the system).
    variable : str
        Specific variable to read from the device.
    expected_value : Union[str, float, int]
        Optional expected value for validation purposes.

    Notes
    -----
    Get steps support:
    - Dynamic device variable retrieval
    - Optional value validation
    - Flexible data type handling
    - Experimental state verification

    Examples
    --------
    >>> get_step = GetStep(
    ...     action="get",
    ...     device="Thermometer",
    ...     variable="temperature",
    ...     expected_value=25.0
    ... )
    >>> # Retrieves temperature, optionally validating against 25.0

    See Also
    --------
    SetStep : Configure device variables
    ActionSequence : Define sequences of action steps
    """

    action: Literal["get"]
    device: Annotated[
        str, Field(description="Device name (must be known to the system)")
    ]
    variable: Annotated[str, Field(description="Variable to read from the device")]
    expected_value: Annotated[
        Union[str, float, int], Field(description="Expected value for validation")
    ]


# -------------------------------------------------------------------
# Union of all step types
# -------------------------------------------------------------------

ActionStep = Union[WaitStep, ExecuteStep, RunStep, SetStep, GetStep]

# -------------------------------------------------------------------
# Full Action schemas
# -------------------------------------------------------------------


class ActionSequence(BaseModel):
    """
    A comprehensive, ordered sequence of action steps defining an executable experimental procedure.

    This model represents a complete, structured workflow composed of multiple
    action steps, enabling complex, modular experimental protocols.

    Attributes
    ----------
    steps : List[ActionStep]
        An ordered list of action steps to perform in sequence.
        Supports various step types: Wait, Execute, Run, Set, Get.

    Notes
    -----
    ActionSequence provides:
    - Ordered execution of multiple action steps
    - Support for diverse step types
    - Flexible experimental workflow definition
    - Comprehensive validation of step sequences

    Examples
    --------
    >>> sequence = ActionSequence(steps=[
    ...     WaitStep(action="wait", wait=1.0),
    ...     SetStep(action="set", device="Laser", variable="power", value=5.0),
    ...     GetStep(action="get", device="Laser", variable="status")
    ... ])
    >>> # Creates a sequence with wait, set, and get steps

    See Also
    --------
    ActionLibrary : Collection of named action sequences
    ActionStep : Individual step types supported in sequences
    """

    steps: List[ActionStep] = Field(
        ..., description="List of steps to perform in this action."
    )


class ActionLibrary(BaseModel):
    """
    A comprehensive library of named action sequences for experimental workflows.

    This model provides a structured repository of reusable, named action
    sequences, enabling modular and extensible experimental protocol design.

    Attributes
    ----------
    actions : Dict[str, ActionSequence]
        A dictionary mapping action names to their corresponding action sequences.

    Notes
    -----
    ActionLibrary supports:
    - Named, reusable action sequences
    - Centralized action management
    - Flexible experimental protocol composition
    - Dynamic action referencing

    Examples
    --------
    >>> library = ActionLibrary(actions={
    ...     "laser_setup": ActionSequence(steps=[...]),
    ...     "data_acquisition": ActionSequence(steps=[...])
    ... })
    >>> # Creates a library with multiple named action sequences

    See Also
    --------
    ActionSequence : Individual action sequence definition
    ExecuteStep : Step for invoking named actions
    """

    actions: dict[str, ActionSequence] = Field(
        ..., description="Dictionary of all named actions."
    )
