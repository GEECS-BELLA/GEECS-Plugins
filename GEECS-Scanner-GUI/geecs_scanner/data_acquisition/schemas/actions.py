from typing import List, Union, Literal, Optional, Annotated
from pydantic import BaseModel, Field, conint, confloat

# -------------------------------------------------------------------
# Individual Step Types
# -------------------------------------------------------------------

class WaitStep(BaseModel):
    """
    A step that pauses execution for a specified number of seconds.
    """
    action: Literal["wait"]
    wait: Annotated[confloat(gt=0), Field(description="Duration to pause in seconds")]

class ExecuteStep(BaseModel):
    """
    A step that invokes another named action, allowing for reusable and nested procedures.
    """
    action: Literal["execute"]
    action_name: Annotated[str, Field(description="Name of another action defined in the action library")]

class RunStep(BaseModel):
    """
    A step that attemps to exeute scripts.
    """
    action: Literal["run"]
    file_name: Annotated[str, Field(description="Name of file to use for run")]
    class_name: Annotated[str, Field(description="Name of class to to")]

class SetStep(BaseModel):
    """
    A step that sets a variable on a device to a given value.
    """
    action: Literal["set"]
    device: Annotated[str, Field(description="Device name (must be known to the system)")]
    variable: Annotated[str, Field(description="Variable on the device to set")]
    value: Annotated[Union[str, float, int], Field(description="Value to assign to the variable")]
    wait_for_execution: Optional[bool] = Field(
        True,
        description="Whether to wait for the command to complete before proceeding."
    )

class GetStep(BaseModel):
    """
    A step that queries a device variable and optionally verifies its value.
    """
    action: Literal["get"]
    device: Annotated[str, Field(description="Device name (must be known to the system)")]
    variable: Annotated[str, Field(description="Variable to read from the device")]
    expected_value: Annotated[Union[str, float, int], Field(description="Expected value for validation")]


# -------------------------------------------------------------------
# Union of all step types
# -------------------------------------------------------------------

ActionStep = Union[WaitStep, ExecuteStep, SetStep, GetStep]

# -------------------------------------------------------------------
# Full Action schemas
# -------------------------------------------------------------------

class ActionSequence(BaseModel):
    """
    A defined sequence of steps that together define an executable procedure.
    """
    steps: List[ActionStep] = Field(..., description="List of steps to perform in this action.")

class ActionLibrary(BaseModel):
    """
    The full action library schemas, mapping action names to their respective step sequences.
    """
    actions: dict[str, ActionSequence] = Field(..., description="Dictionary of all named actions.")
