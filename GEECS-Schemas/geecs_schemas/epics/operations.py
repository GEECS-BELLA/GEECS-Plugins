"""OperatorPolicy and JournalEntry: who may write a device's roles, and the record of each write.

GeecsCanopy's device module enforces the one and writes the other; the
logbook and GEECS-MCP read both without importing it. The GEECS device
name is the key in both.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional, Union

from pydantic import Field, model_validator

from geecs_schemas._base import SchemaModel, VersionedSchemaModel
from geecs_schemas.epics.family import RoleName, TemplateName
from geecs_schemas.epics.panel_devices import DeviceName

#: The policy's file name in the experiment's configs folder.
OPERATOR_POLICY_FILENAME = "operator_policy.yaml"
_VERSION = "Format version. Leave at 1."


class OperatorGrant(SchemaModel):
    """Some operators may write some roles of some devices, named or by family."""

    operators: list[str] = Field(min_length=1, description="Operators, by name.")
    families: list[TemplateName] = Field([], description="Every device of these.")
    devices: list[DeviceName] = Field([], description="These GEECS devices.")
    roles: list[RoleName] = Field([], description="Only these; empty: all allowed.")

    @model_validator(mode="after")
    def _names_a_scope(self) -> "OperatorGrant":
        if not self.families and not self.devices:
            raise ValueError("a grant names families or devices")
        return self


class OperatorPolicy(VersionedSchemaModel):
    """Which operators may write what (``<experiment>/operator_policy.yaml``); default deny.

    The allow list, hazard confirm, run lock and bounds apply regardless.
    """

    schema_version: Literal[1] = Field(1, description=_VERSION)
    grants: list[OperatorGrant] = Field([], description="Nobody else writes.")

    def permits(self, operator: str, device: str, family: str, role: str) -> bool:
        """Whether a grant names this operator, this device or its family, and this role."""
        return any(
            operator in g.operators
            and (device in g.devices or family in g.families)
            and (not g.roles or role in g.roles)
            for g in self.grants
        )


Scalar = Union[float, int, bool, str]


class JournalEntry(VersionedSchemaModel):
    """One journal row: a request's outcome, or the ``attempted`` row a dispatched write leaves first (its outcome names it in ``attempt``)."""

    schema_version: Literal[1] = Field(1, description=_VERSION)
    id: Optional[int] = Field(None, description="Row number, oldest first.")
    time: datetime = Field(description="When it was decided (UTC).")
    operator: str = Field(description="The operator the request named.")
    key_name: Optional[str] = Field(None, description="The Canopy key's name.")
    device: str = Field(description="The GEECS device name.")
    role: str = Field(description="The family role written.")
    axis: Optional[str] = Field(None, description="The axis, if per-axis.")
    value: Union[Scalar, list[Scalar], None] = Field(description="Value requested.")
    outcome: Literal["attempted", "accepted", "denied", "refused"] = Field(
        description="attempted: written before Hare is called; denied: the module's"
        " guards; refused: bounds, IOC or Hare."
    )
    attempt: Optional[int] = Field(
        None, description="The row of this write's attempted row, once dispatched."
    )
    canopy_code: Optional[str] = Field(None, description="Canopy's or our slug.")
    detail: Optional[str] = Field(None, description="Why; or what a clamp took.")
