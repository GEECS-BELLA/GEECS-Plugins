"""OperatorPolicy and JournalEntry: who may write a device's roles, and the record of each write.

GeecsCanopy's device module enforces the one and writes the other; the
logbook and GEECS-MCP read both without importing it. The GEECS device
name is the key in both. A family is a family template for a native
device and the GEECS DB device type for a DB-served one; a role is a
family role, or a DB device's GEECS variable name.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Optional, Union

from pydantic import Field, StringConstraints, model_validator

from geecs_schemas._base import SchemaModel, VersionedSchemaModel
from geecs_schemas.epics.family import AnyRoleName, FamilyWord
from geecs_schemas.epics.panel_devices import DeviceName

#: The policy's file name in the experiment's configs folder.
OPERATOR_POLICY_FILENAME = "operator_policy.yaml"
_VERSION = "Format version. Leave at 1."
#: In a grant's ``operators``, alone: every operator who gives a name.
ANY_OPERATOR = "*"
#: An operator's name as a policy file writes it: no space at either end.
OperatorName = Annotated[str, StringConstraints(pattern=r"^\S(.*\S)?$")]


class _Scope(SchemaModel):
    """Devices named, or by family (a family template or a DB device type)."""

    families: list[FamilyWord] = Field([], description="Every device of these.")
    devices: list[DeviceName] = Field([], description="These GEECS devices.")

    @model_validator(mode="after")
    def _names_a_scope(self) -> "_Scope":
        if not self.families and not self.devices:
            raise ValueError(f"{type(self).__name__} must name families or devices")
        return self

    def covers(self, device: str, family: str) -> bool:
        """Whether the scope names this device or its family."""
        return device in self.devices or family in self.families


class OperatorGrant(_Scope):
    """Some operators may write some roles of some devices, named or by family.

    ``operators: ["*"]`` grants every operator who gives a name: the write is
    still journalled under that name, and the allow list, hazard confirm, run
    lock and bounds still apply.
    """

    operators: list[OperatorName] = Field(
        min_length=1,
        description='Operators, by name; or ["*"] alone: every named operator.',
    )
    roles: list[AnyRoleName] = Field([], description="Only these; empty: all allowed.")

    @model_validator(mode="after")
    def _names_or_everyone(self) -> "OperatorGrant":
        stars = [o for o in self.operators if ANY_OPERATOR in o]
        if stars and self.operators != [ANY_OPERATOR]:
            raise ValueError(
                f"operators is names or [{ANY_OPERATOR!r}] alone, and "
                f"{ANY_OPERATOR!r} is not a pattern: {stars}"
            )
        return self

    def names(self, operator: str) -> bool:
        """Whether the grant names this operator, or every operator and this one gave a name."""
        if self.operators == [ANY_OPERATOR]:
            name = operator.strip()
            visible = any(c.isprintable() and not c.isspace() for c in name)
            return visible and name != ANY_OPERATOR
        return operator in self.operators


class HazardRule(_Scope):
    """These roles of these devices need the hazard confirm, as a family role with ``confirm: hazard`` does."""

    roles: list[AnyRoleName] = Field(min_length=1, description="The roles a write confirms.")  # fmt: skip


class OperatorPolicy(VersionedSchemaModel):
    """Which operators may write what (``<experiment>/operator_policy.yaml``); default deny.

    The allow list, hazard confirm, run lock and bounds apply regardless.
    ``hazards`` is the one home of the hazard confirm for DB-served devices
    (a native device's comes from its family file; both apply to both).
    """

    schema_version: Literal[1] = Field(1, description=_VERSION)
    grants: list[OperatorGrant] = Field([], description="Nobody else writes.")
    hazards: list[HazardRule] = Field([], description="Roles that need the hazard confirm.")  # fmt: skip

    def permits(self, operator: str, device: str, family: str, role: str) -> bool:
        """Whether a grant names this operator, this device or its family, and this role."""
        return any(
            g.names(operator)
            and g.covers(device, family)
            and (not g.roles or role in g.roles)
            for g in self.grants
        )

    def hazardous(self, device: str, family: str, role: str) -> bool:
        """Whether a hazard rule makes this role of this device need the confirm."""
        return any(h.covers(device, family) and role in h.roles for h in self.hazards)


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
    run: Optional[str] = Field(None, description="The action run this row belongs to.")
