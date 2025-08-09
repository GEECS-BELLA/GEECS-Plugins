"""
Pydantic models for filter specifications with automatic validation.

This module defines the data structures for filter specifications, providing
type safety, validation, and clear documentation of filter arguments.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import Optional, Union, List, Literal
from datetime import date
import pandas as pd


class FilterArgs(BaseModel):
    """Base class for filter arguments."""

    model_config = ConfigDict(extra="forbid")  # v2 style


class ECSValueWithinArgs(FilterArgs):
    """Arguments for ECS value numeric range filtering."""

    device_like: str = Field(..., description="Device name pattern to match")
    variable_like: str = Field(..., description="Variable name pattern to match")
    target: float = Field(..., description="Target numeric value")
    tol: float = Field(..., gt=0, description="Tolerance (must be positive)")


class ECSValueContainsArgs(FilterArgs):
    """Arguments for ECS value text matching."""

    device_like: str = Field(..., description="Device name pattern to match")
    variable_like: str = Field(..., description="Variable name pattern to match")
    text: str = Field(..., description="Text to search for in values")
    case: bool = Field(False, description="Case-sensitive matching")


class ScanParameterContainsArgs(FilterArgs):
    """Arguments for scan parameter text matching."""

    substring: str = Field(..., description="Substring to search for")
    case: bool = Field(False, description="Case-sensitive matching")


class DeviceContainsArgs(FilterArgs):
    """Arguments for device name matching."""

    device_substring: str = Field(..., description="Device name substring to match")


class ExperimentEqualsArgs(FilterArgs):
    """Arguments for experiment name matching."""

    name: str = Field(..., description="Exact experiment name to match")


class CompositeArgs(FilterArgs):
    """Arguments for composite filters."""

    subfilters: List[str] = Field(
        ..., min_items=1, description="List of sub-filter names to combine"
    )


class FilterVersion(BaseModel):
    """Single version of a filter with optional date validity."""

    valid_from: Optional[date] = Field(
        None, description="Start date for filter validity"
    )
    valid_to: Optional[date] = Field(
        None, description="End date for filter validity (null = open-ended)"
    )
    kind: Literal[
        "ecs_value_within",
        "ecs_value_contains",
        "scan_parameter_contains",
        "device_contains",
        "experiment_equals",
        "composite",
    ] = Field(..., description="Type of filter")
    args: Union[
        ECSValueWithinArgs,
        ECSValueContainsArgs,
        ScanParameterContainsArgs,
        DeviceContainsArgs,
        ExperimentEqualsArgs,
        CompositeArgs,
    ] = Field(..., description="Filter-specific arguments")

    @model_validator(mode="after")
    def _validate_self(self):
        # date range check
        if self.valid_to and self.valid_from and self.valid_to < self.valid_from:
            raise ValueError(
                f"valid_to ({self.valid_to}) must be >= valid_from ({self.valid_from})"
            )

        # args type vs kind check
        expected_types = {
            "ecs_value_within": ECSValueWithinArgs,
            "ecs_value_contains": ECSValueContainsArgs,
            "scan_parameter_contains": ScanParameterContainsArgs,
            "device_contains": DeviceContainsArgs,
            "experiment_equals": ExperimentEqualsArgs,
            "composite": CompositeArgs,
        }
        expected_type = expected_types.get(self.kind)
        if expected_type and not isinstance(self.args, expected_type):
            raise ValueError(
                f"Filter kind '{self.kind}' requires {expected_type.__name__} arguments"
            )
        return self


class FilterSpec(BaseModel):
    """Complete filter specification with one or more versions."""

    versions: List[FilterVersion] = Field(
        ..., min_items=1, description="List of filter versions"
    )

    @field_validator("versions")
    @classmethod
    def validate_no_overlaps(cls, versions):
        """Ensure no overlapping date ranges between versions."""
        if len(versions) <= 1:
            return versions

        # Only check dated versions for overlaps
        dated_versions = [v for v in versions if v.valid_from is not None]
        if len(dated_versions) <= 1:
            return versions

        # Sort by valid_from date
        sorted_versions = sorted(dated_versions, key=lambda v: v.valid_from)

        # Check for overlaps
        for i in range(len(sorted_versions) - 1):
            curr = sorted_versions[i]
            next_ver = sorted_versions[i + 1]

            # Calculate effective end date (use current date if null)
            curr_end = curr.valid_to or pd.Timestamp.now().date()
            next_start = next_ver.valid_from

            if curr_end >= next_start:
                raise ValueError(
                    f"Overlapping date ranges: "
                    f"[{curr.valid_from} to {curr.valid_to or 'current'}] "
                    f"overlaps with [{next_ver.valid_from} to {next_ver.valid_to or 'current'}]"
                )

        return versions

    def get_version_for_date(self, target_date: date) -> Optional[FilterVersion]:
        """Get the appropriate filter version for a specific date."""
        for version in self.versions:
            # Handle undated versions (always valid)
            if version.valid_from is None:
                return version

            # Check if date falls within this version's range
            valid_from = version.valid_from
            valid_to = version.valid_to or pd.Timestamp.now().date()

            if valid_from <= target_date <= valid_to:
                return version

        return None

    def get_versions_for_range(
        self, start_date: date, end_date: date
    ) -> List[FilterVersion]:
        """Get all filter versions that overlap with a date range."""
        applicable_versions = []

        for version in self.versions:
            # Handle undated versions (always applicable)
            if version.valid_from is None:
                applicable_versions.append(version)
                continue

            # Check if this version overlaps with the query range
            valid_from = version.valid_from
            valid_to = version.valid_to or pd.Timestamp.now().date()

            # Check for overlap: not (version_end < range_start or version_start > range_end)
            if not (valid_to < start_date or valid_from > end_date):
                applicable_versions.append(version)

        return applicable_versions


def parse_filter_spec_from_yaml(
    name: str, raw_spec: Union[dict, List[dict]]
) -> FilterSpec:
    """
    Parse a filter specification from YAML format into a validated FilterSpec.

    Parameters
    ----------
    name : str
        Filter name (for error reporting)
    raw_spec : Union[dict, List[dict]]
        Raw specification from YAML (single dict or list of dicts)

    Returns
    -------
    FilterSpec
        Validated filter specification

    Raises
    ------
    ValueError
        If the specification is invalid
    """
    # Normalize to list format
    if isinstance(raw_spec, dict):
        versions_data = [raw_spec]
    elif isinstance(raw_spec, list):
        versions_data = raw_spec
    else:
        raise ValueError(f"Filter '{name}' must be a dict or list of dicts")

    # Parse each version
    versions = []
    for i, version_data in enumerate(versions_data):
        try:
            # Extract args based on kind
            kind = version_data.get("kind")
            if not kind:
                raise ValueError(f"Missing 'kind' field in version {i}")

            args_data = version_data.get("args", {})

            # Create the appropriate args object
            args_classes = {
                "ecs_value_within": ECSValueWithinArgs,
                "ecs_value_contains": ECSValueContainsArgs,
                "scan_parameter_contains": ScanParameterContainsArgs,
                "device_contains": DeviceContainsArgs,
                "experiment_equals": ExperimentEqualsArgs,
                "composite": CompositeArgs,
            }

            args_class = args_classes.get(kind)
            if not args_class:
                raise ValueError(f"Unknown filter kind: {kind}")

            args = args_class(**args_data)

            # Create the version
            version = FilterVersion(
                valid_from=version_data.get("valid_from"),
                valid_to=version_data.get("valid_to"),
                kind=kind,
                args=args,
            )
            versions.append(version)

        except Exception as e:
            raise ValueError(f"Error in filter '{name}' version {i}: {e}")

    # Create and validate the complete spec
    return FilterSpec(versions=versions)
