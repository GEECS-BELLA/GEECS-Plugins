"""Shared base classes for all GEECS config schemas.

Every config file the scanner reads is validated against one of the models in
this package.  The two base classes here give every model the same two
guarantees:

- **Typos fail loudly.**  Unknown keys are rejected (``extra="forbid"``), so a
  misspelled field name is an immediate validation error instead of a silently
  ignored setting.
- **Files are versioned.**  Every top-level config document carries a
  ``schema_version`` so future format changes can be migrated mechanically.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field


class SchemaModel(BaseModel):
    """Base class for every schema model in this package.

    Rejects unknown fields so a typo in a YAML key fails validation loudly
    instead of being silently ignored.

    Notes
    -----
    All models in ``geecs_schemas`` — both top-level documents and nested
    sub-models — inherit from this class.  Do not subclass ``BaseModel``
    directly inside this package.
    """

    model_config = ConfigDict(extra="forbid")


class VersionedSchemaModel(SchemaModel):
    """Base class for top-level config documents (one YAML file = one model).

    Adds the ``schema_version`` marker that identifies which format revision
    a saved file was written against.  You normally never edit this field;
    tools bump it when the format changes.

    Notes
    -----
    Nested sub-models (individual steps, entries, targets) deliberately do
    *not* carry ``schema_version`` — the version of the enclosing document
    governs the whole file.
    """

    schema_version: int = Field(
        default=1,
        description=(
            "Format version of this config file. Leave at 1 — tools update "
            "this automatically when the file format changes."
        ),
    )

    @classmethod
    def from_path(cls, path: str | Path) -> Self:
        """Load and validate one document from a YAML (default) or ``.json`` file.

        An empty file is the empty document (every field at its default); a
        non-empty file whose root is not a mapping is a validation error.
        The one loader every consumer of a config kind shares — the gateway's
        derived channels, the archiver's policy — so "how a file becomes a
        model" is spelled once.  YAML needs PyYAML, which this package keeps
        out of its runtime dependencies (serialization is not the contract);
        the error says so, and ``.json`` needs nothing.
        """
        file = Path(path)
        text = file.read_text(encoding="utf-8")
        if file.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "PyYAML is required to load YAML documents; geecs-schemas keeps it "
                    "out of its runtime dependencies (YAML is serialization, not the "
                    "contract). Install pyyaml, or pass a .json file."
                ) from exc
            data = yaml.safe_load(text)
        # Only an EMPTY file is the empty document. A non-empty file whose root
        # is not a mapping ([] / false / 0 / "x") must fail validation, not
        # quietly become the defaults — for an archive policy that would widen
        # the archive set by dropping every exclusion (#1035 review).
        return cls.model_validate({} if data is None else data)


def declared_schema_version(data: Mapping[str, object]) -> int | None:
    """Return the ``schema_version`` a raw document declares, or ``None``.

    A quoted digit (``"2"`` from YAML or JSON) counts the same as the int —
    pydantic's lax mode coerces it at field validation, so every reader of
    the raw stamp must see it the same way. An absent or unparseable
    version is ``None``: the field default (the current version) applies.
    """
    version = data.get("schema_version")
    if isinstance(version, str) and version.isdigit():
        version = int(version)
    if isinstance(version, int) and not isinstance(version, bool):
        return version
    return None


def stale_schema_version(data: Mapping[str, object], current: int) -> bool:
    """Whether *data* declares a ``schema_version`` older than *current*.

    The one definition every document kind's lifting validator uses to
    decide "normalize the stamp up".  A quoted digit (``"1"`` from YAML or
    JSON) counts the same as the int — pydantic's lax mode coerces it at
    field validation, so the staleness check must see it the same way.  An
    absent or unparseable version is *not* stale: the field default (the
    current version) applies.

    Parameters
    ----------
    data : Mapping
        The raw document.
    current : int
        The format version the model is at.

    Returns
    -------
    bool
        ``True`` when a declared version is strictly older than *current*.
    """
    version = declared_schema_version(data)
    return version is not None and version < current
