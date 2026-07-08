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
        1,
        description=(
            "Format version of this config file. Leave at 1 — tools update "
            "this automatically when the file format changes."
        ),
    )
