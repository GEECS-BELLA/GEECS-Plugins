"""The ``.himg`` job document: one data-management step over a HASO device folder.

Three jobs, chosen by the top-level ``kind``, each one step over
``scans/ScanNNN/<device>/`` with no per-frame analysis, no scalars and no
products under ``analysis/``:

- ``himg_to_stack`` converts the device's ``.himg`` files into its capture
  stack ``<device>/<device>.h5`` beside them (the sources stay), verified
  after writing; a second run verifies the existing stack.
- ``himg_compact`` rebuilds every frame of the stack, checks it against the
  SHA-256 recorded at conversion and against the ``.himg`` still on disk,
  and only when every frame passes deletes the ``.himg`` files, leaving
  ``himg_manifest.json``.  The one **destructive** job: a host asks for the
  scan number before running it, and the task queue never runs it.
- ``himg_restore`` rebuilds the ``.himg`` files from the stack,
  byte-identical, and removes the manifest.

The jobs sit in the analyzers tree beside the recipes (groups and the
portal's Analysis tab list them by file stem), but they are not analyses:
nothing here is a frame, a step or a measure, so the document is its own
kind rather than a recipe.  ``load_analysis_document`` reads a top-level
``kind`` as this document before it looks at ``schema_version``.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, PrivateAttr

from geecs_schemas._base import SchemaModel, VersionedSchemaModel

HimgJobKind = Literal["himg_to_stack", "himg_compact", "himg_restore"]


class HimgJobRuntime(SchemaModel):
    """How the job runs among a group's analyses."""

    priority: int = Field(
        default=100,
        ge=0,
        description="Run order within a group: lower runs first. 100 is the background default.",
    )


class HimgJob(VersionedSchemaModel):
    """One ``.himg`` data-management job over a HASO device folder.

    The file's stem is the ID groups and the portal refer to.  ``device``
    is the HASO device whose folder under ``scans/ScanNNN/`` holds the
    ``.himg`` files; it also names the stamp column of the scan's rows and
    the stack file.
    """

    schema_version: int = Field(
        1,
        description=(
            "Format version of this job file. Leave at 1 — tools update this "
            "automatically when the file format changes."
        ),
    )
    kind: HimgJobKind = Field(
        ...,
        description=(
            "Which job: himg_to_stack (folder → capture stack), himg_compact "
            "(verify, then delete the .himg files — destructive) or "
            "himg_restore (rebuild the .himg files from the stack)."
        ),
    )
    device: str = Field(
        ...,
        min_length=1,
        description="The HASO device whose folder under scans/ScanNNN/ holds the .himg files.",
    )
    output_name: Optional[str] = Field(
        None,
        description=(
            "Label the portal's Analysis tab lists the job under. Defaults to "
            "device; the job writes nothing under analysis/."
        ),
    )
    description: Optional[str] = Field(
        None, description="Free-text note about this job."
    )
    scan: HimgJobRuntime = Field(
        default_factory=HimgJobRuntime,
        description="How the job runs among a group's analyses.",
    )
    _source_id: Optional[str] = PrivateAttr(default=None)

    # The format-neutral names the analysis documents carry, so a host
    # lists a job beside the recipes without asking which document it holds.

    @property
    def source_id(self) -> Optional[str]:
        """Filename-derived job ID when loaded from disk, if known."""
        return self._source_id

    @property
    def effective_output_name(self) -> str:
        """``output_name`` if set, else ``device``."""
        return self.output_name if self.output_name is not None else self.device

    @property
    def data_folder(self) -> str:
        """The subfolder under scans/ScanNNN/ the job works in: the device's."""
        return self.device

    @property
    def destructive(self) -> bool:
        """Whether a run deletes data files — the compaction alone."""
        return self.kind == "himg_compact"


__all__ = ["HimgJob", "HimgJobKind", "HimgJobRuntime"]
