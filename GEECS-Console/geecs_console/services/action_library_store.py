"""Action-library persistence: named :class:`~geecs_schemas.ActionPlan` YAMLs.

The experiment's action plans all live in **one** library file — the same
one ``geecs_bluesky.config_resolver.ConfigsRepoResolver`` reads::

    scanner_configs/experiments/<Experiment>/action_library/actions.yaml

A file whose top level carries ``schema_version`` loads directly as a
:class:`~geecs_schemas.ActionPlanLibrary`; anything else is treated as the
legacy ``actions:`` dialect and goes through
:func:`geecs_schemas.convert.convert_action_library` — mirroring the
resolver, so the existing corpus opens unchanged.  Saving always writes the
new schema (``model_dump(mode="json")``), which round-trips losslessly
through ``ActionPlanLibrary.model_validate``.

Offline-safety mirrors :class:`~geecs_console.services.presets.PresetStore`:
listing degrades to empty with no configs root; ``load``/``save``/``delete``/
``rename`` raise :class:`ActionLibraryStoreError` with a message fit for
inline display.  The schema's library validator rejects dangling ``run``
references, so this store pre-checks deletes and saves and raises the
clearer message itself.  Directory creation and the scan-folder-invariant
rationale live in :mod:`geecs_console.services.config_store`.

This module is **editing only** — it never executes a plan and never touches
Channel Access or hardware.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from geecs_console.services.config_store import ExperimentConfigStore

# The base resolves the configs root through this module's namespace, so
# tests can monkeypatch ``action_library_store._configs_base``.
from geecs_console.services.configs import _configs_base  # noqa: F401
from geecs_schemas import ActionPlan, ActionPlanLibrary
from geecs_schemas.convert import SchemaConversionError, convert_action_library

logger = logging.getLogger(__name__)

ACTION_FOLDER = "action_library"
LIBRARY_FILE = "actions.yaml"

# Identity sentinel for _read_mapping's empty_as: distinguishes an all-empty
# YAML document (safe_load returns None → empty library) from a literal
# ``{}`` mapping, which must still flow through the legacy converter.
_EMPTY_DOCUMENT: dict = {}


class ActionLibraryStoreError(RuntimeError):
    """An action-library operation cannot be carried out (shown inline)."""


def referencing_plans(library: ActionPlanLibrary, name: str) -> list[str]:
    """List the plans whose ``run`` steps reference plan *name*.

    Parameters
    ----------
    library : ActionPlanLibrary
        The library to search.
    name : str
        The referenced plan name.

    Returns
    -------
    list of str
        Names of plans containing a ``run`` step targeting *name*, sorted.
    """
    return sorted(
        plan_name
        for plan_name, plan in library.plans.items()
        if any(getattr(step, "plan", None) == name for step in plan.steps)
    )


class ActionLibraryStore(ExperimentConfigStore):
    """Load/save named action plans for one experiment's library file."""

    FOLDER = ACTION_FOLDER
    ERROR = ActionLibraryStoreError

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _library_path(self) -> Optional[Path]:
        """Return the library file path, or ``None`` offline/unselected.

        Raises
        ------
        ActionLibraryStoreError
            When the experiment name would escape the experiments root —
            checked before any path join.
        """
        folder = self._folder()
        return None if folder is None else folder / LIBRARY_FILE

    def _library_path_or_raise(self) -> Path:
        """Return the library file path, raising a clear error when unresolvable.

        Returns
        -------
        Path
            ``<experiments root>/<experiment>/action_library/actions.yaml``.

        Raises
        ------
        ActionLibraryStoreError
            When the configs repo is not found, no experiment is selected,
            or the experiment name would escape the experiments root.
        """
        return self._folder_or_raise() / LIBRARY_FILE

    @staticmethod
    def _validate_name(name: str) -> str:
        """Validate and normalise a plan name.

        Plan names are dict keys inside one file — no path is built from
        them — but a name with separators or traversal parts is always a
        mistake and would confuse every downstream name reference, so it is
        rejected here.

        Parameters
        ----------
        name : str
            The candidate plan name.

        Returns
        -------
        str
            The stripped name.

        Raises
        ------
        ActionLibraryStoreError
            On an empty name or one containing path separators / traversal.
        """
        cleaned = name.strip()
        if not cleaned:
            raise ActionLibraryStoreError("Plan name must not be empty.")
        if any(sep in cleaned for sep in ("/", "\\")) or cleaned in (".", ".."):
            raise ActionLibraryStoreError(
                f"Plan name {name!r} must be a plain name (no path separators)."
            )
        return cleaned

    # ------------------------------------------------------------------
    # Whole-library I/O
    # ------------------------------------------------------------------

    def load_library(self) -> ActionPlanLibrary:
        """Load the experiment's action-plan library.

        Returns
        -------
        ActionPlanLibrary
            The validated library; **empty** when the library file does not
            exist yet (a new experiment simply has no plans).

        Raises
        ------
        ActionLibraryStoreError
            Missing configs repo / experiment, unparsable YAML, an
            unconvertible legacy document, or a document the schema rejects
            — always with a message fit for inline display.
        """
        path = self._library_path_or_raise()
        if not path.exists():
            return ActionPlanLibrary(plans={})
        document = self._read_mapping(
            path, describe="Action library", empty_as=_EMPTY_DOCUMENT
        )
        if document is _EMPTY_DOCUMENT:
            return ActionPlanLibrary(plans={})
        if "schema_version" in document:
            try:
                return ActionPlanLibrary.model_validate(document)
            except (ValidationError, ValueError) as exc:
                raise ActionLibraryStoreError(
                    f"Action library ({path}) is not a valid ActionPlanLibrary: {exc}"
                ) from exc
        try:
            return convert_action_library(document)
        except (SchemaConversionError, ValidationError, ValueError) as exc:
            raise ActionLibraryStoreError(
                f"Action library ({path}) could not be converted from the "
                f"legacy dialect: {exc}"
            ) from exc

    def save_library(self, library: ActionPlanLibrary) -> Path:
        """Write *library* as the experiment's ``actions.yaml`` (new schema).

        Parameters
        ----------
        library : ActionPlanLibrary
            The validated library to persist.

        Returns
        -------
        Path
            The YAML file written.

        Raises
        ------
        ActionLibraryStoreError
            Missing configs repo / experiment, or an OS-level write failure.
        """
        path = self._library_path_or_raise()
        self._write_document(
            path, library.model_dump(mode="json"), "the action library"
        )
        logger.info("saved action library (%d plans) to %s", len(library.plans), path)
        return path

    # ------------------------------------------------------------------
    # Named-plan surface
    # ------------------------------------------------------------------

    def list_names(self) -> list[str]:
        """List the saved plan names, sorted; never raises.

        Returns
        -------
        list of str
            Plan names in the library — empty when the configs repo,
            experiment, or library file is missing or unreadable.
        """
        try:
            return sorted(self.load_library().plans)
        except ActionLibraryStoreError as exc:
            logger.info("action library unavailable: %s", exc)
            return []

    def load(self, name: str) -> ActionPlan:
        """Load plan *name* as a validated :class:`ActionPlan`.

        Parameters
        ----------
        name : str
            The plan name.

        Returns
        -------
        ActionPlan
            The schema-validated plan.

        Raises
        ------
        ActionLibraryStoreError
            Missing configs repo / experiment / library / plan, or an
            invalid library file.
        """
        cleaned = self._validate_name(name)
        library = self.load_library()
        plan = library.plans.get(cleaned)
        if plan is None:
            raise ActionLibraryStoreError(
                f"Action plan {name!r} not found. Known plans: {sorted(library.plans)}"
            )
        return plan

    def save(self, name: str, plan: ActionPlan) -> Path:
        """Write *plan* as *name* into the library (overwriting an existing one).

        Parameters
        ----------
        name : str
            The plan name.
        plan : ActionPlan
            The plan to persist.

        Returns
        -------
        Path
            The YAML file written.

        Raises
        ------
        ActionLibraryStoreError
            Missing configs repo / experiment, a bad name, a ``run`` step
            referencing a plan that would not exist in the saved library, or
            an OS-level write failure.
        """
        cleaned = self._validate_name(name)
        library = self.load_library()
        plans = dict(library.plans)
        plans[cleaned] = plan
        dangling = sorted(
            {
                step.plan
                for candidate in plans.values()
                for step in candidate.steps
                if getattr(step, "plan", None) is not None and step.plan not in plans
            }
        )
        if dangling:
            raise ActionLibraryStoreError(
                f"Cannot save {name!r}: 'run' steps reference plans not in "
                f"the library: {dangling}."
            )
        try:
            updated = ActionPlanLibrary(
                schema_version=library.schema_version, plans=plans
            )
        except (ValidationError, ValueError) as exc:
            raise ActionLibraryStoreError(f"Cannot save {name!r}: {exc}") from exc
        return self.save_library(updated)

    def delete(self, name: str) -> None:
        """Delete plan *name* from the library.

        Parameters
        ----------
        name : str
            The plan name.

        Raises
        ------
        ActionLibraryStoreError
            Missing configs repo / experiment / plan, a plan still
            referenced by another plan's ``run`` step, or an OS-level write
            failure.
        """
        cleaned = self._validate_name(name)
        library = self.load_library()
        if cleaned not in library.plans:
            raise ActionLibraryStoreError(f"Action plan {name!r} not found.")
        referrers = referencing_plans(library, cleaned)
        referrers = [ref for ref in referrers if ref != cleaned]
        if referrers:
            raise ActionLibraryStoreError(
                f"Cannot delete {name!r}: still referenced by 'run' steps in "
                f"{referrers}."
            )
        plans = {k: v for k, v in library.plans.items() if k != cleaned}
        updated = ActionPlanLibrary(schema_version=library.schema_version, plans=plans)
        self.save_library(updated)
        logger.info("deleted action plan %r", name)

    def rename(self, old: str, new: str) -> None:
        """Rename plan *old* to *new*, updating every ``run`` reference to it.

        The plan keeps its position in the library file, and every other
        plan's ``run`` step targeting *old* is rewritten to *new* — a rename
        can therefore never leave a dangling reference behind.

        Parameters
        ----------
        old : str
            The current plan name.
        new : str
            The new plan name.

        Raises
        ------
        ActionLibraryStoreError
            Missing configs repo / experiment / plan, a bad new name, a
            *new* name already in the library, or an OS-level write failure.
        """
        cleaned_old = self._validate_name(old)
        cleaned_new = self._validate_name(new)
        if cleaned_new == cleaned_old:
            return
        library = self.load_library()
        if cleaned_old not in library.plans:
            raise ActionLibraryStoreError(f"Action plan {old!r} not found.")
        if cleaned_new in library.plans:
            raise ActionLibraryStoreError(
                f"Cannot rename {old!r}: plan {new!r} already exists."
            )
        plans: dict[str, ActionPlan] = {}
        for plan_name, plan in library.plans.items():
            retargeted = _retarget_run_steps(plan, cleaned_old, cleaned_new)
            plans[cleaned_new if plan_name == cleaned_old else plan_name] = retargeted
        updated = ActionPlanLibrary(schema_version=library.schema_version, plans=plans)
        self.save_library(updated)
        logger.info("renamed action plan %r to %r", old, new)


def _retarget_run_steps(plan: ActionPlan, old: str, new: str) -> ActionPlan:
    """Return *plan* with every ``run`` step targeting *old* rewritten to *new*.

    Parameters
    ----------
    plan : ActionPlan
        The plan to rewrite.
    old : str
        The referenced plan name to replace.
    new : str
        The replacement plan name.

    Returns
    -------
    ActionPlan
        The rewritten plan (the original object when nothing referenced *old*).
    """
    if not any(getattr(step, "plan", None) == old for step in plan.steps):
        return plan
    steps = [
        step.model_copy(update={"plan": new})
        if getattr(step, "plan", None) == old
        else step
        for step in plan.steps
    ]
    return plan.model_copy(update={"steps": steps})
