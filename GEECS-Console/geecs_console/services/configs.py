"""Configs-repo listings for the console's combos and lists.

Names come from scanning the configs-repo folders (the same roots
``geecs_bluesky.scanner_configs`` resolves); validation happens on demand
through ``ConfigsRepoResolver`` — listing is cheap and never parses YAML,
resolving a specific name does.  Offline-safety is first-class: a missing
configs root (or ``geecs-bluesky`` unimportable) degrades to empty listings
plus a status message, never an exception out of this module's listing
surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SAVE_SET_FOLDER = "save_devices"
TRIGGER_FOLDER = "shot_control_configurations"


@dataclass(frozen=True)
class ConfigListing:
    """Everything the main window's combos and lists need, in one snapshot."""

    experiments: list[str] = field(default_factory=list)
    save_sets: list[str] = field(default_factory=list)
    trigger_profiles: list[str] = field(default_factory=list)
    scan_variables: list[str] = field(default_factory=list)
    configs_root: Optional[Path] = None
    message: Optional[str] = None


@dataclass(frozen=True)
class UnionPreview:
    """The R2 preview line: union device count plus a conflict/role hint."""

    device_count: Optional[int] = None
    hint: str = ""


def _configs_base() -> Path | None:
    """Resolve the configs-repo experiments root, or ``None`` offline."""
    try:
        from geecs_bluesky.scanner_configs import scanner_configs_base

        return scanner_configs_base()
    except Exception as exc:  # RuntimeError (no root) or ImportError (no ca)
        logger.info("configs root unavailable: %s", exc)
        return None


def _yaml_stems(folder: Path) -> list[str]:
    """List the YAML file stems in *folder*, sorted; empty when absent."""
    if not folder.is_dir():
        return []
    return sorted(
        path.stem for path in folder.iterdir() if path.suffix in (".yaml", ".yml")
    )


class ConsoleConfigs:
    """Configs-repo access for one experiment (offline-safe).

    Parameters
    ----------
    experiment : str, optional
        Experiment folder under ``scanner_configs/experiments``.  May be
        empty at construction (before the operator picks one) — listings
        then carry experiments only.
    """

    def __init__(self, experiment: str = "") -> None:
        self._experiment = experiment
        self._resolver: Any = None

    @property
    def experiment(self) -> str:
        """The experiment this service reads configs for."""
        return self._experiment

    def set_experiment(self, experiment: str) -> None:
        """Switch to *experiment* (drops the cached resolver).

        Parameters
        ----------
        experiment : str
            The new experiment folder name.
        """
        self._experiment = experiment
        self._resolver = None

    # ------------------------------------------------------------------
    # Listings (folder scans — no YAML parsing, never raise)
    # ------------------------------------------------------------------

    def listing(self) -> ConfigListing:
        """Scan the configs repo for everything the window populates from.

        Returns
        -------
        ConfigListing
            Names for the combos/lists; empty with a ``message`` when the
            configs root is unresolvable or the experiment folder is missing.
        """
        base = _configs_base()
        if base is None:
            return ConfigListing(
                message=(
                    "Configs repo not found — set GEECS_SCANNER_CONFIG_DIR "
                    "or config.ini [Paths] scanner_config_root_path."
                )
            )
        experiments = sorted(p.name for p in base.iterdir() if p.is_dir())
        if not self._experiment:
            return ConfigListing(
                experiments=experiments,
                configs_root=base,
                message="No experiment selected.",
            )
        root = base / self._experiment
        if not root.is_dir():
            return ConfigListing(
                experiments=experiments,
                configs_root=base,
                message=f"No configs folder for experiment {self._experiment!r}.",
            )
        return ConfigListing(
            experiments=experiments,
            save_sets=_yaml_stems(root / SAVE_SET_FOLDER),
            trigger_profiles=_yaml_stems(root / TRIGGER_FOLDER),
            scan_variables=self._scan_variable_names(),
            configs_root=base,
        )

    def _scan_variable_names(self) -> list[str]:
        """Load the scan-variable catalog names (needs YAML, so resolver-backed)."""
        resolver = self._get_resolver()
        if resolver is None:
            return []
        try:
            return sorted(resolver._scan_variables_catalog().variables)
        except Exception as exc:
            logger.info("scan-variable catalog unavailable: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Validation-on-demand (resolver-backed)
    # ------------------------------------------------------------------

    def _get_resolver(self) -> Any:
        """Build (once) the ``ConfigsRepoResolver``; ``None`` offline."""
        if self._resolver is not None:
            return self._resolver
        if not self._experiment:
            return None
        try:
            from geecs_bluesky.config_resolver import ConfigsRepoResolver

            self._resolver = ConfigsRepoResolver(self._experiment)
        except Exception as exc:
            logger.info("resolver unavailable: %s", exc)
            return None
        return self._resolver

    def union_preview(self, save_set_names: list[str]) -> UnionPreview:
        """Preview the device union of the selected save sets (R2 line).

        Parameters
        ----------
        save_set_names : list of str
            The selected save-set names, in list order.

        Returns
        -------
        UnionPreview
            Device count of the union plus a hint line — a role-conflict
            message when the engine's merge would raise, a reference note
            when an entry pins the free-run pacemaker.
        """
        if not save_set_names:
            return UnionPreview(device_count=0)
        resolver = self._get_resolver()
        if resolver is None:
            return UnionPreview(hint="configs unavailable")
        try:
            from geecs_bluesky.scan_request_runner import merge_save_sets

            resolved = [resolver.resolve_save_set(name) for name in save_set_names]
            merged = merge_save_sets(resolved)
        except ValueError as exc:  # conflicting explicit roles across sets
            return UnionPreview(hint=f"role conflict: {exc}")
        except Exception as exc:
            return UnionPreview(hint=str(exc))
        references = [
            entry.device
            for entry in merged.entries
            if getattr(entry.role, "value", None) == "reference"
        ]
        hint = f"reference: {', '.join(references)}" if references else ""
        return UnionPreview(device_count=len(merged.entries), hint=hint)

    def trigger_variants(self, profile_name: str) -> list[str]:
        """List the variants of one trigger profile (R1 variant combo).

        Parameters
        ----------
        profile_name : str
            The trigger-profile name to resolve.

        Returns
        -------
        list of str
            Sorted variant names; empty offline or on any resolution error.
        """
        if not profile_name:
            return []
        resolver = self._get_resolver()
        if resolver is None:
            return []
        try:
            return sorted(resolver.resolve_trigger_profile(profile_name).variants)
        except Exception as exc:
            logger.info("trigger profile %r unavailable: %s", profile_name, exc)
            return []
