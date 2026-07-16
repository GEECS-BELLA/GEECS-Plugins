"""Operator-language tooltips for the main window, and their kill switch.

Main-window operator controls carry hand-written what-it-does tooltips
(:data:`OPERATOR_TOOLTIPS`, applied by :func:`apply_operator_tooltips`) —
GUI concepts with no schema counterpart.  Editor form fields are different:
their tooltips come from the geecs-schemas field descriptions via
:mod:`~geecs_console.services.schema_tooltips` — single source of truth
(issue #497 phase 1); when a tooltip reads poorly there, fix the schema
description, never hardcode GUI text.

**Preferences → Show tooltips** gates them all via
:class:`ToolTipSuppressor`, an application-level event filter installed
**only while tooltips are off** — an always-installed per-window app filter
measurably slowed the offscreen suite (every event crossing into a Python
``eventFilter``), so presence = suppression.

Both pieces moved here from ``app/main_window.py`` in the issue #534
slimming (step 1).
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject


class ToolTipSuppressor(QObject):
    """Application-level event filter that swallows every tooltip event.

    Installed on the ``QApplication`` **only while tooltips are turned
    off** — presence means suppression, so one switch covers every console
    widget (the main window *and* the editor dialogs, whose schema-derived
    tooltips are applied unconditionally) and the default-on path pays no
    per-event Python filter cost at all.  Parented to the window: Qt
    removes a destroyed filter from the application automatically, so a
    closed window can never leave a dangling suppressor behind.

    Parameters
    ----------
    parent : QObject
        The owning window (keeps the wrapper referenced — the usual
        PySide6 GC hazard — and bounds the filter's lifetime).
    """

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 — Qt override
        """Swallow ``QEvent.ToolTip``; pass everything else through.

        Parameters
        ----------
        obj : QObject
            The event's target (unused — suppression is global).
        event : QEvent
            The event under consideration.

        Returns
        -------
        bool
            ``True`` (consume the event) for tooltip events.
        """
        if event.type() == QEvent.Type.ToolTip:
            return True
        return super().eventFilter(obj, event)


#: Main-window widget attribute name → its operator tooltip.  Pure data —
#: :func:`apply_operator_tooltips` resolves each name on the window and a
#: missing widget raises loudly at construction (same fail-loud convention
#: as the editors' schema tooltips).
OPERATOR_TOOLTIPS: dict[str, str] = {
    # R1 session bar
    "experiment_combo": (
        "Which experiment's configs, presets, devices, and gateway "
        "PVs the console works with. Changing it repopulates "
        "everything below."
    ),
    "rep_rate": (
        "The machine repetition rate in Hz — informational for "
        "free-run pacing; it does not command any hardware."
    ),
    "trigger_profile_combo": (
        "Trigger profile the scan drives the machine with (OFF / "
        "STANDBY / SCAN / ... device writes). Empty means the scan "
        "leaves the trigger alone."
    ),
    "trigger_variant_combo": (
        "Named operating condition of the trigger profile (e.g. "
        "laser_off) — overlays a few writes on the base profile. "
        "Empty runs the base behaviour."
    ),
    "gateway_chip": (
        "CA gateway health: reads the experiment's heartbeat PV "
        "every few seconds. WARN means the gateway runs but reports "
        "zero connected devices."
    ),
    "tiled_chip": (
        "Tiled data-server health: an HTTP check of the configured "
        "Tiled URI every few seconds."
    ),
    "db_chip": (
        "GEECS experiment database health: a cheap MySQL query every few seconds."
    ),
    # R2 save sets
    "available_list": (
        "The experiment's save sets (named device groups) not yet "
        "picked for this scan. Select and Add to require them."
    ),
    "selected_list": (
        "Save sets this scan records. Their devices are unioned; "
        "each required device gets guarantees (completeness, "
        "dialogs, images). At least one set is needed to start — "
        "except in Optimization mode, where the optimizer config "
        "provisions its own diagnostics."
    ),
    "add_button": "Require the selected save sets for this scan.",
    "remove_button": (
        "Drop the selected save sets from this scan (their devices "
        "may still be logged as background telemetry)."
    ),
    "union_label": (
        "How many distinct devices the selected save sets add up to "
        "after merging duplicates."
    ),
    # R3 scan form
    "radio_noscan": (
        "Collect shots without moving anything — statistics at the "
        "current machine settings."
    ),
    "radio_1d": (
        "Sweep one scan variable through start → stop in step-sized "
        "moves, taking a batch of shots at each position."
    ),
    "radio_grid": (
        "Sweep two variables as a grid visiting every combination — "
        "axis 1 is the outer (slow) loop, axis 2 the inner (fast) "
        "one."
    ),
    "radio_optimization": (
        "Let an optimizer pick the settings each iteration, per the "
        "optimizer config chosen below. Submission is accepted only "
        "when the scan engine supports optimize scans."
    ),
    "radio_background": (
        "A no-scan whose data is marked as background/calibration "
        "shots so analysis can find them later."
    ),
    "optimization_combo": (
        "Which optimizer config to run — the YAML files in this "
        "experiment's optimizer_configs folder. Empty when offline "
        "or the experiment has none (Start stays disabled)."
    ),
    "iterations_spin": (
        "How many optimizer iterations to run — total shots = "
        "iterations × shots per step. Picking a config seeds this "
        "from the config's own limit; 'auto' (0) submits no limit "
        "and the engine's default budget applies."
    ),
    "shots_per_step": (
        "How many shots to take at each scan position / grid point "
        "(or in total for a no-scan or background run)."
    ),
    "acquisition_combo": (
        "free_run: the trigger runs at the machine rate and device "
        "rows are matched by timestamp afterwards. strict: the scan "
        "fires each shot itself and waits for every device — "
        "slower, but nothing is ever missing."
    ),
    "shot_count_label": (
        "Total shots this form implies (positions × shots per "
        "step). Above the runaway-scan limit, Start is disabled."
    ),
    "description_edit": (
        "Free-text note about this scan — it ends up in the scan's "
        "metadata and the experiment log."
    ),
    # R4 presets
    "preset_combo": (
        "Saved scan requests for this experiment (one YAML each in "
        "the configs repo's presets folder)."
    ),
    "apply_button": (
        "Load the selected preset into the form. Anything the form "
        "cannot express (action bindings, position lists) is "
        "refused, leaving the form untouched."
    ),
    "save_as_button": (
        "Save the current form as a named preset — the exact scan "
        "request it would submit."
    ),
    "delete_button": "Delete the selected preset's YAML file.",
    # R5 submit row
    "start_button": (
        "Build the scan request from this form and hand it to the "
        "scan engine. Needs a valid shot count and at least one "
        "selected save set; engine refusals show in the status bar."
    ),
    "stop_button": (
        "Ask the engine to stop the running scan at the next safe "
        "point (closeout actions still run)."
    ),
    # R6 now panel
    "state_pill": "What the scan engine is doing right now.",
    "progress_bar": ("Shots completed out of the running scan's announced total."),
    "scan_number_label": (
        "The scan folder claimed by the running scan; '(previous)' "
        "is the last scan found in today's data folder."
    ),
    "log_tail": "The most recent scan-engine and console messages.",
    # R7 device panel
    "device_combo": (
        "Type or pick 'DeviceName:Variable Name' to watch its live "
        "readback from the gateway (updates on commit, not per "
        "keystroke)."
    ),
    "readback_label": (
        "Live readback of the selected device variable, streamed from the gateway."
    ),
    "set_field": (
        "Value to write to the selected device variable — a number, "
        "or a word the device understands (e.g. 'on')."
    ),
    "set_button": (
        "Write the value via the gateway setpoint and report the "
        "outcome in the status bar. Disabled while a write is in "
        "flight."
    ),
}


def apply_operator_tooltips(window: QObject) -> None:
    """Set the operator-language tooltips on the main-window controls.

    These are hand-written (what it does / what happens), not copies of
    the widget labels.  A name in :data:`OPERATOR_TOOLTIPS` that no longer
    matches a window attribute raises ``AttributeError`` at construction —
    the mapping and the widgets must move together.

    Parameters
    ----------
    window : QObject
        The main window carrying the widgets named in
        :data:`OPERATOR_TOOLTIPS` as attributes.
    """
    for attr, text in OPERATOR_TOOLTIPS.items():
        getattr(window, attr).setToolTip(text)
