"""MovablePanelController — the catalog-aware R7 movable panel (issue #534 shape).

Owns the R7 surface the window previously ran inline: the editable
device/variable combo (completions + selection commits), the readback
label, and the set field/button — and makes the panel **catalog-aware**.
A selection is resolved in this order:

1. **A catalog scan-variable name** (the same names the R3 axis picker
   lists): a plain :class:`~geecs_schemas.scan_variables.ScanVariable`
   monitors its target readback; a composite
   :class:`~geecs_schemas.scan_variables.PseudoScanVariable` monitors
   *every* component readback (``subscribe_many``) and renders them
   compactly.  Sets go through ``Submitter.move_variable`` — the engine's
   manual-move seam (GeecsBluesky ≥ 0.48.0), carrying scan-identical
   completion semantics (motor poll, confirm poll, pseudo fan-out with a
   fresh relative baseline per move).
2. **A raw ``device:variable`` string**: exactly the historical panel —
   readback monitor + a plain gateway ``:SP`` put through the
   :class:`~geecs_console.services.device_panel.DevicePanelBackend`
   (no engine required).

The R3 axis combos auto-select the panel: picking a scan variable for a
scan re-points the panel at it (the legacy scanner behavior), composites
included.

Controller rules (the #534 pattern — mirrors ``NowPanelController`` /
``ActionsMenuController``): a plain ``QObject`` with **no Qt parent**
(the window holds the only Python reference; parenting would create the
window↔controller cycle that defers C++ teardown to the cyclic GC — a
segfault under offscreen pytest), every dependency constructor-injected,
blocking work on the controller's own :class:`BackgroundResult` worker
(issue #510: the daemon thread emits on the worker, never the window),
monitor values delivered through the controller's **queued** ``value_ready``
signal, and an idempotent :meth:`dispose` (called from the window's
``closeEvent``) that unsubscribes, disconnects, and severs every
controller→window reference.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtWidgets import QComboBox, QLabel, QLineEdit, QPushButton

from geecs_console.services.background import BackgroundResult
from geecs_console.services.device_panel import (
    DevicePanelBackend,
    format_readback,
    format_target_readbacks,
    parse_device_variable,
    parse_set_value,
)
from geecs_console.services.device_completions import (
    CompletionsProvider,
    EmptyCompletions,
)

logger = logging.getLogger(__name__)


def _inert() -> Any:
    """Replacement closure installed by :meth:`dispose` (returns nothing)."""
    return None


class MovablePanelController(QObject):
    """Catalog-aware movable panel: selection, readback(s), manual moves.

    Parameters
    ----------
    device_combo, readback_label, set_field, set_button :
        The R7 widgets (stay attributes on the window — tests and tooltips
        address them there; this controller is the only writer).
    backend :
        The :class:`DevicePanelBackend` (readback monitors + raw sets).
    current_experiment :
        Zero-arg callable returning the selected experiment name.
    ensure_submitter :
        Zero-arg callable returning the engine ``Submitter`` (built lazily)
        or ``None`` when the engine is unavailable — catalog moves need it;
        raw ``device:variable`` sets do not.
    catalog_specs :
        Zero-arg callable returning ``{name: ScanVariableSpec}`` for the
        current experiment (empty offline).
    completions_provider :
        One-arg callable ``experiment -> CompletionsProvider`` (the window
        resolves the injected/test factory).
    report :
        One-arg callable for status-bar + log-tail messages.
    """

    #: (target_index, value) from the CA monitor thread — connected queued.
    value_ready = Signal(int, object)

    def __init__(
        self,
        *,
        device_combo: QComboBox,
        readback_label: QLabel,
        set_field: QLineEdit,
        set_button: QPushButton,
        backend: DevicePanelBackend,
        current_experiment: Callable[[], str],
        ensure_submitter: Callable[[], Any],
        catalog_specs: Callable[[], dict],
        completions_provider: Callable[[str], CompletionsProvider],
        report: Callable[[str], None],
    ) -> None:
        super().__init__()  # no Qt parent — the window holds the only ref
        self._combo = device_combo
        self._readback_label = readback_label
        self._set_field = set_field
        self._set_button = set_button
        self._backend = backend
        self._current_experiment = current_experiment
        self._ensure_submitter = ensure_submitter
        self._catalog_specs = catalog_specs
        self._completions_provider = completions_provider
        self._report = report

        #: The committed selection's monitored targets and their values.
        self._targets: list[tuple[str, str]] = []
        self._values: list[Any] = []
        #: Catalog name when the selection is a catalog variable, else None.
        self._catalog_name: Optional[str] = None
        self._set_in_flight = False
        self._disposed = False

        self.value_ready.connect(self._apply_value, Qt.ConnectionType.QueuedConnection)
        self._set_worker = BackgroundResult()
        self._set_worker.result_ready.connect(
            self._apply_set_result, Qt.ConnectionType.QueuedConnection
        )
        self._completions_worker = BackgroundResult()
        self._completions_worker.result_ready.connect(
            self._apply_completions, Qt.ConnectionType.QueuedConnection
        )

        # Selection commits (dropdown pick / Enter / focus leave)
        # resubscribe; per-keystroke edits only regate the Set button —
        # never churn CA monitors while the operator types.
        self._set_button.clicked.connect(self._on_set_clicked)
        self._set_field.returnPressed.connect(self._on_set_clicked)
        self._set_field.textChanged.connect(self.refresh_set_enabled)
        self._combo.editTextChanged.connect(self.refresh_set_enabled)
        self._combo.activated.connect(self.resubscribe)
        line_edit = self._combo.lineEdit()
        if line_edit is not None:
            line_edit.editingFinished.connect(self.resubscribe)

        self.refresh_set_enabled()

    # ------------------------------------------------------------------
    # Selection & readback
    # ------------------------------------------------------------------

    def _resolve_selection(
        self, text: str
    ) -> tuple[Optional[str], list[tuple[str, str]]]:
        """Resolve combo text → ``(catalog_name, monitored_targets)``.

        Catalog names win over the raw ``device:variable`` parse (a name
        never contains a colon, so the two cannot collide in practice).
        An unresolvable text returns ``(None, [])``.
        """
        text = text.strip()
        if not text:
            return None, []
        specs = self._catalog_specs() or {}
        spec = specs.get(text)
        if spec is not None:
            if getattr(spec, "kind", None) == "pseudo":
                targets = [
                    parse_device_variable(component.target) or ("", "")
                    for component in spec.targets
                ]
                return text, [t for t in targets if t != ("", "")]
            parsed = parse_device_variable(getattr(spec, "target", ""))
            return text, [parsed] if parsed else []
        parsed = parse_device_variable(text)
        return None, [parsed] if parsed else []

    def resubscribe(self, *_args: object) -> None:
        """Re-point the readback monitor(s) at the combo's current selection.

        Closes the previous monitors first (the backend drops straggler
        callbacks from them) and resets the label to the em-dash until new
        values land.  An unresolvable selection leaves the panel
        unsubscribed with a visible format hint (a silent no-op looked
        like a dead panel — user report).
        """
        try:
            self._backend.unsubscribe()
        except Exception as exc:  # noqa: BLE001 — teardown must not break the GUI
            self._report(f"Readback unsubscribe failed: {exc}")
        self._readback_label.setText("—")
        self.refresh_set_enabled()
        text = self._combo.currentText()
        self._catalog_name, self._targets = self._resolve_selection(text)
        self._values = [None] * len(self._targets)
        if not self._targets:
            if text.strip():
                self._warn_format()
            return
        try:
            subscribe_many = getattr(self._backend, "subscribe_many", None)
            if subscribe_many is not None:
                subscribe_many(
                    self._current_experiment(),
                    list(self._targets),
                    self.value_ready.emit,
                )
            elif len(self._targets) == 1:
                # Older/injected single-monitor backends (duck-typed seam).
                device, variable = self._targets[0]
                self._backend.subscribe(
                    self._current_experiment(),
                    device,
                    variable,
                    lambda value: self.value_ready.emit(0, value),
                )
            else:
                self._report(
                    "This device-panel backend cannot monitor a composite "
                    "(no subscribe_many)"
                )
        except Exception as exc:  # noqa: BLE001 — surface, don't crash
            self._report(f"Readback subscribe failed for {text.strip()}: {exc}")

    def select_from_scan_combo(self, text: str) -> None:
        """Auto-select *text* when the operator picks a scan variable (R3).

        The legacy scanner behavior: the variable chosen for a scan is the
        one the operator wants on the panel — composites included.  A
        blank text (mid-repopulation) is ignored; re-selecting the current
        selection is a no-op (no CA-monitor churn).
        """
        text = text.strip()
        if not text or text == self._combo.currentText().strip():
            return
        name, targets = self._resolve_selection(text)
        if name is None and not targets:
            # Unresolvable here (e.g. no catalog offline) — leave the panel
            # alone rather than hijack it with a dead selection.
            return
        line_edit = self._combo.lineEdit()
        self._combo.blockSignals(True)
        if line_edit is not None:
            line_edit.setText(text)
        else:  # pragma: no cover — the combo is editable by construction
            self._combo.setCurrentText(text)
        self._combo.blockSignals(False)
        self.resubscribe()

    @Slot(int, object)
    def _apply_value(self, index: int, value: object) -> None:
        """Render one monitor value (GUI-thread slot, delivered queued)."""
        if not (0 <= index < len(self._values)):
            return  # straggler from a retired subscription shape
        self._values[index] = value
        if len(self._targets) == 1:
            self._readback_label.setText(format_readback(value))
        else:
            self._readback_label.setText(
                format_target_readbacks(self._targets, self._values)
            )

    # ------------------------------------------------------------------
    # Set / move
    # ------------------------------------------------------------------

    def refresh_set_enabled(self, *_args: object) -> None:
        """Gate the Set button: resolvable selection, a value, nothing in flight."""
        name, targets = self._resolve_selection(self._combo.currentText())
        ready = (
            not self._set_in_flight
            and (name is not None or bool(targets))
            and bool(self._set_field.text().strip())
        )
        self._set_button.setEnabled(ready)

    def _warn_format(self) -> None:
        """Status-bar hint for an unresolvable R7 selection."""
        self._report(
            "Device format: DeviceName:Variable Name — or a catalog scan-variable name"
        )

    def _on_set_clicked(self) -> None:
        """Dispatch the blocking set/move through the set worker.

        Catalog selections go through ``Submitter.move_variable`` (the
        engine seam — scan-identical completion semantics; refusals like
        ``"scan in progress — move not started"`` surface verbatim).  Raw
        ``device:variable`` selections keep the historical direct gateway
        put.  Either way the blocking call runs on a short-lived daemon
        thread via :class:`BackgroundResult` (issue #510) and the
        ``(ok, message)`` outcome returns queued to
        :meth:`_apply_set_result`.
        """
        text = self._set_field.text().strip()
        if not text or self._set_in_flight:
            return
        name, targets = self._resolve_selection(self._combo.currentText())
        if name is None and not targets:
            if self._combo.currentText().strip():
                self._warn_format()
            return
        value = parse_set_value(text)

        if name is not None:
            submitter = self._ensure_submitter()
            if submitter is None:
                self._report(f"Cannot move {name!r}: scan engine unavailable")
                return

            def run_move(name: str = name, value: Any = value) -> tuple[bool, str]:
                try:
                    result = submitter.move_variable(name, value)
                except Exception as exc:  # noqa: BLE001 — any failure is a report
                    return (False, f"Move {name} failed: {exc}")
                targets_text = ", ".join(
                    f"{target}={commanded:g}"
                    for target, commanded in (result.get("targets") or {}).items()
                )
                return (True, f"Moved {name} = {result.get('value')} ({targets_text})")

            job = run_move
        else:
            device, variable = targets[0]
            backend = self._backend
            experiment = self._current_experiment()

            def run_set(
                device: str = device, variable: str = variable, value: Any = value
            ) -> tuple[bool, str]:
                try:
                    backend.set(experiment, device, variable, value)
                except Exception as exc:  # noqa: BLE001 — any failure is a report
                    return (False, f"Set {device}:{variable} failed: {exc}")
                return (True, f"Set {device}:{variable} = {value}")

            job = run_set

        self._set_in_flight = True
        self.refresh_set_enabled()
        self._set_worker.run_async(job, name="console-movable-set")

    @Slot(object)
    def _apply_set_result(self, payload: object) -> None:
        """Report one finished set/move and re-arm the button (GUI slot)."""
        _ok, message = payload
        self._set_in_flight = False
        self._report(message)
        self.refresh_set_enabled()

    # ------------------------------------------------------------------
    # Completions
    # ------------------------------------------------------------------

    def start_completions_fetch(self) -> None:
        """Fetch the dropdown entries off the GUI thread.

        The dropdown lists the experiment's **catalog scan-variable names
        first** (composites are first-class selections here), then the DB
        ``device:variable`` completions.  One blocking provider call on the
        worker; no experiment (or the :class:`EmptyCompletions` test
        default) answers inline with the catalog names only.
        """
        experiment = self._current_experiment()
        catalog_names = sorted(self._catalog_specs() or {})
        provider = (
            self._completions_provider(experiment) if experiment else EmptyCompletions()
        )
        if isinstance(provider, EmptyCompletions):
            self._apply_completions((experiment, catalog_names))
            return

        def fetch() -> tuple[str, list[str]]:
            try:
                mapping = provider.device_variables()
            except Exception as exc:  # noqa: BLE001 — completions are best-effort
                logger.info("device completions failed: %s", exc)
                mapping = {}
            words = sorted(
                f"{device}:{variable}"
                for device, variables in (mapping or {}).items()
                for variable in variables
            )
            return (experiment, catalog_names + words)

        self._completions_worker.run_async(fetch, name="console-device-completions")

    @Slot(object)
    def _apply_completions(self, payload: object) -> None:
        """Populate the combo's dropdown (GUI-thread slot, delivered queued).

        A result tagged with an experiment that is no longer selected is
        dropped (a stale fetch racing an experiment change); typed text
        survives repopulation.
        """
        experiment, words = payload
        if experiment != self._current_experiment():
            return
        current = self._combo.currentText()
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItems(list(words))
        self._combo.setCurrentIndex(-1)
        line_edit = self._combo.lineEdit()
        if line_edit is not None:
            line_edit.setText(current)
        self._combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_experiment_changed(self) -> None:
        """Re-point the monitors and refresh completions (new PV prefix)."""
        self.resubscribe()
        self.start_completions_fetch()

    def dispose(self) -> None:
        """Idempotent teardown: unsubscribe, disconnect, sever window refs."""
        if self._disposed:
            return
        self._disposed = True
        try:
            self._backend.unsubscribe()
        except Exception:  # noqa: BLE001 — teardown must not raise
            pass
        for worker, slot in (
            (self._set_worker, self._apply_set_result),
            (self._completions_worker, self._apply_completions),
        ):
            try:
                worker.result_ready.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        try:
            self.value_ready.disconnect(self._apply_value)
        except (RuntimeError, TypeError):
            pass
        # Sever every controller → window edge (the #534 lifetime rule).
        self._current_experiment = _inert
        self._ensure_submitter = _inert
        self._catalog_specs = _inert
        self._report = lambda _message: None
