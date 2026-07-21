"""The Actions menu: arming switch, per-plan entries, and their dialogs.

Owns everything behind the menu-bar "Actions" entry so the main window
stays a composition root: the "Enable action execution" arming switch,
the per-plan menu entries (fetched off the GUI thread, stale results
dropped by experiment tag), and the non-modal :class:`ActionRunDialog`
windows those entries open.

The arming switch is deliberately **not** persisted (unlike the
Preferences toggles): a fresh console session must never start with
action execution armed.  Do not "fix" this by adding it to
``ConsoleSettings``.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from PySide6.QtCore import QObject, Qt, Slot
from PySide6.QtWidgets import QMenu, QWidget

from geecs_console.app.action_dialog import ActionRunDialog
from geecs_console.services.action_library_store import ActionLibraryStore
from geecs_console.services.background import BackgroundResult
from geecs_console.submission import Submitter

logger = logging.getLogger(__name__)


class ActionsMenuController(QObject):
    """Populate and run one Actions menu on behalf of the main window.

    A ``QObject`` with **no Qt parent**, kept alive only by the window's
    Python reference (``self._actions``) — the same lifetime rule as the
    ``BackgroundResult`` workers.

    Lifetime constraint: holding the window and its bound methods here
    creates a Python reference cycle (window → controller → window) that
    refcounting cannot free — the dead window's whole C++ tree would
    then be torn down whenever the *cyclic* GC happens to run, including
    mid-event-processing of a later window (a segfault under offscreen
    pytest).  The window's ``closeEvent`` therefore calls
    :meth:`dispose`, which severs every controller → window edge and
    restores deterministic refcount destruction.

    Parameters
    ----------
    menu : QMenu
        The already-created menu-bar entry (the window keeps it in
        ``self._menus``); this controller owns its contents.
    window : QWidget
        The main window — Qt parent and the dialogs' parent.
    store : ActionLibraryStore
        The plan-name source; the window re-points it on experiment
        change (``set_experiment``), so the reference stays valid.
    current_experiment : callable
        Returns the currently selected experiment name ("" for none).
    ensure_submitter : callable
        Returns the engine :class:`Submitter` (or ``None`` when the
        engine is unavailable — reported to the operator by the window).
    report : callable
        One operator-facing status line (status bar + log tail).
    """

    def __init__(
        self,
        menu: QMenu,
        *,
        window: QWidget,
        store: ActionLibraryStore,
        current_experiment: Callable[[], str],
        ensure_submitter: Callable[[], Optional[Submitter]],
        report: Callable[[str], None],
    ) -> None:
        super().__init__()
        self._menu = menu
        self._window = window
        self._store = store
        self._current_experiment = current_experiment
        self._ensure_submitter = ensure_submitter
        self._report = report

        self.enable_action = menu.addAction("Enable action execution")
        self.enable_action.setCheckable(True)
        self.enable_action.setChecked(False)
        self.enable_action.toggled.connect(self._on_enable_actions_toggled)
        menu.addSeparator()
        #: The per-plan entries below the arming switch (rebuilt per fetch).
        self._plan_actions: list = []
        #: Open ActionRunDialogs — referenced here so GC cannot take them.
        self.open_dialogs: list = []

        self._worker = BackgroundResult()
        self._worker.result_ready.connect(
            self._apply_action_names, Qt.ConnectionType.QueuedConnection
        )
        #: Experiment tag of an in-flight fetch, or None.  One fetch at a
        #: time: startup can request twice back-to-back (the restored
        #: experiment fires the experiment-changed path just before the
        #: explicit construction-time call), and two concurrent fetch
        #: threads race the lazy ``geecs_bluesky`` import inside the
        #: store's configs-root resolution — a native-extension init that
        #: aborts the process when raced.  Dedup also spares the configs
        #: mount a redundant YAML parse.
        self._fetch_inflight: Optional[str] = None
        self._set_action_names([])

    # ------------------------------------------------------------------
    # Plan-name fetch (off the GUI thread, stale-dropped by experiment)
    # ------------------------------------------------------------------

    def start_fetch(self) -> None:
        """Fetch the experiment's action-plan names off the GUI thread.

        ``list_names`` parses the library YAML on a possibly slow configs
        mount, so it runs on a short-lived daemon thread (via the
        :class:`BackgroundResult` worker), tagged with the experiment and
        delivered queued to :meth:`_apply_action_names`.  No experiment
        answers inline with an empty list, spawning no thread.
        """
        experiment = self._current_experiment()
        if not experiment:
            self._apply_action_names((experiment, []))
            return
        if self._fetch_inflight == experiment:
            return
        self._fetch_inflight = experiment
        store = self._store

        def fetch() -> tuple[str, list[str]]:
            # Never raise: BackgroundResult swallows exceptions without
            # emitting, which would leave the in-flight tag stuck.
            try:
                return (experiment, store.list_names())
            except Exception as exc:  # noqa: BLE001 — deliver the empty answer
                logger.info("action-name fetch failed: %s", exc)
                return (experiment, [])

        self._worker.run_async(fetch, name="console-action-names")

    @Slot(object)
    def _apply_action_names(self, payload: object) -> None:
        """Rebuild the menu entries (GUI-thread slot, delivered queued).

        Parameters
        ----------
        payload : tuple
            ``(experiment, [plan_name, ...])``; a result tagged with an
            experiment that is no longer selected is dropped (a stale
            fetch racing an experiment change).
        """
        experiment, names = payload
        if self._fetch_inflight == experiment:
            self._fetch_inflight = None
        if experiment != self._current_experiment():
            return
        self._set_action_names(list(names))

    def _set_action_names(self, names: list[str]) -> None:
        """Replace the per-plan menu entries below the arming switch.

        Parameters
        ----------
        names : list of str
            The experiment's action-plan names; empty (or offline)
            renders one disabled ``(no actions)`` entry.
        """
        for action in self._plan_actions:
            self._menu.removeAction(action)
            action.deleteLater()
        self._plan_actions = []
        if not names:
            placeholder = self._menu.addAction("(no actions)")
            placeholder.setEnabled(False)
            self._plan_actions.append(placeholder)
            return
        for name in names:
            action = self._menu.addAction(name)
            action.triggered.connect(
                lambda checked=False, plan=name: self._on_action_plan(plan)
            )
            self._plan_actions.append(action)

    # ------------------------------------------------------------------
    # Arming + dialogs
    # ------------------------------------------------------------------

    def _on_enable_actions_toggled(self, checked: bool) -> None:
        """Push the arming state into every open action dialog.

        Parameters
        ----------
        checked : bool
            The "Enable action execution" state.
        """
        self.open_dialogs = [d for d in self.open_dialogs if d.isVisible()]
        for dialog in self.open_dialogs:
            dialog.set_execution_enabled(checked)

    def _on_action_plan(self, name: str) -> None:
        """Open the Run/Preview dialog for action plan *name*.

        The dialog is shown non-modally and kept in :attr:`open_dialogs`
        (the PySide6 GC hazard).  Preview is always available; Run is
        gated by the arming switch.

        Parameters
        ----------
        name : str
            The action-plan name (a menu entry).
        """
        submitter = self._ensure_submitter()
        if submitter is None:
            return
        dialog = ActionRunDialog(
            self._window,
            name,
            describe=submitter.describe_action,
            run=submitter.run_action,
            execution_enabled=self.enable_action.isChecked(),
            report=self._report,
            request_during_scan=submitter.request_action_during_scan,
            is_scanning=submitter.is_scanning_active,
        )
        self.open_dialogs = [d for d in self.open_dialogs if d.isVisible()]
        self.open_dialogs.append(dialog)
        dialog.show()

    def set_scanning(self, scanning: bool) -> None:
        """Push live scan state into every open dialog.

        The window's scan-lifecycle hub calls this so each dialog's Run
        button flips between "Run" and "Pause scan & run".

        Parameters
        ----------
        scanning : bool
            Whether a scan is currently active (non-terminal, non-idle).
        """
        self.open_dialogs = [d for d in self.open_dialogs if d.isVisible()]
        for dialog in self.open_dialogs:
            dialog.set_scanning(scanning)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def disconnect_worker(self) -> None:
        """Detach the fetch worker so a straggling result lands nowhere.

        Safe to call more than once.
        """
        try:
            self._worker.result_ready.disconnect(self._apply_action_names)
        except (RuntimeError, TypeError):
            pass

    def dispose(self) -> None:
        """Sever every controller → window reference (see the class docstring).

        Called from the window's ``closeEvent``; idempotent.  After this
        the controller renders nothing and answers no fetches — it only
        awaits collection alongside the window that owns it.
        """
        self.disconnect_worker()
        self._window = None
        self._store = None
        self._current_experiment = lambda: ""
        self._ensure_submitter = lambda: None
        self._report = lambda message: None
        self.open_dialogs = []
        self._plan_actions = []
