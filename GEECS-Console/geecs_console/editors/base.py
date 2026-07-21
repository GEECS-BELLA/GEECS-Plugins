"""Shared scaffolding for the config-editor dialogs.

The one home for the dialog machinery all four editors (save sets, scan
variables, trigger profiles, action plans) share; each editor keeps only
its store binding, its widgets, and its dialect rules.  A new editor
should subclass :class:`ConfigEditorDialog` rather than re-growing any
of this.

What lives here:

- ``.ui`` loading (:meth:`_load_ui` from :attr:`UI_PATH`, optional
  :attr:`INITIAL_SIZE`) and the fail-loud :meth:`_child` lookup.
- The Enter-key guard, both mechanisms applied uniformly: no
  default/auto-default buttons (:meth:`_guard_enter_keys`) and a
  :meth:`keyPressEvent` that swallows Return/Enter.
- The completions scaffold: one queued ``completions_ready`` signal, a
  daemon-thread fetch with the ``EmptyCompletions`` inline fast-path (no
  thread for a constant — offline construction stays thread-free), the
  normalized ``{device: [variable, ...]}`` word lists in
  :attr:`_device_vars`, the public ``completions_applied`` flag tests
  wait on, and the one-shot signal disconnect at close.
- The unified unsaved-changes flow: :meth:`_prompt_unsaved`
  (Save/Discard/Cancel, three-way string result) resolved by
  :meth:`_resolve_unsaved` through the per-editor hooks
  :meth:`_has_unsaved` / :meth:`_save_unsaved` / :meth:`_discard_unsaved`,
  wired into both close paths (:meth:`closeEvent` and Esc via
  :meth:`reject`).
- One name-prompt convention (:meth:`_prompt_name`: stripped text or
  ``None`` on cancel) and one Yes/No confirm (:meth:`_confirm`).

The prompt methods are plain bound methods so tests can monkeypatch them
per instance (``editor._prompt_unsaved = lambda: "discard"``) — the
pattern the editor test suites already use.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QFile, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent, QKeyEvent
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QDialog,
    QInputDialog,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from geecs_console.services.device_completions import (
    CompletionsProvider,
    EmptyCompletions,
)

logger = logging.getLogger(__name__)


class ConfigEditorDialog(QDialog):
    """Base dialog for the per-experiment config editors.

    Subclasses set :attr:`UI_PATH` (and optionally :attr:`INITIAL_SIZE`
    and :attr:`COMPLETIONS_THREAD_NAME`), call :meth:`_init_completions`
    early in ``__init__`` and :meth:`_start_completions_fetch` once their
    widgets exist, and implement the small hooks documented on
    :meth:`_has_unsaved`, :meth:`_save_unsaved`, :meth:`_discard_unsaved`,
    :meth:`_unsaved_prompt_text`, and :meth:`_install_completions`.
    """

    #: The hand-authored ``.ui`` file this dialog loads (subclass-set).
    UI_PATH: Path
    #: Optional ``(width, height)`` applied after the ``.ui`` loads.
    INITIAL_SIZE: Optional[tuple[int, int]] = None
    #: Name of the completions-fetch daemon thread (subclass-set for
    #: recognizable thread dumps).
    COMPLETIONS_THREAD_NAME: str = "editor-completions-fetch"

    #: One ``{device: [variable, ...]}`` mapping per finished provider
    #: call (emitted from the fetch daemon thread; delivered queued to
    #: :meth:`_apply_completions`).
    completions_ready = Signal(object)

    # ------------------------------------------------------------------
    # UI loading
    # ------------------------------------------------------------------

    def _load_ui(self) -> None:
        """Load :attr:`UI_PATH` as this dialog's only child."""
        loader = QUiLoader()
        ui_file = QFile(str(self.UI_PATH))
        ui_file.open(QFile.OpenModeFlag.ReadOnly)
        try:
            self._ui: QWidget = loader.load(ui_file, self)
        finally:
            ui_file.close()
        if self._ui is None:
            raise RuntimeError(f"Failed to load {self.UI_PATH}: {loader.errorString()}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._ui)
        if self.INITIAL_SIZE is not None:
            self.resize(*self.INITIAL_SIZE)

    def _child(self, cls: type, name: str):
        """Return the named child widget, failing loudly when missing."""
        widget = self._ui.findChild(cls, name)
        if widget is None:
            raise LookupError(f"{name!r} ({cls.__name__}) not found in {self.UI_PATH}")
        return widget

    # ------------------------------------------------------------------
    # Enter must never accept (close) the dialog
    # ------------------------------------------------------------------

    def _guard_enter_keys(self) -> None:
        """Strip default/auto-default from every button (call after binding).

        Every ``QPushButton`` in a ``QDialog`` is auto-default by default,
        so pressing Enter in any line edit would "click" the first button.
        :meth:`keyPressEvent` swallows whatever this loop cannot reach.
        """
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 (Qt override)
        """Swallow Return/Enter so typing in a field never closes the dialog."""
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            event.accept()
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Completions (injectable provider, fetched off the GUI thread)
    # ------------------------------------------------------------------

    def _init_completions(self, provider: Optional[CompletionsProvider]) -> None:
        """Install the completions state and the queued signal connection.

        Call early in ``__init__`` (before any fetch).  The connection is
        forced queued: the fetch emits from a daemon thread, and a direct
        delivery would touch state on the wrong thread.

        Parameters
        ----------
        provider : CompletionsProvider, optional
            The word-list source; ``None`` means
            :class:`~geecs_console.services.device_completions.EmptyCompletions`.
        """
        self._completions: CompletionsProvider = (
            provider if provider is not None else EmptyCompletions()
        )
        #: The fetched completion word lists (empty until the fetch lands).
        self._device_vars: dict[str, list[str]] = {}
        #: True once the (async) completions fetch has been delivered on
        #: the GUI thread — tests wait on this instead of sleeping.
        self.completions_applied = False
        #: Guards the one-shot signal disconnect in :meth:`closeEvent`.
        self._completions_disconnected = False
        self.completions_ready.connect(
            self._apply_completions, Qt.ConnectionType.QueuedConnection
        )

    def _start_completions_fetch(self) -> None:
        """Fetch completions on a short-lived daemon thread (queued delivery).

        The :class:`EmptyCompletions` default is answered inline (no
        thread to spawn for a constant) — offline construction stays
        thread-free.

        Known debt: the daemon thread emits a *dialog-owned* signal
        instead of routing through the blessed
        ``services/background.py::BackgroundResult`` worker, and an
        Esc-closed dialog (``reject`` → ``done()``, no ``QCloseEvent``)
        never runs the close-time disconnect.  Consolidate onto
        ``BackgroundResult`` here, once — not per editor.
        """
        if isinstance(self._completions, EmptyCompletions):
            self.completions_applied = True
            return
        provider = self._completions

        def fetch() -> None:
            """Run the provider's one blocking call off the GUI thread."""
            try:
                mapping = provider.device_variables()
            except Exception as exc:  # noqa: BLE001 — providers should not raise
                logger.info("completions fetch failed: %s", exc)
                mapping = {}
            self.completions_ready.emit(mapping or {})

        threading.Thread(
            target=fetch, name=self.COMPLETIONS_THREAD_NAME, daemon=True
        ).start()

    @Slot(object)
    def _apply_completions(self, mapping: object) -> None:
        """Store the fetched word lists (GUI-thread slot, delivered queued).

        Parameters
        ----------
        mapping : dict
            The provider result delivered by the queued signal.
        """
        self.completions_applied = True
        if not isinstance(mapping, dict):
            return
        self._device_vars = {
            str(device): [str(variable) for variable in variables]
            for device, variables in mapping.items()
        }
        self._install_completions()

    def _install_completions(self) -> None:
        """Hook: point completer models at the fresh :attr:`_device_vars`.

        Default is a no-op — an editor whose cell editors read
        :attr:`_device_vars` lazily at edit-open needs no replumbing.
        """

    # ------------------------------------------------------------------
    # Unsaved-changes flow (one prompt, both close paths)
    # ------------------------------------------------------------------

    def _has_unsaved(self) -> bool:
        """Hook: whether there are unsaved edits to resolve."""
        raise NotImplementedError

    def _save_unsaved(self) -> bool:
        """Hook: run the editor's save; ``True`` when it landed."""
        raise NotImplementedError

    def _discard_unsaved(self) -> None:
        """Hook: side effects of discarding (default: none)."""

    def _unsaved_prompt_text(self) -> str:
        """Hook: the sentence describing what has unsaved changes."""
        return "This editor has unsaved changes."

    def _prompt_unsaved(self) -> str:
        """Ask what to do with unsaved changes.

        Returns
        -------
        str
            ``"save"``, ``"discard"``, or ``"cancel"``.
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Unsaved changes")
        box.setText(self._unsaved_prompt_text())
        save = box.addButton("Save", QMessageBox.ButtonRole.AcceptRole)
        discard = box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
        cancel = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        # The keyboard default is the safe no-op (the convention three of
        # the four pre-base editors already had) — a reflexive Enter must
        # never write a config to disk.
        box.setDefaultButton(cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is save:
            return "save"
        if clicked is discard:
            return "discard"
        return "cancel"

    def _resolve_unsaved(self) -> bool:
        """Settle unsaved edits before leaving the current document.

        Returns
        -------
        bool
            ``True`` when it is safe to proceed (nothing dirty, saved, or
            discarded); ``False`` on cancel or a failed save.
        """
        if not self._has_unsaved():
            return True
        choice = self._prompt_unsaved()
        if choice == "cancel":
            return False
        if choice == "save":
            return self._save_unsaved()
        self._discard_unsaved()
        return True

    # ------------------------------------------------------------------
    # Small modal prompts (instance-monkeypatchable test seams)
    # ------------------------------------------------------------------

    def _prompt_name(self, title: str, label: str, initial: str = "") -> Optional[str]:
        """Ask the operator for one name.

        Parameters
        ----------
        title : str
            Dialog window title.
        label : str
            Prompt label (e.g. ``"Plan name:"``).
        initial : str, optional
            Prefilled text.

        Returns
        -------
        str or None
            The stripped answer (may be ``""``), or ``None`` on cancel.
        """
        name, accepted = QInputDialog.getText(self, title, label, text=initial)
        return name.strip() if accepted else None

    def _confirm(self, title: str, text: str) -> bool:
        """Ask a yes/no question (used for deletes).

        Parameters
        ----------
        title : str
            Dialog title.
        text : str
            The question.

        Returns
        -------
        bool
            ``True`` on Yes.
        """
        answer = QMessageBox.question(
            self,
            title,
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    # ------------------------------------------------------------------
    # Close paths
    # ------------------------------------------------------------------

    def reject(self) -> None:
        """Route Esc through the unsaved-changes prompt.

        Visible closes land here too — see :meth:`closeEvent`.
        """
        if not self._resolve_unsaved():
            return
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt override)
        """Close via QDialog (one prompt), then disconnect the completions signal.

        :meth:`reject` is the **only** prompt owner: ``QDialog::closeEvent``
        routes a *visible* close through the virtual ``reject()`` (ours,
        above) and ignores the close when the dialog survives it — so
        prompting here too would ask twice on Discard in the editors whose
        dirty predicate is computed (draft ≠ snapshot) rather than a flag
        the discard hook clears.  A hidden dialog (test teardown, app
        shutdown) closes without ``reject()`` and therefore never blocks
        on a modal nobody can answer.
        """
        super().closeEvent(event)
        if not event.isAccepted():
            return  # the operator canceled — the dialog stays open
        # A still-running completions fetch must not queue an event onto a
        # dialog being torn down.  Once only: closeEvent can run twice
        # (explicit close + owner teardown) and a second disconnect warns.
        if not self._completions_disconnected:
            self._completions_disconnected = True
            try:
                self.completions_ready.disconnect(self._apply_completions)
            except (RuntimeError, TypeError):
                pass
