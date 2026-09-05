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
- The completions scaffold: one fetch on the blessed
  ``services/background.py::BackgroundResult`` worker (the daemon thread
  never emits toward the dialog — the result hops onto the GUI thread
  first; #787), with the ``EmptyCompletions`` inline fast-path (no thread
  for a constant — offline construction stays thread-free), the
  normalized ``{device: [variable, ...]}`` word lists in
  :attr:`_device_vars`, the public ``completions_applied`` flag tests
  wait on, and the one-shot worker disconnect at close.
- The unified unsaved-changes flow: :meth:`_prompt_unsaved`
  (Save/Discard/Cancel, three-way string result) resolved by
  :meth:`_resolve_unsaved` through the per-editor hooks
  :meth:`_has_unsaved` / :meth:`_save_unsaved` / :meth:`_discard_unsaved`,
  wired into both close paths (:meth:`closeEvent` and Esc via
  :meth:`reject`).
- One name-prompt convention (:meth:`_prompt_name`: stripped text or
  ``None`` on cancel) and one Yes/No confirm (:meth:`_confirm`).
- The save-time **target check** (#772): :func:`target_problem` and the
  :meth:`_target_problem` convenience — does a ``device`` / ``variable``
  pair name something in the fetched listing?  Free text + completer
  stays (the editors must work with the DB unreachable); the gate is at
  Save, where an unknown name blocks with a near-miss hint and an empty
  listing downgrades to :data:`UNCHECKED_TARGETS_NOTE`.

The prompt methods are plain bound methods so tests can monkeypatch them
per instance (``editor._prompt_unsaved = lambda: "discard"``) — the
pattern the editor test suites already use.
"""

from __future__ import annotations

import difflib
import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QFile, Qt, Slot
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

from geecs_console.services.background import BackgroundResult, disconnect_quietly
from geecs_console.services.device_completions import (
    CompletionsProvider,
    EmptyCompletions,
)

logger = logging.getLogger(__name__)

#: Appended to a successful save's message when the device/variable names
#: could not be checked (no completions listing: offline, DB down, fetch
#: failed).  The save goes through — the editors must stay usable off the
#: lab network — but the operator is told what was skipped.
UNCHECKED_TARGETS_NOTE = (
    "device and variable names were not checked against the database "
    "(completions unavailable)"
)

#: ``difflib`` similarity floor for a "did you mean" hint.  The live case
#: (#772) was ``Position.Axis1`` vs ``Position.Axis 1`` — ratio 0.97; the
#: default 0.6 still rejects unrelated names.
_NEAR_MISS_CUTOFF = 0.6


def resolve_device(
    device_vars: Mapping[str, Sequence[str]], device: str
) -> Optional[str]:
    """Return the listing's key for *device*, or ``None`` when unknown.

    An exact key wins; otherwise a *unique* case-insensitive match (the
    completer already tolerates case, so the check must too — see the
    editors' ``_refresh_variable_model``).

    Parameters
    ----------
    device_vars : mapping
        The fetched ``{device: [variable, ...]}`` listing.
    device : str
        The device text as typed (already stripped).

    Returns
    -------
    str or None
        The listing's spelling of the device, or ``None``.
    """
    if device in device_vars:
        return device
    lowered = device.lower()
    matches = [known for known in device_vars if known.lower() == lowered]
    return matches[0] if len(matches) == 1 else None


def near_miss(text: str, candidates: Iterable[str]) -> Optional[str]:
    """The candidate closest to *text* (case-insensitive), or ``None``.

    Parameters
    ----------
    text : str
        The unknown name.
    candidates : iterable of str
        The known names to compare against.

    Returns
    -------
    str or None
        The best candidate in its own spelling when one is close enough
        (:data:`_NEAR_MISS_CUTOFF`), else ``None``.
    """
    by_lower = {candidate.lower(): candidate for candidate in candidates}
    matches = difflib.get_close_matches(
        text.lower(), list(by_lower), n=1, cutoff=_NEAR_MISS_CUTOFF
    )
    return by_lower[matches[0]] if matches else None


def target_problem(
    device_vars: Mapping[str, Sequence[str]],
    device: str,
    variable: Optional[str] = None,
) -> Optional[str]:
    """Why ``device`` (and ``variable``) name nothing in the listing, or ``None``.

    The save-time check behind #772: the editors' device/variable fields are
    free text with a *suggesting* completer, so a near miss like
    ``Position.Axis1`` for ``Position.Axis 1`` is accepted silently and only
    fails later — as a 20 s ophyd connect timeout on the worker that never
    says "no such variable".  This names the problem at Save.

    Parameters
    ----------
    device_vars : mapping
        The fetched ``{device: [variable, ...]}`` listing.  **Empty means
        unchecked**: with nothing to compare against the result is ``None``
        and the caller downgrades to :data:`UNCHECKED_TARGETS_NOTE`.
    device : str
        The device text (stripped).
    variable : str, optional
        The variable text (stripped); ``None`` checks the device only (a
        save-set entry names a device, its scalars name variables).

    Returns
    -------
    str or None
        One sentence naming the unknown name, with a "did you mean" hint
        when a close spelling exists; ``None`` when the target checks out
        or the listing is empty.
    """
    if not device_vars:
        return None
    known = resolve_device(device_vars, device)
    if known is None:
        hint = near_miss(device, device_vars)
        suffix = f" — did you mean {hint!r}?" if hint else ""
        return f"unknown device {device!r}{suffix}"
    if variable is None:
        return None
    variables = device_vars[known]
    if variable in variables:
        return None
    hint = near_miss(variable, variables)
    suffix = f" — did you mean {hint!r}?" if hint else ""
    return f"{variable!r} is not a variable of {known!r}{suffix}"


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
        """Install the completions state and the fetch worker.

        Call early in ``__init__`` (before any fetch).  The worker's
        ``result_ready`` is emitted on the GUI thread (the
        ``BackgroundResult`` hop) and connected queued, so the mapping
        lands in :meth:`_apply_completions` one event-loop turn later —
        never touching dialog state from the fetch thread.

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
        #: Whether a listing was ever expected: a dialog built on the
        #: :class:`EmptyCompletions` default (no experiment, tests) has
        #: nothing to check *by design* and gets no "unchecked" note; a
        #: real provider that came back empty (offline, DB down) does.
        self._completions_expected = not isinstance(self._completions, EmptyCompletions)
        #: True once the (async) completions fetch has been delivered on
        #: the GUI thread — tests wait on this instead of sleeping.
        self.completions_applied = False
        #: Guards the one-shot worker disconnect in :meth:`closeEvent`.
        self._completions_disconnected = False
        self._completions_worker = BackgroundResult()
        self._completions_worker.result_ready.connect(
            self._apply_completions, Qt.ConnectionType.QueuedConnection
        )

    def _start_completions_fetch(self) -> None:
        """Fetch completions on the worker's short-lived daemon thread.

        The :class:`EmptyCompletions` default is answered inline (no
        thread to spawn for a constant) — offline construction stays
        thread-free.  The fetch callable never raises (a failing provider
        yields ``{}``) — ``BackgroundResult`` swallows a raise without
        emitting, which would leave ``completions_applied`` False forever.
        Since 0.30.0 (#787) this rides the blessed worker instead of a
        dialog-owned signal emitted from the thread; a dialog closed by
        Esc (``reject``, no ``QCloseEvent``) before the fetch lands is
        therefore safe too — the result is emitted on the GUI thread, at
        a dialog that is hidden, not half-destroyed.
        """
        if isinstance(self._completions, EmptyCompletions):
            self.completions_applied = True
            return
        provider = self._completions

        def fetch() -> dict:
            """Run the provider's one blocking call off the GUI thread."""
            try:
                mapping = provider.device_variables()
            except Exception as exc:  # noqa: BLE001 — providers should not raise
                logger.info("completions fetch failed: %s", exc)
                mapping = {}
            return mapping or {}

        self._completions_worker.run_async(fetch, name=self.COMPLETIONS_THREAD_NAME)

    @Slot(object)
    def _apply_completions(self, mapping: object) -> None:
        """Store the fetched word lists (GUI-thread slot, delivered queued).

        Parameters
        ----------
        mapping : dict
            The provider result delivered by the worker's ``result_ready``.
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

    @property
    def completions_available(self) -> bool:
        """Whether a non-empty listing arrived (so targets can be checked)."""
        return bool(self._device_vars)

    @property
    def targets_unchecked(self) -> bool:
        """Whether a save should carry :data:`UNCHECKED_TARGETS_NOTE`.

        True only when a listing was expected (a real provider) and none
        arrived — the offline / DB-down case.  The ``EmptyCompletions``
        default never expected one, so it stays quiet.
        """
        return self._completions_expected and not self.completions_available

    def _target_problem(
        self, device: str, variable: Optional[str] = None
    ) -> Optional[str]:
        """:func:`target_problem` against this dialog's fetched listing.

        Parameters
        ----------
        device : str
            The device text (stripped).
        variable : str, optional
            The variable text (stripped); ``None`` checks the device only.

        Returns
        -------
        str or None
            The problem sentence, or ``None`` (known, or nothing to check).
        """
        return target_problem(self._device_vars, device, variable)

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
        """Close via QDialog (one prompt), then detach the completions worker.

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
        # A still-running completions fetch must not land on a dialog being
        # torn down.  Once only: closeEvent can run twice (explicit close +
        # owner teardown) and a second disconnect warns.
        if not self._completions_disconnected:
            self._completions_disconnected = True
            disconnect_quietly(
                self._completions_worker.result_ready, self._apply_completions
            )
