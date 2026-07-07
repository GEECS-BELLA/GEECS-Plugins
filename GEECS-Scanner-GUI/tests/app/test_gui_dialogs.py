"""Pin the dialog-request shape shared with the Bluesky pre-flight dialogs.

``BlueskyScanner`` (GeecsBluesky) raises pre-claim operator dialogs through
the same channel the legacy engine uses: a ``DialogRequest`` carried by a
``ScanDialogEvent`` into ``show_device_error_dialog``.  These tests pin the
request fields it relies on (``title`` / ``continue_label`` / ``abort_label``)
and the content resolution the GUI handler performs — without any Qt display
(``_resolve_dialog_content`` is the pure part of ``show_device_error_dialog``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from geecs_scanner.engine.dialog_request import DialogRequest
from geecs_scanner.engine.scan_events import ScanDialogEvent

# ``geecs_scanner.app``'s package __init__ imports the editor windows and so
# PyQt5 (a Windows-only dependency), but ``gui_dialogs`` itself is import-safe
# without Qt (PyQt5 is imported inside the display function).  Load the module
# straight from its file to keep this test hermetic on non-Windows CI.
_GUI_DIALOGS_PATH = (
    Path(__file__).parents[2] / "geecs_scanner" / "app" / "gui_dialogs.py"
)
_spec = importlib.util.spec_from_file_location(
    "_gui_dialogs_under_test", _GUI_DIALOGS_PATH
)
_gui_dialogs = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_gui_dialogs)
_resolve_dialog_content = _gui_dialogs._resolve_dialog_content


def test_plain_request_keeps_legacy_device_error_message() -> None:
    """Requests without custom fields render exactly as before."""
    request = DialogRequest(exc=RuntimeError("boom"))

    title, body, continue_label, abort_label = _resolve_dialog_content(request)

    assert title == "Device Error"
    assert "boom" in body
    assert "Click  Continue  to proceed" in body  # legacy instruction footer
    assert continue_label == "Continue"
    assert abort_label == "Abort"


def test_plain_request_context_is_appended() -> None:
    request = DialogRequest(exc=RuntimeError("boom"), context="var list: a, b")

    _title, body, _continue, _abort = _resolve_dialog_content(request)

    assert body.endswith("var list: a, b")


def test_custom_request_uses_exc_message_and_labels_verbatim() -> None:
    """The Bluesky pre-flight request shape: title + labels + full body."""
    message = (
        "Synchronous device(s) look dead: U_Cam4 (last frame 12 s ago). "
        "Drop them and continue the scan without their data, or abort."
    )
    request = DialogRequest(
        exc=RuntimeError(message),
        title="Dead Contributor Device(s)",
        continue_label="Drop && Continue",
        abort_label="Abort Scan",
    )

    title, body, continue_label, abort_label = _resolve_dialog_content(request)

    assert title == "Dead Contributor Device(s)"
    assert body == message  # verbatim — no device-command boilerplate
    assert continue_label == "Drop && Continue"
    assert abort_label == "Abort Scan"


def test_custom_request_context_still_appended() -> None:
    request = DialogRequest(
        exc=RuntimeError("body"),
        context="extra detail",
        title="Trigger May Be Off",
        continue_label="Start Anyway",
    )

    _title, body, continue_label, abort_label = _resolve_dialog_content(request)

    assert body == "body\n\nextra detail"
    assert continue_label == "Start Anyway"
    assert abort_label == "Abort"  # unset label falls back to the default


def test_scan_dialog_event_carries_the_extended_request() -> None:
    """The event shape the Bluesky backend emits is accepted unchanged."""
    request = DialogRequest(
        exc=RuntimeError("reference looks dead"),
        title="Reference Device Looks Dead",
        continue_label="Try Anyway",
        abort_label="Abort Scan",
    )
    event = ScanDialogEvent(request=request)

    assert event.request is request
    # Response channel unchanged: consumer writes abort[0] + sets the event.
    assert event.request.abort == [False]
    assert not event.request.response_event.is_set()
    event.request.abort[0] = True
    event.request.response_event.set()
    assert request.abort == [True]
    assert request.response_event.is_set()


def test_legacy_positional_construction_is_unchanged() -> None:
    """Existing callers pass (exc, context) positionally — must still work."""
    request = DialogRequest(RuntimeError("x"), "some context")

    assert request.context == "some context"
    assert request.title is None
    assert request.continue_label is None
    assert request.abort_label is None
