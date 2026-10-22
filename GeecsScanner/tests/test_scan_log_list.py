"""The scan-log list: the shared store, its optimistic save, the picker, and the stamp at submit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from geecs_scanner.service.demo import DemoQueueClient
from geecs_scanner.service.errors import ScannerError
from geecs_scanner.service.scan_log_list import (
    DbLoggedVariables,
    FileListStore,
    ScanLogList,
    build_logged_variables,
    normalise,
    version_of,
)


def test_the_list_round_trips_and_a_stale_save_is_a_conflict(
    client: TestClient,
) -> None:
    first = client.get("/api/scan-log-variables").json()
    assert first["variables"] == []
    saved = client.put(
        "/api/scan-log-variables",
        json={
            "variables": ["U_Hexapod:ypos", "U_S1H:current"],
            "version": first["version"],
        },
    )
    assert saved.status_code == 200, saved.text
    assert client.get("/api/scan-log-variables").json() == saved.json()

    # A second page still holding the first version is refused with the current list.
    stale = client.put(
        "/api/scan-log-variables",
        json={"variables": ["U_HP_Daq:Jet pressure"], "version": first["version"]},
    )
    assert stale.status_code == 409
    err = stale.json()["error"]
    assert err["kind"] == "conflict"
    assert err["current"] == saved.json()


def test_names_are_cleaned_and_a_malformed_one_refused() -> None:
    assert normalise([" U_A:x ", "U_A:x", "", "U_B:Position.Axis 1"]) == [
        "U_A:x",
        "U_B:Position.Axis 1",
    ]
    for bad in ("nodevice", ":x", "U_A:", 3):
        with pytest.raises(ScannerError) as info:
            normalise([bad])
        assert info.value.kind == "invalid_request"


def test_the_file_store_is_replaced_atomically_and_a_hand_edit_is_a_new_version(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state" / "scan_log_variables.json"
    lst = ScanLogList(FileListStore(path))
    assert lst.get().variables == []
    lst.put(["U_A:x"], version_of([]))
    assert json.loads(path.read_text()) == {"variables": ["U_A:x"]}
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]

    path.write_text(json.dumps({"variables": ["U_A:x", "U_B:y"]}))
    with pytest.raises(ScannerError) as info:
        lst.put(["U_C:z"], version_of(["U_A:x"]))
    assert info.value.kind == "conflict"


@pytest.mark.parametrize(
    "content",
    [b"{not json", b'{"variables": ["U_A:\xff"]}', b'["U_A:x"]', b'{"variables": 3}'],
    ids=["bad json", "not utf-8", "not an object", "not a list"],
)
def test_a_broken_file_never_refuses_a_scan(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "scan_log_variables.json"
    path.write_bytes(content)
    lst = ScanLogList(FileListStore(path))
    with pytest.raises(ScannerError):
        lst.get()
    assert lst.for_submission() == []


def test_submit_stamps_the_list_into_the_run_metadata(
    client: TestClient, preset_doc: dict, manager: DemoQueueClient
) -> None:
    version = client.get("/api/scan-log-variables").json()["version"]
    client.put(
        "/api/scan-log-variables",
        json={"variables": ["U_Hexapod:ypos"], "version": version},
    )
    r = client.post(
        "/api/submit",
        json={"preset": preset_doc, "acknowledged": ["gateway_liveness"]},
    )
    assert r.status_code == 200, r.text
    md = manager._running["kwargs"]["md"]
    assert md["geecs"]["scan_log_variables"] == ["U_Hexapod:ypos"]


def test_an_empty_list_stamps_nothing(
    client: TestClient, preset_doc: dict, manager: DemoQueueClient
) -> None:
    r = client.post(
        "/api/submit",
        json={"preset": preset_doc, "acknowledged": ["gateway_liveness"]},
    )
    assert r.status_code == 200, r.text
    assert "scan_log_variables" not in manager._running["kwargs"]["md"]["geecs"]


def test_the_picker_lists_logged_variables_first() -> None:
    items = build_logged_variables(
        {"U_B": ["y", "img"], "U_A": ["x"]}, {"U_B": ["y"], "U_A": ["x"], "U_C": ["z"]}
    )
    assert [(i.name, i.logged) for i in items] == [
        ("U_A:x", True),
        ("U_B:y", True),
        ("U_C:z", True),
        ("U_B:img", False),
    ]


def test_a_failed_db_read_is_reported_and_not_cached() -> None:
    class FlakyDb:
        calls = 0

        @classmethod
        def get_all_experiment_variables(cls, experiment):
            cls.calls += 1
            if cls.calls == 1:
                raise ConnectionError("down")
            return {"U_A": ["x"]}

        @classmethod
        def get_subscribed_variables(cls, experiment):
            return {"U_A": ["x"]}

    source = DbLoggedVariables("Exp", db=FlakyDb)
    out = source.logged_variables()
    assert out.items == [] and "down" in out.detail
    assert [i.name for i in source.logged_variables().items] == ["U_A:x"]


def _pick_variables(clicks: str) -> list[str]:
    """Run the scan-log drawer's selection logic under node; return what is picked.

    The list mirrors the device drawer's test: ``U_C:not_logged`` and
    ``U_D:listed`` cannot be picked.
    """
    import json as _json
    import re
    import shutil
    import subprocess

    if shutil.which("node") is None:  # pragma: no cover - CI and dev machines have it
        pytest.skip("node not available")
    source = (
        Path(__file__).parents[1] / "geecs_scanner" / "static" / "scan-log-list.js"
    ).read_text()
    body = re.search(r"\n  function pick\([^)]*\) \{\n(.*?)\n  \}\n", source, re.S)
    assert body, "scan-log-list.js: no function pick()"
    harness = (
        """
var shown = [
  {name: 'U_A:x', pickable: true}, {name: 'U_B:y', pickable: true},
  {name: 'U_C:not_logged', pickable: false}, {name: 'U_D:listed', pickable: false},
  {name: 'U_E:z', pickable: true},
];
var picked = {}, anchor = null;
function renderPicks() {}
function pick(name, index, range) {
"""
        + body.group(1)
        + "\n}\n"
        + clicks
        + "\nconsole.log(JSON.stringify(Object.keys(picked).sort()));"
    )
    done = subprocess.run(
        ["node", "-"], input=harness, text=True, capture_output=True, check=True
    )
    return _json.loads(done.stdout)


def test_the_drawer_picks_many_like_the_device_drawer() -> None:
    """A click toggles a row; Add commits them all — the device picker's model."""
    assert _pick_variables("pick('U_A:x', 0, false); pick('U_B:y', 1, false);") == [
        "U_A:x",
        "U_B:y",
    ]
    assert _pick_variables("pick('U_A:x', 0, false); pick('U_A:x', 0, false);") == []


def test_a_shift_range_skips_unlogged_and_listed_variables() -> None:
    picked = _pick_variables("pick('U_A:x', 0, false); pick('U_E:z', 4, true);")
    assert picked == ["U_A:x", "U_B:y", "U_E:z"]


def test_the_panel_opens_a_drawer_not_a_combobox(client: TestClient) -> None:
    html = client.get("/").text
    assert 'data-drawer-open="drw-scanlog"' in html and 'id="drw-scanlog"' in html
    assert 'id="sl-list"' in html and 'class="picklist" id="sl-list"' in html
    assert 'role="combobox"' not in html
