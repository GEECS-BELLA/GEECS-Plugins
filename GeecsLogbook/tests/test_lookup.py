"""Start values on a scan's card and the day's variable lookup — both from the s-file's first row."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from geecs_logbook.app import create_app
from geecs_logbook.lookup import day_lookup
from geecs_logbook.scan_reader import read_day

DAY = date(2026, 9, 11)

#: A finished scan's ini, as the Bluesky path writes it (trimmed to the keys read here).
SUCCESS_INI = """[Scan Info]
Scan No = {n}
ScanStartInfo = "lookup test"
Scan Parameter = "U_S1H Current"
ScanEndInfo = "success"
Scanner = "bluesky"
Plan = "scan"
"""

#: Claimed and written, never finalised.
UNFINALISED_INI = SUCCESS_INI.replace('ScanEndInfo = "success"', 'ScanEndInfo = ""')


def _day(root: Path) -> Path:
    return root / "Undulator" / "Y2026" / "09-Sep" / "26_0911"


def _scan(
    root: Path,
    n: int,
    *,
    listed: list[str] | None = None,
    sfile: list[list[str]] | None = None,
    ini: str = SUCCESS_INI,
) -> Path:
    folder = _day(root) / "scans" / f"Scan{n:03d}"
    folder.mkdir(parents=True)
    text = ini.format(n=n)
    if listed is not None:
        text += f"\n[Scan Log]\nVariables = {json.dumps(listed)}\n"
    (folder / f"ScanInfoScan{n:03d}.ini").write_text(text)
    if sfile is not None:
        (folder / f"ScanDataScan{n:03d}.txt").write_text(
            "\n".join("\t".join(row) for row in sfile) + "\n"
        )
    return folder


@pytest.fixture
def day(tmp_path: Path) -> Path:
    listed = ["U_Hexapod:ypos", "U_S1H:Current", "U_Gone:x"]
    header = [
        "Bin #",
        "Shotnumber",
        "U_Hexapod ypos",
        "U_S1H Current",
        "U_Jet Pressure",
    ]
    _scan(
        tmp_path,
        1,
        listed=listed,
        sfile=[
            header,
            ["1", "1", "14.23", "1.25", "305"],
            ["1", "2", "14.23", "1.25", "306"],
        ],
    )
    _scan(tmp_path, 2, listed=listed, sfile=[header, ["1", "1", "15.1", "1.25", "312"]])
    # Still running: the ini names the list, no s-file yet.
    _scan(tmp_path, 3, listed=listed, ini=UNFINALISED_INI)
    # Before the feature: no [Scan Log] section.
    _scan(tmp_path, 4, sfile=[header, ["1", "1", "15.1", "1.30", "312"]])
    return tmp_path


class TestStartValues:
    def test_listed_variables_read_from_the_first_row(self, day: Path) -> None:
        scans = {
            s.number: s for s in read_day(DAY, "Undulator", base_directory=day).scans
        }
        assert scans[1].scan_log_variables == [
            "U_Hexapod:ypos",
            "U_S1H:Current",
            "U_Gone:x",
        ]
        assert scans[1].start_values == {
            "U_Hexapod:ypos": "14.23",
            "U_S1H:Current": "1.25",
            "U_Gone:x": None,  # listed, but the s-file has no such column
        }
        assert scans[2].start_values["U_Hexapod:ypos"] == "15.1"

    def test_a_running_scan_has_its_list_and_no_values_yet(self, day: Path) -> None:
        scan = next(
            s
            for s in read_day(DAY, "Undulator", base_directory=day).scans
            if s.number == 3
        )
        assert scan.scan_log_variables and scan.start_values is None
        assert scan.values_state == "recording"

    def test_a_failed_scan_without_an_sfile_does_not_promise_values(
        self, tmp_path: Path
    ) -> None:
        ini = SUCCESS_INI.replace('"success"', '"fail: RuntimeError: boom"')
        _scan(tmp_path, 8, listed=["U_A:x"], ini=ini)
        scan = read_day(DAY, "Undulator", base_directory=tmp_path).scans[0]
        assert scan.values_state == "no_sfile" and scan.start_values is None

    def test_a_run_with_no_shots_says_so(self, tmp_path: Path) -> None:
        _scan(tmp_path, 9, listed=["U_A:x"], sfile=[["Shotnumber", "U_A x"]])
        scan = read_day(DAY, "Undulator", base_directory=tmp_path).scans[0]
        assert scan.values_state == "no_shots" and scan.start_values is None

    def test_a_scan_before_the_feature_shows_nothing(self, day: Path) -> None:
        scan = next(
            s
            for s in read_day(DAY, "Undulator", base_directory=day).scans
            if s.number == 4
        )
        assert scan.scan_log_variables == [] and scan.start_values is None

    def test_the_sfile_arriving_or_changing_is_seen(self, day: Path) -> None:
        folder = _day(day) / "scans" / "Scan003"
        sfile = folder / "ScanDataScan003.txt"
        sfile.write_text("U_Hexapod ypos\tU_S1H\n")  # a read racing the in-place write
        os.utime(sfile, (1, 1))
        scan = next(
            s
            for s in read_day(DAY, "Undulator", base_directory=day).scans
            if s.number == 3
        )
        assert scan.values_state == "no_shots"  # a header and no row yet
        sfile.write_text("U_Hexapod ypos\tU_S1H Current\n16.0\t1.4\n")
        scan = next(
            s
            for s in read_day(DAY, "Undulator", base_directory=day).scans
            if s.number == 3
        )
        assert scan.start_values["U_Hexapod:ypos"] == "16.0"

    def test_the_analysis_copy_is_preferred_like_the_portal(
        self, tmp_path: Path
    ) -> None:
        """ScanAnalysis appends its scalars to analysis/sNNN.txt only."""
        _scan(tmp_path, 5, listed=["U_A:x"], sfile=[["U_A x"], ["1.0"]])
        analysis = _day(tmp_path) / "analysis"
        analysis.mkdir()
        (analysis / "s5.txt").write_text("U_A x\tUC_Cam peak\n1.0\t42\n")
        summary = read_day(DAY, "Undulator", base_directory=tmp_path)
        assert summary.scans[0].sfile.endswith("s5.txt")
        table = day_lookup(summary, ["UC_Cam peak"])
        assert table.rows[0].values == {"UC_Cam peak": "42"}

    def test_master_control_aliased_columns_and_analysis_sfile(
        self, tmp_path: Path
    ) -> None:
        _scan(tmp_path, 7, listed=["U_Hexapod:ypos"])
        analysis = _day(tmp_path) / "analysis"
        analysis.mkdir()
        (analysis / "s7.txt").write_text(
            "Shotnumber\tU_Hexapod ypos Alias:Hex Y\n1\t9.5\n"
        )
        scan = read_day(DAY, "Undulator", base_directory=tmp_path).scans[0]
        assert scan.start_values == {"U_Hexapod:ypos": "9.5"}


class TestDayLookup:
    def test_default_columns_are_the_lists_and_every_column_is_offered(
        self, day: Path
    ) -> None:
        table = day_lookup(read_day(DAY, "Undulator", base_directory=day))
        assert table.default_columns == ["U_Hexapod ypos", "U_S1H Current", "U_Gone x"]
        assert table.shown == table.default_columns
        offered = {c.name: (c.scans, c.listed) for c in table.columns}
        assert offered["U_Jet Pressure"] == (3, False)
        assert offered["U_Hexapod ypos"] == (3, True)
        rows = {r.number: r for r in table.rows}
        assert rows[1].values["U_Hexapod ypos"] == "14.23"
        assert (
            rows[3].values_state == "recording"
            and rows[3].values["U_Hexapod ypos"] is None
        )
        assert rows[4].values["U_S1H Current"] == "1.30"
        assert rows[1].status == "success" and rows[1].state == "ok"

    def test_asked_columns_replace_the_defaults(self, day: Path) -> None:
        table = day_lookup(
            read_day(DAY, "Undulator", base_directory=day), ["U_Jet Pressure"]
        )
        assert table.shown == ["U_Jet Pressure"]
        assert [r.values for r in table.rows] == [
            {"U_Jet Pressure": "305"},
            {"U_Jet Pressure": "312"},
            {"U_Jet Pressure": None},
            {"U_Jet Pressure": "312"},
        ]

    def test_the_route(self, day: Path) -> None:
        client = TestClient(create_app("Undulator", base_directory=day))
        r = client.get(
            f"/api/day/{DAY.isoformat()}/lookup", params=[("col", "U_Jet Pressure")]
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["shown"] == ["U_Jet Pressure"] and len(body["rows"]) == 4
        default = client.get(f"/api/day/{DAY.isoformat()}/lookup").json()
        assert default["shown"] == ["U_Hexapod ypos", "U_S1H Current", "U_Gone x"]

    def test_an_empty_day(self, tmp_path: Path) -> None:
        table = day_lookup(
            read_day(date(2026, 9, 12), "Undulator", base_directory=tmp_path)
        )
        assert table.rows == [] and table.columns == []

    def test_the_lookup_never_creates_a_folder(self, day: Path, monkeypatch) -> None:
        def explode(*args, **kwargs):
            raise AssertionError("the logbook must never create scan folders")

        before = set(day.rglob("*"))
        monkeypatch.setattr(Path, "mkdir", explode)
        day_lookup(read_day(DAY, "Undulator", base_directory=day), ["U_Jet Pressure"])
        assert set(day.rglob("*")) == before


class TestTheDayPage:
    def test_the_card_and_the_panel_render(self, day: Path) -> None:
        client = TestClient(create_app("Undulator", base_directory=day))
        html = client.get(f"/day/{DAY.isoformat()}").text
        assert "Start values · first shot" in html
        assert ">14.23<" in html.replace("\n", "").replace(" ", "")
        assert "not recorded" in html  # U_Gone:x
        assert "recording — the values arrive when the scan closes" in html  # Scan003
        assert "data-lookup" in html and "/static/lookup.js" in html
        # Expand-all never opens the lookup: opening it reads every s-file.
        assert "details.scan:not(.lookup)" in html


#: A DOM just big enough for lookup.js: the elements it looks up, listeners
#: that tests can fire, and a fetch that fails once then answers.
_HARNESS = r"""
const listeners = {};
const el = (id) => ({
  id, hidden: false, textContent: "", value: "", dataset: { api: "/api", day: "2026-09-11" },
  open: false, className: "", title: "",
  setAttribute() {}, appendChild() {},
  addEventListener(kind, fn) { (listeners[id + ":" + kind] = listeners[id + ":" + kind] || []).push(fn); },
});
const els = {};
const panel = el("lookup");
global.document = {
  getElementById: (id) => (els[id] = els[id] || el(id)),
  querySelector: () => panel,
  createElement: () => el("x"),
};
global.location = { search: "", pathname: "/day/2026-09-11", hash: "" };
global.history = { replaceState() {} };
let calls = 0;
global.fetch = async () => {
  calls += 1;
  if (calls === 1) throw new Error("share unavailable");
  return { ok: true, json: async () => ({ columns: [], default_columns: [], shown: [], rows: [] }) };
};
const tick = () => new Promise((r) => setTimeout(r, 0));
const toggle = async (open) => { panel.open = open; for (const fn of listeners["lookup:toggle"] || []) fn(); await tick(); await tick(); };
(async () => {
  require(process.argv[2]);
  await toggle(true);            // first open: the read fails
  await toggle(false);
  await toggle(true);            // reopen: must try again
  await toggle(false);
  await toggle(true);            // and once it has data, never again
  console.log(JSON.stringify({ calls }));
})();
"""


def test_a_failed_first_read_is_retried_on_reopen(tmp_path: Path) -> None:
    """A share blip on the first open must not leave the panel dead until a reload."""
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:  # pragma: no cover - CI and dev machines have it
        pytest.skip("node not available")
    script = Path(__file__).parents[1] / "geecs_logbook" / "static" / "lookup.js"
    harness = tmp_path / "harness.js"
    harness.write_text(_HARNESS)
    done = subprocess.run(
        [node, str(harness), str(script)], capture_output=True, text=True, timeout=30
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == {"calls": 2}


def _pick_columns(clicks: str) -> list[str]:
    """Run the column drawer's ``pick`` under node over a fixed list; return the picks."""
    import re
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:  # pragma: no cover - CI and dev machines have it
        pytest.skip("node not available")
    source = (
        Path(__file__).parents[1] / "geecs_logbook" / "static" / "lookup.js"
    ).read_text()
    body = re.search(r"\n  function pick\([^)]*\) \{\n(.*?)\n  \}\n", source, re.S)
    assert body, "lookup.js: no function pick()"
    harness = (
        "let shownCols = [{name: 'U_A x', pickable: true}, {name: 'U_B y', pickable: true},"
        " {name: 'U_C z', pickable: false}, {name: 'U_D w', pickable: true}];\n"
        "let picked = {}, anchor = null;\nfunction renderPicks() {}\n"
        "function pick(name, index, range) {\n"
        + body.group(1)
        + "\n}\n"
        + clicks
        + "\nconsole.log(JSON.stringify(Object.keys(picked).sort()));"
    )
    done = subprocess.run(
        [node, "-"], input=harness, text=True, capture_output=True, check=True
    )
    return json.loads(done.stdout)


def test_the_column_drawer_picks_many_like_the_device_drawer() -> None:
    assert _pick_columns("pick('U_A x', 0, false); pick('U_A x', 0, false);") == []
    # shift-click reaches across, skipping a column already in the table
    assert _pick_columns("pick('U_A x', 0, false); pick('U_D w', 3, true);") == [
        "U_A x",
        "U_B y",
        "U_D w",
    ]


def test_the_panel_opens_a_drawer_not_a_combobox(day: Path) -> None:
    html = (
        TestClient(create_app("Undulator", base_directory=day))
        .get(f"/day/{DAY.isoformat()}")
        .text
    )
    assert 'data-drawer-open="drw-lookup"' in html and 'id="drw-lookup"' in html
    assert 'class="picklist" id="lk-list"' in html
    assert 'role="combobox"' not in html
