"""The day page's scan filters: each scan's s-file, judged on demand."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
from geecs_data_utils.scan_conditions import labels_path, write_condition_labels

from geecs_portal.day_scans import DayScans
from test_app import TEST_DAY, FakeCatalog, _client

LABELS = {"APL": ["in", "out"]}


def day_tree(root: Path) -> Path:
    """Scan001 (APL in, charge 10/12) and Scan002 (mixed, charge 30/40)."""
    day = root / "day"
    for number, apl, charge in ((1, [0, 0], [10.0, 12.0]), (2, [0, 1], [30.0, 40.0])):
        (day / "scans" / f"Scan{number:03d}").mkdir(parents=True)
        (day / "analysis").mkdir(exist_ok=True)
        pd.DataFrame({"Shotnumber": [1, 2], "APL": apl, "charge": charge}).to_csv(
            day / "analysis" / f"s{number}.txt", sep="\t", index=False
        )
        out = day / "analysis" / f"Scan{number:03d}"
        out.mkdir()
        write_condition_labels(labels_path(out, out.name, "job"), "job", LABELS)
    return day


def client_over(root: Path):
    day = day_tree(root)
    scans = DayScans(scans_folder=lambda experiment, d: day / "scans")
    return _client(FakeCatalog(), default_experiment="Undulator", day_scans=scans), day


def get(client, where=None):
    params = {"where": json.dumps(where)} if where is not None else {}
    return client.get(f"/api/day/{TEST_DAY.isoformat()}/scans", params=params)


def by_scan(body):
    return {r["scan_number"]: r for r in body["runs"]}


def test_conditions_without_rules(tmp_path):
    client, _ = client_over(tmp_path)
    body = get(client).json()
    assert body["condition_labels"] == LABELS and "charge" in body["columns"]
    runs = by_scan(body)
    assert runs[1]["conditions"][0]["label"] == "in"
    assert runs[2]["conditions"][0]["mixed"] is True
    assert {r["verdict"] for r in runs.values()} == {"pass"}


def test_condition_and_number_rules(tmp_path):
    client, _ = client_over(tmp_path)
    runs = by_scan(get(client, [{"column": "APL", "op": "is", "label": "in"}]).json())
    assert runs[1]["verdict"] == "pass" and runs[2]["verdict"] == "mixed"
    body = get(client, [{"column": "charge", "op": "between", "low": 20}]).json()
    runs = by_scan(body)
    assert runs[1]["verdict"] == "fail" and runs[2]["verdict"] == "pass"
    assert runs[2]["values"]["charge"] == 35.0
    assert body["counts"] == {"fail": 1, "pass": 1}


def test_a_bad_rule_is_a_400(tmp_path):
    client, _ = client_over(tmp_path)
    assert get(client, [{"column": "APL", "op": "is"}]).status_code == 400
    bad = client.get(f"/api/day/{TEST_DAY.isoformat()}/scans", params={"where": "{"})
    assert bad.status_code == 400


def test_a_reanalysed_scan_reads_afresh(tmp_path):
    client, day = client_over(tmp_path)
    assert by_scan(get(client).json())[1]["conditions"][0]["label"] == "in"
    sfile = day / "analysis" / "s1.txt"
    pd.DataFrame({"Shotnumber": [1, 2], "APL": [1, 1], "charge": [1.0, 1.0]}).to_csv(
        sfile, sep="\t", index=False
    )
    later = time.time() + 5
    os.utime(sfile, (later, later))
    assert by_scan(get(client).json())[1]["conditions"][0]["label"] == "out"


def test_the_page_offers_the_button_and_keeps_the_rules(tmp_path):
    client, _ = client_over(tmp_path)
    where = '[{"column":"APL","op":"is","label":"in"}]'
    page = client.get(f"/day/{TEST_DAY.isoformat()}", params={"where": where}).text
    assert 'id="scanf-open"' in page and 'id="scanf"' in page
    assert '<th class="scanf-cond" hidden>Conditions</th>' in page
    # the day arrows and the text filter keep the rules
    assert "where=%5B%7B%22column%22" in page
    assert 'name="where" value=' in page
