"""Pin ``scripts/lib/lab_env.sh``: the config.ini reader both fleet scripts share."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LIB = REPO_ROOT / "scripts" / "lib" / "lab_env.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _run(config: Path, snippet: str) -> str:
    script = f'CONFIG="{config}"; . "{LIB}"; {snippet}'
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    ).stdout


def test_endpoints_and_experiment_derive_from_config(tmp_path: Path) -> None:
    cfg = tmp_path / "config.ini"
    cfg.write_text(
        "[Experiment]\nexpt = Undulator\n\n[tiled]\nuri = http://192.168.6.14:8000\n"
        "[qserver]\nhost = 192.168.6.14\n[Paths]\nGEECS_DATA_LOCAL_BASE_PATH = /Volumes/hdna2/data/\n"
    )
    out = _run(
        cfg, 'echo "$LAB_HOST $TILED_PORT $WORKER_HOST $DATA_ROOT $(config_experiment)"'
    )
    assert out.split() == [
        "192.168.6.14",
        "8000",
        "192.168.6.14",
        "/Volumes/hdna2/data/",
        "Undulator",
    ]


def test_legacy_exp_name_and_defaults(tmp_path: Path) -> None:
    cfg = tmp_path / "config.ini"
    cfg.write_text("[Experiment]\nexp_name = Thomson\n[tiled]\nuri = http://lab\n")
    out = _run(cfg, 'echo "$LAB_HOST $TILED_PORT $(config_experiment)"')
    assert out.split() == ["lab", "8000", "Thomson"]


def test_missing_config_is_empty_not_an_error(tmp_path: Path) -> None:
    out = _run(tmp_path / "absent.ini", 'echo "[$LAB_HOST][$(config_experiment)]"')
    assert out.strip() == "[][]"


def test_printers_and_bounded(tmp_path: Path) -> None:
    out = _run(
        tmp_path / "absent.ini",
        "ok a; bad b; warn c; skip d; info e; bounded 5 true && echo bounded-ok",
    )
    assert out.splitlines() == [
        "  [ OK ] a",
        "  [DOWN] b",
        "  [WARN] c",
        "  [ -- ] d",
        "         e",
        "bounded-ok",
    ]
