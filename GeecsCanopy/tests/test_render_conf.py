"""deploy/render_conf.sh against the fleet's example site.env."""

import subprocess
from pathlib import Path

import yaml

from geecs_canopy.pin import CANOPY_COMMIT, CANOPY_IMAGE

PKG = Path(__file__).resolve().parents[1]
RENDER = PKG / "deploy" / "render_conf.sh"
EXAMPLE = PKG.parent / "deploy" / "site.env.example"


def render(tmp_path: Path, drop: str = "") -> subprocess.CompletedProcess[str]:
    site = tmp_path / "site.env"
    lines = EXAMPLE.read_text(encoding="utf-8").splitlines()
    site.write_text("\n".join(ln for ln in lines if not (drop and ln.startswith(drop))))
    return subprocess.run(
        ["bash", str(RENDER), str(site), str(tmp_path / "out")],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "RENDER_QUIET": "1"},
    )


def test_the_pin_and_the_site_reach_the_compose_file(tmp_path: Path) -> None:
    result = render(tmp_path)
    assert result.returncode == 0, result.stderr
    compose = yaml.safe_load((tmp_path / "out" / "compose.yaml").read_text())
    api = compose["services"]["api"]
    assert (
        api["build"]["args"] == {"CANOPY_IMAGE": CANOPY_IMAGE}
        and "@sha256:" in CANOPY_IMAGE
    )
    assert api["image"] == f"geecs-canopy:{CANOPY_COMMIT}"
    assert api["environment"]["CANOPY_EPICS__BACKEND"] == "epics"
    assert "--root-path" in api["command"]
    assert all(s.get("network_mode") == "host" for s in compose["services"].values())


def test_a_missing_canopy_key_fails_before_rendering(tmp_path: Path) -> None:
    result = render(tmp_path, drop="GEECS_CANOPY_DB=")
    assert result.returncode == 2 and "GEECS_CANOPY_DB" in result.stderr
