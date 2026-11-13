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
    cmd = ["bash", str(RENDER), str(site), str(tmp_path / "out")]
    env = {"PATH": "/usr/bin:/bin", "RENDER_QUIET": "1"}
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


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
    assert not (tmp_path / "out" / "compose.yaml").exists()


def test_a_worktree_configs_root_mounts_its_git_dir_at_the_same_path(tmp_path: Path) -> None:  # fmt: skip
    """The configurator's live checkout: its rewrite lock must be visible inside the container."""
    live = tmp_path / "state" / "live"
    gitdir = tmp_path / "state" / "repo.git" / "worktrees" / "live"
    live.mkdir(parents=True)
    gitdir.mkdir(parents=True)
    (live / ".git").write_text(f"gitdir: {gitdir}\n", encoding="utf-8")
    site = tmp_path / "site.env"
    lines = EXAMPLE.read_text(encoding="utf-8").splitlines()
    site.write_text("\n".join(f"GEECS_CONFIGS_ROOT={live}" if ln.startswith("GEECS_CONFIGS_ROOT=") else ln for ln in lines))  # fmt: skip
    env = {"PATH": "/usr/bin:/bin", "RENDER_QUIET": "1"}
    result = subprocess.run(["bash", str(RENDER), str(site), str(tmp_path / "out")], capture_output=True, text=True, env=env)  # fmt: skip
    assert result.returncode == 0, result.stderr
    api = yaml.safe_load((tmp_path / "out" / "compose.yaml").read_text())["services"]["api"]  # fmt: skip
    assert f"{gitdir}:{gitdir}:ro" in api["volumes"]


def test_a_plain_configs_root_mounts_nothing_more(tmp_path: Path) -> None:
    result = render(tmp_path)
    api = yaml.safe_load((tmp_path / "out" / "compose.yaml").read_text())["services"]["api"]  # fmt: skip
    assert result.returncode == 0 and len(api["volumes"]) == 2
