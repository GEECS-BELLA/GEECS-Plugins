"""A scratch configs tree (one experiment, the analysis root), an app over it, and scratch git repos.

The commit path's tests run against local bare repositories as "origin":
no network, no GitHub; the pull-request opener is a fake.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from geecs_configurator.app import create_app
from geecs_configurator.repository import ConfigsRepository, Identity
from geecs_configurator.save import PullRequest
from geecs_configurator.settings import Settings

#: path below the configs root -> YAML text
FILES = {
    "scanner_configs/experiments/Bench/shot_control_configurations/laser_off.yaml": """\
schema_version: 2
name: laser_off
states:
  SCAN:
    - {device: U_DG645, variable: Amplitude.Ch AB, value: '4.0'}
  STANDBY:
    - {device: U_DG645, variable: Amplitude.Ch AB, value: '0.5'}
""",
    "scanner_configs/experiments/Bench/shot_control_configurations/spare.yml": "schema_version: 2\nname: spare\n",
    # a bare OFF: key, which YAML 1.1 reads as a boolean (the model maps it back)
    "scanner_configs/experiments/Bench/shot_control_configurations/bare_off.yaml": """\
schema_version: 2
name: bare_off
states:
  OFF:
    - {device: U_DG645, variable: Amplitude.Ch AB, value: '0.5'}
  SCAN:
    - {device: U_DG645, variable: Amplitude.Ch AB, value: '4.0'}
""",
    # a malformed legacy profile: the converter meets a list where it wants a mapping
    "scanner_configs/experiments/Bench/shot_control_configurations/malformed.yaml": "device: A\nvariables: [bad]\n",
    # unquoted YAML timestamps: a datetime and a date, served to the browser as text
    "scanner_configs/experiments/Bench/setups/stamped.yaml": "name: stamped\nsteps:\n  - {do: wait, seconds: 1}\ncaptured: {by: operator, at: 2026-11-10T00:00:00}\n",
    "scanner_configs/experiments/Bench/setups/dated.yaml": "name: dated\nsteps:\n  - {do: wait, seconds: 1}\ncaptured: {by: operator, at: 2026-11-10}\n",
    # the legacy dialect (no schema_version), which the reader converts
    "scanner_configs/experiments/Bench/shot_control_configurations/legacy.yaml": """\
device: U_DG645
variables:
  Amplitude.Ch AB: {'OFF': '0.5', SCAN: '4.0', STANDBY: '0.5'}
""",
    "scanner_configs/experiments/Bench/setups/low_power.yaml": """\
name: low_power
steps:
  - {do: set, device: U_PS, variable: Current, value: 1.5}
  - {do: wait, seconds: 2}
""",
    # a non-finite float, which must never come back as null
    "scanner_configs/experiments/Bench/setups/far.yaml": "name: far\nsteps:\n  - {do: set, device: A, variable: V, value: .inf}\n  - {do: wait, seconds: 1}\n",  # fmt: skip
    # an unquoted date in a text field: invalid as the file, valid as served text
    "scanner_configs/experiments/Bench/setups/dated_text.yaml": "name: dated_text\ndescription: 2026-11-10\nsteps:\n  - {do: wait, seconds: 1}\n",  # fmt: skip
    "scanner_configs/experiments/Bench/setups/renamed.yaml": "name: other\nsteps:\n  - {do: wait, seconds: 1}\n",
    "scanner_configs/experiments/Bench/setups/broken.yaml": "name: [unclosed\n",
    "scanner_configs/experiments/Bench/archiver/archive_policy.yaml": "include_setpoints: true\n",
    "scan_analysis_configs/analyzers/Ns/compact.yaml": "kind: himg_compact\ndevice: CamA\n",
    "scan_analysis_configs/analyzers/Ns/recipe.yaml": "device: CamB\n",
    "scan_analysis_configs/groups/base.yaml": "name: base\nanalyzers: []\n",
}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for rel, text in FILES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (tmp_path / "scanner_configs/experiments/Empty").mkdir()
    return tmp_path


@pytest.fixture
def client(root: Path) -> TestClient:
    settings = Settings(configs_root=root, experiment="Bench", cors_origins=("http://surface.test",))  # fmt: skip
    with TestClient(create_app(settings)) as c:
        yield c


# ------------------------------------------------- the commit path's scratch repos


def git(*args: str, cwd: Path) -> str:
    """Run git in a test (identity and config isolated by ``git_env``)."""
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()  # fmt: skip


@pytest.fixture(autouse=True)
def git_env(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    """No user or system git config reaches a test (signing, hooks, a default branch)."""
    empty = tmp_path_factory.mktemp("gitconfig") / "config"
    empty.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for key in ("NAME", "EMAIL"):
        monkeypatch.setenv(f"GIT_AUTHOR_{key}", "seed@test.invalid" if key == "EMAIL" else "Seed")  # fmt: skip
        monkeypatch.setenv(f"GIT_COMMITTER_{key}", "seed@test.invalid" if key == "EMAIL" else "Seed")  # fmt: skip


@pytest.fixture
def origin(root: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A bare "origin" holding the scratch configs tree on main."""
    bare = tmp_path_factory.mktemp("origin") / "configs.git"
    git("init", "--bare", "--quiet", "--initial-branch=main", str(bare), cwd=root)
    git("init", "--quiet", "--initial-branch=main", cwd=root)
    git("add", "-A", cwd=root)
    git("commit", "--quiet", "-m", "seed", cwd=root)
    git("push", "--quiet", str(bare), "main", cwd=root)
    return bare


@pytest.fixture
def other(origin: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Someone else's clone of origin, pushing behind the service's back."""
    path = tmp_path_factory.mktemp("other") / "clone"
    git("clone", "--quiet", str(origin), str(path), cwd=origin)
    return path


def push_change(clone: Path, rel: str, text: str, message: str = "elsewhere") -> str:
    """Commit ``text`` at ``rel`` in ``clone`` on top of origin's main and push it; the commit."""
    git("pull", "--quiet", "--ff-only", cwd=clone)
    (clone / rel).parent.mkdir(parents=True, exist_ok=True)
    (clone / rel).write_text(text, encoding="utf-8")
    git("add", "-A", cwd=clone)
    git("commit", "--quiet", "-m", message, cwd=clone)
    git("push", "--quiet", "origin", "HEAD:main", cwd=clone)
    return git("rev-parse", "HEAD", cwd=clone)


@pytest.fixture
def repo(origin: Path, tmp_path_factory: pytest.TempPathFactory) -> ConfigsRepository:
    """The service's repository over origin, set up."""
    r = ConfigsRepository(tmp_path_factory.mktemp("state"), str(origin), "main", Identity("GEECS configurator", "configurator@test.invalid"))  # fmt: skip
    r.ensure()
    return r


class FakePullRequests:
    """The App's pull-request opener, recorded instead of called."""

    def __init__(self) -> None:
        self.opened: list[tuple[str, str, str]] = []

    def open(self, branch: str, title: str, body: str) -> PullRequest:
        self.opened.append((branch, title, body))
        return PullRequest(opened=True, url=f"https://example.invalid/pull/{len(self.opened)}")  # fmt: skip


@pytest.fixture
def prs() -> FakePullRequests:
    return FakePullRequests()


@pytest.fixture
def committing(origin: Path, prs: FakePullRequests, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> TestClient:  # fmt: skip
    """An app with the commit path on over origin (no timer: tests reconcile by hand; every read fetches)."""
    monkeypatch.setattr("geecs_configurator.app.FRESH_S", -1.0)
    state = tmp_path_factory.mktemp("service")
    env = {"GEECS_CONFIGURATOR_STATE_DIR": str(state), "GEECS_CONFIGURATOR_ORIGIN": str(origin), "GEECS_CONFIGURATOR_RECONCILE_SECONDS": "0", "GEECS_CONFIGURATOR_EXPERIMENT": "Bench"}  # fmt: skip
    with TestClient(create_app(Settings.from_env(env), pull_requests=prs)) as c:
        yield c


def files_of(tree: Path) -> dict[str, bytes]:
    """Every file of a checkout (its ``.git`` aside)."""
    return {p.relative_to(tree).as_posix(): p.read_bytes() for p in tree.rglob("*") if p.is_file() and ".git" not in p.relative_to(tree).parts}  # fmt: skip
