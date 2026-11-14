"""One configs repo, many facilities: a box commits, judges and logs only what it owns.

Two experiments on one origin: ``Bench`` (this box's, with the analysis
folder ``Ns``) and ``Other`` (another facility's box). Every write outside
the owned paths is refused before git; another facility's commits, valid or
not, fast-forward unjudged and are never logged or applied here.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from conftest import git, push_change
from fastapi.testclient import TestClient

from geecs_configurator.app import create_app
from geecs_configurator.log import Answer, Row
from geecs_configurator.ownership import Ownership, OwnershipError, parse_names
from geecs_configurator.repository import ConfigsRepository, Identity, NotWritable
from geecs_configurator.service import CommitPath
from geecs_configurator.settings import Settings, SettingsError

B, OQ = "?experiment=Bench", "?experiment=Other"
EXP = "scanner_configs/experiments"
OURS, THEIRS = f"{EXP}/Bench/setups", f"{EXP}/Other/setups"
SETUP = "name: {name}\ndescription: {text}\nsteps:\n  - {{do: wait, seconds: 1}}\n"
INVALID = "name: {name}\nsteps: []\n"  # a setup needs a step
OWNED = {"GEECS_CONFIGURATOR_EXPERIMENTS": "Bench", "GEECS_CONFIGURATOR_ANALYSIS_FOLDERS": "Ns"}  # fmt: skip


class Recorder:
    """The apply seam, recording what it was asked."""

    def __init__(self) -> None:
        self.asked: list[str] = []
        self.settled: list[str] = []

    def apply(self, kind: str | None, path: str) -> Answer:
        self.asked.append(path)
        return Answer(state="queued", detail="faked")

    def settle(self, rows: list[Row]) -> list[Answer]:
        self.settled.extend(r.path for r in rows)
        return [Answer(state="applied", detail="faked") for _ in rows]


def _seed_other(clone: Path) -> str:
    """Another facility's experiment on main: a setup and an archive policy."""
    push_change(clone, f"{THEIRS}/theirs.yaml", SETUP.format(name="theirs", text="seed"), "Other's seed")  # fmt: skip
    return push_change(clone, f"{EXP}/Other/archiver/archive_policy.yaml", "include_setpoints: true\n", "Other's policy")  # fmt: skip


@pytest.fixture
def box(origin: Path, other: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:  # fmt: skip
    """This box (Bench + Ns) over an origin that also holds Other; every read fetches."""
    monkeypatch.setattr("geecs_configurator.app.FRESH_S", -1.0)
    _seed_other(other)
    env = {**OWNED, "GEECS_CONFIGURATOR_STATE_DIR": str(tmp_path / "s"), "GEECS_CONFIGURATOR_ORIGIN": str(origin), "GEECS_CONFIGURATOR_RECONCILE_SECONDS": "0", "GEECS_CONFIGURATOR_FETCH_SECONDS": "0"}  # fmt: skip
    recorder = Recorder()
    with TestClient(create_app(Settings.from_env(env), applier=recorder)) as c:
        c.app.state.recorder = recorder
        c.app.state.commit.unverified_marker.unlink()  # the maintainer looked at main
        yield c


def _cp(c: TestClient) -> CommitPath:
    return c.app.state.commit


def _put(c: TestClient, kind: str, doc_id: str, q: str, change: dict, **extra: object):  # fmt: skip
    doc = c.get(f"/api/documents/{kind}/{doc_id}{q}").json()
    return c.put(f"/api/documents/{kind}/{doc_id}{q}", json={"document": {**doc["document"], **change}, "etag": doc["etag"], "operator": "Ada", **extra})  # fmt: skip


def _branches(origin: Path) -> list[str]:
    return git("for-each-ref", "--format=%(refname:short)", "refs/heads", cwd=origin).split()  # fmt: skip


# ------------------------------------------------------------------ the setting


def test_the_commit_path_refuses_to_start_owning_nothing(tmp_path: Path) -> None:
    on = {"GEECS_CONFIGURATOR_STATE_DIR": str(tmp_path), "GEECS_CONFIGURATOR_ORIGIN": "o"}  # fmt: skip
    for empty in ({}, {"GEECS_CONFIGURATOR_EXPERIMENTS": "  , ,"}, {"GEECS_CONFIGURATOR_EXPERIMENTS": " ", "GEECS_CONFIGURATOR_ANALYSIS_FOLDERS": ","}):  # fmt: skip
        with pytest.raises(SettingsError, match="never read as everything"):
            Settings.from_env({**on, **empty})
    # read-only (no commit path) needs no ownership
    assert Settings.from_env({"GEECS_CONFIGS_ROOT": str(tmp_path)}).ownership.empty
    # the analysis folders alone are an owned set
    assert Settings.from_env({**on, "GEECS_CONFIGURATOR_ANALYSIS_FOLDERS": "Ns"}).ownership.analysis_folders == ("Ns",)  # fmt: skip
    s = Settings.from_env(
        {**on, "GEECS_CONFIGURATOR_EXPERIMENTS": " Bench , Bench,Two "}
    )
    assert s.ownership.experiments == ("Bench", "Two") and s.experiment == "Bench"
    # and the commit path itself refuses an empty set, however the settings were made
    with pytest.raises(ValueError, match="never everything"):
        CommitPath(Settings(configs_root=tmp_path, state_dir=tmp_path, origin="o"))


@pytest.mark.parametrize(
    "bad", ["../Other", "Bench/sub", ".hidden", "..", "*", "Ben?h", "a\\b"]
)
def test_an_owned_name_is_one_plain_folder(bad: str) -> None:
    with pytest.raises(OwnershipError):
        parse_names(f"Bench,{bad}", "KEY")


def test_owned_by_name() -> None:
    own = Ownership(("Bench",), ("Ns",))
    yes = [f"{EXP}/Bench/setups/a.yaml", f"{EXP}/Bench/shot_offsets.yaml", "scan_analysis_configs/analyzers/Ns/x/r.yaml", "scan_analysis_configs/groups/Ns/g.yaml"]  # fmt: skip
    no = [
        f"{EXP}/Bench", f"{EXP}/Benchmark/a.yaml", f"{EXP}/Other/a.yaml", f"{EXP}/Bench/../Other/a.yaml",
        f"{EXP}/Bench//a.yaml", f"./{EXP}/Bench/a.yaml", f"/{EXP}/Bench/a.yaml", "scan_analysis_configs/groups/base.yaml",
        "scan_analysis_configs/analyzers/Nsx/r.yaml", "scan_analysis_configs/calibrations/Ns/c.yaml", "README.md", "",
    ]  # fmt: skip
    assert [p for p in yes if not own.owns(p)] == [] and [p for p in no if own.owns(p)] == []  # fmt: skip
    assert Ownership().owns(f"{EXP}/Bench/a.yaml") is False


def test_owned_on_disk_never_through_a_link_out_of_the_folder(tmp_path: Path) -> None:
    (tmp_path / OURS).mkdir(parents=True)
    (tmp_path / THEIRS).mkdir(parents=True)
    (tmp_path / THEIRS / "t.yaml").write_text("x")
    os.symlink("../../Other/setups/t.yaml", tmp_path / OURS / "link.yaml")
    os.symlink("../../Other/setups", tmp_path / f"{EXP}/Bench/elsewhere")
    own = Ownership(("Bench",))
    assert own.owns(f"{OURS}/link.yaml") and not own.owns_file(tmp_path, f"{OURS}/link.yaml")  # fmt: skip
    assert not own.owns_file(tmp_path, f"{EXP}/Bench/elsewhere/t.yaml")
    assert own.owns_file(tmp_path, f"{OURS}/new.yaml")


# ------------------------------------------------------------------ writes


def test_a_save_outside_the_owned_paths_is_refused_before_git(box: TestClient, origin: Path, prs) -> None:  # fmt: skip
    head, branches = git("rev-parse", "main", cwd=origin), _branches(origin)
    # an operator-tier save, a maintainer-tier proposal and a create, all in Other
    r = _put(box, "setup", "theirs.yaml", OQ, {"description": "from the wrong box"})
    assert r.status_code == 403 and "not this services box's" in r.json()["detail"]
    assert _put(box, "archive_policy", "archive_policy.yaml", OQ, {"exclude": ["U_A:B"]}).status_code == 403  # fmt: skip
    created = box.put(f"/api/documents/setup/new.yaml{OQ}", json={"document": {"name": "new", "steps": [{"do": "wait", "seconds": 1}]}, "etag": "", "operator": "Ada", "create": True})  # fmt: skip
    assert created.status_code == 403
    # the shared analysis tree: only this box's facility folder
    assert _put(box, "analysis_group", "base.yaml", "", {"name": "renamed"}).status_code == 403  # fmt: skip
    assert git("rev-parse", "main", cwd=origin) == head and _branches(origin) == branches and prs.opened == []  # fmt: skip
    assert box.get("/api/log").json() == []
    # this box's own: saved
    assert _put(box, "setup", "low_power.yaml", B, {"description": "ours"}).json()["outcome"] == "committed"  # fmt: skip


def test_a_link_from_an_owned_folder_to_another_facility_is_not_owned(box: TestClient, other: Path, origin: Path) -> None:  # fmt: skip
    git("pull", "--quiet", "--ff-only", cwd=other)
    os.symlink("../../Other/setups/theirs.yaml", other / OURS / "theirs.yaml")
    git("add", "-A", cwd=other)
    git("commit", "--quiet", "-m", "a link into Other", cwd=other)
    git("push", "--quiet", "origin", "HEAD:main", cwd=other)
    head = git("rev-parse", "main", cwd=origin)
    doc = box.get(f"/api/documents/setup/theirs.yaml{B}").json()
    assert doc["read_only"] is True  # listed (inside the root), never saved here
    r = box.put(f"/api/documents/setup/theirs.yaml{B}", json={"document": {**doc["document"], "description": "through the link"}, "etag": doc["etag"], "operator": "Ada"})  # fmt: skip
    assert (
        r.status_code == 403 and "not this services box's" in r.json()["detail"]
    )  # the save's own answer, before any staging
    assert git("rev-parse", "main", cwd=origin) == head


def test_reads_list_every_experiment_and_mark_another_facilitys_read_only(box: TestClient) -> None:  # fmt: skip
    assert "Other" in box.get("/api/experiments").json()
    assert box.get(f"/api/documents/setup/theirs.yaml{OQ}").json()["read_only"] is True
    assert box.get(f"/api/documents/setup/low_power.yaml{B}").json()["read_only"] is False  # fmt: skip
    assert (
        box.get("/api/documents/analysis_group/base.yaml").json()["read_only"] is True
    )
    assert box.get("/api/documents/analysis_recipe/Ns/recipe.yaml").json()["read_only"] is False  # fmt: skip
    page = box.get("/?experiment=Other&kind=setup").text
    assert "Read-only here." in page and "Other is not this services box&#39;s" in page
    assert "Read-only here." not in box.get("/?experiment=Bench&kind=setup").text
    assert (
        box.get("/healthz").json()["owns"] == "experiments Bench; analysis folders Ns"
    )


def test_healthz_names_an_owned_experiment_with_no_folder(origin: Path, tmp_path: Path) -> None:  # fmt: skip
    env = {"GEECS_CONFIGURATOR_EXPERIMENTS": "Bench,bench", "GEECS_CONFIGURATOR_STATE_DIR": str(tmp_path / "s"), "GEECS_CONFIGURATOR_ORIGIN": str(origin), "GEECS_CONFIGURATOR_RECONCILE_SECONDS": "0", "GEECS_CONFIGURATOR_FETCH_SECONDS": "0"}  # fmt: skip
    with TestClient(create_app(Settings.from_env(env))) as c:
        assert c.get("/healthz").json()["owns"] == "experiments Bench, bench (no folder in the configs repo: bench)"  # fmt: skip


def test_the_repository_refuses_a_commit_it_may_not_write(origin: Path, tmp_path: Path) -> None:  # fmt: skip
    me = Identity("Ada via configurator", "a@test.invalid")
    repo = ConfigsRepository(tmp_path / "s", str(origin), "main", me, writable=lambda rel: rel.startswith(f"{EXP}/Bench/"))  # fmt: skip
    repo.ensure()
    head = repo.remote_head()
    with pytest.raises(NotWritable, match="not this service's"):
        repo.commit_file(f"{THEIRS}/x.yaml", b"name: x\n", author=me, message="m", base=head)  # fmt: skip
    with pytest.raises(NotWritable, match="not this service's"):
        repo.push_branch(f"{THEIRS}/x.yaml", b"name: x\n", author=me, message="m", base=head, branch="b")  # fmt: skip
    assert repo.remote_head() == head and list(repo.staging_dir.iterdir()) == []


def test_the_repository_never_writes_through_a_link(origin: Path, other: Path, tmp_path: Path) -> None:  # fmt: skip
    git("pull", "--quiet", "--ff-only", cwd=other)
    os.symlink("setups", other / f"{EXP}/Bench/alias")
    git("add", "-A", cwd=other)
    git("commit", "--quiet", "-m", "a folder link", cwd=other)
    git("push", "--quiet", "origin", "HEAD:main", cwd=other)
    me = Identity("Ada via configurator", "a@test.invalid")
    repo = ConfigsRepository(tmp_path / "s", str(origin), "main", me)
    repo.ensure()
    head = repo.remote_head()
    with pytest.raises(NotWritable, match="symbolic link"):
        repo.commit_file(f"{EXP}/Bench/alias/low_power.yaml", b"name: x\n", author=me, message="m", base=head)  # fmt: skip
    assert repo.remote_head() == head


# ------------------------------------------------------------------ the gate


def test_another_facilitys_invalid_commit_never_blocks_this_box(box: TestClient, other: Path) -> None:  # fmt: skip
    cp = _cp(box)
    bad = push_change(other, f"{THEIRS}/theirs.yaml", INVALID.format(name="theirs"), "Other breaks its own setup")  # fmt: skip
    cp.fetch()
    assert (
        cp.repo.live_head() == bad and cp.main_state() == "applied"
    )  # fast-forwarded unjudged
    assert box.get("/api/log").json() == []  # neither refused nor merged here
    saved = _put(box, "setup", "low_power.yaml", B, {"description": "still saving"}).json()  # fmt: skip
    assert saved["outcome"] == "committed" and cp.repo.live_head() == saved["commit"]


def test_this_boxs_own_invalid_commit_still_blocks(box: TestClient, other: Path) -> None:  # fmt: skip
    cp = _cp(box)
    here = cp.repo.live_head()
    bad = push_change(other, f"{OURS}/low_power.yaml", INVALID.format(name="low_power"), "Bench broken by a push")  # fmt: skip
    cp.fetch()
    assert cp.repo.live_head() == here and cp.main_state().startswith(f"main at {bad[:10]} is not applied")  # fmt: skip
    assert _put(box, "setup", "far.yaml", B, {"description": "x"}).status_code == 502


def test_a_commit_touching_both_is_judged_on_the_owned_paths_alone(box: TestClient, other: Path) -> None:  # fmt: skip
    cp = _cp(box)
    git("pull", "--quiet", "--ff-only", cwd=other)
    (other / THEIRS / "theirs.yaml").write_text(INVALID.format(name="theirs"))
    (other / OURS / "low_power.yaml").write_text(
        SETUP.format(name="low_power", text="both")
    )
    git("commit", "--quiet", "-am", "valid here, invalid there", cwd=other)
    git("push", "--quiet", "origin", "HEAD:main", cwd=other)
    cp.fetch()
    mixed = git("rev-parse", "HEAD", cwd=other)
    assert cp.repo.live_head() == mixed
    # the other way round: invalid here, valid there, refused whole
    (other / THEIRS / "theirs.yaml").write_text(
        SETUP.format(name="theirs", text="fixed")
    )
    (other / OURS / "low_power.yaml").write_text(INVALID.format(name="low_power"))
    git("commit", "--quiet", "-am", "invalid here, valid there", cwd=other)
    git("push", "--quiet", "origin", "HEAD:main", cwd=other)
    cp.fetch()
    assert cp.repo.live_head() == mixed
    (row,) = [r for r in box.get("/api/log").json() if r["source"] == "refused"]
    assert row["path"] == f"{OURS}/low_power.yaml"  # names the owned path, not Other's


def test_an_owned_link_is_judged_when_the_file_it_points_at_changes(box: TestClient, other: Path) -> None:  # fmt: skip
    cp = _cp(box)
    git("pull", "--quiet", "--ff-only", cwd=other)
    os.symlink("../../Other/setups/theirs.yaml", other / OURS / "theirs.yaml")
    git("add", "-A", cwd=other)
    git("commit", "--quiet", "-m", "a link into Other", cwd=other)
    git("push", "--quiet", "origin", "HEAD:main", cwd=other)
    cp.fetch()
    linked = cp.repo.live_head()
    assert linked == git("rev-parse", "HEAD", cwd=other)
    push_change(other, f"{THEIRS}/theirs.yaml", INVALID.format(name="theirs"), "Other's file, which Bench reads by its link")  # fmt: skip
    cp.fetch()
    assert cp.repo.live_head() == linked  # Bench's document changed: judged, refused


# ------------------------------------------------------------------ the log and the share


def test_another_facilitys_commits_are_never_logged_or_applied(box: TestClient, other: Path) -> None:  # fmt: skip
    cp = _cp(box)
    theirs = push_change(other, f"{THEIRS}/theirs.yaml", SETUP.format(name="theirs", text="changed"), "Other's change")  # fmt: skip
    ours = push_change(other, f"{OURS}/low_power.yaml", SETUP.format(name="low_power", text="merged"), "Bench's merge")  # fmt: skip
    cp.fetch()
    assert cp.repo.live_head() == ours
    rows = box.get("/api/log").json()
    assert [(r["path"], r["commit"]) for r in rows] == [
        (f"{OURS}/low_power.yaml", ours)
    ]
    assert box.app.state.recorder.asked == [f"{OURS}/low_power.yaml"]
    assert theirs not in {r["commit"] for r in rows}


def test_the_share_follows_main_when_only_other_facilities_moved_it(origin: Path, other: Path, tmp_path: Path) -> None:  # fmt: skip
    _seed_other(other)
    share = tmp_path / "share"
    git("clone", "--quiet", str(origin), str(share), cwd=tmp_path)
    env = {**OWNED, "GEECS_CONFIGURATOR_STATE_DIR": str(tmp_path / "s"), "GEECS_CONFIGURATOR_ORIGIN": str(origin), "GEECS_CONFIGURATOR_RECONCILE_SECONDS": "0", "GEECS_CONFIGURATOR_FETCH_SECONDS": "0", "GEECS_CONFIGURATOR_SHARE_CLONE": str(share)}  # fmt: skip
    with TestClient(create_app(Settings.from_env(env))) as c:
        cp = _cp(c)
        assert not cp.unverified  # started at the share's commit
        theirs = push_change(other, f"{THEIRS}/theirs.yaml", INVALID.format(name="theirs"), "Other's alone")  # fmt: skip
        cp.fetch()
        cp.wait_share()
        assert git("rev-parse", "HEAD", cwd=share) == theirs == cp.repo.live_head()
        assert c.get("/healthz").json()["share"] == "current"


# ------------------------------------------------------------------ the inbox


def test_the_inbox_publishes_only_the_owned_experiments(box: TestClient, origin: Path) -> None:  # fmt: skip
    cp = _cp(box)
    assert (cp.repo.inbox / f"{EXP}/Bench").is_dir() and not (cp.repo.inbox / f"{EXP}/Other").exists()  # fmt: skip
    foreign = cp.repo.inbox / f"{EXP}/Other/shot_offsets.yaml"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("offsets: {}\n")
    head = git("rev-parse", "main", cwd=origin)
    assert cp.reconcile() == [] and git("rev-parse", "main", cwd=origin) == head
    assert foreign.exists()  # left alone, never published from this box


def test_settle_never_sees_another_facilitys_row(box: TestClient, other: Path) -> None:  # fmt: skip
    cp, recorder = _cp(box), box.app.state.recorder
    ours = push_change(other, f"{OURS}/low_power.yaml", SETUP.format(name="low_power", text="merged"), "Bench's merge")  # fmt: skip
    cp.fetch()  # queued by the recorder's apply
    # a queued row of another facility's path, as a log from before the box owned less would hold
    cp.log.add(Row(source="merged", who="Seed", kind="setup", path=f"{THEIRS}/theirs.yaml", commit=ours, state="queued"))  # fmt: skip
    assert cp.settle_queued() == 1
    assert recorder.settled == [f"{OURS}/low_power.yaml"]


def test_a_foreign_save_is_refused_before_the_dry_run(origin: Path, other: Path, tmp_path: Path) -> None:  # fmt: skip
    _seed_other(other)
    ran: list[list[str]] = []

    def run(argv: object, timeout: float) -> tuple[int, str]:
        ran.append(list(argv))  # type: ignore[call-overload]
        return 0, "plan: 1 to archive"

    env = {**OWNED, "GEECS_CONFIGURATOR_STATE_DIR": str(tmp_path / "s"), "GEECS_CONFIGURATOR_ORIGIN": str(origin), "GEECS_CONFIGURATOR_RECONCILE_SECONDS": "0", "GEECS_CONFIGURATOR_FETCH_SECONDS": "0", "GEECS_CONFIGURATOR_APPLY_SECONDS": "0", "GEECS_CONFIGURATOR_ARCHIVER_COMMAND": "geecs-archiver"}  # fmt: skip
    with TestClient(create_app(Settings.from_env(env), archiver_run=run)) as c:
        r = _put(c, "archive_policy", "archive_policy.yaml", OQ, {"include_setpoints": False})  # fmt: skip
        assert (
            r.status_code == 403 and ran == []
        )  # no consumer is asked about another box's draft
        assert _put(c, "archive_policy", "archive_policy.yaml", B, {"include_setpoints": False}).status_code != 403  # fmt: skip
        assert ran  # this box's own draft reaches the dry run


def _link(
    clone: Path, target: str, at: str, message: str, *, replace: bool = False
) -> str:
    """Commit a symbolic link ``at -> target`` (``replace``: the folder there removed first) and push it."""
    git("pull", "--quiet", "--ff-only", cwd=clone)
    if replace:
        shutil.rmtree(clone / at)
    (clone / at).parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, clone / at)
    git("add", "-A", cwd=clone)
    git("commit", "--quiet", "-m", message, cwd=clone)
    git("push", "--quiet", "origin", "HEAD:main", cwd=clone)
    return git("rev-parse", "HEAD", cwd=clone)


def test_a_folder_link_carries_its_targets_changes_to_the_gate_and_the_log(box: TestClient, other: Path) -> None:  # fmt: skip
    cp, recorder = _cp(box), box.app.state.recorder
    linked = _link(other, "../Other/setups", OURS, "Bench's setups are Other's", replace=True)  # fmt: skip
    cp.fetch()
    assert cp.repo.live_head() == linked
    assert (
        f"{OURS}/theirs.yaml" in recorder.asked
    )  # the link made Other's setup Bench's: judged and logged
    here = cp.repo.live_head()
    push_change(other, f"{THEIRS}/theirs.yaml", INVALID.format(name="theirs"), "Other's file, Bench's document")  # fmt: skip
    cp.fetch()
    assert cp.repo.live_head() == here  # judged as Bench's, refused
    recorder.asked.clear()
    fixed = push_change(other, f"{THEIRS}/theirs.yaml", SETUP.format(name="theirs", text="fixed"), "fixed")  # fmt: skip
    cp.fetch()
    # logged and applied as Bench's (once per commit of the range: the refused one and its fix)
    assert cp.repo.live_head() == fixed and set(recorder.asked) == {
        f"{OURS}/theirs.yaml"
    }


def test_an_owned_experiment_that_is_a_link_is_judged_through_it(origin: Path, other: Path, tmp_path: Path) -> None:  # fmt: skip
    _seed_other(other)
    _link(other, "Other", f"{EXP}/Mine", "Mine is Other")
    env = {"GEECS_CONFIGURATOR_EXPERIMENTS": "Mine", "GEECS_CONFIGURATOR_STATE_DIR": str(tmp_path / "s"), "GEECS_CONFIGURATOR_ORIGIN": str(origin), "GEECS_CONFIGURATOR_RECONCILE_SECONDS": "0", "GEECS_CONFIGURATOR_FETCH_SECONDS": "0"}  # fmt: skip
    with TestClient(create_app(Settings.from_env(env))) as c:
        cp = _cp(c)
        cp.unverified_marker.unlink()
        here = cp.repo.live_head()
        push_change(other, f"{THEIRS}/theirs.yaml", INVALID.format(name="theirs"), "Other's file, read as Mine's")  # fmt: skip
        cp.fetch()
        assert cp.repo.live_head() == here
