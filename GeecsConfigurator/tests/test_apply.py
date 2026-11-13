"""The applies and the dry runs: the worker's reopen through the scanner, the archiver's onboard, the save's gate.

The scanner is an httpx mock transport and the archiver a recorded
subprocess; nothing reaches a network, a manager or an appliance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from geecs_configurator.app import create_app
from geecs_configurator.apply import Archiver, ScannerClient, ServiceApplier
from geecs_configurator.log import Reader, Row
from geecs_configurator.settings import Settings

B = "?experiment=Bench"


def _row(kind: str, path: str, **readers: str) -> Row:
    """A committed row, queued, with any readers it already holds (name -> state)."""
    return Row(source="saved", who="Ada", kind=kind, path=path, state="queued", readers={n: Reader(state=s) for n, s in readers.items()})  # type: ignore[misc]  # fmt: skip


PROFILE = "scanner_configs/experiments/Bench/shot_control_configurations/laser_off.yaml"
POLICY = "scanner_configs/experiments/Bench/archiver/archive_policy.yaml"


class Scanner:
    """The scanner's routes, recorded: the reopen answers from a script, the dry run from ``refusal``."""

    def __init__(
        self, reopens: list[int] | None = None, refusal: str | None = None
    ) -> None:
        self.reopens = list(reopens or [200])
        self.refusal = refusal
        self.calls: list[tuple[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"null")
        self.calls.append((request.url.path, body))
        if request.url.path == "/api/environment/reopen":
            status = self.reopens.pop(0) if len(self.reopens) > 1 else self.reopens[0]
            if status == 200:
                return httpx.Response(
                    200, json={"ok": True, "message": "reopened", "seconds": 31.0}
                )
            return httpx.Response(status, json={"error": {"kind": "policy_refusal", "message": "a reopen is idle-only: a plan is running"}})  # fmt: skip
        if request.url.path == "/api/preflight/trigger-profile":
            return httpx.Response(200, json={"refusal": self.refusal, "outcomes": [{"check": "validate", "result": "passed", "detail": ""}], "questions": []})  # fmt: skip
        if request.url.path == "/api/readback":
            return httpx.Response(200, json={"ok": True, "value": 4.0})
        return httpx.Response(404, json={"detail": "no route"})

    def reopen_calls(self) -> int:
        return sum(1 for path, _ in self.calls if path == "/api/environment/reopen")


class Run:
    """The archiver's CLI, recorded: each call answers the next scripted exit code."""

    def __init__(self, *codes: int, output: str = "plan: 3 to archive") -> None:
        self.codes, self.output = list(codes) or [0], output
        self.argv: list[list[str]] = []

    def __call__(self, argv: Any, timeout: float) -> tuple[int, str]:
        self.argv.append(list(argv))
        code = self.codes.pop(0) if len(self.codes) > 1 else self.codes[0]
        return code, self.output


def _http(scanner: Scanner) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(scanner))


def _app(origin: Path, state: Path, scanner: Scanner | None = None, run: Run | None = None) -> TestClient:  # fmt: skip
    env = {"GEECS_CONFIGURATOR_STATE_DIR": str(state), "GEECS_CONFIGURATOR_ORIGIN": str(origin), "GEECS_CONFIGURATOR_RECONCILE_SECONDS": "0", "GEECS_CONFIGURATOR_FETCH_SECONDS": "0", "GEECS_CONFIGURATOR_APPLY_SECONDS": "0", "GEECS_CONFIGURATOR_EXPERIMENT": "Bench"}  # fmt: skip
    if scanner is not None:
        env["GEECS_CONFIGURATOR_SCANNER_URL"] = "http://scanner.test"
    if run is not None:
        env["GEECS_CONFIGURATOR_ARCHIVER_COMMAND"] = "geecs-archiver"
    return TestClient(create_app(Settings.from_env(env), consumer_http=_http(scanner) if scanner else None, archiver_run=run))  # fmt: skip


def _save_profile(c: TestClient, value: str = "4.5") -> httpx.Response:
    doc = c.get(f"/api/documents/trigger_profile/laser_off.yaml{B}").json()
    doc["document"]["states"]["SCAN"][0]["value"] = value
    return c.put(f"/api/documents/trigger_profile/laser_off.yaml{B}", json={"document": doc["document"], "etag": doc["etag"], "operator": "Ada"})  # fmt: skip


def _rows(c: TestClient) -> list[dict]:
    return c.get("/api/log").json()


# ------------------------------------------------------------------ applies


def test_a_saved_trigger_profile_is_queued_then_applied_by_one_reopen(origin: Path, tmp_path: Path) -> None:  # fmt: skip
    scanner = Scanner()
    with _app(origin, tmp_path / "s", scanner) as c:
        assert _save_profile(c, "4.5").status_code == 200
        assert _save_profile(c, "4.6").status_code == 200
        assert [r["state"] for r in _rows(c)] == ["queued", "queued"]
        assert scanner.reopen_calls() == 0  # a save never waits on the worker
        assert c.app.state.commit.settle_queued() == 2
        assert scanner.reopen_calls() == 1  # one reopen applies both
        assert [r["state"] for r in _rows(c)] == ["applied", "applied"]
        assert "reopened" in _rows(c)[0]["detail"]
        assert c.app.state.commit.settle_queued() == 0  # nothing left to try


def test_a_busy_worker_keeps_the_row_queued_and_it_is_tried_again(origin: Path, tmp_path: Path) -> None:  # fmt: skip
    scanner = Scanner(reopens=[409, 200])
    with _app(origin, tmp_path / "s", scanner) as c:
        _save_profile(c)
        c.app.state.commit.settle_queued()
        (row,) = _rows(c)
        assert row["state"] == "queued" and "idle-only" in row["detail"]
        c.app.state.commit.settle_queued()
        assert _rows(c)[0]["state"] == "applied" and scanner.reopen_calls() == 2


def test_an_unreachable_scanner_keeps_the_row_queued(
    origin: Path, tmp_path: Path
) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    applier = ServiceApplier(tmp_path, scanners={"Bench": ScannerClient("http://scanner.test", httpx.Client(transport=httpx.MockTransport(down)))})  # fmt: skip
    (answer,) = applier.settle([_row("trigger_profile", PROFILE)])
    assert answer.state == "queued" and "did not answer" in answer.detail


def test_without_a_scanner_the_reopen_is_the_maintainers(
    origin: Path, tmp_path: Path
) -> None:
    with _app(origin, tmp_path / "s") as c:
        _save_profile(c)
        (row,) = _rows(c)
        assert row["state"] == "pending" and "no scanner configured" in row["detail"]


def test_the_settle_pass_holds_the_repository_lock(
    origin: Path, tmp_path: Path
) -> None:
    """No fast-forward may rewrite a file while the worker reopens and reads it."""
    held: list[bool] = []

    class Watching(Scanner):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/environment/reopen":
                import threading

                def probe() -> (
                    None
                ):  # another thread: an RLock is re-entrant for its holder
                    got = c.app.state.repo.lock.acquire(blocking=False)
                    if got:
                        c.app.state.repo.lock.release()
                    held.append(not got)

                t = threading.Thread(target=probe)
                t.start()
                t.join()
            return super().__call__(request)

    with _app(origin, tmp_path / "s", Watching()) as c:
        _save_profile(c)
        c.app.state.commit.settle_queued()
    assert held == [True]


def test_a_reader_still_pending_keeps_the_row_pending_after_the_reopen(
    tmp_path: Path,
) -> None:
    """native_devices: the worker reopens, the archiver onboards, Canopy's cache is not surveyed: pending, every word named."""
    run = Run(0)
    applier = ServiceApplier(tmp_path, scanners={"Bench": ScannerClient("http://s.test", _http(Scanner()))}, archiver=Archiver("geecs-archiver", run=run))  # fmt: skip
    (answer,) = applier.settle([_row("native_devices", "scanner_configs/experiments/Bench/native_devices.yaml")])  # fmt: skip
    assert answer.state == "pending"
    assert "reopened" in answer.detail and "onboard" in answer.detail and "not yet surveyed" in answer.detail  # fmt: skip


@pytest.mark.parametrize(
    ("code", "state", "word"),
    [
        (0, "applied", "sent the plan"),
        (1, "applied", "drift remains"),
        (2, "pending", "maintainer"),
        (3, "queued", "tried again"),
    ],  # fmt: skip
)
def test_the_archive_policys_onboard_answers_by_exit_code(tmp_path: Path, code: int, state: str, word: str) -> None:  # fmt: skip
    run = Run(code)
    (tmp_path / POLICY).parent.mkdir(parents=True)
    (tmp_path / POLICY).write_text("include_setpoints: true\n")
    applier = ServiceApplier(tmp_path, archiver=Archiver("poetry run geecs-archiver", run=run))  # fmt: skip
    (answer,) = applier.settle([_row("archive_policy", POLICY)])
    assert (answer.state, word in answer.detail) == (state, True)
    assert run.argv == [["poetry", "run", "geecs-archiver", "onboard", "--experiment", "Bench", "--policy", str(tmp_path / POLICY), "--wait", "0"]]  # fmt: skip


def test_one_pass_onboards_once_for_many_rows(tmp_path: Path) -> None:
    run = Run(0)
    applier = ServiceApplier(tmp_path, archiver=Archiver("geecs-archiver", run=run))
    applier.settle([_row("archive_policy", POLICY), _row("archive_policy", POLICY), _row("panel_devices", "scanner_configs/experiments/Bench/panel_devices.yaml")])  # fmt: skip
    assert len(run.argv) == 1


def test_a_per_request_kind_is_nothing_to_do(tmp_path: Path) -> None:
    applier = ServiceApplier(tmp_path, scanners={"Bench": ScannerClient("http://s.test", _http(Scanner()))})  # fmt: skip
    assert applier.apply("setup", "x.yaml").state == "nothing to do"
    (answer,) = applier.settle([_row("setup", "x.yaml")])
    assert answer.state == "nothing to do"


# ------------------------------------------------------------------ dry runs


def test_a_trigger_profile_save_is_refused_when_the_workers_check_refuses(origin: Path, tmp_path: Path) -> None:  # fmt: skip
    scanner = Scanner(refusal="trigger profile 'laser_off' names no trigger device")
    with _app(origin, tmp_path / "s", scanner) as c:
        head = c.app.state.repo.remote_head()
        r = _save_profile(c)
        assert (
            r.status_code == 422
            and "names no trigger device" in r.json()["detail"]["reason"]
        )
        assert c.app.state.repo.remote_head() == head and _rows(c) == []
        sent = [
            body
            for path, body in scanner.calls
            if path == "/api/preflight/trigger-profile"
        ]
        assert (
            sent[0]["document"]["states"]["SCAN"][0]["value"] == "4.5"
        )  # the unsaved draft


def test_the_dry_run_route_checks_a_draft_and_says_when_there_is_none(origin: Path, tmp_path: Path) -> None:  # fmt: skip
    with _app(origin, tmp_path / "s", Scanner()) as c:
        r = c.post(f"/api/dry-run/trigger_profile{B}", json={"document": {"name": "x"}})
        assert r.status_code == 200 and r.json()["ran"] and r.json()["by"] == "scanner"
        r = c.post(f"/api/dry-run/setup{B}", json={"document": {}})
        assert r.json()["ran"] is False and "no consumer check" in r.json()["detail"]
        assert c.post(f"/api/dry-run/nope{B}", json={"document": {}}).status_code == 404


def test_the_archivers_dry_run_reads_the_draft_from_a_temporary_file(
    tmp_path: Path,
) -> None:
    seen: list[str] = []

    def run(argv: Any, timeout: float) -> tuple[int, str]:
        seen.append(Path(argv[argv.index("--policy") + 1]).read_text(encoding="utf-8"))
        assert "--dry-run" in argv
        return 0, "plan: 2 to archive\n"

    applier = ServiceApplier(tmp_path, archiver=Archiver("geecs-archiver", run=run))
    out = applier.dry_run("archive_policy", "Bench", {"include_setpoints": False})
    assert out.ran and out.refusal is None and "include_setpoints: false" in seen[0]


@pytest.mark.parametrize(
    ("code", "output", "refused", "ran"),
    [
        (2, "geecs-archiver: no appliance URL (set GEECS_ARCHIVER_URL)", False, False),
        (2, "refusing to pause or abort 300 PVs (> 50) without --yes", False, True),
        (3, "geecs-archiver: unreachable", False, False),
        (
            1,
            "Traceback (most recent call last):\nmysql.connector.errors.InterfaceError",
            False,
            False,
        ),
    ],
)
def test_the_archivers_dry_run_never_refuses_and_runs_only_with_a_plan(tmp_path: Path, code: int, output: str, refused: bool, ran: bool) -> None:  # fmt: skip
    applier = ServiceApplier(tmp_path, archiver=Archiver("geecs-archiver", run=Run(code, output=output)))  # fmt: skip
    out = applier.dry_run("archive_policy", "Bench", {})
    assert (out.refusal is not None, out.ran) == (refused, ran)


def test_a_dry_run_that_could_not_run_is_noted_never_a_refusal(origin: Path, tmp_path: Path) -> None:  # fmt: skip
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    env = {"GEECS_CONFIGURATOR_STATE_DIR": str(tmp_path / "s"), "GEECS_CONFIGURATOR_ORIGIN": str(origin), "GEECS_CONFIGURATOR_RECONCILE_SECONDS": "0", "GEECS_CONFIGURATOR_FETCH_SECONDS": "0", "GEECS_CONFIGURATOR_APPLY_SECONDS": "0", "GEECS_CONFIGURATOR_SCANNER_URL": "http://scanner.test", "GEECS_CONFIGURATOR_EXPERIMENT": "Bench"}  # fmt: skip
    with TestClient(create_app(Settings.from_env(env), consumer_http=httpx.Client(transport=httpx.MockTransport(down)))) as c:  # fmt: skip
        r = _save_profile(c)
        assert r.status_code == 200 and any(
            "did not run" in n for n in r.json()["notes"]
        )


def test_the_readback_goes_through_the_scanner(origin: Path, tmp_path: Path) -> None:
    with _app(origin, tmp_path / "s", Scanner()) as c:
        r = c.get("/api/readback", params={"variable": "U_DG645:Amplitude.Ch AB"})
        assert r.status_code == 200 and r.json()["value"] == 4.0
    with _app(origin, tmp_path / "t") as c:
        assert c.get("/api/readback", params={"variable": "A:B"}).status_code == 503


# ------------------------------------------------------------ review round 1

NATIVE = "scanner_configs/experiments/Bench/native_devices.yaml"


def test_a_reader_already_applied_is_never_redone_while_another_waits(tmp_path: Path) -> None:  # fmt: skip
    """The worker reopened; the appliance is down: the next pass retries the onboard alone, never the reopen."""
    scanner, run = Scanner(), Run(3, output="geecs-archiver: appliance unreachable")
    applier = ServiceApplier(tmp_path, scanners={"Bench": ScannerClient("http://s.test", _http(scanner))}, archiver=Archiver("geecs-archiver", run=run))  # fmt: skip
    (first,) = applier.settle([_row("native_devices", NATIVE)])
    assert first.state == "queued" and first.readers["worker"].state == "applied"
    assert first.readers["archiver (its registry pass)"].state == "queued"
    row = Row(source="saved", who="Ada", kind="native_devices", path=NATIVE, **first.row_fields())  # fmt: skip
    (second,) = applier.settle([row])
    assert (
        scanner.reopen_calls() == 1 and len(run.argv) == 2
    )  # the onboard again, the reopen not
    assert second.readers["worker"].state == "applied"


def test_an_archiver_crash_is_never_applied(tmp_path: Path) -> None:
    """A traceback exits 1 like drift, but sends nothing: no plan line, no applied."""
    run = Run(1, output="Traceback (most recent call last):\nmysql.connector.errors.InterfaceError: 2003")  # fmt: skip
    applier = ServiceApplier(tmp_path, archiver=Archiver("geecs-archiver", run=run))
    (answer,) = applier.settle([_row("archive_policy", POLICY)])
    assert answer.state == "queued" and "did not complete" in answer.detail


def test_without_a_policy_file_onboard_uses_the_archivers_default(
    tmp_path: Path,
) -> None:
    run = Run(0)
    applier = ServiceApplier(tmp_path, archiver=Archiver("geecs-archiver", run=run))
    applier.settle([_row("native_devices", NATIVE)])
    assert "--policy" not in run.argv[0]


def test_another_experiments_reopen_is_its_own_scanners_or_the_maintainers(tmp_path: Path) -> None:  # fmt: skip
    """One scanner per experiment: a PulsedWire row never reads applied off the main worker's reopen."""
    main, bench = Scanner(), Scanner()
    other = "scanner_configs/experiments/PulsedWire/shot_control_configurations/laser_off.yaml"
    applier = ServiceApplier(tmp_path, scanners={"Bench": ScannerClient("http://main.test", _http(main))})  # fmt: skip
    (answer,) = applier.settle([_row("trigger_profile", other)])
    assert answer.state == "pending" and "PulsedWire" in answer.detail and main.reopen_calls() == 0  # fmt: skip
    applier = ServiceApplier(tmp_path, scanners={"Bench": ScannerClient("http://main.test", _http(main)), "PulsedWire": ScannerClient("http://pw.test", _http(bench))})  # fmt: skip
    (answer,) = applier.settle([_row("trigger_profile", other)])
    assert answer.state == "applied" and bench.reopen_calls() == 1 and main.reopen_calls() == 0  # fmt: skip


def test_the_scanner_url_names_each_experiments_scanner() -> None:
    from geecs_configurator.settings import Settings as S

    one = S(configs_root=Path("."), experiment="Bench", scanner_url="http://a:8300/")
    assert one.scanner_urls() == {"Bench": "http://a:8300"}
    many = S(configs_root=Path("."), scanner_url="Bench=http://a:8300, PulsedWire=http://a:8310")  # fmt: skip
    assert many.scanner_urls() == {"Bench": "http://a:8300", "PulsedWire": "http://a:8310"}  # fmt: skip


def test_a_save_during_an_apply_pass_is_answered_busy_at_once(origin: Path, tmp_path: Path) -> None:  # fmt: skip
    with _app(origin, tmp_path / "s", Scanner()) as c:
        head = c.app.state.repo.remote_head()
        c.app.state.commit.settling.set()
        r = _save_profile(c)
        assert r.status_code == 503 and r.headers["Retry-After"] == "60"
        assert c.app.state.repo.remote_head() == head


# ------------------------------------------------------------ review round 2


@pytest.mark.parametrize(
    "output",
    [
        "Bench: 120 PVs; plan: 3 to archive\nrefusing to change the appliance while 1 registry device(s) are unresolved (nothing was sent; fix the registry)",  # fmt: skip
        "Bench: 120 PVs; plan: 3 to archive\nTraceback (most recent call last):\nMgmtTimeout",
    ],
)
def test_a_plan_printed_then_nothing_sent_is_never_applied(tmp_path: Path, output: str) -> None:  # fmt: skip
    applier = ServiceApplier(tmp_path, archiver=Archiver("geecs-archiver", run=Run(1, output=output)))  # fmt: skip
    (answer,) = applier.settle([_row("archive_policy", POLICY)])
    # a refusal no retry clears is the maintainer's; a crash is tried again
    assert answer.state == ("pending" if "nothing was sent" in output else "queued")


def test_a_pass_that_began_during_the_dry_run_answers_the_save_busy(origin: Path, tmp_path: Path) -> None:  # fmt: skip
    class Starting(Scanner):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/preflight/trigger-profile":
                c.app.state.commit.settling.set()  # the timer's pass starts meanwhile
            return super().__call__(request)

    with _app(origin, tmp_path / "s", Starting()) as c:
        head = c.app.state.repo.remote_head()
        assert _save_profile(c).status_code == 503
        assert c.app.state.repo.remote_head() == head


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "http://a:8300/?x=1",
            {"Bench": "http://a:8300/?x=1"},
        ),  # one URL, an "=" in its query
        (
            "Bench=http://a:8300,PulsedWire=http://a:8310",
            {"Bench": "http://a:8300", "PulsedWire": "http://a:8310"},
        ),  # fmt: skip
    ],
)
def test_the_scanner_url_is_one_url_or_every_part_named(value: str, expected: dict) -> None:  # fmt: skip
    env = {"GEECS_CONFIGURATOR_CONFIGS_ROOT": ".", "GEECS_EXPERIMENT": "Bench", "GEECS_CONFIGURATOR_SCANNER_URL": value}  # fmt: skip
    assert Settings.from_env(env).scanner_urls() == expected


def test_a_scanner_url_naming_only_some_experiments_is_refused() -> None:
    from geecs_configurator.settings import SettingsError

    env = {"GEECS_CONFIGURATOR_CONFIGS_ROOT": ".", "GEECS_EXPERIMENT": "Bench", "GEECS_CONFIGURATOR_SCANNER_URL": "Bench=http://a,http://b"}  # fmt: skip
    with pytest.raises(SettingsError, match="not every one"):
        Settings.from_env(env)
