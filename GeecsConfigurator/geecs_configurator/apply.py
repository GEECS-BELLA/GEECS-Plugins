"""The apply seam implemented, and the consumers' dry runs: the scanner over HTTP, the archiver by its CLI.

Each file kind's readers and their apply verbs are ``kinds.CONSUMERS``; this
module turns a verb into what the service does about it:

- **nothing to do** (the reader reads per request): nothing;
- **the worker's environment reopen** (:data:`kinds.REOPEN`: trigger
  profiles, the scan-variable catalogue, the native-device registry, the
  shot offsets): the scanner's ``POST /api/environment/reopen``, which
  closes and reopens the environment only while nothing runs and nothing
  waits, under its own lock. Refused (busy), the row stays **queued** and
  is tried again; a reopen the scanner could not finish stays queued too
  (the readiness unit reopens a closed environment, and the next try then
  finds it idle);
- **the archiver's onboard** (:data:`kinds.ONBOARD`): ``geecs-archiver
  onboard --policy <the committed file> --wait 0`` as a subprocess (the
  archiver has no HTTP API); its mass-pause guard (exit 2) hands the row to
  the maintainer, **pending**;
- anything else (a gateway restart, Canopy's not-yet-surveyed cache): **pending**,
  the verb named, the maintainer's.

:meth:`ServiceApplier.apply` is fast and blocks on nothing: a verb with
work is answered **queued**; :meth:`ServiceApplier.settle` does the work
(the service's apply thread, off every request), retrying only the readers a
row still holds as queued. One settle pass reopens each experiment's worker at
most once and onboards each at most once for all the rows it was given, and the
service holds the repository's lock through it, so no fast-forward rewrites
a file while the worker or the archiver reads it. A row's state is the
least finished of its readers': queued, then pending, then applied, then
nothing to do. A committed document is never written up as applied by that
fact alone: applied is the consumer's own answer.

The dry runs are the consumers' own checks of an unsaved draft, never
re-implemented here: the scanner's ``POST /api/preflight/trigger-profile``,
and ``geecs-archiver onboard --dry-run --policy <draft>`` over a temporary
file.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from geecs_configurator.kinds import CONSUMERS, ONBOARD, REOPEN
from geecs_configurator.log import Answer, Reader, Row
from geecs_configurator.store import Json, dump_yaml

logger = logging.getLogger(__name__)

#: The order a row's state is decided in: the least finished reader wins.
_ORDER = ("queued", "pending", "applied", "nothing to do")
#: How long one reopen may take (seconds): the scanner waits out the open, which builds the namespace.
REOPEN_TIMEOUT_S = 900.0
#: How long the archiver's CLI may take (seconds): it reads the DB and asks the appliance.
ARCHIVER_TIMEOUT_S = 300.0
#: The archiver's exit codes (``geecs_archiver.cli``): plan sent (or nothing to do), drift, usage or guard, unreachable.
EXIT_OK, EXIT_DRIFT, EXIT_USAGE, EXIT_UNREACHABLE = 0, 1, 2, 3


class DryRun(BaseModel):
    """A consumer's own check of an unsaved draft."""

    kind: str
    by: str  # who checked: "scanner", "archiver", or "none"
    ran: bool  # False: there is no check, or it could not be asked (then ``detail`` says why)
    refusal: str | None = None  # the consumer refuses the draft: the save is refused
    outcomes: list[dict[str, str]] = []
    questions: list[dict[str, str]] = []
    output: str = ""
    detail: str = ""


def _one_line(text: str, limit: int = 300) -> str:
    return " ".join(str(text).split())[:limit]


class ScannerClient:
    """The scanner's routes the configurator calls (``GEECS_CONFIGURATOR_SCANNER_URL``)."""

    def __init__(self, base_url: str, http: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self.http = http or httpx.Client(timeout=REOPEN_TIMEOUT_S)

    def reopen(self, reason: str) -> Answer:
        """Ask for the reopen: ``("applied", …)``, or ``("queued", why)`` when it did not happen (busy, failed, unreachable)."""
        try:
            r = self.http.post(f"{self.base_url}/api/environment/reopen", json={"operator": "configurator", "reason": reason}, timeout=REOPEN_TIMEOUT_S)  # fmt: skip
        except httpx.HTTPError as exc:
            return Answer(
                state="queued",
                detail=f"the scanner did not answer ({_one_line(exc)}); tried again",
            )
        if r.status_code == 200:
            return Answer(
                state="applied",
                detail=f"the worker environment reopened ({r.json().get('seconds', '?')} s)",
            )
        message = _error_message(r)
        if r.status_code == 409:
            return Answer(
                state="queued", detail=f"waits for the worker idle and empty: {message}"
            )
        return Answer(
            state="queued",
            detail=f"the reopen did not complete ({r.status_code}: {message}); tried again",
        )

    def trigger_profile_dry_run(self, document: Json) -> DryRun:
        """The worker's checks of a draft trigger profile, through the scanner."""
        try:
            r = self.http.post(f"{self.base_url}/api/preflight/trigger-profile", json={"document": document}, timeout=60.0)  # fmt: skip
        except httpx.HTTPError as exc:
            return DryRun(kind="trigger_profile", by="scanner", ran=False, detail=f"the scanner did not answer: {_one_line(exc)}")  # fmt: skip
        if r.status_code != 200:
            return DryRun(kind="trigger_profile", by="scanner", ran=False, detail=f"the scanner answered {r.status_code}: {_error_message(r)}")  # fmt: skip
        out = r.json()
        return DryRun(kind="trigger_profile", by="scanner", ran=True, refusal=out.get("refusal"), outcomes=out.get("outcomes") or [], questions=out.get("questions") or [])  # fmt: skip

    def readback(self, variable: str, *, word: bool = False) -> tuple[int, Any]:  # Any: the scanner's JSON, passed through  # fmt: skip
        """One live readback of ``Device:Variable`` through the scanner; ``word`` reads it as text (an enum's state)."""
        params = {"variable": variable, **({"word": "true"} if word else {})}
        r = self.http.get(f"{self.base_url}/api/readback", params=params, timeout=10.0)  # fmt: skip
        return r.status_code, r.json()

    def relay(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:  # Any: the scanner's JSON, passed through  # fmt: skip
        """One scanner route; its status and JSON body as they came.

        The body goes as given and nothing is added to it: an apply's
        ``acknowledged`` is the operator's answer, never this service's.
        """
        r = self.http.request(method, f"{self.base_url}{path}", json=body, timeout=30.0)  # fmt: skip
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {"detail": _one_line(r.text)}


def _error_message(r: httpx.Response) -> str:
    """The scanner's ``{"error": {"message"}}`` or FastAPI's ``detail``, one line."""
    try:
        body = r.json()
    except ValueError:
        return _one_line(r.text)
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return _one_line(error.get("message", ""))
    return _one_line(body.get("detail", body) if isinstance(body, dict) else body)


#: ``(argv, timeout) -> (exit code, combined output)``: the subprocess seam tests replace.
Run = Callable[[Sequence[str], float], tuple[int, str]]


def _run(argv: Sequence[str], timeout: float) -> tuple[int, str]:
    done = subprocess.run(
        list(argv), capture_output=True, text=True, timeout=timeout, check=False
    )
    return done.returncode, (done.stdout + done.stderr)


class Archiver:
    """The archiver's CLI (``GEECS_CONFIGURATOR_ARCHIVER_COMMAND``, run in the archiver's own environment)."""

    def __init__(self, command: str, run: Run | None = None):
        self.command = shlex.split(command)
        self._run = run or _run

    def _onboard(self, experiment: str, policy: Path | None, *extra: str) -> tuple[int, str]:  # fmt: skip
        """One ``onboard``; no ``--policy`` when there is no file, so the archiver uses its own default."""
        argv = [*self.command, "onboard", "--experiment", experiment, *(["--policy", str(policy)] if policy else []), *extra]  # fmt: skip
        try:
            return self._run(argv, ARCHIVER_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError) as exc:
            return EXIT_UNREACHABLE, f"the archiver's command did not run: {exc}"

    def dry_run(self, experiment: str, document: Json) -> DryRun:
        """``onboard --dry-run --policy <draft>``: the plan the unsaved policy would send; nothing is sent.

        Never a refusal: the draft has already passed the schema the archiver
        loads it with, so what the archiver says (its plan, its mass-pause
        guard, a missing appliance URL) is information about the plan, shown
        on the save.
        """
        with tempfile.TemporaryDirectory(prefix="configurator-draft-") as tmp:
            draft = Path(tmp) / "archive_policy.yaml"
            draft.write_text(dump_yaml(document), encoding="utf-8")
            code, output = self._onboard(experiment, draft, "--dry-run")
        out = output[-4000:]
        if _planned(code, output):
            drift = "; drift exists already (never-connected or unresolved PVs, not the draft's)" if code == EXIT_DRIFT else ""  # fmt: skip
            return DryRun(kind="archive_policy", by="archiver", ran=True, output=out, detail=f"{_plan_line(output)}{drift}")  # fmt: skip
        guard = next(
            (ln for ln in output.splitlines() if "refusing to pause" in ln), ""
        )
        if code == EXIT_USAGE and guard:  # the mass-pause guard: a fact about the plan
            return DryRun(kind="archive_policy", by="archiver", ran=True, output=out, detail=_one_line(guard))  # fmt: skip
        return DryRun(kind="archive_policy", by="archiver", ran=False, output=out, detail=f"the archiver could not plan (exit {code}): {_last_line(output)}")  # fmt: skip

    def onboard(self, experiment: str, policy: Path | None) -> Answer:
        """Reconcile the appliance with the committed policy; this reader's state.

        Applied only when the archiver printed its plan (exit 0, or 1 for
        drift it reports): a crash also exits 1 and sends nothing, so it is
        queued and tried again, never applied.
        """
        code, output = self._onboard(experiment, policy, "--wait", "0")
        if _planned(code, output):
            drift = "; drift remains (see its output)" if code == EXIT_DRIFT else ""
            return Answer(state="applied", detail=f"geecs-archiver onboard sent the plan ({_plan_line(output)}){drift}")  # fmt: skip
        if code == EXIT_USAGE:
            return Answer(state="pending", detail=f"geecs-archiver onboard refused ({_last_line(output)}); the maintainer's, by hand (--yes only if a mass pause is meant)")  # fmt: skip
        if (
            "nothing was sent" in output
        ):  # a refusal no retry can clear (unresolved registry devices)
            return Answer(state="pending", detail=f"geecs-archiver onboard refused ({_last_line(output)}); the maintainer fixes the registry, then runs it by hand")  # fmt: skip
        return Answer(state="queued", detail=f"geecs-archiver onboard did not complete (exit {code}: {_last_line(output)}); tried again")  # fmt: skip


def _planned(code: int, output: str) -> bool:
    """Whether onboard got past its plan: exit 0 or 1, its ``plan:`` line, and nothing after it that sent nothing.

    Exit 1 is the archiver's drift code, but a traceback exits 1 too, before
    or after the plan line, and the unresolved-registry guard prints the
    plan, refuses ("nothing was sent") and exits 1.
    """
    stopped = "Traceback (most recent call last)" in output or "nothing was sent" in output  # fmt: skip
    return code in (EXIT_OK, EXIT_DRIFT) and bool(_plan_line(output)) and not stopped


def _plan_line(output: str) -> str:
    return _one_line(next((ln for ln in output.splitlines() if "plan:" in ln), ""))


def _last_line(text: str) -> str:
    return _one_line(next((ln for ln in reversed(text.splitlines()) if ln.strip()), ""))


def _least_finished(answers: dict[str, Answer]) -> Answer:
    """One state for a row: the least finished reader's; every reader's own kept, its word in the detail."""
    if not answers:
        return Answer(state="pending", detail="no reader known")
    state = min((a.state for a in answers.values()), key=_ORDER.index)
    detail = "; ".join(f"{name}: {a.detail}" for name, a in answers.items() if a.detail)
    readers = {
        name: Reader(state=a.state, detail=a.detail) for name, a in answers.items()
    }
    return Answer(state=state, detail=detail, readers=readers)


class ServiceApplier:
    """The applies of the service: fast answers (:meth:`apply`) and the work (:meth:`settle`).

    Parameters
    ----------
    live : Path
        The live checkout (the configs root the readers read).
    scanners : mapping, optional
        Each experiment's scanner (its worker's reopen, its dry runs); a row
        of an experiment with none is pending for its reopen, named.
    archiver : Archiver, optional
        The archiver's CLI, for onboard; ``None``: those readers are pending, named.
    """

    def __init__(self, live: Path, scanners: dict[str, ScannerClient] | None = None, archiver: Archiver | None = None):  # fmt: skip
        self.live, self.scanners, self.archiver = (
            Path(live),
            dict(scanners or {}),
            archiver,
        )

    def _reader(self, verb: str, experiment: str) -> Answer:
        """The fast answer for one reader's verb, for a document of ``experiment``."""
        if verb == "nothing to do":
            return Answer(state="nothing to do", detail="")
        if verb == REOPEN and experiment in self.scanners:
            return Answer(state="queued", detail="the worker reopens its environment when idle and empty")  # fmt: skip
        if verb == ONBOARD and self.archiver is not None and experiment:
            return Answer(state="queued", detail="geecs-archiver onboard runs on the service's next apply pass")  # fmt: skip
        configured = {REOPEN: f"no scanner configured for {experiment or 'it'} (GEECS_CONFIGURATOR_SCANNER_URL)", ONBOARD: "no archiver command configured (GEECS_CONFIGURATOR_ARCHIVER_COMMAND)"}  # fmt: skip
        why = f": {configured[verb]}" if verb in configured else ""
        return Answer(state="pending", detail=f"the maintainer's: {verb}{why}")

    def apply(self, kind: str | None, path: str) -> Answer:
        """The row's state now; work is queued for :meth:`settle`, never done here."""
        experiment = _experiment_of(path)
        answer = _least_finished({c.name: self._reader(c.apply, experiment) for c in CONSUMERS.get(kind or "", [])})  # fmt: skip
        if answer.state == "nothing to do":
            answer.detail = "every reader reads it per request"
        return answer

    def settle(self, rows: Sequence[Row]) -> list[Answer]:
        """Do the queued work for these rows (committed, live); each row's state after it.

        Only a reader the row still holds as queued is retried (a reader
        already applied is never redone: a worker reopened while the
        appliance was down is not reopened again for it). At most one
        reopen per experiment and one onboard per experiment, for all the
        rows. Blocking: the service's apply thread calls it, holding the
        repository's lock so nothing moves the live checkout meanwhile.
        """
        work: dict[str, Answer] = {}  # "<verb>\0<experiment>" -> this pass's one answer
        reasons = self._reopen_reasons(rows)

        def done(verb: str, experiment: str) -> Answer:
            key = f"{verb}\0{experiment}"
            if key not in work:
                work[key] = self._do(verb, experiment, reasons.get(experiment, []))
            return work[key]

        out = []
        for row in rows:
            experiment = _experiment_of(row.path)
            answers: dict[str, Answer] = {}
            for c in CONSUMERS.get(row.kind or "", []):
                held = row.readers.get(c.name)
                if held is not None and held.state != "queued":
                    answers[c.name] = Answer(state=held.state, detail=held.detail)
                    continue
                fresh = self._reader(c.apply, experiment)
                answers[c.name] = done(c.apply, experiment) if fresh.state == "queued" else fresh  # fmt: skip
            out.append(_least_finished(answers))
        return out

    def _do(self, verb: str, experiment: str, names: list[str]) -> Answer:
        """The work of one queued verb for one experiment: the reopen, or the onboard."""
        if verb == REOPEN:
            return self.scanners[experiment].reopen(f"configurator: {', '.join(sorted(set(names)))}")  # fmt: skip
        policy = self.live / _policy_path(experiment)
        return self.archiver.onboard(experiment, policy if policy.is_file() else None)  # type: ignore[union-attr]  # fmt: skip

    def _reopen_reasons(self, rows: Sequence[Row]) -> dict[str, list[str]]:
        """Per experiment, the files its reopen is for (the reason the scanner logs)."""
        out: dict[str, list[str]] = {}
        for row in rows:
            for c in CONSUMERS.get(row.kind or "", []):
                if c.apply == REOPEN and self._still_queued(row, c.name):
                    out.setdefault(_experiment_of(row.path), []).append(row.path.rsplit("/", 1)[-1])  # fmt: skip
        return out

    @staticmethod
    def _still_queued(row: Row, reader: str) -> bool:
        """Whether this reader of the row has work left (a row from before readers were kept: all of them)."""
        held = row.readers.get(reader)
        return held is None or held.state == "queued"

    def dry_run(self, kind: str, experiment: str | None, document: Json) -> DryRun:
        """The kind's consumer check of an unsaved draft (``kinds.DRY_RUNS``), or ``ran=False`` saying why not."""
        scanner = self.scanners.get(experiment or "")
        if kind == "trigger_profile" and scanner is not None:
            return scanner.trigger_profile_dry_run(document)
        if kind == "archive_policy" and self.archiver is not None and experiment:
            return self.archiver.dry_run(experiment, document)
        if kind in ("trigger_profile", "archive_policy"):
            return DryRun(kind=kind, by="none", ran=False, detail=f"its consumer is not configured here for {experiment or 'this experiment'}")  # fmt: skip
        return DryRun(kind=kind, by="none", ran=False, detail="no consumer check for this kind")  # fmt: skip


def _experiment_of(path: str) -> str:
    """``scanner_configs/experiments/<experiment>/…`` → the experiment ("" for another root)."""
    parts = path.split("/")
    return parts[2] if len(parts) > 3 and parts[:2] == ["scanner_configs", "experiments"] else ""  # fmt: skip


def _policy_path(experiment: str) -> str:
    """The experiment's archive policy below the configs root (GEECS-Schemas' storage table)."""
    from geecs_schemas import KIND_STORAGE

    return KIND_STORAGE["archive_policy"].path_pattern().replace("<experiment>", experiment)  # type: ignore[union-attr]  # fmt: skip
