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
(the service's apply thread, off every request). One settle pass reopens at
most once and onboards at most once for all the rows it was given, and the
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
from geecs_configurator.log import Answer
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

    def readback(
        self, variable: str
    ) -> tuple[int, Any]:  # Any: the scanner's JSON, passed through
        """One live readback of ``Device:Variable`` through the scanner (the form's current value)."""
        r = self.http.get(f"{self.base_url}/api/readback", params={"variable": variable}, timeout=10.0)  # fmt: skip
        return r.status_code, r.json()


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

    def _onboard(self, experiment: str, policy: Path, *extra: str) -> tuple[int, str]:
        argv = [*self.command, "onboard", "--experiment", experiment, "--policy", str(policy), *extra]  # fmt: skip
        try:
            return self._run(argv, ARCHIVER_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError) as exc:
            return EXIT_UNREACHABLE, f"the archiver's command did not run: {exc}"

    def dry_run(self, experiment: str, document: Json) -> DryRun:
        """``onboard --dry-run --policy <draft>``: the plan the unsaved policy would send; nothing is sent."""
        with tempfile.TemporaryDirectory(prefix="configurator-draft-") as tmp:
            draft = Path(tmp) / "archive_policy.yaml"
            draft.write_text(dump_yaml(document), encoding="utf-8")
            code, output = self._onboard(experiment, draft, "--dry-run")
        out = output[-4000:]
        if code in (EXIT_OK, EXIT_DRIFT):
            drift = "; drift exists already (never-connected or unresolved PVs, not the draft's)" if code == EXIT_DRIFT else ""  # fmt: skip
            return DryRun(kind="archive_policy", by="archiver", ran=True, output=out, detail=f"the plan the draft would send{drift}")  # fmt: skip
        if (
            code == EXIT_USAGE and "refusing to pause" in output
        ):  # the mass-pause guard: a fact about the plan, not the draft's fault
            return DryRun(kind="archive_policy", by="archiver", ran=True, output=out, detail=_one_line(next((ln for ln in output.splitlines() if "refusing to pause" in ln), "")))  # fmt: skip
        if code == EXIT_USAGE:
            return DryRun(kind="archive_policy", by="archiver", ran=True, refusal=_last_line(output) or "the archiver refused the draft", output=out)  # fmt: skip
        return DryRun(kind="archive_policy", by="archiver", ran=False, output=out, detail=f"the archiver could not plan (exit {code}): {_last_line(output)}")  # fmt: skip

    def onboard(self, experiment: str, policy: Path) -> Answer:
        """Reconcile the appliance with the committed policy; the row's state for this reader."""
        code, output = self._onboard(experiment, policy, "--wait", "0")
        if code == EXIT_OK:
            return Answer(
                state="applied",
                detail=f"geecs-archiver onboard sent the plan: {_first_line(output)}",
            )
        if code == EXIT_DRIFT:
            return Answer(state="applied", detail=f"geecs-archiver onboard sent the plan; drift remains (see its output): {_first_line(output)}")  # fmt: skip
        if code == EXIT_USAGE:
            return Answer(state="pending", detail=f"geecs-archiver onboard refused ({_last_line(output)}); the maintainer runs it by hand (--yes, if the pause is meant)")  # fmt: skip
        return Answer(state="queued", detail=f"geecs-archiver onboard did not reach the appliance (exit {code}: {_last_line(output)}); tried again")  # fmt: skip


def _first_line(text: str) -> str:
    return _one_line(next((ln for ln in text.splitlines() if ln.strip()), ""))


def _last_line(text: str) -> str:
    return _one_line(next((ln for ln in reversed(text.splitlines()) if ln.strip()), ""))


def _least_finished(answers: list[Answer]) -> Answer:
    """One state for a row: the least finished reader's, every reader's word in the detail."""
    if not answers:
        return Answer(state="pending", detail="no reader known")
    state = min((a.state for a in answers), key=_ORDER.index)
    return Answer(state=state, detail="; ".join(a.detail for a in answers if a.detail))


class ServiceApplier:
    """The applies of the service: fast answers (:meth:`apply`) and the work (:meth:`settle`).

    Parameters
    ----------
    live : Path
        The live checkout (the configs root the readers read).
    scanner : ScannerClient, optional
        The scanner, for the reopen; ``None``: those rows are pending, named.
    archiver : Archiver, optional
        The archiver's CLI, for onboard; ``None``: those rows are pending, named.
    """

    def __init__(self, live: Path, scanner: ScannerClient | None = None, archiver: Archiver | None = None):  # fmt: skip
        self.live, self.scanner, self.archiver = Path(live), scanner, archiver

    def _reader(self, verb: str) -> Answer:
        """The fast answer for one reader's verb."""
        if verb == "nothing to do":
            return Answer(state="nothing to do", detail="")
        if verb == REOPEN and self.scanner is not None:
            return Answer(state="queued", detail="the worker reopens its environment when idle and empty")  # fmt: skip
        if verb == ONBOARD and self.archiver is not None:
            return Answer(state="queued", detail="geecs-archiver onboard runs on the service's next apply pass")  # fmt: skip
        configured = {REOPEN: "no scanner configured (GEECS_CONFIGURATOR_SCANNER_URL)", ONBOARD: "no archiver command configured (GEECS_CONFIGURATOR_ARCHIVER_COMMAND)"}  # fmt: skip
        why = f": {configured[verb]}" if verb in configured else ""
        return Answer(state="pending", detail=f"the maintainer's: {verb}{why}")

    def apply(self, kind: str | None, path: str) -> Answer:
        """The row's state now; work is queued for :meth:`settle`, never done here."""
        consumers = CONSUMERS.get(kind or "", [])
        answer = _least_finished([self._reader(c.apply) for c in consumers])
        if answer.state == "nothing to do":
            answer.detail = "every reader reads it per request"
        return answer

    def settle(self, rows: Sequence[tuple[str | None, str]]) -> list[Answer]:
        """Do the queued work for these rows (committed, live); each row's state after it.

        One reopen and one onboard per experiment at most, for all the rows.
        Blocking: the service's apply thread calls it, holding the
        repository's lock so nothing moves the live checkout meanwhile.
        """
        verbs = {c.apply for kind, _ in rows for c in CONSUMERS.get(kind or "", [])}
        reopened: Answer | None = None
        if REOPEN in verbs and self.scanner is not None:
            names = ", ".join(sorted({p.rsplit("/", 1)[-1] for k, p in rows if any(c.apply == REOPEN for c in CONSUMERS.get(k or "", []))}))  # fmt: skip
            reopened = self.scanner.reopen(f"configurator: {names}")
        onboarded: dict[str, Answer] = {}
        out = []
        for kind, path in rows:
            answers = []
            for consumer in CONSUMERS.get(kind or "", []):
                if consumer.apply == REOPEN and reopened is not None:
                    answers.append(reopened)
                elif consumer.apply == ONBOARD and self.archiver is not None:
                    experiment = _experiment_of(path)
                    if experiment not in onboarded:
                        policy = self.live / _policy_path(experiment)
                        onboarded[experiment] = self.archiver.onboard(
                            experiment, policy
                        )
                    answers.append(onboarded[experiment])
                else:
                    answers.append(self._reader(consumer.apply))
            out.append(_least_finished(answers))
        return out

    def dry_run(self, kind: str, experiment: str | None, document: Json) -> DryRun:
        """The kind's consumer check of an unsaved draft (``kinds.DRY_RUNS``), or ``ran=False`` saying why not."""
        if kind == "trigger_profile" and self.scanner is not None:
            return self.scanner.trigger_profile_dry_run(document)
        if kind == "archive_policy" and self.archiver is not None and experiment:
            return self.archiver.dry_run(experiment, document)
        if kind in ("trigger_profile", "archive_policy"):
            return DryRun(kind=kind, by="none", ran=False, detail="its consumer is not configured here")  # fmt: skip
        return DryRun(kind=kind, by="none", ran=False, detail="no consumer check for this kind")  # fmt: skip


def _experiment_of(path: str) -> str:
    """``scanner_configs/experiments/<experiment>/…`` → the experiment ("" for another root)."""
    parts = path.split("/")
    return parts[2] if len(parts) > 3 and parts[:2] == ["scanner_configs", "experiments"] else ""  # fmt: skip


def _policy_path(experiment: str) -> str:
    from geecs_schemas import KIND_STORAGE

    return (
        KIND_STORAGE["archive_policy"]
        .path_pattern()
        .replace("<experiment>", experiment)
    )  # type: ignore[union-attr]
