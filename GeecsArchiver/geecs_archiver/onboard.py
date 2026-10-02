"""Idempotent onboarding: reconcile the appliance with the derived archive set.

``plan_onboarding`` compares what the rule wants with what the appliance has
and produces the minimal set of management calls; ``apply`` issues them; and
``verify`` waits for the new PVs to reach *Being archived with a live
connection*.  A request the appliance cannot complete because the PV never
answers on CA — the rule wants a PV the gateway does not serve — stays in
its archive-request workflow and appears in the appliance's own
never-connected list (``getNeverConnectedPVs``); :func:`stuck_requests`
reads that list, on every run, so the drift cannot hide behind a no-op.
Nothing here deletes data: a PV the rule no longer wants is **paused**, and
only PVs under the experiment's own prefix are ever touched.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from geecs_archiver.archive_set import Sampling
from geecs_archiver.mgmt_client import MgmtClient, PVStatus


@dataclass
class OnboardPlan:
    """What ``apply`` will do, in the order it does it."""

    to_archive: list[dict[str, str]] = field(default_factory=list)
    to_resume: list[str] = field(default_factory=list)
    to_retune: list[tuple[str, Sampling]] = field(default_factory=list)
    to_pause: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)

    @property
    def is_noop(self) -> bool:
        """Nothing to send (pending requests are the appliance's to finish, not ours to resend)."""
        return not (
            self.to_archive or self.to_resume or self.to_retune or self.to_pause
        )

    def summary(self) -> str:
        """One line, every bucket."""
        return (
            f"archive {len(self.to_archive)}, resume {len(self.to_resume)}, "
            f"retune {len(self.to_retune)}, pause {len(self.to_pause)}, "
            f"pending {len(self.pending)}, unchanged {len(self.unchanged)}"
        )


def plan_onboarding(
    desired: Mapping[str, Sampling],
    *,
    statuses: Iterable[PVStatus],
    archived_pvs: Iterable[str],
    prefix: str,
) -> OnboardPlan:
    """Diff the desired set against the appliance's state.

    Parameters
    ----------
    desired : mapping
        ``{pv: Sampling}`` from the rule and the policy (the policy's
        ``include`` PVs are in it, so they are never paused).
    statuses : iterable of PVStatus
        ``getPVStatus`` rows for every desired PV and for every archived PV
        under *prefix* (so paused ones are recognised).
    archived_pvs : iterable of str
        ``getAllPVs``: every PV the appliance has a record for.
    prefix : str
        The experiment's PV prefix (``"undulator:"``) — the only namespace
        this tool pauses in.
    """
    status_by_pv = {s.pv: s for s in statuses}
    archived = set(archived_pvs)
    plan = OnboardPlan()
    for pv in sorted(desired):
        sampling = desired[pv]
        status = status_by_pv.get(pv)
        if status is None or status.unknown:
            if pv in archived:
                plan.unchanged.append(
                    pv
                )  # known to the appliance but no status row: leave it
            else:
                plan.to_archive.append(sampling.request(pv))
        elif status.paused:
            plan.to_resume.append(pv)
        elif status.archived:
            period_differs = (
                status.sampling_period is not None
                and abs(status.sampling_period - sampling.period) > 1e-9
            )
            method_differs = (
                status.sampling_method is not None
                and status.sampling_method != sampling.method
            )
            if period_differs or method_differs:
                plan.to_retune.append((pv, sampling))
            else:
                plan.unchanged.append(pv)
        else:
            plan.pending.append(pv)
    for pv in sorted(archived):
        if not pv.lower().startswith(prefix.lower()) or pv in desired:
            continue
        status = status_by_pv.get(pv)
        if status is None or not status.paused:
            plan.to_pause.append(pv)
    return plan


@dataclass
class ApplyReport:
    """What the appliance answered."""

    archive_responses: list[dict[str, object]] = field(default_factory=list)
    resumed: list[str] = field(default_factory=list)
    retuned: list[str] = field(default_factory=list)
    paused: list[str] = field(default_factory=list)

    @property
    def rejected(self) -> list[dict[str, object]]:
        """``archivePV`` answers that were not a submission."""
        return [
            r
            for r in self.archive_responses
            if "submitted" not in str(r.get("status", "")).lower()
        ]


def apply(client: MgmtClient, plan: OnboardPlan, *, batch: int = 500) -> ApplyReport:
    """Send the plan to the appliance (archive requests in batches)."""
    report = ApplyReport()
    for start in range(0, len(plan.to_archive), batch):
        report.archive_responses.extend(
            client.archive_pvs(plan.to_archive[start : start + batch])
        )
    for pv in plan.to_resume:
        client.resume(pv)
        report.resumed.append(pv)
    for pv, sampling in plan.to_retune:
        client.change_archival_params(pv, sampling.period, sampling.method)
        report.retuned.append(pv)
    for pv in plan.to_pause:
        client.pause(pv)
        report.paused.append(pv)
    return report


def stuck_requests(client: MgmtClient, pvs: Iterable[str]) -> list[str]:
    """The subset of *pvs* in the appliance's never-connected list.

    A requested PV that never answers on CA stays in the archive-request
    workflow indefinitely; this is the authoritative drift signal between the
    rule and the gateway, and it is read on every run.
    """
    wanted = set(pvs)
    names = {str(row.get("pvName", "")) for row in client.never_connected()}
    return sorted(wanted & names)


@dataclass
class VerifyReport:
    """Where the submitted PVs ended up after the wait."""

    archived: list[str] = field(default_factory=list)
    never_connected: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    other: list[PVStatus] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Every PV is being archived with a live connection."""
        return not (self.never_connected or self.pending or self.other)


def verify(
    client: MgmtClient,
    pvs: Sequence[str],
    *,
    wait_s: float = 300.0,
    poll_s: float = 10.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> VerifyReport:
    """Poll ``getPVStatus`` until every PV is archived and connected, or *wait_s* passes.

    Afterwards the appliance's never-connected list says which of the
    leftovers are the drift case (the gateway does not serve them) and which
    are merely still in the workflow.
    """
    deadline = clock() + wait_s
    remaining = list(pvs)
    done: dict[str, PVStatus] = {}
    while remaining:
        for status in client.get_pv_status(remaining):
            if status.archived and status.connected:
                done[status.pv] = status
        remaining = [pv for pv in remaining if pv not in done]
        if not remaining or clock() >= deadline:
            break
        sleep(poll_s)
    report = VerifyReport(archived=sorted(done))
    if remaining:
        stuck = set(stuck_requests(client, remaining))
        for status in client.get_pv_status(remaining):
            if status.pv in stuck or (status.archived and status.connected is False):
                report.never_connected.append(status.pv)
            elif status.pending:
                report.pending.append(status.pv)
            else:
                report.other.append(status)
    return report
