"""Planning, applying and verifying against fakes."""

from geecs_archiver.archive_set import Sampling
from geecs_archiver.mgmt_client import PVStatus
from geecs_archiver.onboard import apply, plan_onboarding, verify


def st(pv, status="Being archived", connected=True, period=1.0):
    return PVStatus(
        pv=pv,
        status=status,
        connected=connected,
        last_event=None,
        sampling_period=period,
        appliance="appliance0",
    )


S1 = Sampling(1.0, "MONITOR")


def test_plan_buckets():
    desired = {
        "undulator:a:new": S1,
        "undulator:a:kept": S1,
        "undulator:a:paused": S1,
        "undulator:a:retune": Sampling(10.0, "MONITOR", "Slow"),
        "undulator:a:pending": S1,
    }
    statuses = [
        st("undulator:a:new", "Not being archived", None, None),
        st("undulator:a:kept"),
        st("undulator:a:paused", "Paused", False),
        st("undulator:a:retune", period=1.0),
        st("undulator:a:pending", "Initial sampling", None, None),
        st("undulator:a:stale"),
        st("undulator:a:alreadypaused", "Paused", False),
        st("other:x:y"),
    ]
    archived = [
        "undulator:a:kept",
        "undulator:a:paused",
        "undulator:a:retune",
        "undulator:a:stale",
        "undulator:a:alreadypaused",
        "other:x:y",
    ]
    plan = plan_onboarding(
        desired, statuses=statuses, archived_pvs=archived, prefix="undulator:"
    )
    assert [r["pv"] for r in plan.to_archive] == ["undulator:a:new"]
    assert plan.to_resume == ["undulator:a:paused"]
    assert plan.to_retune == [("undulator:a:retune", Sampling(10.0, "MONITOR", "Slow"))]
    assert plan.to_pause == [
        "undulator:a:stale"
    ]  # never another experiment's PV, never an already-paused one
    assert plan.pending == ["undulator:a:pending"]
    assert plan.unchanged == ["undulator:a:kept"]
    assert not plan.is_noop
    assert "archive 1" in plan.summary()


def test_plan_is_noop_when_in_step():
    plan = plan_onboarding(
        {"undulator:a:b": S1},
        statuses=[st("undulator:a:b")],
        archived_pvs=["undulator:a:b"],
        prefix="undulator:",
    )
    assert plan.is_noop and plan.unchanged == ["undulator:a:b"]


def test_known_pv_without_status_row_is_left_alone():
    plan = plan_onboarding(
        {"undulator:a:b": S1},
        statuses=[],
        archived_pvs=["undulator:a:b"],
        prefix="undulator:",
    )
    assert plan.is_noop


class FakeClient:
    def __init__(self, flips_after=1):
        self.archived = []
        self.resumed = []
        self.retuned = []
        self.paused = []
        self.polls = 0
        self.flips_after = flips_after

    def archive_pvs(self, requests):
        self.archived.extend(requests)
        return [
            {
                "pvName": r["pv"],
                "status": "Archive request submitted"
                if "bad" not in r["pv"]
                else "PV is not archivable",
            }
            for r in requests
        ]

    def resume(self, pv):
        self.resumed.append(pv)

    def change_archival_params(self, pv, period, method):
        self.retuned.append((pv, period, method))

    def pause(self, pv):
        self.paused.append(pv)

    def get_pv_status(self, pvs):
        self.polls += 1
        out = []
        for pv in pvs:
            if "ghost" in pv:
                out.append(st(pv, "Being archived", False))
            elif "slow" in pv:
                out.append(st(pv, "Initial sampling", None))
            elif self.polls >= self.flips_after:
                out.append(st(pv))
            else:
                out.append(st(pv, "Initial sampling", None))
        return out


def test_apply_batches_and_reports_rejections():
    plan = plan_onboarding(
        {f"undulator:a:v{i}": S1 for i in range(3)}
        | {
            "undulator:a:bad": S1,
            "undulator:a:p": S1,
            "undulator:a:r": Sampling(5.0, "SCAN"),
        },
        statuses=[
            st("undulator:a:p", "Paused", False),
            st("undulator:a:r", period=1.0),
            st("undulator:a:old"),
        ],
        archived_pvs=["undulator:a:p", "undulator:a:r", "undulator:a:old"],
        prefix="undulator:",
    )
    client = FakeClient()
    report = apply(client, plan, batch=2)
    assert len(client.archived) == 4
    assert [r["pvName"] for r in report.rejected] == ["undulator:a:bad"]
    assert client.resumed == ["undulator:a:p"]
    assert client.retuned == [("undulator:a:r", 5.0, "SCAN")]
    assert client.paused == ["undulator:a:old"]


def test_verify_waits_then_classifies():
    client = FakeClient(flips_after=2)
    slept = []
    clock = iter(range(0, 100, 10))
    report = verify(
        client,
        ["undulator:a:ok", "undulator:a:ghost", "undulator:a:slow"],
        wait_s=25,
        poll_s=10,
        sleep=slept.append,
        clock=lambda: next(clock),
    )
    assert report.archived == ["undulator:a:ok"]
    assert report.never_connected == ["undulator:a:ghost"]
    assert report.pending == ["undulator:a:slow"]
    assert not report.ok
    assert slept  # it waited at least once


def test_verify_returns_early_when_everything_connects():
    client = FakeClient(flips_after=1)
    report = verify(
        client,
        ["undulator:a:x"],
        wait_s=1000,
        poll_s=10,
        sleep=lambda s: (_ for _ in ()).throw(AssertionError("should not sleep")),
        clock=lambda: 0.0,
    )
    assert report.ok and report.archived == ["undulator:a:x"] and client.polls == 1
