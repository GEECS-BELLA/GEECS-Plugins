"""The monitor's states: waiting at start, trips, settling, off, staleness, the verdict."""

import pytest
from conftest import interlock_set, threshold

from geecs_interlocks.monitor import Monitor


def make(source, *entries):
    return Monitor(interlock_set(*entries), source, clock=source.clock)


def states(snapshot):
    return {s.name: s.state for s in snapshot.statuses}


class TestStart:
    def test_starts_waiting_and_unsafe(self, source):
        monitor = make(source, threshold())
        snapshot = monitor.snapshot()
        assert states(snapshot) == {"vac": "waiting"}
        assert not snapshot.safe

    def test_first_safe_reading_is_ok_without_settling(self, source):
        monitor = make(source, threshold(settle_s=10))
        source.push("Gauge", P=1.0)
        assert states(monitor.step()) == {"vac": "ok"}
        assert monitor.step().safe

    def test_first_unsafe_reading_trips(self, source):
        monitor = make(source, threshold())
        source.push("Gauge", P=50.0)
        assert states(monitor.step()) == {"vac": "tripped"}


class TestSettling:
    def test_holds_for_settle_s_then_ok(self, source):
        monitor = make(source, threshold(settle_s=3))
        source.push("Gauge", P=50.0)
        monitor.step()
        source.push("Gauge", P=1.0)
        assert states(monitor.step()) == {"vac": "settling"}
        assert not monitor.snapshot().safe
        source.now += 2.9
        assert states(monitor.step()) == {"vac": "settling"}
        source.now += 0.2
        assert states(monitor.step()) == {"vac": "ok"}

    def test_a_relapse_restarts_the_timer(self, source):
        monitor = make(source, threshold(settle_s=3))
        source.push("Gauge", P=50.0)
        monitor.step()
        source.push("Gauge", P=1.0)
        monitor.step()
        source.now += 2.0
        source.push("Gauge", P=50.0)
        assert states(monitor.step()) == {"vac": "tripped"}
        source.push("Gauge", P=1.0)
        monitor.step()
        source.now += 2.0
        assert states(monitor.step()) == {"vac": "settling"}

    def test_hysteresis_applies_while_settling(self, source):
        entry = threshold(settle_s=1, condition={"hysteresis": 2.0})
        monitor = make(source, entry)
        source.push("Gauge", P=50.0)
        monitor.step()
        source.push("Gauge", P=9.0)  # below the limit, inside the band
        assert states(monitor.step()) == {"vac": "tripped"}


class TestVerdict:
    def test_off_is_not_counted(self, source):
        monitor = make(source, threshold(), threshold("spare", enabled=False))
        source.push("Gauge", P=1.0)
        snapshot = monitor.step()
        assert states(snapshot) == {"vac": "ok", "spare": "off"}
        assert snapshot.safe

    def test_nothing_armed_is_not_safe(self, source):
        monitor = make(source, threshold(enabled=False))
        assert not monitor.step().safe

    def test_no_interlocks_is_not_safe(self, source):
        assert not make(source).step().safe


class TestDevices:
    def test_disconnect_trips(self, source):
        monitor = make(source, threshold())
        source.push("Gauge", P=1.0)
        monitor.step()
        source.drop("Gauge")
        snapshot = monitor.step()
        assert states(snapshot) == {"vac": "tripped"}
        assert "disconnected" in snapshot.statuses[0].diagnostic

    def test_stale_age(self, source):
        monitor = make(
            source,
            {
                "name": "alive",
                "conditions": [{"kind": "stale", "device": "Gauge", "max_age_s": 2}],
            },
        )
        source.push("Gauge", acq_timestamp=1)
        assert states(monitor.step()) == {"alive": "ok"}
        source.now += 2.5
        assert states(monitor.step()) == {"alive": "tripped"}
        assert monitor.snapshot().statuses[0].value == 2.5


class TestHistory:
    def test_events_and_trend(self, source):
        monitor = make(source, threshold())
        source.push("Gauge", P=1.0)
        monitor.step()
        source.push("Gauge", P=50.0)
        monitor.step()
        changes = [(e.old, e.new) for e in monitor.events()]
        assert changes == [("waiting", "ok"), ("ok", "tripped")]
        (series,) = monitor.trend("vac")
        assert [v for _, v in series.points] == [1.0, 50.0]
        assert series.limit == 10.0 and series.condition == "Gauge P > 10"
        assert monitor.trend("nope") == []
        assert monitor.snapshot().statuses[0].limit == 10.0

    def test_lifecycle_drives_the_source(self, source):
        monitor = make(source, threshold())
        monitor.start()
        monitor.stop()
        assert source.started and source.stopped


def group(*conditions, name="cam", **overrides):
    return {"name": name, "conditions": list(conditions), **overrides}


SAT = {
    "kind": "threshold",
    "device": "Cam",
    "variable": "Max",
    "operator": ">=",
    "limit": 4000,
}
DRIFT = {
    "kind": "alignment",
    "device": "Cam",
    "variable": "X",
    "target_variable": "X0",
    "tolerance": 5,
}


class TestGroups:
    def test_safe_only_when_every_condition_is(self, source):
        monitor = make(source, group(SAT, DRIFT))
        source.push("Cam", Max=100, X=1, X0=0)
        assert states(monitor.step()) == {"cam": "ok"}

    def test_any_unsafe_condition_trips_and_is_named(self, source):
        monitor = make(source, group(SAT, DRIFT))
        source.push("Cam", Max=100, X=9, X0=0)
        snapshot = monitor.step()
        assert states(snapshot) == {"cam": "tripped"}
        assert snapshot.statuses[0].diagnostic.startswith("X=9")

    def test_the_reason_lists_every_failing_condition(self, source):
        monitor = make(source, group(SAT, DRIFT))
        source.push("Cam", Max=4095, X=9, X0=0)
        reason = monitor.step().statuses[0].diagnostic
        assert "Max=4095" in reason and "X=9" in reason and "; " in reason

    def test_a_trip_beside_a_silent_variable_is_a_trip(self, source):
        monitor = make(source, group(SAT, DRIFT))
        source.push("Cam", Max=4095)  # X never arrives
        monitor.step()
        assert states(monitor.step()) == {"cam": "tripped"}

    def test_waiting_while_every_unsafe_one_is_only_waiting(self, source):
        monitor = make(source, group(SAT, DRIFT))
        source.push("Cam", Max=100)  # X never arrives
        assert states(monitor.step()) == {"cam": "waiting"}

    def test_a_group_shows_every_condition(self, source):
        status = make(source, group(SAT, DRIFT)).snapshot().statuses[0]
        assert status.condition.count("; ") == 1


class TestImplicitStaleness:
    def test_a_silent_device_trips_after_stale_after_s(self, source):
        monitor = make(source, threshold())
        source.push("Gauge", P=1.0)
        assert states(monitor.step()) == {"vac": "ok"}
        source.now += 5.5
        snapshot = monitor.step()
        assert states(snapshot) == {"vac": "tripped"}
        assert "no new data for 5.5 s" in snapshot.statuses[0].diagnostic

    def test_off_when_the_file_says_so(self, source):
        interlocks = interlock_set(threshold()).model_copy(
            update={"stale_after_s": None}
        )
        monitor = Monitor(interlocks, source, clock=source.clock)
        source.push("Gauge", P=1.0)
        monitor.step()
        source.now += 1e4
        assert states(monitor.step()) == {"vac": "ok"}

    def test_a_stale_condition_keeps_its_own_rule(self, source):
        monitor = make(
            source,
            {"name": "alive", "conditions": [{"kind": "stale", "device": "Gauge"}]},
        )
        source.push("Gauge", acq_timestamp=1)
        monitor.step()
        source.now += 60
        assert states(monitor.step()) == {"alive": "ok"}


EXT = {"kind": "external", "port": 5050, "source": "stage"}


class TestExternal:
    def make(self, source, signals, **condition):
        return Monitor(
            interlock_set(group({**EXT, **condition}, name="stage")),
            source,
            signals=signals,
            clock=source.clock,
        )

    def test_waits_for_the_first_packet_then_follows_it(self, source, signals):
        monitor = self.make(source, signals)
        assert states(monitor.step()) == {"stage": "waiting"}
        signals.send(5050, "SAFE")
        assert states(monitor.step()) == {"stage": "ok"}
        signals.send(5050, "Warning: moving")
        assert states(monitor.step()) == {"stage": "tripped"}

    def test_an_unrecognised_packet_keeps_the_last_state(self, source, signals):
        monitor = self.make(source, signals)
        signals.send(5050, "safe")
        signals.send(5050, "hello")
        assert states(monitor.step()) == {"stage": "ok"}

    def test_max_age(self, source, signals):
        monitor = self.make(source, signals, max_age_s=2)
        signals.send(5050, "safe")
        monitor.step()
        source.now += 3
        assert states(monitor.step()) == {"stage": "tripped"}
        assert monitor.snapshot().statuses[0].value == 3

    def test_no_signals_means_waiting(self, source):
        assert states(self.make(source, None).step()) == {"stage": "waiting"}


HOLD = {"kind": "config_change", "device": "W", "variables": ["Pos", "Mode"]}


class TestArming:
    def test_unarmed_trips_and_is_armable(self, source):
        monitor = make(source, group(HOLD, name="windmill"))
        source.push("W", Pos=1, Mode="a")
        snapshot = monitor.step()
        assert states(snapshot) == {"windmill": "tripped"}
        assert snapshot.statuses[0].armable

    def test_arm_then_a_change_trips_and_a_return_clears(self, source):
        monitor = make(source, group(HOLD, name="windmill"))
        source.push("W", Pos=1, Mode="a")
        assert monitor.arm() == ["windmill"]
        assert states(monitor.step()) == {"windmill": "ok"}
        source.push("W", Pos=2)
        snapshot = monitor.step()
        assert states(snapshot) == {"windmill": "tripped"}
        assert "Pos changed from 1 to 2" in snapshot.statuses[0].diagnostic
        source.push("W", Pos=1)
        assert states(monitor.step()) == {"windmill": "ok"}

    def test_rearm_accepts_the_new_settings(self, source):
        monitor = make(source, group(HOLD, name="windmill"))
        source.push("W", Pos=1, Mode="a")
        monitor.arm()
        source.push("W", Pos=2)
        monitor.step()
        monitor.arm("windmill")
        assert states(monitor.step()) == {"windmill": "ok"}

    def test_all_or_nothing(self, source):
        other = {**HOLD, "device": "V"}
        monitor = make(source, group(HOLD, name="a"), group(other, name="b"))
        source.push("W", Pos=1, Mode="a")
        source.push("V", Pos=1)  # Mode never arrives
        with pytest.raises(ValueError, match="V Mode"):
            monitor.arm()
        assert monitor.baseline("a", 0) is None

    def test_a_disconnected_device_cannot_be_armed(self, source):
        monitor = make(source, group(HOLD, name="a"))
        source.push("W", Pos=1, Mode="a")
        source.drop("W")
        with pytest.raises(ValueError, match="disconnected"):
            monitor.arm()

    def test_nothing_to_arm(self, source):
        with pytest.raises(ValueError, match="no armed interlock"):
            make(source, threshold()).arm()
        with pytest.raises(ValueError, match="holds no settings"):
            make(source, threshold()).arm("vac")


class TestCheckStatuses:
    def test_each_condition_reports_on_its_own(self, source):
        monitor = make(source, group(SAT, DRIFT))
        source.push("Cam", Max=4095, X=1, X0=0)
        checks = monitor.step().statuses[0].checks
        assert [c.unsafe for c in checks] == [True, False]
        assert checks[0].value == 4095 and checks[0].limit == 4000
        assert checks[1].value == 1 and checks[1].diagnostic == ""

    def test_off_has_no_checks(self, source):
        status = make(source, threshold(enabled=False)).step().statuses[0]
        assert status.checks == ()


def test_a_group_keeps_one_trend_per_check(source):
    monitor = make(source, group(SAT, DRIFT))
    source.push("Cam", Max=100, X=3, X0=0)
    monitor.step()
    source.push("Cam", Max=200)
    monitor.step()
    sat, drift = monitor.trend("cam")
    assert [v for _, v in sat.points] == [100, 200] and sat.limit == 4000
    assert [v for _, v in drift.points] == [3, 3] and drift.limit == 5
