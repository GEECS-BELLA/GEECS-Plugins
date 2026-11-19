"""The monitor's states: waiting at start, trips, settling, off, staleness, the verdict."""

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
                "condition": {"kind": "stale", "device": "Gauge", "max_age_s": 2},
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
        assert [v for _, v in monitor.trend("vac")] == [1.0, 50.0]
        assert monitor.snapshot().statuses[0].limit == 10.0

    def test_lifecycle_drives_the_source(self, source):
        monitor = make(source, threshold())
        monitor.start()
        monitor.stop()
        assert source.started and source.stopped
