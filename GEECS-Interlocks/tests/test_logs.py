"""The daily log file and the in-memory buffer the window's pane reads."""

import datetime as dt
import logging

from geecs_interlocks.logs import DailyFileHandler, LogBuffer


def record(message, level=logging.INFO):
    return logging.LogRecord("x", level, __file__, 1, message, None, None)


class TestDailyFile:
    def test_one_file_per_day(self, tmp_path):
        day = {"d": dt.date(2026, 11, 20)}
        handler = DailyFileHandler(tmp_path, today=lambda: day["d"])
        handler.emit(record("first"))
        day["d"] = dt.date(2026, 11, 21)
        handler.emit(record("second"))
        handler.close()
        assert (tmp_path / "interlocks-2026-11-20.log").read_text().strip() == "first"
        assert (tmp_path / "interlocks-2026-11-21.log").read_text().strip() == "second"

    def test_old_files_are_pruned_and_others_kept(self, tmp_path):
        old = tmp_path / "interlocks-2026-10-01.log"
        recent = tmp_path / "interlocks-2026-11-01.log"
        other = tmp_path / "interlocks-notes.log"
        for path in (old, recent, other):
            path.write_text("x")
        handler = DailyFileHandler(tmp_path, today=lambda: dt.date(2026, 11, 20))
        handler.emit(record("now"))
        handler.close()
        assert not old.exists() and recent.exists() and other.exists()

    def test_creates_the_folder(self, tmp_path):
        DailyFileHandler(tmp_path / "a" / "b").close()
        assert (tmp_path / "a" / "b").is_dir()


class TestBuffer:
    def test_keeps_the_last_records(self):
        buffer = LogBuffer(size=3)
        for n in range(5):
            buffer.emit(record(str(n)))
        assert [r.getMessage() for r in buffer.records()] == ["2", "3", "4"]

    def test_listeners_see_every_record(self):
        buffer, seen = LogBuffer(), []
        buffer.subscribe(seen.append)
        buffer.emit(record("a"))
        buffer.unsubscribe(seen.append)
        buffer.emit(record("b"))
        assert [r.getMessage() for r in seen] == ["a"]

    def test_a_failing_listener_is_dropped_and_the_rest_still_hear(self):
        buffer, seen, calls = LogBuffer(), [], []

        def broken(_):
            calls.append(1)
            raise RuntimeError("boom")

        buffer.subscribe(broken)
        buffer.subscribe(seen.append)
        buffer.emit(record("a"))
        buffer.emit(record("b"))
        assert calls == [1]
        assert [r.getMessage() for r in seen] == ["a", "b"]

    def test_clear(self):
        buffer = LogBuffer()
        buffer.emit(record("a"))
        buffer.clear()
        assert buffer.records() == []
