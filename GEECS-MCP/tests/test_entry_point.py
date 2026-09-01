"""``python -m geecs_mcp`` transport wiring: the HTTP-mode stream warm-up (#685).

Hermetic like the rest of the suite: the server, the queue client and the
process-wide cache are all patched on their modules — no zmq, no threads.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from geecs_mcp import __main__ as entry
from geecs_mcp import runtime, server
from geecs_mcp.scans import progress_stream


class _FakeCache:
    def __init__(self, log):
        self._log = log
        self.started_with = None

    def ensure_started(self, doc_addr, info_addr):
        self.started_with = (doc_addr, info_addr)
        self._log.append("ensure_started")

    def snapshot(self):
        return {"available": False, "detail": ""}


class _FakeServer:
    def __init__(self, log):
        self._log = log
        self.run_kwargs = None

    def run(self, **kwargs):
        self.run_kwargs = kwargs
        self._log.append("run")


@pytest.fixture
def wired(monkeypatch):
    """Patched server + client + cache; returns (log, cache, fake_server)."""
    log: list[str] = []
    cache = _FakeCache(log)
    fake_server = _FakeServer(log)
    client = SimpleNamespace(
        doc_addr="localhost:5568", info_addr="tcp://localhost:60625"
    )
    monkeypatch.setattr(server, "create_server", lambda: fake_server)
    monkeypatch.setattr(runtime, "get_queue_client", lambda: client)
    monkeypatch.setattr(progress_stream, "get_progress_cache", lambda: cache)
    return log, cache, fake_server


def test_http_transport_warms_the_stream_before_serving(wired, monkeypatch):
    log, cache, fake_server = wired
    monkeypatch.setattr(
        "sys.argv", ["geecs_mcp", "--transport", "http", "--port", "8123"]
    )
    entry.main()
    # Started from the queue client's own addresses, and BEFORE the server
    # blocks in run() — the whole point: consuming before the first start doc.
    assert cache.started_with == ("localhost:5568", "tcp://localhost:60625")
    assert log == ["ensure_started", "run"]
    assert fake_server.run_kwargs == {
        "transport": "http",
        "host": "0.0.0.0",
        "port": 8123,
    }


def test_stdio_transport_keeps_the_lazy_start(wired, monkeypatch):
    log, cache, fake_server = wired
    monkeypatch.setattr("sys.argv", ["geecs_mcp"])
    entry.main()
    assert cache.started_with is None
    assert log == ["run"]
    assert fake_server.run_kwargs == {}


def test_warm_up_failure_never_stops_the_server(wired, monkeypatch, caplog):
    log, cache, fake_server = wired

    def boom():
        raise RuntimeError("config.ini unreadable")

    monkeypatch.setattr(runtime, "get_queue_client", boom)
    monkeypatch.setattr("sys.argv", ["geecs_mcp", "--transport", "http"])
    with caplog.at_level(logging.WARNING, logger="geecs_mcp.main"):
        entry.main()
    assert cache.started_with is None
    assert log == ["run"]
    assert "not warmed at startup" in caplog.text
    assert "config.ini unreadable" in caplog.text


def test_warm_up_without_addresses_degrades_honestly(wired, monkeypatch, caplog):
    log, cache, _ = wired
    # The stub client of an unconfigured [qserver]: addresses are None.
    monkeypatch.setattr(runtime, "get_queue_client", lambda: SimpleNamespace())
    monkeypatch.setattr("sys.argv", ["geecs_mcp", "--transport", "http"])
    with caplog.at_level(logging.WARNING, logger="geecs_mcp.main"):
        entry.main()
    # Latched unconfigured (the cache's own honest available=false), served anyway.
    assert cache.started_with == (None, None)
    assert log == ["ensure_started", "run"]
    assert "no document-stream address" in caplog.text
