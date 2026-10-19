"""The scans domain against the scanner's real routes (in process).

Each test calls a tool's impl and asserts on the envelope; the scanner
app underneath is the real one over its demo manager, stepped by hand.
"""

from __future__ import annotations

import json

import httpx
import pytest

from geecs_mcp import clients, config, guard
from geecs_mcp.scans import tools

OPERATOR = "Ada Lovelace"
IDENTITY = "agent, admin supervision (Ada Lovelace)"


def _j(envelope: str) -> dict:
    return json.loads(envelope)


async def run(impl, *args, **kwargs) -> str:
    """Through the guard, as the wrappers call: refusals arrive as envelopes."""
    return await guard.run_async(impl, *args, **kwargs)


async def _submit(
    acknowledged=("gateway_liveness",), name="jet_pressure_sweep"
) -> dict:
    return _j(await run(tools._submit_preset, name, list(acknowledged), OPERATOR))


# ----------------------------------------------------------------- reads


class TestReads:
    async def test_scan_status_is_the_scanner_snapshot(self, wired) -> None:
        out = _j(await run(tools._scan_status))
        assert out["ok"] is True
        assert out["connected"] is True
        assert out["re_state"] == "idle"
        assert out["identity"] == "geecs-scanner test"
        # /api reports the scanner PACKAGE's version (the service's injected
        # string is /health's, which costs a second manager round trip)
        assert isinstance(out["scanner_version"], str) and out["scanner_version"]

    async def test_scan_progress_before_any_scan(self, wired) -> None:
        out = _j(await run(tools._scan_progress))
        assert out["ok"] is True
        assert "shots_done" in out and "state" in out

    async def test_list_scan_configs_names_and_one_preset(self, wired) -> None:
        names = _j(await run(tools._list_scan_configs, "presets", None))
        assert names["ok"] is True
        assert "jet_pressure_sweep" in names["names"]
        doc = _j(await run(tools._list_scan_configs, "presets", "jet_pressure_sweep"))
        assert doc["ok"] is True
        assert doc["preset"]["name"] == "jet_pressure_sweep"
        assert doc["preset"]["devices"]

    async def test_list_scan_configs_refusals(self, wired) -> None:
        bad_kind = _j(await run(tools._list_scan_configs, "save_sets", None))
        assert bad_kind["ok"] is False
        assert bad_kind["error_kind"] == "not_found"
        assert "presets" in bad_kind["kinds"]
        missing = _j(await run(tools._list_scan_configs, "presets", "nope"))
        assert missing["ok"] is False
        assert missing["error_kind"] == "not_found"
        by_name = _j(await run(tools._list_scan_configs, "actions", "x"))
        assert by_name["error_kind"] == "invalid_request"

    async def test_scan_history_rows_carry_the_keys(self, stepping) -> None:
        submitted = await _submit()
        assert submitted["ok"]
        for _ in range(80):  # the demo sweep is 7 steps × 10 shots
            stepping.step()
        out = _j(await run(tools._scan_history, 5))
        assert out["ok"] is True
        assert out["running"] is None
        finished = out["finished"][0]
        assert finished["item_uid"] == submitted["item_uid"]
        assert len(finished["scan_numbers"]) == 1
        assert finished["run_uids"]
        assert finished["user"] == "geecs-scanner test"


# ---------------------------------------------------------- submit_preset


class TestSubmitPreset:
    async def test_the_question_comes_back_verbatim_and_is_never_acknowledged_here(
        self, wired
    ) -> None:
        out = await _submit(acknowledged=())
        assert out["ok"] is False
        assert out["error_kind"] == "policy_refusal"
        question = out["needs_acknowledgement"][0]
        assert question["check"] == "gateway_liveness"
        assert "Disconnected" in question["title"]
        assert out["planned_shots"]
        # nothing was queued
        assert _j(await run(tools._scan_history, 5))["waiting"] == []

    async def test_an_acknowledged_submit_queues_under_the_identity_string(
        self, wired
    ) -> None:
        out = await _submit()
        assert out["ok"] is True
        assert out["item_uid"]
        assert out["operator"] == IDENTITY
        assert out["submitted_as"] == "geecs-scanner test"
        assert out["plan"]["name"] in ("sweep", "count", "optimize")
        assert out["acknowledged"] == ["gateway_liveness"]
        assert {o["check"] for o in out["preflight"]} >= {"validate", "worker_ready"}
        # the scanner stamped the operator on the item's provenance (the
        # demo manager starts an item at once when idle, so look there too)
        item = wired.running_item() or wired.queue_items()[0]
        assert item["kwargs"]["md"]["geecs"]["operator"] == IDENTITY

    async def test_a_submit_while_one_waits_is_refused_with_the_rows(
        self, wired
    ) -> None:
        assert (await _submit())["ok"]  # starts at once on the idle demo manager
        behind = await _submit()  # queued behind the running one: allowed
        assert behind["ok"] is True
        out = await _submit()  # something already WAITS: refused, with the rows
        assert out["ok"] is False
        assert out["error_kind"] == "policy_refusal"
        assert out["pending_items"][0]["plan"] == "sweep"

    async def test_unknown_preset_lists_the_catalog(self, wired) -> None:
        out = await _submit(name="not_a_preset")
        assert out["error_kind"] == "not_found"
        assert "jet_pressure_sweep" in out["names"]

    async def test_a_blank_operator_is_refused_before_any_call(self, wired) -> None:
        out = _j(
            await run(
                tools._submit_preset, "jet_pressure_sweep", ["gateway_liveness"], "  "
            )
        )
        assert out["error_kind"] == "invalid_request"
        assert "operator" in out["message"]
        assert _j(await run(tools._scan_history, 5))["waiting"] == []


# ---------------------------------------------------------- wait_for_scan


class TestWaitForScan:
    async def test_waits_to_the_end_and_returns_the_keys(self, stepping) -> None:
        item_uid = (await _submit())["item_uid"]
        out = _j(await run(tools._wait_for_scan, item_uid, 240))
        assert out["ok"] is True
        assert out["finished"] is True
        assert out["state"] == "ok"
        assert len(out["scan_numbers"]) == 1
        assert len(out["run_uids"]) == 1
        assert out["progress"]["state"] == "done"
        assert out["progress"]["scan_number"] == out["scan_numbers"][0]

    async def test_a_timeout_is_a_result_with_the_progress(
        self, wired, frozen_clock, monkeypatch
    ) -> None:
        item_uid = (await _submit())["item_uid"]  # started at once (idle manager)
        wired.step()  # one shot

        async def tick(_s: float) -> None:
            frozen_clock["t"] += 10.0

        monkeypatch.setattr(tools, "_sleep", tick)
        out = _j(await run(tools._wait_for_scan, item_uid, 5))
        assert out["ok"] is True
        assert out["finished"] is False
        assert out["where"] == "running"
        assert out["progress"]["shots_done"] == 1
        assert out["progress"]["scan_number"]

    async def test_progress_is_attached_only_when_it_is_this_item_s(
        self, stepping
    ) -> None:
        """/api/progress is the LATEST run's picture: never another scan's under this item."""
        first = (await _submit())["item_uid"]  # running at once
        second = (await _submit())["item_uid"]  # waiting behind it
        # waiting: the live snapshot is the first scan's, not the second's
        waiting = _j(await run(tools._wait_for_scan, second, 0))
        assert waiting["finished"] is False and waiting["where"] == "waiting"
        assert waiting["progress"] is None
        assert waiting["live_scan"]["state"] in ("running", "paused")
        # run the first to its end; the second starts right behind it
        done = _j(await run(tools._wait_for_scan, first, 240))
        assert done["finished"] is True
        for _ in range(3):
            stepping.step()  # the second scan is now the live one
        again = _j(await run(tools._wait_for_scan, first, 240))
        assert again["finished"] is True
        assert again["scan_numbers"] == done["scan_numbers"]
        assert again["progress"] is None  # the snapshot is the second scan's
        assert again["live_scan"]["scan_number"] != done["scan_numbers"][0]
        # and the running second item gets the live picture as ITS progress
        running = _j(await run(tools._wait_for_scan, second, 0))
        assert running["where"] == "running"
        assert running["progress"]["scan_number"] == again["live_scan"]["scan_number"]
        assert running["live_scan"] is None

    async def test_an_item_between_two_queue_reads_is_re_read_not_lost(
        self, monkeypatch
    ) -> None:
        """The scanner reads running/waiting/history in separate round trips.

        An item moving waiting → running can be in none of them for one
        read; the wait must re-read before it says not_found.
        """
        reads = {"n": 0}
        empty = {"running": None, "waiting": [], "finished": []}
        running = {
            "running": {"item_uid": "it-1", "plan": "sweep"},
            "waiting": [],
            "finished": [],
        }
        done = {
            "running": None,
            "waiting": [],
            "finished": [
                {
                    "item_uid": "it-1",
                    "state": "ok",
                    "word": "ok",
                    "scan_numbers": [9],
                    "run_uids": ["u"],
                }
            ],
        }

        async def fake_call(service, method, path, **kwargs):
            if path == "/api/queue":
                reads["n"] += 1
                return [empty, empty, running, done][min(reads["n"] - 1, 3)]
            return {"state": "done", "scan_number": 9}

        async def no_sleep(_s):
            return None

        monkeypatch.setattr(clients, "call", fake_call)
        monkeypatch.setattr(tools, "_sleep", no_sleep)
        out = _j(await run(tools._wait_for_scan, "it-1", 60))
        assert out["ok"] is True and out["finished"] is True
        assert out["scan_numbers"] == [9]

    async def test_an_unknown_item_is_not_found(self, stepping) -> None:
        # `stepping`: the vanished re-reads sleep through the patched pause,
        # not the wall clock
        out = _j(await run(tools._wait_for_scan, "no-such-item", 1))
        assert out["error_kind"] == "not_found"
        assert "history" in out["message"]

    async def test_the_budget_is_capped(self) -> None:
        from geecs_mcp import waits

        assert waits.bounded_wait(10_000) == waits.MAX_WAIT_S
        assert waits.bounded_wait(-3) == 0.0
        assert waits.bounded_wait("x") == 0.0


# -------------------------------------------------------------------- move


class TestMove:
    async def test_a_move_finishes_and_reads_back(self, stepping) -> None:
        out = _j(await run(tools._move, "U_Hexapod:xpos", 1.25, OPERATOR, True, 60))
        assert out["ok"] is True, out
        assert out["finished"] is True
        assert out["reference"] == "U_Hexapod.xpos"
        assert out["readback"]["value"] == 1.25
        assert out["readback"]["variable"] == "U_Hexapod:xpos"
        assert out["operator"] == IDENTITY

    async def test_a_bare_device_moves_but_has_no_readback(self, stepping) -> None:
        out = _j(await run(tools._move, "U_S1H", 0.5, OPERATOR, True, 60))
        assert out["ok"] is True and out["finished"] is True
        assert out["readback"] is None
        assert "not a Device:Variable" in out["readback_detail"]

    async def test_refused_while_a_scan_runs(self, wired) -> None:
        assert (await _submit())["ok"]  # running at once on the idle demo manager
        out = _j(await run(tools._move, "U_Hexapod:xpos", 1.0, OPERATOR, True, 60))
        assert out["ok"] is False
        assert out["error_kind"] == "policy_refusal"
        assert out["re_state"] == "running"

    async def test_argument_checks_happen_before_any_call(self, wired) -> None:
        nan = _j(
            await run(tools._move, "U_Hexapod:xpos", float("nan"), OPERATOR, True, 60)
        )
        assert nan["error_kind"] == "invalid_request"
        blank = _j(await run(tools._move, "U_Hexapod:xpos", 1.0, "", True, 60))
        assert blank["error_kind"] == "invalid_request"
        text = _j(await run(tools._move, "U_Hexapod:xpos", "fast", OPERATOR, True, 60))
        assert text["error_kind"] == "invalid_request"

    async def test_no_wait_returns_the_queued_item(self, wired) -> None:
        out = _j(await run(tools._move, "U_Hexapod:xpos", 2.0, OPERATOR, False, 60))
        assert out["ok"] is True
        assert out["finished"] is False
        assert out["item_uid"]


# ---------------------------------------------------- halts and the queue


class TestHaltsAndQueue:
    async def test_nothing_running_is_a_refusal_in_the_scanner_words(
        self, wired
    ) -> None:
        out = _j(
            await run(tools._verb, "/api/stop", {"force": False}, read_timeout_s=1)
        )
        assert out["ok"] is False
        assert out["error_kind"] == "policy_refusal"
        assert out["message"] == "nothing is running"

    async def test_pause_resume_stop_through_a_scan(self, wired) -> None:
        assert (await _submit())["ok"]  # running at once
        paused = _j(
            await run(tools._verb, "/api/pause", tools._verb_body(False, OPERATOR))
        )
        assert paused["ok"] is True
        for _ in range(10):
            wired.step()  # reaches a step boundary → paused
        assert _j(await run(tools._scan_status))["re_state"] == "paused"
        resumed = _j(
            await run(tools._verb, "/api/resume", tools._verb_body(False, OPERATOR))
        )
        assert resumed["ok"] is True
        stopped = _j(
            await run(
                tools._verb, "/api/stop", tools._verb_body(False, ""), read_timeout_s=5
            )
        )
        assert stopped["ok"] is True
        history = _j(await run(tools._scan_history, 5))
        assert history["finished"][0]["word"] == "stopped"

    async def test_clear_queue_removes_the_waiting_items(self, wired) -> None:
        assert (await _submit())["ok"]  # running
        assert (await _submit())["ok"]  # waiting behind it
        assert len(_j(await run(tools._scan_history, 5))["waiting"]) == 1
        out = _j(await run(tools._verb, "/api/clear", None))
        assert out["ok"] is True
        assert "removed" in out["message"]
        assert _j(await run(tools._scan_history, 5))["waiting"] == []

    async def test_a_malformed_optional_name_never_blocks_a_halt(self, wired) -> None:
        """Through the WRAPPERS: an over-long operator is dropped, the halt still goes out."""
        long_name = "A" * 100
        for halt in (tools.stop_scan, tools.pause_scan):
            out = _j(await halt(operator=long_name))
            # the request reached the scanner: its own "nothing is running" verdict
            assert out["error_kind"] == "policy_refusal", out
            assert out["message"] == "nothing is running"
        # and with a scan running, the halt lands and the note says what was dropped
        assert (await _submit())["ok"]
        out = _j(await tools.pause_scan(operator=long_name))
        assert out["ok"] is True
        assert "operator not recorded" in out["note"]

    async def test_resume_with_a_blank_operator_is_an_envelope_not_a_raise(
        self, wired
    ) -> None:
        """Through the WRAPPER: the identity check must sit inside the guard."""
        out = _j(await tools.resume_scan(operator="   "))
        assert out["ok"] is False
        assert out["error_kind"] == "invalid_request"
        assert "operator" in out["message"]

    def test_the_verb_body_carries_the_identity_string_only_when_named(self) -> None:
        assert tools._verb_body(True, "") == {"force": True}
        assert tools._verb_body(False, OPERATOR) == {
            "force": False,
            "operator": IDENTITY,
        }


# ------------------------------------------------------- the transport seam


class TestServiceFailures:
    async def test_an_unconfigured_service_names_the_key(self, monkeypatch) -> None:
        monkeypatch.setattr(config, "service_url", lambda service: None)
        out = _j(await run(tools._scan_status))
        assert out["error_kind"] == "service_unreachable"
        assert out["service"] == "scanner"
        assert "scanner_url" in out["message"]

    async def test_a_dead_scanner_is_unreachable(self, monkeypatch) -> None:
        def dead(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(config, "service_url", lambda service: "http://scanner")
        monkeypatch.setattr(
            clients, "transport_for", lambda service: httpx.MockTransport(dead)
        )
        try:
            out = _j(await run(tools._scan_status))
        finally:
            await clients.reset_clients()
        assert out["error_kind"] == "service_unreachable"
        assert "ConnectError" in out["message"]

    async def test_a_scanner_500_is_unreachable_not_a_refusal(
        self, monkeypatch
    ) -> None:
        def wedged(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="store is wedged")

        monkeypatch.setattr(config, "service_url", lambda service: "http://scanner")
        monkeypatch.setattr(
            clients, "transport_for", lambda service: httpx.MockTransport(wedged)
        )
        try:
            out = _j(await run(tools._scan_status))
        finally:
            await clients.reset_clients()
        assert out["error_kind"] == "service_unreachable"
        assert "500" in out["message"]

    async def test_a_non_json_success_body_is_the_service_fault(
        self, monkeypatch
    ) -> None:
        """A proxy page on the scanner's port is a deployment fault, not a bad request."""

        def html(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>maintenance</html>")

        monkeypatch.setattr(config, "service_url", lambda service: "http://scanner")
        monkeypatch.setattr(
            clients, "transport_for", lambda service: httpx.MockTransport(html)
        )
        try:
            out = _j(await run(tools._scan_status))
        finally:
            await clients.reset_clients()
        assert out["error_kind"] == "service_unreachable"
        assert "non-JSON" in out["message"]

    async def test_a_portal_style_detail_maps_by_status(self, monkeypatch) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                409, json={"detail": {"message": "busy", "job": {"state": "running"}}}
            )

        monkeypatch.setattr(config, "service_url", lambda service: "http://portal")
        monkeypatch.setattr(
            clients, "transport_for", lambda service: httpx.MockTransport(refuse)
        )
        try:
            with pytest.raises(clients.ServiceRefusal) as caught:
                await clients.call("portal", "GET", "/x")
        finally:
            await clients.reset_clients()
        assert caught.value.kind == "policy_refusal"
        assert caught.value.message == "busy"
        assert caught.value.extra == {"job": {"state": "running"}}


async def test_a_wait_reports_progress_to_an_mcp_client(stepping) -> None:
    """The heartbeat reaches the client: what resets its idle timer."""
    from fastmcp import Client

    from geecs_mcp.server import create_server

    item_uid = (await _submit())["item_uid"]
    seen = []

    async def on_progress(progress, total, message):
        seen.append((progress, total, message))

    async with Client(create_server()) as client:
        result = await client.call_tool(
            "wait_for_scan",
            {"item_uid": item_uid, "timeout_s": 60},
            progress_handler=on_progress,
        )
    assert json.loads(result.content[0].text)["finished"] is True
    assert seen, "no progress notification reached the client"
    assert all(total == 60 for _, total, _ in seen)
    assert item_uid in seen[0][2]


async def test_a_stop_the_scanner_has_not_confirmed_is_reported_not_landed(
    wired, monkeypatch
) -> None:
    """Past the stop's read budget the request is in: ok, landed false — never an outage."""
    from geecs_mcp import clients as clients_mod

    async def unanswered(service, method, path, **kwargs):
        assert path == "/api/stop"
        assert kwargs["read_timeout_s"] == tools.STOP_READ_TIMEOUT_S
        raise clients_mod.ServiceTimeout(service, "http://scanner", "no answer")

    monkeypatch.setattr(clients_mod, "call", unanswered)
    out = _j(await run(tools._stop, False, OPERATOR))
    assert out["ok"] is True, out
    assert out["landed"] is False
    assert "scan_status" in out["message"]


async def test_an_accepted_move_keeps_its_handle_when_the_scanner_stops_answering(
    stepping, monkeypatch
) -> None:
    """The move is queued: losing the scanner while following it still returns the item_uid."""
    from geecs_mcp import clients as clients_mod

    real = clients_mod.call

    async def queue_down(service, method, path, **kwargs):
        if path == "/api/queue":
            raise clients_mod.ServiceUnavailable(service, "http://scanner", "gone")
        return await real(service, method, path, **kwargs)

    monkeypatch.setattr(clients_mod, "call", queue_down)
    out = _j(await run(tools._move, "U_Hexapod:xpos", 1.25, OPERATOR, True, 60))
    assert out["ok"] is True, out
    assert out["item_uid"]
    assert out["finished"] is False and out["where"] == "unknown"
    assert out["item_uid"] in out["readback_detail"]


def _trickling_scanner(monkeypatch, scanner_app, slow_paths):
    """The real scanner, except *slow_paths*, which answer one byte every 30 ms."""
    import anyio
    import httpx

    from geecs_mcp import clients as clients_mod

    asgi = httpx.ASGITransport(app=scanner_app)

    async def trickle():
        for byte in b'{"running": null, "waiting": [], "finished": []}' * 20:
            await anyio.sleep(0.03)
            yield bytes([byte])

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path in slow_paths:
            return httpx.Response(
                200, headers={"content-type": "application/json"}, content=trickle()
            )
        return await asgi.handle_async_request(request)

    previous = clients_mod.transport_for
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        clients_mod,
        "transport_for",
        lambda service: transport if service == "scanner" else previous(service),
    )


async def test_a_wait_is_bounded_by_its_wall_clock_against_a_trickling_scanner(
    wired, scanner, monkeypatch
) -> None:
    """The call budget is absolute: a peer that keeps bytes coming cannot hold the call."""
    import time

    from geecs_mcp import clients as clients_mod
    from geecs_mcp import waits

    _, _, scanner_app = scanner
    _trickling_scanner(monkeypatch, scanner_app, {"/api/queue"})
    monkeypatch.setattr(waits, "CALL_BUDGET_S", 0.4)
    monkeypatch.setattr(clients_mod, "CONNECT_TIMEOUT_S", 0.05)
    await clients_mod.reset_clients()
    started = time.monotonic()
    out = _j(await run(tools._wait_for_scan, "some-item", 60))
    assert time.monotonic() - started < 1.5
    assert out["ok"] is True and out["finished"] is False, out
    assert out["where"] == "unknown"
    assert "budget ran out" in out["note"]


async def test_an_accepted_move_keeps_its_handle_when_its_budget_runs_out(
    wired, scanner, monkeypatch
) -> None:
    """The POST lands; following it against a trickling queue ends at the clock with the item_uid."""
    import time

    from geecs_mcp import clients as clients_mod
    from geecs_mcp import waits

    _, _, scanner_app = scanner
    _trickling_scanner(monkeypatch, scanner_app, {"/api/queue"})
    monkeypatch.setattr(waits, "CALL_BUDGET_S", 0.6)
    monkeypatch.setattr(clients_mod, "CONNECT_TIMEOUT_S", 0.05)
    await clients_mod.reset_clients()
    started = time.monotonic()
    out = _j(await run(tools._move, "U_Hexapod:xpos", 1.25, OPERATOR, True, 60))
    assert time.monotonic() - started < 1.5
    assert out["ok"] is True, out
    assert out["item_uid"] and out["where"] == "unknown"
    assert out["item_uid"] in out["readback_detail"]
