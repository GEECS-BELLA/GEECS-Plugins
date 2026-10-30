"""The module's test doubles, shared: GeecsCanopy's own tests and the Control Room's contract test.

``FakeHare`` stands in for Canopy's Hare, ``FakeSquirrel`` for Squirrel's
REST (an httpx transport), ``write_documents`` writes an experiment's
device documents and operator policy, and ``make_env`` builds the module
and its routes over them (the caller mounts ``env.app`` on an
``httpx.ASGITransport``). A suite that drives the module imports these, so
a route or field change here fails every such suite in the same PR. Imports
no Canopy name; FastAPI only inside ``make_env``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

from geecs_canopy.journal import Journal
from geecs_canopy.module import Module, ModuleSettings

EXPERIMENT = "Test"
REGISTRY = """\
schema_version: 1
devices:
  Cam1: {kind: aravis_camera, trigger_source: line0}
  Mot1: {kind: epics_motor, axes: [m1]}
"""
PANELS = """\
schema_version: 1
devices:
  ps1: {family: easydriver, prefix: "bench:ps1:", allow: [current, reset]}
"""
POLICY = """\
grants:
  - {operators: [ann], families: [ad_camera_chain, motor_record, easydriver]}
"""
#: An operator's key: Canopy read, never Canopy write.
KEY = SimpleNamespace(name="display", read=True, write=False)


def live(value: Any, **wire: Any) -> dict[str, Any]:
    """A connected, unalarmed Hare wire carrying ``value``; ``wire`` overrides any field."""
    return {
        "value": value,
        "connected": True,
        "severity": "NO_ALARM",
        "alarmStatus": 0,
    } | wire


class FakeHare:
    """Hare in memory: ``values`` and ``meta`` by PV, the ``puts`` made, ``refuse``/``slow`` to fail one."""

    def __init__(self) -> None:
        self.values: dict[str, dict[str, Any]] = {}
        self.meta: dict[str, dict[str, Any]] = {}
        self.puts: list[tuple[str, Any, bool, float]] = []
        self.clamp: dict[str, float] = {}
        self.refuse: Exception | None = None
        self.slow = False

    async def get(self, pv: str) -> dict[str, Any]:
        """The PV's wire; never seen is disconnected."""
        return self.values.get(pv, {"pv": pv, "value": None, "connected": False})

    async def describe(self, pv: str) -> dict[str, Any]:
        """The PV's metadata."""
        return self.meta.get(pv, {"pv": pv})

    async def put(self, pv: str, value: Any, *, wait: bool, timeout: float) -> None:
        """Record the put and read it back (clamped); or raise ``refuse``, after ``timeout`` when ``slow``."""
        if self.refuse:
            await asyncio.sleep(timeout if self.slow else 0)
            raise self.refuse
        self.puts.append((pv, value, wait, timeout))
        self.values[pv] = live(min(value, self.clamp.get(pv, value)))

    async def watch(
        self, pvs: list[str], on_frame: Any
    ) -> list[str]:  # Any: Hare's frame callback
        """Keep the callback: a test calls ``hare.on_frame({pv: wire})``."""
        self.on_frame = on_frame
        return pvs


def write_documents(
    configs_root: Path,
    *,
    registry: str = REGISTRY,
    panels: str = PANELS,
    policy: str = POLICY,
) -> Path:
    """Write the experiment's ``native_devices``, ``panel_devices`` and ``operator_policy``; its folder."""
    folder = configs_root / "scanner_configs" / "experiments" / EXPERIMENT
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "native_devices.yaml").write_text(registry)
    (folder / "panel_devices.yaml").write_text(panels)
    (folder / "operator_policy.yaml").write_text(policy)
    return folder


def make_env(
    tmp_path: Path, *, key: Any = KEY, squirrel: FakeSquirrel | None = None
) -> SimpleNamespace:  # Any: Canopy's key object
    """The module over the documents and a ``FakeHare`` (and a ``FakeSquirrel``), and its routes: ``module, hare, app, clock, folder``."""
    from geecs_canopy.routes import build_app

    folder = write_documents(tmp_path)
    clock = [100.0]
    squirrel_settings = {"squirrel_url": "http://canopy.test", "squirrel_key": "service-write-key"} if squirrel else {}  # fmt: skip
    settings = ModuleSettings(EXPERIMENT, tmp_path, tmp_path, 3.0, settle_timeout_s=0.05, **squirrel_settings)  # fmt: skip
    hare = FakeHare()
    module = Module(settings, hare, Journal(tmp_path / "j.sqlite3"), clock=lambda: clock[0])  # fmt: skip
    app = build_app(module, read_access=lambda: key, squirrel_transport=squirrel.transport() if squirrel else None)  # fmt: skip
    return SimpleNamespace(
        module=module, hare=hare, app=app, clock=clock, folder=folder
    )


class FakeSquirrel:
    """Squirrel's registry and snapshot routes in memory, as an httpx transport.

    ``pvs`` by id (Squirrel's wire shape), ``groups`` by id with their tags,
    ``snapshots`` by id with the values a save read from ``values`` (a PV
    not in ``values`` saves as ``None``, as an unreachable one does),
    ``restores`` every call of Squirrel's own restore route (the module
    must never make one), ``keys`` the key each request carried.
    """

    def __init__(self) -> None:
        self.pvs: dict[str, dict[str, Any]] = {}
        self.groups: dict[str, dict[str, Any]] = {}
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.values: dict[str, Any] = {}
        self.restores: list[tuple[str, Any]] = []
        self.keys: list[str | None] = []
        self._n = 0

    def transport(self) -> httpx.MockTransport:
        """The transport a ``SquirrelClient`` takes."""
        return httpx.MockTransport(self.handle)

    def _id(self) -> str:
        self._n += 1
        return f"{self._n:032x}"

    def _tag(self, tag_id: str) -> dict[str, Any]:
        for g in self.groups.values():
            for t in g["tags"]:
                if t["id"] == tag_id:
                    return {"id": t["id"], "name": t["name"], "description": None}
        raise KeyError(tag_id)

    def _pvs(self, request: httpx.Request, body: Any) -> httpx.Response:
        m, path = request.method, request.url.path
        if path == "/squirrel/v1/pvs/paged":
            return httpx.Response(200, json={"results": list(self.pvs.values()), "continuationToken": None, "totalCount": len(self.pvs)})  # fmt: skip
        if path == "/squirrel/v1/pvs" and m == "POST":
            if any(
                p["setpointAddress"] == body["setpointAddress"]
                for p in self.pvs.values()
            ):
                return httpx.Response(409, json={"code": "pv_exists", "detail": body["setpointAddress"]})  # fmt: skip
            row = {"id": self._id(), "setpointAddress": body["setpointAddress"], "readbackAddress": body.get("readbackAddress")}  # fmt: skip
            row |= {"device": body.get("device"), "description": body.get("description"), "absTolerance": body.get("absTolerance", 0.0), "relTolerance": body.get("relTolerance", 0.0)}  # fmt: skip
            row["tags"] = [self._tag(t) for t in body.get("tags", [])]
            self.pvs[row["id"]] = row
            return httpx.Response(200, json=row)
        pv_id = path.rsplit("/", 1)[1]
        if (row := self.pvs.get(pv_id)) is None:
            return httpx.Response(404, json={"code": "not_found", "detail": pv_id})
        if m == "DELETE":
            del self.pvs[pv_id]
            for snap in self.snapshots.values():  # the cascade
                snap["pvValues"] = [v for v in snap["pvValues"] if v["pvId"] != pv_id]
            return httpx.Response(200, json=True)
        for k in ("device", "description", "absTolerance", "relTolerance"):
            if body.get(k) is not None:
                row[k] = body[k]
        if body.get("tags") is not None:
            row["tags"] = [self._tag(t) for t in body["tags"]]
        return httpx.Response(200, json=row)

    def _tags(self, request: httpx.Request, body: Any) -> httpx.Response:
        path = request.url.path
        if path == "/squirrel/v1/tags":
            return httpx.Response(200, json=[{"id": g["id"], "name": g["name"], "tagCount": len(g["tags"])} for g in self.groups.values()])  # fmt: skip
        if path == "/squirrel/v1/tags/bulk":
            for name, tags in body["groups"].items():
                group = next(
                    (g for g in self.groups.values() if g["name"] == name), None
                )
                if group is None:
                    gid = self._id()
                    group = self.groups[gid] = {"id": gid, "name": name, "tags": []}
                have = {t["name"] for t in group["tags"]}
                group["tags"] += [{"id": self._id(), "name": t} for t in tags if t not in have]  # fmt: skip
            return httpx.Response(200, json={"groupsCreated": 0, "tagsCreated": 0, "tagsSkipped": 0, "warnings": []})  # fmt: skip
        group = self.groups.get(path.rsplit("/", 1)[1])
        return httpx.Response(200, json=[group]) if group else httpx.Response(404, json={"code": "not_found", "detail": path})  # fmt: skip

    def _snapshots(self, request: httpx.Request, body: Any) -> httpx.Response:
        m, path = request.method, request.url.path
        if path == "/squirrel/v1/snapshots" and m == "GET":
            rows = sorted(
                self.snapshots.values(), key=lambda s: s["createdDate"], reverse=True
            )
            summaries = [{k: v for k, v in s.items() if k != "pvValues"} for s in rows]
            return httpx.Response(200, json={"results": summaries, "continuationToken": None, "totalCount": len(rows)})  # fmt: skip
        if (
            path == "/squirrel/v1/snapshots" and m == "POST"
        ):  # the save: a job, done at once
            values = [{"pvId": p["id"], "pvName": p["setpointAddress"], "device": p["device"], "setpointName": p["setpointAddress"], "readbackName": p["readbackAddress"], "setpointValue": None if p["setpointAddress"] not in self.values else {"value": self.values[p["setpointAddress"]]}, "readbackValue": None, "tags": []} for p in self.pvs.values()]  # fmt: skip
            sid = self._id()
            self.snapshots[sid] = {"id": sid, "title": body["title"], "description": body.get("description"), "createdDate": f"2026-10-29T00:00:{self._n:02d}Z", "createdBy": None, "pvCount": len(values), "pvValues": values}  # fmt: skip
            return httpx.Response(200, json={"jobId": f"job-{sid}", "statusUrl": f"/jobs/v1/job-{sid}"})  # fmt: skip
        sid = path.split("/")[4]
        if (snap := self.snapshots.get(sid)) is None:
            return httpx.Response(404, json={"code": "not_found", "detail": sid})
        if path.endswith("/restore"):
            self.restores.append((sid, body))
            return httpx.Response(
                200, json={"jobId": "never", "statusUrl": "/jobs/v1/never"}
            )
        if m == "PUT":
            snap |= {
                k: body[k] for k in ("title", "description") if body.get(k) is not None
            }
            return httpx.Response(200, json={k: v for k, v in snap.items() if k != "pvValues"})  # fmt: skip
        return httpx.Response(200, json=snap)

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Answer one request as Squirrel (or the jobs module) would."""
        self.keys.append(request.headers.get("X-API-Key"))
        body = json.loads(request.content) if request.content else None
        path = request.url.path
        if path.startswith("/jobs/v1/"):
            sid = path.rsplit("/", 1)[1].removeprefix("job-")
            done = sid in self.snapshots
            return httpx.Response(200, json={"id": path.rsplit("/", 1)[1], "status": "completed" if done else "failed", "result": {"type": "snapshot", "id": sid} if done else None, "error": None if done else "no such job"})  # fmt: skip
        if path.startswith("/squirrel/v1/pvs"):
            return self._pvs(request, body)
        if path.startswith("/squirrel/v1/tags"):
            return self._tags(request, body)
        if path.startswith("/squirrel/v1/snapshots"):
            return self._snapshots(request, body)
        return httpx.Response(404, json={"code": "not_found", "detail": path})
