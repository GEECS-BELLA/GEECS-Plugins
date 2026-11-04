"""Screenshot bench for raw-frames R2: the real camera page over GeecsCanopy's own test module,
the test-only frame.raw fixture, and a fake Hare socket in the same process. Never a device."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, WebSocket

sys.path.insert(0, sys.argv[1])  # the GeecsControlRoom folder (tests/ importable)
from geecs_canopy.testing import EXPERIMENT, live, make_env  # noqa: E402
from geecs_control_room.app import create_app  # noqa: E402
from geecs_control_room.images import Images  # noqa: E402
from geecs_control_room.module_client import ModuleClient  # noqa: E402
from geecs_control_room.settings import Settings  # noqa: E402
from tests.raw_fixture import RawFixture  # noqa: E402

PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8791
TYPES = ["None", "Rot90", "Rot180", "Rot270", "Mirror", "Rot90Mirror", "Rot180Mirror", "Rot270Mirror"]
tmp = Path(tempfile.mkdtemp())
env = make_env(tmp)
canopy = FastAPI()
canopy.mount("/geecs", env.app)
module = ModuleClient("http://canopy.test", "read-key", transport=httpx.ASGITransport(app=canopy))


class NoCa:
    async def text(self, pv): raise TimeoutError("bench: no IOC")
    async def host(self, pv): raise TimeoutError("bench: no IOC")
    async def arrays(self, pvs): raise TimeoutError("bench: no IOC")


async def nofetch(url: str) -> bytes:
    raise TimeoutError("bench: no MJPEG server")


settings = Settings(experiment=EXPERIMENT, configs_root=tmp, canopy_url="http://canopy.test", canopy_browser_url=f"http://127.0.0.1:{PORT}", read_key="read-key")
app = create_app(settings, module, camera=Images(NoCa(), nofetch))
app.include_router(RawFixture().router())

VALUES = {
    "transform": 0, "state": 1, "acquire": 1, "exposure": 0.01, "period": 1.0, "gain": 0, "rate": 1.0,
    "width": 600, "height": 600, "image_counter": 1, "pixel_format": 1,
    "crosshair1_use": 1, "crosshair1_x": 180, "crosshair1_y": 420, "crosshair1_r": 0, "crosshair1_g": 200, "crosshair1_b": 255,
    "roi_minx": 340, "roi_miny": 120, "roi_sizex": 160, "roi_sizey": 140,
}
META = {"transform": {"enumChoices": TYPES}}


@app.websocket("/hare/v1/ws/pvs")
async def hare(ws: WebSocket) -> None:
    await ws.accept()
    _, route = await module.device("Cam1")
    roles = {r["role"]: r["pv"] for r in route.get("roles", []) if r.get("pv")}
    wires = {roles[k]: live(v) for k, v in VALUES.items() if k in roles}
    env.hare.values.update(wires)
    meta = {roles[k]: m for k, m in META.items() if k in roles}

    async def beat() -> None:
        while True:
            await asyncio.sleep(4)
            await ws.send_json({"type": "heartbeat"})

    task = asyncio.create_task(beat())
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "subscribe":
                names = msg.get("pvNames", [])
                await ws.send_json({"type": "initial", "data": {p: wires.get(p, {"value": None, "connected": True, "severity": "NO_ALARM"}) for p in names}, "meta": {p: meta.get(p, {}) for p in names}})
    except Exception:
        task.cancel()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
