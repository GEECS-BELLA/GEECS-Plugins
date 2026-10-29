"""The module client: the put's shape, the module's answer passed through, no retry."""

from __future__ import annotations

import httpx
import pytest

from geecs_control_room.module_client import ModuleClient, ModuleUnavailable
from tests.conftest import body_of


async def test_a_put_goes_once_to_the_module_route_with_the_read_key(
    module: ModuleClient, sent: list[httpx.Request]
) -> None:
    body = {"role": "output_sp", "value": 1, "operator": "A. Operator"}
    status, answer = await module.put("fps 1", body)
    assert (status, answer) == (428, {"outcome": "denied", "code": "confirm_required"})
    (request,) = sent  # sent once: a refusal is shown, never retried
    assert request.method == "POST"
    assert request.url.raw_path == b"/geecs/v1/devices/fps%201/put"
    assert request.headers["X-API-Key"] == "read-key-for-tests"
    assert body_of(request) == body


async def test_an_unreachable_module_is_unavailable_not_an_answer() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = ModuleClient(
        "http://canopy.test", "k", transport=httpx.MockTransport(down)
    )
    with pytest.raises(ModuleUnavailable):
        await client.devices()
