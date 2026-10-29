"""The skeleton: settings, /healthz, the page shell and what it hands the browser."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from geecs_web_theme import PANE_STATES, STATES
from geecs_web_theme.testing import (
    bare_url_for_calls,
    inline_scripts,
    javascript_syntax_error,
    node_available,
    unknown_data_states,
)

from geecs_control_room.home import SCANNER_TIMEOUT_S
from geecs_control_room.module_client import READ_TIMEOUT_S
from geecs_control_room.settings import Settings, SettingsError

_TEMPLATES = sorted(
    (Path(__file__).resolve().parents[1] / "geecs_control_room" / "templates").glob(
        "*.html"
    )
)
_ENV = {
    "GEECS_EXPERIMENT": "Bench",
    "GEECS_CONFIGS_ROOT": "/configs",
    "GEECS_CONTROL_ROOM_CANOPY_URL": "http://canopy.test/",
    "GEECS_CONTROL_ROOM_READ_KEY": "k",
}


def test_settings_take_the_shared_keys_beneath_their_own() -> None:
    s = Settings.from_env(_ENV)
    assert (s.experiment, s.canopy_url, s.browser_url) == (
        "Bench",
        "http://canopy.test",
        "http://canopy.test",
    )
    assert s.experiment_dir == Path("/configs/scanner_configs/experiments/Bench")
    own = Settings.from_env(_ENV | {"GEECS_CONTROL_ROOM_EXPERIMENT": "Other"})
    assert own.experiment == "Other"


@pytest.mark.parametrize("key", sorted(_ENV))
def test_settings_refuse_a_missing_value(key: str) -> None:
    with pytest.raises(SettingsError, match=key.removeprefix("GEECS_")):
        Settings.from_env({k: v for k, v in _ENV.items() if k != key})


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok", "experiment": "Bench"}


def test_the_page_hands_the_browser_hare_and_the_read_key(client: TestClient) -> None:
    html = client.get("/").text
    assert '<body class="kit"' in html and "Control Room" in html
    assert 'data-hare-url="http://canopy.test/hare/v1/ws/pvs"' in html
    assert 'data-hare-key="read-key-for-tests"' in html
    for asset in ("/static/hare.js", "/static/control_room.css", "/theme/kit.css"):
        assert asset in html and client.get(asset).status_code == 200


def test_the_page_carries_the_proxy_prefix(client: TestClient) -> None:
    html = client.get("/", headers={"X-Forwarded-Prefix": "/room"}).text
    assert 'src="/room/static/hare.js"' in html and 'data-root="/room"' in html


@pytest.mark.parametrize("template", _TEMPLATES, ids=lambda p: p.name)
def test_templates_keep_the_kit_rules(template: Path) -> None:
    assert bare_url_for_calls(template) == []
    assert unknown_data_states(template, [*STATES, *PANE_STATES]) == []
    if node_available():
        for code in inline_scripts(template):
            assert javascript_syntax_error(code) is None


def test_every_polled_read_answers_inside_the_page_timeout() -> None:
    # poller.js gives up at 1.5 x the period; a route's worst case is two reads in
    # sequence (the home's roster, then details + journal; the strip's two scanner
    # GETs), and a slow device must reach the page as unread, not as a page that gave up
    static = Path(__file__).resolve().parents[1] / "geecs_control_room/static"
    assert "timeoutMs = 1.5 * pollMs" in (static / "poller.js").read_text()
    for page in ("screen.js", "home.js"):
        poll_s = int(re.search(r"POLL_MS = (\d+)", (static / page).read_text())[1]) / 1000  # fmt: skip
        assert 2 * max(READ_TIMEOUT_S, SCANNER_TIMEOUT_S) < 1.5 * poll_s, page
