"""The launcher: which address opens, in which browser, and when."""

from __future__ import annotations

import webbrowser

import pytest

from geecs_plotter import __main__ as launcher


class _Recorder:
    def __init__(self) -> None:
        self.opened: list[str] = []

    def open(self, url: str) -> bool:
        self.opened.append(url)
        return True


@pytest.fixture
def browsers(monkeypatch: pytest.MonkeyPatch) -> dict[object, _Recorder]:
    """``webbrowser.get`` answering from a dict instead of launching anything."""
    known: dict[object, _Recorder] = {None: _Recorder(), "chrome": _Recorder()}

    def get(name: object = None) -> _Recorder:
        if name not in known:
            raise webbrowser.Error(f"could not locate runnable browser {name}")
        return known[name]

    monkeypatch.setattr(launcher.webbrowser, "get", get)
    return known


@pytest.mark.parametrize(
    ("host", "root", "url"),
    [
        ("127.0.0.1", "", "http://127.0.0.1:8600/"),
        ("0.0.0.0", "", "http://127.0.0.1:8600/"),
        ("::1", "/plot/", "http://[::1]:8600/plot/"),
    ],
)
def test_the_page_url(host: str, root: str, url: str) -> None:
    assert launcher.page_url(host, 8600, root) == url


def test_default_is_the_system_browser(browsers: dict) -> None:
    assert launcher.open_browser("http://x/", "default")
    assert browsers[None].opened == ["http://x/"]


def test_a_named_browser(browsers: dict) -> None:
    assert launcher.open_browser("http://x/", "chrome")
    assert browsers["chrome"].opened == ["http://x/"]
    assert browsers[None].opened == []


def test_none_opens_nothing(browsers: dict) -> None:
    assert not launcher.open_browser("http://x/", "none")
    assert launcher.open_when_started(lambda: True, lambda: False, "u", "None") is None
    assert all(not b.opened for b in browsers.values())


def test_an_unknown_browser_warns_and_serves_on(
    browsers: dict, caplog: pytest.LogCaptureFixture
) -> None:
    assert not launcher.open_browser("http://x/", "netscape")
    assert "open it by hand" in caplog.text


def test_it_opens_only_once_the_server_has_started(browsers: dict) -> None:
    calls = iter([False, False, True])
    thread = launcher.open_when_started(
        lambda: next(calls), lambda: False, "http://x/", "chrome", poll_s=0
    )
    assert thread is not None
    thread.join(timeout=5)
    assert browsers["chrome"].opened == ["http://x/"]


def test_a_server_that_never_starts_opens_nothing(browsers: dict) -> None:
    """A port already in use: uvicorn exits, and no tab opens on someone else's page."""
    thread = launcher.open_when_started(
        lambda: False, lambda: True, "http://x/", "chrome", poll_s=0
    )
    assert thread is not None
    thread.join(timeout=5)
    assert browsers["chrome"].opened == []
