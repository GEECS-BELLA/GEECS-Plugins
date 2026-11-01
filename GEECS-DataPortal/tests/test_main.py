"""The launcher: which address opens, in which browser, and when (GeecsPlotter's, moved here)."""

from __future__ import annotations

import webbrowser

import pytest

from geecs_portal import __main__ as launcher


class _Recorder:
    def __init__(self) -> None:
        self.opened: list[str] = []

    def open(self, url: str) -> bool:
        self.opened.append(url)
        return True


@pytest.fixture
def browsers(monkeypatch: pytest.MonkeyPatch) -> dict[object, _Recorder]:
    """``webbrowser.get`` answering from a dict instead of launching anything."""
    monkeypatch.setattr(launcher, "has_display", lambda: True)
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
        ("127.0.0.1", "", "http://127.0.0.1:8200/"),
        ("0.0.0.0", "", "http://127.0.0.1:8200/"),
        ("::1", "/plot/", "http://[::1]:8200/plot/"),
    ],
)
def test_the_page_url(host: str, root: str, url: str) -> None:
    assert launcher.page_url(host, 8200, root) == url


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


def test_a_server_stopped_before_it_starts_opens_nothing(browsers: dict) -> None:
    """A Ctrl-C during startup: the thread gives up without opening a tab."""
    thread = launcher.open_when_started(
        lambda: False, lambda: True, "http://x/", "chrome", poll_s=0
    )
    assert thread is not None
    thread.join(timeout=5)
    assert browsers["chrome"].opened == []


def test_default_matches_in_any_case(browsers: dict) -> None:
    assert launcher.open_browser("http://x/", "Default")
    assert browsers[None].opened == ["http://x/"]


def test_a_linux_host_with_no_display_opens_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher.sys, "platform", "linux")
    assert not launcher.has_display({})
    assert launcher.has_display({"WAYLAND_DISPLAY": "wayland-0"})
    monkeypatch.setattr(launcher.sys, "platform", "darwin")
    assert launcher.has_display({})


def test_open_browser_honours_a_missing_display(
    browsers: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launcher, "has_display", lambda: False)
    assert not launcher.open_browser("http://x/", "chrome")
    assert browsers["chrome"].opened == []


def test_ctrl_c_exits_quietly(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """uvicorn re-raises the Ctrl-C after shutdown; main() must not show it."""
    import uvicorn

    class _Server:
        started = False
        should_exit = False

        def __init__(self, _config: object) -> None:
            pass

        def run(self) -> None:
            raise KeyboardInterrupt

    monkeypatch.setattr(uvicorn, "Server", _Server)
    monkeypatch.setattr(uvicorn, "Config", lambda *a, **k: None)
    # Hermetic: the settings sets go to a tmp folder, never this machine's.
    argv = ["geecs-data-portal", "--browser", "none", "--settings-dir", str(tmp_path)]
    monkeypatch.setattr(launcher.sys, "argv", argv)
    launcher.main()  # returns instead of raising
