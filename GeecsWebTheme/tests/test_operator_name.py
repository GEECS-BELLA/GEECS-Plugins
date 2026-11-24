"""One operator name for every surface: ``window.GeecsOperator`` in ``theme-boot.js``.

The scanner, the portal, the logbook, the Control Room and the configurator
each remembered the operator's name under a key of their own, so behind the
front door (one origin) an operator typed it on every surface. The boot
script now owns the one key and the helper. Pinned here:

- the helper itself, run under node: the shared key, the read-through of the
  old per-surface keys while it is unset, a cleared name staying cleared,
  storage that throws;
- no surface script or template reads or writes ``localStorage`` under the
  shared key or an old key: the boot script's fallback list is the only
  copy of the old ones;
- every page that uses the helper loads ``theme-boot.js`` undeferred, so the
  helper exists before any page script runs.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from geecs_web_theme import theme_boot_js
from geecs_web_theme.testing import _JINJA_COMMENT, node_available

_REPO = Path(__file__).resolve().parents[2]

#: The shared key, and the per-surface keys it replaced (the scanner's was
#: already the shared one).
_KEY = "geecs.operator"
_OLD_KEYS = (
    "geecs-control-room-operator",
    "geecs.configurator.operator",
    "geecs-portal:author",
    "geecs.author",
)

#: Each surface's template and static folders. Its own list: the colour
#: guard's ``_SURFACES`` names files, not folders, and lives in a test module.
_SURFACES = {
    "GeecsScanner": ["GeecsScanner/geecs_scanner"],
    "GEECS-DataPortal": [
        "GEECS-DataPortal/geecs_portal",
        "GEECS-DataPortal/geecs_portal/plotter",
    ],
    "GeecsLogbook": ["GeecsLogbook/geecs_logbook"],
    "GeecsControlRoom": ["GeecsControlRoom/geecs_control_room"],
    "GeecsConfigurator": ["GeecsConfigurator/geecs_configurator"],
}


def _files(root: str, folder: str, suffix: str) -> list[Path]:
    return sorted((_REPO / root / folder).glob(f"*{suffix}"))


def _strip_jinja_comments(text: str) -> str:
    return _JINJA_COMMENT.sub(" ", text)


# ---------------------------------------------------------------- the helper

_HARNESS = r"""
const store = Object.assign({}, STORE);
const storage = THROWS
  ? { getItem() { throw new Error("blocked"); }, setItem() { throw new Error("blocked"); } }
  : { getItem: (k) => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); } };
global.window = { localStorage: storage, matchMedia: () => ({ matches: false }) };
global.document = { documentElement: { setAttribute() {} } };
"""


def _run(store: dict[str, str], steps: str, throws: bool = False) -> dict:
    if not node_available():  # pragma: no cover - CI and dev machines have it
        pytest.skip("node not available")
    script = (
        _HARNESS.replace("STORE", json.dumps(store)).replace(
            "THROWS", "true" if throws else "false"
        )
        + theme_boot_js().read_text()
        + "\nconst out = {};\n"
        + steps
        + "\nconsole.log(JSON.stringify({ out, store }));"
    )
    done = subprocess.run(
        ["node", "-"], input=script, text=True, capture_output=True, timeout=30
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_the_shared_key_is_read_and_written() -> None:
    r = _run(
        {_KEY: "ann", "geecs.author": "bob"},
        'out.first = window.GeecsOperator.get(); window.GeecsOperator.set("  cat ");'
        " out.second = window.GeecsOperator.get();",
    )
    assert r["out"] == {"first": "ann", "second": "cat"}
    assert r["store"][_KEY] == "cat"
    assert r["store"]["geecs.author"] == "bob"  # an old key is never written


@pytest.mark.parametrize("old", _OLD_KEYS)
def test_an_old_key_fills_an_unset_shared_key_once(old: str) -> None:
    r = _run({old: " dee "}, "out.name = window.GeecsOperator.get();")
    assert r["out"]["name"] == "dee"
    assert r["store"][_KEY] == "dee"


def test_the_old_keys_are_read_in_order() -> None:
    r = _run(
        {"geecs.author": "logbook", "geecs-control-room-operator": "control room"},
        "out.name = window.GeecsOperator.get();",
    )
    assert r["out"]["name"] == "control room"


def test_a_cleared_name_is_never_refilled_from_an_old_key() -> None:
    r = _run(
        {_KEY: "ann", "geecs-portal:author": "bob"},
        'window.GeecsOperator.set(""); out.name = window.GeecsOperator.get();',
    )
    assert r["out"]["name"] == ""


def test_no_name_anywhere_is_the_empty_string() -> None:
    r = _run({}, "out.name = window.GeecsOperator.get();")
    assert r["out"]["name"] == "" and _KEY not in r["store"]


def test_blocked_storage_reads_empty_and_writes_nothing() -> None:
    r = _run(
        {},
        'window.GeecsOperator.set("ann"); out.name = window.GeecsOperator.get();',
        throws=True,
    )
    assert r["out"]["name"] == ""


def test_the_fallback_list_is_the_old_keys() -> None:
    """The boot script's list is the one copy the guard below exempts."""
    boot = theme_boot_js().read_text()
    listed = re.search(r"OLD_OPERATOR_KEYS = \[(.*?)\]", boot, re.S).group(1)
    assert tuple(re.findall(r'"([^"]+)"', listed)) == _OLD_KEYS
    assert f'"{_KEY}"' in boot


# ---------------------------------------------------------------- the surfaces


def _surface_sources() -> list[Path]:
    found = []
    for roots in _SURFACES.values():
        for root in roots:
            found += _files(root, "static", ".js") + _files(root, "templates", ".html")
    return found


@pytest.mark.parametrize(
    "path", _surface_sources(), ids=lambda p: str(p.relative_to(_REPO))
)
def test_no_surface_stores_under_an_operator_key(path: Path) -> None:
    """A surface reads and writes the name through ``GeecsOperator`` only.

    Only a ``localStorage`` access is judged: its key written in the call,
    or a constant the call names. A quoted ``geecs.operator`` elsewhere (the
    run metadata key of that name) is not a storage access.
    """
    assert _storage_keys(path.read_text()) & {_KEY, *_OLD_KEYS} == set(), path.name


def _storage_keys(text: str) -> set[str]:
    """Every key a script's ``localStorage`` calls name, constants resolved."""
    constants = dict(
        re.findall(r"(?:const|let|var)\s+(\w+)\s*=\s*[\"'`]([^\"'`]+)[\"'`]", text)
    )
    keys = set()
    for arg in re.findall(r"localStorage\.\w+\(\s*([^,)]+)", text):
        arg = arg.strip()
        literal = re.fullmatch(r"[\"'`]([^\"'`]+)[\"'`]", arg)
        keys.add(literal.group(1) if literal else constants.get(arg, arg))
    return keys


def test_the_storage_guard_bites() -> None:
    """The guard sees a literal key, a constant, and not a metadata string."""
    assert _storage_keys('localStorage.getItem("geecs.author")') == {"geecs.author"}
    assert _storage_keys(
        'const K = "geecs.operator"; try { localStorage.setItem(K, v); } catch (e) {}'
    ) == {"geecs.operator"}
    assert _storage_keys('md["geecs.operator"] = name;') == set()


def _users(root: str) -> set[str]:
    """The stems of a surface's scripts that call the helper."""
    return {
        p.stem
        for p in _files(root, "static", ".js")
        if "GeecsOperator" in p.read_text()
    }


def _chain(template: Path) -> str:
    """A template's text with every template it extends, comments stripped."""
    text = _strip_jinja_comments(template.read_text())
    parent = re.search(r'\{%\s*extends\s+"([^"]+)"\s*%\}', text)
    if parent:
        text += _chain(template.parent / parent.group(1))
    return text


def _pages_using_the_name() -> dict[str, list[Path]]:
    pages: dict[str, list[Path]] = {}
    for surface, roots in _SURFACES.items():
        for root in roots:
            stems = _users(root)
            for template in _files(root, "templates", ".html"):
                text = _strip_jinja_comments(template.read_text())
                if "GeecsOperator" in text or any(
                    re.search(rf"[\"'/]{re.escape(stem)}(\.js)?[\"']", text)
                    for stem in stems
                ):
                    pages.setdefault(surface, []).append(template)
    return pages


def test_every_surface_uses_the_helper() -> None:
    """The walk below finds a page on each of the five surfaces.

    Without this, a renamed folder or a changed script tag would leave the
    load check with nothing to check, and pass.
    """
    assert set(_pages_using_the_name()) == set(_SURFACES)


@pytest.mark.parametrize(
    "page",
    [p for pages in _pages_using_the_name().values() for p in pages],
    ids=lambda p: str(p.relative_to(_REPO)),
)
def test_a_page_using_the_name_loads_the_boot_script_undeferred(page: Path) -> None:
    """The helper must exist before any page script runs, inline ones included."""
    tags = re.findall(r"<script\b[^>]*theme-boot\.js[^>]*>", _chain(page))
    assert tags, f"{page.name} uses GeecsOperator without loading theme-boot.js"
    for tag in tags:
        assert "defer" not in tag and "async" not in tag, tag
