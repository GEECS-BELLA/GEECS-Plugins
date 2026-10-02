"""deploy/render_conf.sh against the fleet's example site.env."""

import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO_ROOT = PKG.parent
RENDER = PKG / "deploy" / "render_conf.sh"
EXAMPLE = REPO_ROOT / "deploy" / "site.env.example"
ARCHIVER_KEYS = "GEECS_ARCHIVER_HOST=10.1.2.3\nGEECS_ARCHIVER_DATA_ROOT=/srv/arch root\nGEECS_ARCHIVER_JAVA_OPTS=-Xmx2g\n"


def site_env(tmp_path: Path, extra: str = ARCHIVER_KEYS) -> Path:
    text = EXAMPLE.read_text(encoding="utf-8")
    if "GEECS_ARCHIVER_HOST=" in text:
        # once the example carries the keys itself, override them for the assertions
        text = "\n".join(
            line for line in text.splitlines() if not line.startswith("GEECS_ARCHIVER_")
        )
    p = tmp_path / "site.env"
    p.write_text(text + "\n" + extra, encoding="utf-8")
    return p


def render(site: Path, out: Path):
    return subprocess.run(
        ["bash", str(RENDER), str(site), str(out)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "RENDER_QUIET": "0"},
    )


def test_renders_compose_and_appliances_and_copies_static_conf(tmp_path):
    out = tmp_path / "out"
    result = render(site_env(tmp_path), out)
    assert result.returncode == 0, result.stderr
    compose = (out / "compose.yaml").read_text()
    assert '"/srv/arch root/lts:/usr/local/tomcat/storage/lts"' in compose
    assert "@" not in "\n".join(
        line for line in compose.splitlines() if not line.lstrip().startswith("#")
    )
    assert 'user: "' in compose
    appliances = (out / "appliances.xml").read_text()
    assert (
        "<data_retrieval_url>http://10.1.2.3:17665/retrieval</data_retrieval_url>"
        in appliances
    )
    for static in ("server.xml", "context.xml", "policies.py", "archappl.properties"):
        assert (out / static).exists()
    assert 'port="17665"' in (out / "server.xml").read_text()
    assert "sudo install" in result.stdout


def test_missing_archiver_key_fails_before_rendering(tmp_path):
    out = tmp_path / "out"
    result = render(site_env(tmp_path, extra="GEECS_ARCHIVER_HOST=10.1.2.3\n"), out)
    assert result.returncode == 2
    assert "GEECS_ARCHIVER_DATA_ROOT" in result.stderr
    assert not out.exists()


@pytest.mark.parametrize("args", [[], ["only-one"]])
def test_usage(args):
    result = subprocess.run(
        ["bash", str(RENDER), *args], capture_output=True, text=True
    )
    assert result.returncode == 2 and "usage" in result.stderr


def test_every_conf_file_is_ascii():
    """The appliance reads policies.py with Jython 2 (a non-ASCII byte without an encoding
    declaration fails every archive request: first production start, 2026-10-02) and
    archappl.properties as ISO-8859-1. Keep the whole conf directory ASCII."""
    offenders = {
        p.name: [c for c in p.read_text(encoding="utf-8") if not c.isascii()][:3]
        for p in (PKG / "deploy").iterdir()
        if p.is_file() and not p.read_text(encoding="utf-8").isascii()
    }
    assert offenders == {}, offenders
