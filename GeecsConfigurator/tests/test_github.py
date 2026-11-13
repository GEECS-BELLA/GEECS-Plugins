"""The GitHub App against a fake GitHub (an httpx mock transport): tokens, git's credential, pull requests."""

from __future__ import annotations

import base64
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import jwt
import pytest
from conftest import git
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from geecs_configurator.app import create_app
from geecs_configurator.github import AppCredential, AppPullRequests, GitHubError
from geecs_configurator.repository import ConfigsRepository, Identity
from geecs_configurator.settings import Settings


class FakeGitHub:
    """GitHub's two endpoints the App uses, checking the JWT against the App's public key."""

    def __init__(
        self, public_key: bytes, app_id: str = "123", installation: str = "456"
    ):
        self.public_key, self.app_id, self.installation = (
            public_key,
            app_id,
            installation,
        )
        self.minted, self.pulls, self.refuse_pulls = 0, [], False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("Authorization", "")
        if request.url.path == f"/app/installations/{self.installation}/access_tokens":
            claims = jwt.decode(auth.removeprefix("Bearer "), self.public_key, algorithms=["RS256"])  # raises if forged  # fmt: skip
            assert claims["iss"] == self.app_id
            self.minted += 1
            expires = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")  # fmt: skip
            return httpx.Response(201, json={"token": f"ghs_token{self.minted}", "expires_at": expires})  # fmt: skip
        if request.url.path == "/repos/org/configs/pulls":
            assert auth == f"token ghs_token{self.minted}"
            if self.refuse_pulls:
                return httpx.Response(422, json={"message": "Validation Failed"})
            self.pulls.append(json.loads(request.content))
            return httpx.Response(201, json={"html_url": f"https://github.invalid/org/configs/pull/{len(self.pulls)}"})  # fmt: skip
        return httpx.Response(404, json={"message": "Not Found"})


@pytest.fixture
def key(tmp_path: Path) -> tuple[Path, bytes]:
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())  # fmt: skip
    path = tmp_path / "app.pem"
    path.write_bytes(pem)
    path.chmod(0o600)
    public = private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)  # fmt: skip
    return path, public


@pytest.fixture
def github(key: tuple[Path, bytes]) -> FakeGitHub:
    return FakeGitHub(key[1])


def _credential(key: tuple[Path, bytes], github: FakeGitHub) -> AppCredential:
    return AppCredential("123", "456", key[0], httpx.Client(transport=httpx.MockTransport(github)))  # fmt: skip


def test_a_token_is_minted_from_the_key_and_reused_until_near_expiry(key: tuple[Path, bytes], github: FakeGitHub) -> None:  # fmt: skip
    cred = _credential(key, github)
    assert cred.token() == cred.token() == "ghs_token1" and github.minted == 1
    cred._expires = 0.0  # expired
    assert cred.token() == "ghs_token2" and github.minted == 2


def test_a_forged_key_is_refused_by_github(tmp_path: Path, github: FakeGitHub) -> None:  # fmt: skip
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = tmp_path / "other.pem"
    pem.write_bytes(other.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))  # fmt: skip
    with pytest.raises(jwt.InvalidSignatureError):
        AppCredential("123", "456", pem, httpx.Client(transport=httpx.MockTransport(github))).token()  # fmt: skip


def test_github_unreachable_is_a_github_error(key: tuple[Path, bytes]) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(GitHubError, match="did not answer"):
        AppCredential("123", "456", key[0], httpx.Client(transport=httpx.MockTransport(down))).token()  # fmt: skip


def test_git_gets_the_token_in_its_environment_never_on_its_command_line(key: tuple[Path, bytes], github: FakeGitHub, origin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    cred = _credential(key, github)
    repo = ConfigsRepository(tmp_path / "state", str(origin), "main", Identity("c", "c@test.invalid"), credential=cred.git_env)  # fmt: skip
    seen: list[tuple[list[str], dict[str, str]]] = []
    real = subprocess.run

    def spy(args: list[str], **kw: object) -> subprocess.CompletedProcess:
        seen.append((list(args), dict(kw.get("env") or {})))  # type: ignore[arg-type]
        return real(args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(subprocess, "run", spy)
    repo.ensure()
    fetches = [(a, e) for a, e in seen if a[1] == "fetch"]
    assert fetches and all("ghs_token1" not in " ".join(a) for a, _ in seen)
    header = fetches[0][1]["GIT_CONFIG_VALUE_0"]
    assert fetches[0][1]["GIT_CONFIG_KEY_0"] == "http.extraHeader"
    assert base64.b64decode(header.removeprefix("Authorization: Basic ")).decode() == "x-access-token:ghs_token1"  # fmt: skip
    local = [(a, e) for a, e in seen if a[1] in ("rev-parse", "worktree", "update-ref")]
    assert local and all(
        "GIT_CONFIG_VALUE_0" not in e for _, e in local
    )  # only network commands carry it


def test_the_pull_request_is_opened_as_the_app(key: tuple[Path, bytes], github: FakeGitHub) -> None:  # fmt: skip
    pr = AppPullRequests(_credential(key, github), "org/configs", "main").open("configurator/x", "t", "Proposed by Ada")  # fmt: skip
    assert pr.opened and pr.url.endswith("/pull/1")
    assert github.pulls == [{"title": "t", "head": "configurator/x", "base": "main", "body": "Proposed by Ada"}]  # fmt: skip


def _app_env(origin: Path, state: Path, key: Path) -> dict[str, str]:
    return {
        "GEECS_CONFIGURATOR_STATE_DIR": str(state), "GEECS_CONFIGURATOR_ORIGIN": str(origin),
        "GEECS_CONFIGURATOR_RECONCILE_SECONDS": "0", "GEECS_CONFIGURATOR_FETCH_SECONDS": "0",
        "GEECS_CONFIGURATOR_GITHUB_REPO": "org/configs", "GEECS_CONFIGURATOR_GITHUB_APP_ID": "123",
        "GEECS_CONFIGURATOR_GITHUB_INSTALLATION_ID": "456", "GEECS_CONFIGURATOR_GITHUB_APP_KEY_FILE": str(key),
    }  # fmt: skip


def test_a_maintainer_save_opens_the_pull_request_as_the_app_and_logs_it_pending(key: tuple[Path, bytes], github: FakeGitHub, origin: Path, tmp_path: Path) -> None:  # fmt: skip
    http = httpx.Client(transport=httpx.MockTransport(github))
    with TestClient(create_app(Settings.from_env(_app_env(origin, tmp_path / "s", key[0])), http=http)) as c:  # fmt: skip
        doc = c.get("/api/documents/archive_policy/archive_policy.yaml?experiment=Bench").json()  # fmt: skip
        saved = c.put("/api/documents/archive_policy/archive_policy.yaml?experiment=Bench", json={"document": {**doc["document"], "exclude": ["U_A:B"]}, "etag": doc["etag"], "operator": "Ada"}).json()  # fmt: skip
        assert saved["outcome"] == "proposed" and saved["pull_request"].endswith(
            "/pull/1"
        )
        assert github.pulls[0]["head"] == saved["branch"] and "Proposed by Ada" in github.pulls[0]["body"]  # fmt: skip
        (row,) = c.get("/api/log").json()
        assert (row["source"], row["state"], row["link"]) == ("proposed", "pending", saved["pull_request"])  # fmt: skip
        assert c.get("/healthz").json()["credential"] == "GitHub App"
        everything = json.dumps([saved, c.get("/api/log").json(), c.get("/healthz").json()]) + c.get("/log").text  # fmt: skip
        assert "ghs_token" not in everything  # the token is never served
    assert "ghs_token" not in git("log", "--all", "--format=%B", cwd=origin)


def test_a_refused_pull_request_keeps_the_pushed_branch_and_says_so(key: tuple[Path, bytes], github: FakeGitHub, origin: Path, tmp_path: Path) -> None:  # fmt: skip
    github.refuse_pulls = True
    http = httpx.Client(transport=httpx.MockTransport(github))
    with TestClient(create_app(Settings.from_env(_app_env(origin, tmp_path / "s", key[0])), http=http)) as c:  # fmt: skip
        doc = c.get("/api/documents/archive_policy/archive_policy.yaml?experiment=Bench").json()  # fmt: skip
        r = c.put("/api/documents/archive_policy/archive_policy.yaml?experiment=Bench", json={"document": {**doc["document"], "exclude": ["U_A:B"]}, "etag": doc["etag"], "operator": "Ada"})  # fmt: skip
    saved = r.json()
    assert r.status_code == 200 and saved["outcome"] == "proposed" and saved["pull_request"] is None  # fmt: skip
    assert any("Validation Failed" in n for n in saved["notes"])
    assert git("rev-parse", "--verify", saved["branch"], cwd=origin)


def test_github_down_is_a_git_failure_everywhere_never_a_500(key: tuple[Path, bytes], github: FakeGitHub, origin: Path, tmp_path: Path) -> None:  # fmt: skip
    from geecs_configurator import inbox
    from geecs_configurator.app import reconcile

    state = tmp_path / "s"
    env = _app_env(origin, state, key[0])
    with TestClient(create_app(Settings.from_env(env), http=httpx.Client(transport=httpx.MockTransport(github)))) as c:  # fmt: skip
        etag = c.get("/api/documents/setup/low_power.yaml?experiment=Bench").json()["etag"]  # fmt: skip

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to GitHub", request=request)

    # a restart while GitHub's API is down: the live checkout on disk serves
    with TestClient(create_app(Settings.from_env(env), http=httpx.Client(transport=httpx.MockTransport(down)))) as c:  # fmt: skip
        r = c.get("/api/documents/setup/low_power.yaml?experiment=Bench")
        assert r.status_code == 200 and "no credential" in r.headers["X-Configs-Stale"] and r.json()["etag"] == etag  # fmt: skip
        doc = r.json()
        saved = c.put("/api/documents/setup/low_power.yaml?experiment=Bench", json={"document": {**doc["document"], "description": "x"}, "etag": etag, "operator": "Ada"})  # fmt: skip
        assert saved.status_code == 502 and "no credential" in saved.json()["detail"]
        repo = c.app.state.repo
        (repo.inbox / "scanner_configs/experiments/Bench/shot_offsets.yaml").write_text("schema_version: 1\nreference: A\ndevices:\n  A: {offset_s: 0.0}\n")  # fmt: skip
        assert [p.outcome for p in reconcile(c.app)] == [
            "failed"
        ]  # kept, and backing off
        assert reconcile(c.app) == []
        inbox.reset_backoff(repo)


def test_the_app_settings_come_together_or_not_at_all(origin: Path, tmp_path: Path, key: tuple[Path, bytes]) -> None:  # fmt: skip
    from geecs_configurator.settings import SettingsError

    env = _app_env(origin, tmp_path / "s", key[0])
    del env["GEECS_CONFIGURATOR_GITHUB_INSTALLATION_ID"]
    with pytest.raises(SettingsError, match="together"):
        Settings.from_env(env)
    env = _app_env(origin, tmp_path / "s", key[0])
    del env["GEECS_CONFIGURATOR_GITHUB_REPO"]
    with pytest.raises(SettingsError, match="GITHUB_REPO"):
        Settings.from_env(env)
