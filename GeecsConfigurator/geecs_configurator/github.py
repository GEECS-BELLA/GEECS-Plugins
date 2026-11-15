"""The GitHub App: installation tokens for git, and the maintainer tier's pull requests.

The App (the maintainer creates it once) is installed on the configs repo
alone, with contents and pull requests write, and sits on the repo's
ruleset bypass list: that bypass is the whole operator tier. Its private
key is a file beside site.env, readable by the service account only; the
service mints an **installation token** from it as needed (an RS256 JWT
for the App, exchanged for a token valid an hour, reused until five
minutes before it expires) and never stores one.

- :meth:`AppCredential.git_env` authenticates git's fetch and push over
  HTTPS through ``GIT_CONFIG_*`` in the environment: the token never
  appears on a command line.
- :class:`AppPullRequests` opens the maintainer tier's pull request as the
  App, the operator named in its body. :class:`save.CompareLinkOnly` stays
  the deploy-key fallback (it pushes, it cannot open a pull request).

Nothing here is called in tests against GitHub: the HTTP client is
injected, and the tests hand it a fake.
"""

from __future__ import annotations

import base64
import threading
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import httpx
import jwt

from geecs_configurator.save import PullRequest

#: GitHub's REST API (a site with GitHub Enterprise would set its own; none here does).
API = "https://api.github.com"
#: Seconds before expiry that a token is no longer reused.
TOKEN_MARGIN_S = 300.0


class GitHubError(RuntimeError):
    """GitHub refused or did not answer (the status and its message)."""


def _post(http: httpx.Client, url: str, what: str, headers: dict[str, str], json: dict[str, str] | None = None) -> dict:  # fmt: skip
    """POST and return the JSON answer; :class:`GitHubError` on a refusal or no answer."""
    try:
        r = http.post(url, headers=headers, json=json)
    except httpx.HTTPError as exc:
        raise GitHubError(f"{what}: GitHub did not answer: {exc}") from exc
    if r.status_code >= 300:
        try:
            msg = r.json().get("message", r.text)
        except ValueError:
            msg = r.text
        raise GitHubError(f"{what}: {r.status_code} {msg}"[:400])
    return r.json()


class AppCredential:
    """An installation token for the configs repo, minted from the App's private key.

    Parameters
    ----------
    app_id, installation_id : str
        The App's id and its installation on the configs repo (site values).
    key_file : Path
        The App's private key (PEM), readable by the service account only.
    http : httpx.Client
        The client to GitHub (a fake in tests).
    """

    def __init__(self, app_id: str, installation_id: str, key_file: Path, http: httpx.Client, api: str = API):  # fmt: skip
        self.app_id, self.installation_id, self.key_file = app_id, installation_id, Path(key_file)  # fmt: skip
        self.http, self.api = http, api.rstrip("/")
        self._token, self._expires = "", 0.0
        self._lock = threading.Lock()

    def _jwt(self) -> str:
        now = int(time.time())
        claims = {
            "iat": now - 60,
            "exp": now + 540,
            "iss": self.app_id,
        }  # GitHub allows ten minutes
        return jwt.encode(claims, self.key_file.read_bytes(), algorithm="RS256")

    def token(self) -> str:
        """A token valid for at least five more minutes (minted when none is)."""
        with self._lock:
            if time.time() < self._expires - TOKEN_MARGIN_S:
                return self._token
            body = _post(
                self.http, f"{self.api}/app/installations/{self.installation_id}/access_tokens",
                "minting an installation token",
                headers={"Authorization": f"Bearer {self._jwt()}", "Accept": "application/vnd.github+json"},
            )  # fmt: skip
            expires = datetime.fromisoformat(body["expires_at"].replace("Z", "+00:00")).timestamp()  # fmt: skip
            self._token, self._expires = body["token"], expires
            return self._token

    def git_env(self) -> Mapping[str, str]:
        """The environment that makes git's HTTPS fetch and push present the token (``x-access-token``)."""
        basic = base64.b64encode(f"x-access-token:{self.token()}".encode()).decode()
        return {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.extraHeader",
            "GIT_CONFIG_VALUE_0": f"Authorization: Basic {basic}",
        }


class AppPullRequests:
    """Opens the maintainer tier's pull request as the App."""

    def __init__(self, credential: AppCredential, repo: str, base: str):
        self.credential, self.repo, self.base = credential, repo, base

    def open(self, branch: str, title: str, body: str) -> PullRequest:
        """Open a pull request from ``branch`` into the base; :class:`GitHubError` when GitHub refuses."""
        answer = _post(
            self.credential.http, f"{self.credential.api}/repos/{self.repo}/pulls", "opening the pull request",
            headers={"Authorization": f"token {self.credential.token()}", "Accept": "application/vnd.github+json"},
            json={"title": title, "head": branch, "base": self.base, "body": body},
        )  # fmt: skip
        return PullRequest(opened=True, url=answer["html_url"])
