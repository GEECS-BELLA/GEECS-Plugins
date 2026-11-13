"""The commit path assembled: the repository with its gate and hooks, the credential, the log, the share.

:class:`CommitPath` is what the app holds when the commit path is on. It
wires the pieces the other modules keep apart:

- the **gate** (``accept``): before the live checkout moves past a commit
  it did not make, the documents that changed must validate
  (:func:`validation.problems`, the configs repo's CI check too). A refused
  commit is logged once (``refused``, ``pending``: the maintainer fixes main
  by pull request) and never applied; every fast-forward stops there until
  main is fixed;
- the **advance hook**: each foreign commit applied (a merged pull request,
  a maintainer's push) is logged per document, its state the apply seam's
  answer, and the share's clone is fast-forwarded from the service's
  repository;
- the **timer**: the inbox every ``reconcile_seconds``, a fetch every
  ``fetch_seconds`` (how a merge on GitHub takes effect with nobody in the
  editor);
- the **credential**: the GitHub App's installation token for git and its
  pull requests when the App is configured, else git's own (a deploy key)
  and :class:`save.CompareLinkOnly`.
"""

from __future__ import annotations

import logging
import threading
import time

import httpx

from geecs_configurator import inbox, validation
from geecs_configurator.github import AppCredential, AppPullRequests
from geecs_configurator.log import Applier, CommitLog, NoApplier, Row
from geecs_configurator.repository import ConfigsRepository, GitError, Identity
from geecs_configurator.save import CompareLinkOnly, PullRequests, Saved
from geecs_configurator.settings import Settings
from geecs_configurator.store import DocumentStore

logger = logging.getLogger(__name__)


class CommitPath:
    """The repository, its gate, the log, the share and the credential, for one service process."""

    def __init__(self, settings: Settings, *, pull_requests: PullRequests | None = None, http: httpx.Client | None = None, applier: Applier | None = None):  # fmt: skip
        if settings.state_dir is None:
            raise ValueError(
                "the commit path needs a state directory (Settings.commits)"
            )
        self.settings = settings
        self.credential = (
            AppCredential(
                settings.github_app_id,
                settings.github_installation_id,
                settings.github_app_key_file,
                http or httpx.Client(timeout=20),
            )  # fmt: skip
            if settings.github_app_key_file is not None
            else None
        )
        self.repo = ConfigsRepository(
            settings.state_dir, settings.origin, settings.branch,
            Identity(settings.committer_name, settings.committer_email),
            credential=self.credential.git_env if self.credential else None,
            accept=self.accept, on_advanced=self.advanced,
        )  # fmt: skip
        self.store = DocumentStore(self.repo.live)
        self.log = CommitLog(settings.state_dir / "log.jsonl")
        self.applier: Applier = applier or NoApplier()
        self.pull_requests: PullRequests = pull_requests or (
            AppPullRequests(self.credential, settings.github_repo, settings.branch)
            if self.credential
            else CompareLinkOnly(settings.github_repo, settings.branch)
        )
        #: Why the share's clone did not fast-forward last time (``None``: it did, or there is none).
        self.share_error: str | None = None
        #: Why main is not applied (``None``: the live checkout is main).
        self.refused: str | None = None

    # ------------------------------------------------------------ the gate

    def accept(self, old: str, new: str) -> list[str]:
        """The documents changed in ``old..new`` that do not validate at ``new`` (logged once per head)."""
        paths = self.repo.changed(old, new)
        if not paths:
            return []
        with self.repo.staging(new) as tree:
            found = validation.problems(tree, paths)
        if found:
            self.refused = f"main at {new[:10]} is not applied: " + "; ".join(found)
            if not self.log.has_commit(new, "refused"):
                author = self.repo.git("log", "-1", "--format=%an", new)
                self.log.add(Row(source="refused", who=author, kind=None, path="; ".join(p.split(":")[0] for p in found), commit=new, state="pending", detail="fetched, not applied: " + "; ".join(found) + ". The maintainer fixes main by pull request."))  # fmt: skip
        return found

    def advanced(self, old: str, new: str) -> None:
        """Log the foreign commits just applied, then fast-forward the share's clone."""
        self.refused = None
        known = validation.documents(self.store)
        for sha, email, author, paths in self.repo.commits(old, new):
            if email == self.settings.committer_email:
                continue  # the service's own: the save or the inbox logs it
            for path in paths:
                if path in known:
                    kind = known[path][0]
                    answer = self.applier.apply(kind, path)
                    self.log.add(Row(source="merged", who=author, kind=kind, path=path, commit=sha, state=answer.state, detail=answer.detail))  # fmt: skip
        self.fast_forward_share()

    def fast_forward_share(self) -> None:
        """The share's clone to the accepted main; a refusal is kept for the page, never raised."""
        if self.settings.share_clone is None:
            return
        try:
            self.repo.fast_forward_share(self.settings.share_clone)
            self.share_error = None
        except GitError as exc:
            self.share_error = " ".join(str(exc).split())[:240]
            logger.warning(
                "the share's clone was not fast-forwarded: %s", self.share_error
            )

    # ------------------------------------------------------------ the log

    def saved(self, kind: str, saved: Saved) -> None:
        """Log a save that committed or proposed."""
        if saved.outcome == "committed":
            answer = self.applier.apply(kind, saved.path)
            self.log.add(Row(source="saved", who=saved.author, kind=kind, path=saved.path, commit=saved.commit, state=answer.state, detail=answer.detail))  # fmt: skip
        elif saved.outcome == "proposed":
            self.log.add(Row(source="proposed", who=saved.author, kind=kind, path=saved.path, commit=saved.commit, link=saved.pull_request, state="pending", detail=f"the maintainer's review (branch {saved.branch})"))  # fmt: skip

    # ------------------------------------------------------------ the timer

    def reconcile(self) -> list[inbox.Published]:
        """Publish what waits in the inbox, logging each publish (no git when it is empty)."""
        experiments = self.store.experiments()
        if not inbox.pending(self.repo, experiments):
            return []
        author = Identity(self.settings.inbox_author, self.settings.committer_email)
        results = inbox.reconcile(self.repo, experiments, author)
        for r in results:
            if r.outcome == "committed":
                answer = self.applier.apply(r.kind, r.path)
                self.log.add(Row(source="published", who=self.settings.inbox_author, kind=r.kind, path=r.path, commit=r.commit, state=answer.state, detail=answer.detail))  # fmt: skip
        return results

    def fetch(self) -> None:
        """Fetch and apply what main gained (the gate decides); a failure waits for the next tick."""
        try:
            self.repo.sync()
        except GitError as exc:
            logger.warning("the timer fetch did not apply main: %s", exc)

    def run(self, stop: threading.Event) -> None:
        """The timer: the inbox every ``reconcile_seconds``, a fetch every ``fetch_seconds``, until ``stop``."""
        every, fetch_every = (
            self.settings.reconcile_seconds,
            self.settings.fetch_seconds,
        )
        tick = min(t for t in (every, fetch_every) if t > 0)
        last_fetch = time.monotonic()
        while not stop.wait(tick):
            try:
                if every > 0:
                    self.reconcile()
                if fetch_every > 0 and time.monotonic() - last_fetch >= fetch_every:
                    last_fetch = time.monotonic()
                    self.fetch()
            except (
                Exception
            ):  # the timer outlives any one failure; the next tick retries
                logger.exception("the configurator's timer")
