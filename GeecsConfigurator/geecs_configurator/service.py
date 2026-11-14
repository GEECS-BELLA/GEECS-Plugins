"""The commit path assembled: the repository with its gate and hooks, the credential, the log, the share.

:class:`CommitPath` is what the app holds when the commit path is on. It
wires the pieces the other modules keep apart:

- **what this box owns** (:class:`ownership.Ownership`, from settings): one
  configs repo serves many facilities, each with its own box. Commits are
  made at owned paths only; the gate and the log look at owned paths only;
- the **gate** (``accept``): before the live checkout moves past a commit
  it did not make, the **owned** documents that changed must validate
  (:func:`validation.problems`, the configs repo's CI check too); another
  facility's documents fast-forward unjudged (its box and the configs
  repo's CI judge them), so its invalid commit never stops this box. A refused
  commit is logged once (``refused``, ``pending``: the maintainer fixes main
  by pull request) and never applied; every fast-forward stops there until
  main is fixed;
- the **advance hook**: each foreign commit applied (a merged pull request,
  a maintainer's push) is logged per **owned** document (another facility's
  are not this box's record, and never reach the apply seam), its state the apply seam's
  answer, and the share's clone is fast-forwarded from the service's
  repository;
- the **timer**: the inbox every ``reconcile_seconds``, a fetch every
  ``fetch_seconds``, the queued applies every ``apply_seconds`` on a
  thread of their own (:meth:`CommitPath.settle_queued`) (how a merge on GitHub takes effect with nobody in the
  editor);
- the **credential**: the GitHub App's installation token for git and its
  pull requests when the App is configured, else git's own (a deploy key)
  and :class:`save.CompareLinkOnly`.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import httpx

from geecs_configurator import inbox, validation
from geecs_configurator.github import AppCredential, AppPullRequests
from geecs_configurator.log import Answer, Applier, CommitLog, NoApplier, Row
from geecs_configurator.ownership import Ownership
from geecs_configurator.repository import ConfigsRepository, GitError, Identity
from geecs_configurator.save import CompareLinkOnly, PullRequests, Saved
from geecs_configurator.settings import Settings
from geecs_configurator.store import DocumentStore

logger = logging.getLogger(__name__)
#: How a row whose commit was not yet live says so (re-answered when it is).
NOT_LIVE = "on main but not in the live checkout"
#: Seconds GitHub has to mint a token: under a read's fetch bound, so an unreachable API never outwaits it.
MINT_TIMEOUT_S = 8.0


class CommitPath:
    """The repository, its gate, the log, the share and the credential, for one service process."""

    def __init__(self, settings: Settings, *, pull_requests: PullRequests | None = None, http: httpx.Client | None = None, applier: Applier | None = None):  # fmt: skip
        if settings.state_dir is None:
            raise ValueError(
                "the commit path needs a state directory (Settings.commits)"
            )
        if settings.ownership.empty:
            raise ValueError("the commit path needs what this box owns (Settings.ownership): an empty set is never everything")  # fmt: skip
        self.settings = settings
        self.ownership: Ownership = settings.ownership
        self.credential = (
            AppCredential(
                settings.github_app_id,
                settings.github_installation_id,
                settings.github_app_key_file,
                http or httpx.Client(timeout=MINT_TIMEOUT_S),
            )  # fmt: skip
            if settings.github_app_key_file is not None
            else None
        )
        self.repo = ConfigsRepository(
            settings.state_dir, settings.origin, settings.branch,
            Identity(settings.committer_name, settings.committer_email),
            credential=self.credential.git_env if self.credential else None,
            accept=self.accept, on_advanced=self.advanced,
            writable=self.ownership.owns,  # by name; the commit itself refuses a path through a link
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
        #: Present while the live checkout's first commit was never checked (see :meth:`start`).
        self.unverified_marker = settings.state_dir / "UNVERIFIED"
        self._share_thread: threading.Thread | None = None
        self._settle_thread: threading.Thread | None = None
        #: Set while a settle pass holds the repository (a save is then answered "busy" at once, never kept waiting).
        self.settling = threading.Event()

    @property
    def unverified(self) -> bool:
        """Whether the live checkout's baseline was never checked (until the maintainer says it was)."""
        return self.unverified_marker.exists()

    # ------------------------------------------------------------ the start

    def start(self) -> None:
        """Set the repository up, then bring it through the gate to main (the service's start).

        A new live checkout starts at the share's commit when main contains
        it (accepted before, by hand), so main's newer changes pass the gate
        like any other. With no such share it starts at main's tip
        **unverified**: the gate only ever checks what changed, so nothing
        will check that baseline. The state is a marker file in the state
        directory, said on the page and in ``/healthz``, and the share is not
        moved while it is there; later advances never clear it. The
        maintainer clears it (deletes the file) once main has been looked at.
        """
        fresh = not (self.repo.live / ".git").exists()
        self.repo.ensure(start_at=self._share_start)
        if fresh and not self._started_at_share:
            self.unverified_marker.write_text(f"{self.repo.live_head()}\nthe live checkout was first made at main's tip with no share to start from; nothing checked it\n")  # fmt: skip
        self.fetch()
        if not self.unverified:
            self.fast_forward_share()
        elif self.settings.share_clone is not None:
            self.share_error = self._share_held()

    def _share_held(self) -> str:
        """Why the share is not moved while main is unverified (and whether it is ahead of what is served)."""
        try:
            head = self.repo.git(
                "rev-parse", "HEAD", cwd=Path(self.settings.share_clone or ".")
            )
            if not self.repo.is_ancestor(head, self.repo.live_head()):
                return f"the share's clone is at {head[:10]}, ahead of the accepted main or apart from it; not moved"
        except GitError as exc:
            return " ".join(str(exc).split())[:240]
        return "held: main is unverified, so the share is not moved until a change passes the gate"

    _started_at_share = False

    def _share_start(self) -> str | None:
        share = self.settings.share_clone
        if share is None:
            return None
        try:
            head = self.repo.git("rev-parse", "HEAD", cwd=Path(share))
            if self.repo.is_ancestor(head, self.repo.remote_head()):
                self._started_at_share = True
                return head
        except GitError:
            pass
        return None

    def main_state(self) -> str:
        """``applied``, or why not: a refused commit (cleared once main is the live commit again), or unverified."""
        try:
            current = self.repo.remote_head() == self.repo.live_head()
        except GitError:
            current = False
        if self.refused and not current:
            return self.refused
        if self.unverified:
            return f"unverified: the live checkout was first made at main's tip with no share to start from, and the gate checks only changes; once main has been looked at, the maintainer deletes {self.unverified_marker} (the share is held until then)"  # fmt: skip
        return "applied"

    # ------------------------------------------------------------ the gate

    def accept(self, old: str, new: str) -> list[str]:
        """The owned documents changed in ``old..new`` that do not validate at ``new`` (logged once per head).

        Only owned paths are judged: a commit touching other facilities'
        paths alone is accepted unread. An owned document read through a
        symbolic link (a file, a folder, the experiment folder itself) is
        judged when what the link reaches changed (:meth:`_owned_changes`).
        """
        changed = self.repo.changed(old, new)
        if not changed:
            return []
        with self.repo.staging(new) as tree:
            paths = self._owned_changes(tree, new, changed)
            found = validation.problems(tree, paths) if paths else []
        if found:
            self.refused = f"main at {new[:10]} is not applied: " + "; ".join(found)
            bad = [p.split(": ")[0] for p in found]
            if not self.log.has_commit(new, "refused"):
                # who made each failing path what it is: the newest commit in the range that touched it
                authors = sorted({self.repo.git("log", "-1", "--format=%an", f"{old}..{new}", "--", p) for p in bad})  # fmt: skip
                self.log.add(Row(source="refused", who=", ".join(a for a in authors if a) or "?", kind=None, path="; ".join(bad), commit=new, state="pending", detail="fetched, not applied: " + "; ".join(found) + ". The maintainer fixes main by pull request."))  # fmt: skip
        return found

    def _owned_changes(self, tree: Path, rev: str, changed: list[str]) -> list[str]:
        """What ``changed`` changed of this box's: the owned paths, and every owned document that reaches a changed file through a link.

        A link may be a file, a folder below an owned folder, or the owned
        folder itself (``experiments/<owned> -> <another>``): the document
        is read through it, so a change to its target is a change to it.
        ``tree`` is a checkout at ``rev``. The gate judges these, and the log
        and the apply seam take them up: the two never disagree.
        """
        paths = self.ownership.owned(changed)
        prefixes = set(self.ownership.prefixes())
        links = [
            path for meta, _, path in (line.partition("\t") for line in self.repo.git("ls-tree", "-r", "-z", rev).split("\0"))
            if meta.startswith("120000") and (self.ownership.owns(path) or path in prefixes or any(p.startswith(path + "/") for p in prefixes))
        ]  # fmt: skip
        if not links:
            return paths
        moved, base = set(changed), tree.resolve()
        relinked = [
            link for link in links if link in moved
        ]  # a link made or retargeted: everything below it changed
        for path in validation.documents(DocumentStore(tree)):
            if path in paths or not self.ownership.owns(path):
                continue
            target = (tree / path).resolve()
            if (target.is_relative_to(base) and target.relative_to(base).as_posix() in moved) or any(path.startswith(link + "/") for link in relinked):  # fmt: skip
                paths.append(path)
        return paths

    def advanced(self, old: str, new: str) -> None:
        """Log the foreign commits just applied, then fast-forward the share's clone (off the repository's lock)."""
        self.refused = None
        known = validation.documents(self.store)
        self._settle_pending(new)
        for sha, email, author, paths in self.repo.commits(old, new):
            if email == self.settings.committer_email:
                continue  # the service's own: the save or the inbox logs it
            # another facility's paths: not this box's row, never applied here (the live checkout is at ``new``)
            for path in self._owned_changes(self.repo.live, new, paths):
                if path in known:
                    kind = known[path][0]
                    answer = self.applier.apply(kind, path)
                    self.log.add(Row(source="merged", who=author, kind=kind, path=path, commit=sha, **answer.row_fields()))  # fmt: skip
        if self.settings.share_clone is not None and self.unverified:
            self.share_error = self._share_held()
        elif self.settings.share_clone is not None:
            self._share_thread = threading.Thread(target=self.fast_forward_share, name="configurator-share", daemon=True)  # fmt: skip
            self._share_thread.start()

    def _settle_pending(self, new: str) -> None:
        """Re-answer the service's own rows left ``pending`` because their commit was not live yet, now that it is."""
        for row in self.log.rows(limit=10_000):
            if row.source in ("saved", "published") and row.state == "pending" and self.ownership.owns(row.path) and row.detail.startswith(NOT_LIVE) and row.commit and self.repo.is_ancestor(row.commit, new):  # fmt: skip
                answer = self.applier.apply(row.kind, row.path)
                self.log.add(row.model_copy(update=answer.row_fields()))  # fmt: skip

    def wait_share(self, timeout: float = 30.0) -> None:
        """Wait for the share's last fast-forward (tests, and the start)."""
        if self._share_thread is not None:
            self._share_thread.join(timeout)

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
            answer = self._answer(kind, saved.path, saved.commit)
            self.log.add(Row(source="saved", who=saved.author, kind=kind, path=saved.path, commit=saved.commit, **answer.row_fields()))  # fmt: skip
        elif saved.outcome == "proposed":
            self.log.add(Row(source="proposed", who=saved.author, kind=kind, path=saved.path, commit=saved.commit, link=saved.pull_request, state="pending", detail=f"the maintainer's review (branch {saved.branch})"))  # fmt: skip

    def _answer(self, kind: str | None, path: str, commit: str | None) -> Answer:
        """The seam's answer for a commit the live checkout holds; ``pending`` (and no apply) for one it does not."""
        try:
            live = commit is not None and self.repo.is_ancestor(
                commit, self.repo.live_head()
            )
        except GitError:
            live = False
        if not live:
            return Answer(state="pending", detail=f"{NOT_LIVE}: {self.main_state()}")
        return self.applier.apply(kind, path)

    # ------------------------------------------------------------ the timer

    def reconcile(self) -> list[inbox.Published]:
        """Publish what waits in the inbox, logging each publish (no git when it is empty)."""
        experiments = self.inbox_experiments()
        if not inbox.pending(self.repo, experiments):
            return []
        author = Identity(self.settings.inbox_author, self.settings.committer_email)
        results = inbox.reconcile(self.repo, experiments, author)
        for r in results:
            if r.outcome == "committed":
                answer = self._answer(r.kind, r.path, r.commit)
                self.log.add(Row(source="published", who=self.settings.inbox_author, kind=r.kind, path=r.path, commit=r.commit, **answer.row_fields()))  # fmt: skip
        return results

    def inbox_experiments(self) -> list[str]:
        """The experiments whose worker writes this box publishes: the owned ones the repo has a folder for."""
        there = set(self.store.experiments())
        return [e for e in self.ownership.experiments if e in there]

    def settle_queued(self) -> int:
        """Do the queued applies (one pass, the applier's :meth:`~log.Applier.settle`); how many rows were tried.

        Only rows already queued when the pass begins: their commits are in
        the live checkout, so the reopen or onboard that follows reads them.
        The repository's lock is held through the pass, so no fast-forward
        rewrites a file while the worker or the archiver reads it (a save or
        a publish waits for it; a read serves the last accepted main after
        its bounded wait). A row whose answer changed gets a newer line.
        """
        # owned rows only: the log holds no other (a row left from before the box owned less is never applied here)
        rows = [r for r in self.log.rows(limit=10_000) if r.state == "queued" and self.ownership.owns(r.path)]  # fmt: skip
        if not rows:
            return 0
        self.settling.set()
        try:
            with self.repo.lock:
                answers = self.applier.settle(rows)
        finally:
            self.settling.clear()
        for row, answer in zip(rows, answers):
            if (answer.state, answer.detail) != (row.state, row.detail):
                self.log.add(row.model_copy(update=answer.row_fields()))  # fmt: skip
        return len(rows)

    def _settle_in_background(self) -> None:
        """Start a settle pass on its own thread unless one is running (a reopen takes a minute)."""
        if self._settle_thread is not None and self._settle_thread.is_alive():
            return

        def run() -> None:
            try:
                self.settle_queued()
            except (
                Exception
            ):  # the thread outlives any one failure; the next pass retries
                logger.exception("the configurator's apply pass")

        self._settle_thread = threading.Thread(target=run, name="configurator-apply", daemon=True)  # fmt: skip
        self._settle_thread.start()

    def fetch(self) -> None:
        """Fetch and apply what main gained (the gate decides); a failure waits for the next tick."""
        try:
            self.repo.sync()
        except GitError as exc:
            logger.warning("the timer fetch did not apply main: %s", exc)

    def run(self, stop: threading.Event) -> None:
        """The timer: the inbox every ``reconcile_seconds``, a fetch every ``fetch_seconds``, until ``stop``."""
        every, fetch_every, apply_every = (
            self.settings.reconcile_seconds,
            self.settings.fetch_seconds,
            self.settings.apply_seconds,
        )
        tick = min(t for t in (every, fetch_every, apply_every) if t > 0)
        last_fetch = last_apply = time.monotonic()
        while not stop.wait(tick):
            try:
                if every > 0:
                    self.reconcile()
                if fetch_every > 0 and time.monotonic() - last_fetch >= fetch_every:
                    last_fetch = time.monotonic()
                    self.fetch()
                if apply_every > 0 and time.monotonic() - last_apply >= apply_every:
                    last_apply = time.monotonic()
                    self._settle_in_background()
            except (
                Exception
            ):  # the timer outlives any one failure; the next tick retries
                logger.exception("the configurator's timer")
