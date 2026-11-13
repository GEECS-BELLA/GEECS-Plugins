"""The service's git: a bare repository, the live checkout, a staging worktree per edit, one lock.

Layout under the state directory::

    repo.git/   the bare repository; ``origin`` is the configs repo, fetched
                into ``refs/remotes/origin/*``
    live/       the live checkout: a detached worktree at an accepted commit of
                main, the configs root every Linux service reads
    staging/    one throwaway worktree per edit, removed when the edit ends
    inbox/      the worker's write root (shot offsets), see ``inbox.py``

The bare repository's own ``refs/heads/<branch>`` is the **accepted** main:
the commit the live checkout is at. The share's clone fast-forwards from it
(:meth:`ConfigsRepository.fast_forward_share`), never from ``origin``.

Rules (each pinned by ``tests/test_repository.py``):

- **The live checkout holds accepted commits of main only.** Nothing writes
  a file there, and it is never on a branch: it only ever moves by
  fast-forward to ``origin/<branch>``, and refuses to move when it is dirty
  or main was rewritten (it stays where it was, and says why).
- **Every edit is staged in its own worktree** at the base the edit was
  made against, committed with the editor as the git author and the
  service as the committer, and pushed. On a rejected push the service
  fetches: if main did not move the push failed for another reason (no
  retry); if it moved and the edit's file did not change, the edit is
  made again on the new main (bounded tries; the file is rewritten, never
  merged); if the file changed, :class:`BaseMoved` (the save's 409). The
  staging worktree is removed on every path, and nothing reaches the live
  checkout unless the push landed.
- **One lock** (per process: the service runs one uvicorn worker) around
  every git operation, so a fetch, a fast-forward and a commit never
  interleave.
- **A push that reported failure but landed is a success**: after a failed
  push the fetched main is checked for the commit before anything else.
- **The gate.** Every fast-forward asks ``accept(old, new)`` first (the
  app's: the documents changed between the two validate); a refusal leaves
  the live checkout where it was (:class:`Unaccepted`), whoever asked: a
  timer, a read, a save. A commit is fetched, never applied, until main is
  fixed. ``on_advanced(old, new)`` runs after every fast-forward (the log
  and the share's fast-forward).
- **Network commands carry the credential in the environment**
  (``GIT_CONFIG_*``), never on a command line another user could read.

Readers of the live checkout (the other Linux services, which take no lock)
should treat a missing file as transient: a fast-forward rewrites a changed
file by unlink and create, never by rename. A service killed mid-merge can
leave the live checkout dirty; it is then never moved (saves answer 502)
until someone restores it by hand: ``git -C <state>/live reset --hard`` and
``git -C <state>/live clean -fd``, which drop only what nothing should have
written there. A stale ``index.lock`` from such a kill is removed at start,
when no git of this process can be running.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

logger = logging.getLogger(__name__)

#: Seconds before a git call is given up (a hung network push must not hold the lock forever).
GIT_TIMEOUT_S = 120.0
#: How many times an edit is made again on a main that moved under it.
PUSH_TRIES = 3


class GitError(RuntimeError):
    """A git command failed (its arguments and stderr in the message)."""


class PushFailed(GitError):
    """The push did not land; nothing reached the live checkout."""


class BaseMoved(GitError):
    """Main changed the edited file after the edit's base: the editor re-applies their edit."""


class Unaccepted(GitError):
    """Main holds a commit the gate refuses: fetched, never applied."""

    def __init__(self, target: str, problems: list[str]):
        super().__init__(
            f"main at {target[:10]} is not applied: " + "; ".join(problems)
        )
        self.target, self.problems = target, problems


#: The gate: the problems with moving from one commit to another (none: accepted).
Accept = Callable[[str, str], list[str]]
#: Called after the live checkout moved from one commit to another.
Advanced = Callable[[str, str], None]
#: The environment that authenticates a network git command (empty: git's own credential).
Credential = Callable[[], Mapping[str, str]]


@dataclass(frozen=True)
class Identity:
    """A git name and e-mail."""

    name: str
    email: str

    def __str__(self) -> str:
        """The ident: ``Name <email>``."""
        return f"{self.name} <{self.email}>"


#: What follows the operator's name in a git author.
VIA = " via configurator"


def author_name(operator: str) -> str:
    """The git author for an operator: ``"Name via configurator"``; refuses a blank name.

    Characters git refuses in an ident (``<``, ``>``, newlines) are dropped.
    """
    name = " ".join(operator.replace("<", "").replace(">", "").split())
    if not name:
        raise ValueError("the operator's name is required")
    return f"{name}{VIA}"


class ConfigsRepository:
    """The bare repository, its live checkout and its staging worktrees under one state directory.

    Parameters
    ----------
    state_dir : Path
        The service's own directory (its systemd ``StateDirectory``).
    origin : str
        The configs repository's git URL (a site value).
    branch : str
        The branch the live checkout follows.
    committer : Identity
        The service, as the committer of every commit it makes.
    credential, accept, on_advanced : callable, optional
        The network credential, the gate and the advance hook (see the module docstring).
    """

    def __init__(self, state_dir: Path, origin: str, branch: str, committer: Identity, *, credential: Credential | None = None, accept: Accept | None = None, on_advanced: Advanced | None = None):  # fmt: skip
        self.state_dir = Path(state_dir)
        self.origin, self.branch, self.committer = origin, branch, committer
        self.bare = self.state_dir / "repo.git"
        self.live = self.state_dir / "live"
        self.staging_dir = self.state_dir / "staging"
        self.inbox = self.state_dir / "inbox"
        self.lock = threading.RLock()
        self.share_lock = threading.Lock()
        #: ``time.monotonic()`` of the last fetch that answered (0: none yet).
        self.fetched_at = 0.0
        self.credential, self.accept, self.on_advanced = credential, accept, on_advanced

    # ----------------------------------------------------------------- git

    def git(self, *args: str, cwd: Path | None = None, env: Mapping[str, str] | None = None, timeout: float = GIT_TIMEOUT_S) -> str:  # fmt: skip
        """Run one git command (never prompting) and return its stdout; :class:`GitError` on failure."""
        full = {**os.environ, "GIT_TERMINAL_PROMPT": "0", **(env or {})}
        try:
            done = subprocess.run(
                ["git", *args], cwd=cwd or self.bare, env=full, capture_output=True,
                text=True, timeout=timeout,
            )  # fmt: skip
        except subprocess.TimeoutExpired as exc:
            raise GitError(f"git {args[0]}: timed out after {timeout:.0f} s") from exc
        if done.returncode:
            raise GitError(f"git {' '.join(args[:2])}: {done.stderr.strip() or done.stdout.strip()}")  # fmt: skip
        return done.stdout.strip()

    def _remote_ref(self) -> str:
        return f"refs/remotes/origin/{self.branch}"

    def _network(self) -> Mapping[str, str]:
        """The credential's environment for a command that talks to origin; its failure is a :class:`GitError`."""
        if not self.credential:
            return {}
        try:
            return self.credential()
        except (
            RuntimeError,
            OSError,
            ValueError,
        ) as exc:  # GitHub down, the key unreadable or malformed
            raise GitError(f"no credential for origin: {exc}") from exc

    # --------------------------------------------------------------- setup

    def ensure(self, start_at: Callable[[], str | None] | None = None) -> None:
        """Create what is missing (bare repository, live checkout, folders) and clear stale staging worktrees.

        A new live checkout starts at ``start_at()`` (a commit already
        accepted elsewhere, e.g. the share's) when it names one main contains,
        else at main's tip unverified; the caller's next advance passes the
        gate from there.
        """
        with self.lock:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            if not (self.bare / "HEAD").is_file():
                self.bare.mkdir(exist_ok=True)
                self.git("init", "--bare", "--quiet")
                self.git("remote", "add", "origin", self.origin)
            elif self.git("remote", "get-url", "origin") != self.origin:
                self.git("remote", "set-url", "origin", self.origin)
            self.git("config", "remote.origin.fetch", "+refs/heads/*:refs/remotes/origin/*")  # fmt: skip
            try:
                self.fetch()
            except GitError:
                if not (self.live / ".git").exists():
                    raise  # nothing to serve without a first fetch
                logger.warning("origin did not answer at start; serving the last accepted main")  # fmt: skip
            if not (self.live / ".git").exists():
                start = (start_at() if start_at else None) or self._remote_ref()
                self.git("worktree", "add", "--detach", "--quiet", str(self.live), start)  # fmt: skip
            self.git("update-ref", f"refs/heads/{self.branch}", self.live_head())
            stale_lock = self.bare / "worktrees" / "live" / "index.lock"
            if stale_lock.exists():  # a kill mid-merge; no git of this process runs yet
                logger.warning("removing a stale index.lock in the live checkout")
                stale_lock.unlink()
            self.staging_dir.mkdir(exist_ok=True)
            for stale in self.staging_dir.iterdir():
                self._remove_worktree(stale)
            self.git("worktree", "prune")
            self.inbox.mkdir(exist_ok=True)

    # ------------------------------------------------------------ the live

    def fetch(self, timeout: float = GIT_TIMEOUT_S) -> None:
        """Fetch origin (every branch, pruned), giving up after ``timeout`` seconds."""
        with self.lock:
            self.git("fetch", "--quiet", "--prune", "origin", timeout=timeout, env=self._network())  # fmt: skip
            self.fetched_at = time.monotonic()

    def remote_head(self) -> str:
        """The commit ``origin/<branch>`` names after the last fetch or push."""
        return self.git("rev-parse", "--verify", self._remote_ref())

    def live_head(self) -> str:
        """The commit the live checkout is at."""
        return self.git("rev-parse", "HEAD", cwd=self.live)

    def is_ancestor(self, older: str, newer: str) -> bool:
        """Whether ``older`` is ``newer`` or behind it."""
        try:
            self.git("merge-base", "--is-ancestor", older, newer)
        except GitError:
            return False
        return True

    def advance(self) -> str:
        """Fast-forward the live checkout to ``origin/<branch>``; :class:`GitError` (and no move) when it cannot.

        It cannot when the checkout is dirty (something wrote into it), main
        was rewritten (the target is not ahead of it), or the gate refuses
        (:class:`Unaccepted`).
        """
        with self.lock:
            target, here = self.remote_head(), self.live_head()
            if target == here:
                return here
            if self.git(
                "status", "--porcelain", "--untracked-files=all", cwd=self.live
            ):
                raise GitError("the live checkout has local changes; it is never written, so it is not moved")  # fmt: skip
            if not self.is_ancestor(here, target):
                raise GitError(f"main at {target[:10]} does not contain the live checkout's {here[:10]} (rewritten?); not moved")  # fmt: skip
            problems = self.accept(here, target) if self.accept else []
            if problems:
                raise Unaccepted(target, problems)
            self.git("merge", "--ff-only", "--quiet", target, cwd=self.live)
            self.git("update-ref", f"refs/heads/{self.branch}", target)
            if self.on_advanced:
                try:
                    self.on_advanced(here, target)
                except Exception:  # the hook's failure is its own (the log, the share), never the move's
                    logger.exception("after the live checkout moved to %s", target[:10])
            return target

    def commits(self, old: str, new: str) -> list[tuple[str, str, str, list[str]]]:
        """The commits main gained in ``old..new`` (first parent: a merged pull request is one), oldest first.

        Each is (sha, committer e-mail, author, changed paths).
        """
        out = []
        for line in self.git("log", "--first-parent", "--reverse", "--format=%H%x1f%ce%x1f%an", f"{old}..{new}").splitlines():  # fmt: skip
            sha, email, author = line.split("\x1f")
            out.append((sha, email, author, self.changed(f"{sha}^", sha) if self.git("rev-list", "--parents", "-n1", sha).count(" ") else []))  # fmt: skip
        return out

    def changed(self, old: str, new: str) -> list[str]:
        """The paths added, copied, modified or renamed between two commits (not deleted)."""
        return self.git(
            "diff", "--name-only", "--diff-filter=ACMR", old, new
        ).splitlines()

    def fast_forward_share(self, share: Path, timeout: float = GIT_TIMEOUT_S) -> str:
        """Fast-forward the share's clone to the accepted main, from this bare repository (no credential).

        The share's clone must be on ``<branch>`` with no edit in the way, and
        not ahead of the accepted main: otherwise it stays where it was
        (:class:`GitError`). It only reads the bare repository, so it holds
        its own lock, never the repository's: a slow share delays no save.
        """
        with self.share_lock:
            try:
                on = self.git("symbolic-ref", "--short", "-q", "HEAD", cwd=share)
            except GitError:
                on = ""
            if on != self.branch:
                raise GitError(f"the share's clone is on {on or 'no branch (a detached HEAD)'}, not {self.branch!r}; not moved")  # fmt: skip
            self.git("fetch", "--quiet", str(self.bare.resolve()), f"refs/heads/{self.branch}", cwd=share, timeout=timeout)  # fmt: skip
            self.git("merge", "--ff-only", "--quiet", "FETCH_HEAD", cwd=share, timeout=timeout)  # fmt: skip
            head, accepted = self.git("rev-parse", "HEAD", cwd=share), self.git("rev-parse", "FETCH_HEAD", cwd=share)  # fmt: skip
            if head != accepted:
                raise GitError(f"the share's clone is at {head[:10]}, ahead of the accepted main {accepted[:10]}; not moved")  # fmt: skip
            return head

    def sync(self, timeout: float = GIT_TIMEOUT_S) -> str:
        """Fetch and fast-forward the live checkout; the commit it is at."""
        with self.lock:
            self.fetch(timeout)
            return self.advance()

    # ------------------------------------------------------------ staging

    def _remove_worktree(self, path: Path) -> None:
        try:
            self.git("worktree", "remove", "--force", str(path))
        except GitError:
            shutil.rmtree(path, ignore_errors=True)
            self.git("worktree", "prune")

    @contextmanager
    def staging(self, base: str) -> Iterator[Path]:
        """A throwaway detached worktree at ``base``, removed however the block ends."""
        path = self.staging_dir / uuid.uuid4().hex[:12]
        self.git("worktree", "add", "--detach", "--quiet", str(path), base)
        try:
            yield path
        finally:
            self._remove_worktree(path)

    def _commit(self, tree: Path, rel: str, data: bytes, author: Identity, message: str) -> str | None:  # fmt: skip
        """Write ``data`` at ``rel`` in a staging worktree and commit it; ``None`` when it changes nothing."""
        target = tree / PurePosixPath(rel)
        if not target.resolve().is_relative_to(tree.resolve()):
            raise GitError(f"{rel!r} is outside the repository")
        # inside a staging worktree of the configs repo: never a scan folder
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.git("add", "--", rel, cwd=tree)
        try:
            self.git("diff", "--cached", "--quiet", cwd=tree)
            return None
        except GitError:
            pass
        env = {
            "GIT_AUTHOR_NAME": author.name, "GIT_AUTHOR_EMAIL": author.email,
            "GIT_COMMITTER_NAME": self.committer.name, "GIT_COMMITTER_EMAIL": self.committer.email,
        }  # fmt: skip
        self.git("-c", "commit.gpgsign=false", "commit", "--quiet", "--no-verify", "-m", message, cwd=tree, env=env)  # fmt: skip
        return self.git("rev-parse", "HEAD", cwd=tree)

    def _push(self, tree: Path, ref: str) -> None:
        """Push the staging worktree's HEAD to ``ref`` on origin (the seam tests break)."""
        self.git("push", "--quiet", "--porcelain", "origin", f"HEAD:{ref}", cwd=tree, env=self._network())  # fmt: skip

    def file_changed(self, rel: str, old: str, new: str) -> bool:
        """Whether ``rel`` differs between two commits."""
        try:
            self.git("diff", "--quiet", old, new, "--", rel)
        except GitError:
            return True
        return False

    def commit_file(
        self,
        rel: str,
        data: bytes,
        *,
        author: Identity,
        message: str,
        base: str,
        on_moved: Callable[[str, str], None] | None = None,
    ) -> str | None:
        """Commit one file's new bytes on ``base`` and push it to main; the commit, or ``None`` for no change.

        ``on_moved(old, new)`` is called when the push races another commit
        that changed this file: the save passes one that raises
        :class:`BaseMoved`; the inbox passes none, so the measurement wins.
        On success the live checkout is fast-forwarded (a failure there is
        logged, and the next sync catches up). :class:`PushFailed` when the
        push did not land; the staging worktree is gone either way.
        """
        with self.lock:
            for _ in range(PUSH_TRIES):
                with self.staging(base) as tree:
                    sha = self._commit(tree, rel, data, author, message)
                    if sha is None:  # main already holds these bytes
                        self._catch_up()
                        return None
                    try:
                        self._push(tree, f"refs/heads/{self.branch}")
                    except GitError as exc:
                        error = exc
                    else:
                        error = None
                if error is None:
                    break
                moved = self._after_rejected(sha, base, rel, error, on_moved)
                if moved is None:  # it landed after all
                    break
                base = moved
            else:
                raise PushFailed(f"main kept moving: {PUSH_TRIES} tries")
            self._catch_up()
            return sha

    def _after_rejected(self, sha: str, base: str, rel: str, error: GitError, on_moved: Callable[[str, str], None] | None) -> str | None:  # fmt: skip
        """After a push reported failure: ``None`` if it landed after all, else the new base to try again on.

        :class:`PushFailed` when main did not move (a real failure) or was
        rewritten; ``on_moved`` decides when main changed this file.
        """
        try:
            self.fetch()
            moved = self.remote_head()
        except GitError as exc:
            raise PushFailed(f"the push was not confirmed and origin did not answer, so whether it landed shows at the next fetch: {exc}") from exc  # fmt: skip
        if self.is_ancestor(sha, moved):
            logger.warning("the push of %s reported %s but landed", sha[:10], error)
            return None
        if moved == base or not self.is_ancestor(base, moved):
            raise PushFailed(str(error)) from error
        problems = self.accept(base, moved) if self.accept else []
        if problems:  # never build on what is not applied
            raise Unaccepted(moved, problems)
        if self.file_changed(rel, base, moved):
            if on_moved is not None:
                self._catch_up()
                on_moved(base, moved)
            logger.warning("%s changed on main while it was being pushed; the new bytes win", rel)  # fmt: skip
        return moved

    def _catch_up(self) -> None:
        try:
            self.sync()
        except GitError as exc:
            logger.warning("the live checkout did not catch up: %s", exc)

    def push_branch(self, rel: str, data: bytes, *, author: Identity, message: str, base: str, branch: str) -> str | None:  # fmt: skip
        """Commit on ``base`` and push to a new branch on origin; the live checkout is not touched."""
        with self.lock, self.staging(base) as tree:
            sha = self._commit(tree, rel, data, author, message)
            if sha is None:
                return None
            try:
                self._push(tree, f"refs/heads/{branch}")
            except GitError as exc:
                raise PushFailed(str(exc)) from exc
            return sha
