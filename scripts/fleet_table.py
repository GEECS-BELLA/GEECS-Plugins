#!/usr/bin/env python3
"""Render fleet_status.sh records as one box table (the dashboard view).

Reads tab-separated ``key=value`` records on stdin — one line per probe
result, each carrying a ``role=`` — merges them per service role, and
prints a wrapped table plus an attention list. Stdlib only, so it runs
under the system python3 from any terminal pane; ``scripts/fleet_status.sh
--summary`` is the only caller.
"""

from __future__ import annotations

import os
import shutil
import sys
import textwrap

# Display order for the roles fleet_status.sh emits; unknown roles append.
ROLE_ORDER = [
    "CA gateway",
    "Tiled",
    "Data Portal",
    "Queueserver RE Manager",
    "Bluesky doc proxy",
    "Capture daemon",
    "GEECS-MCP",
    "PVA image gateways",
]
HEADERS = ["", "Service", "Runs as", "Checkout", "Version", "Notes"]


def _add_note(rec: dict[str, str], text: str, key: str = "notes") -> None:
    rec[key] = (rec.get(key, "") + "|" + text).strip("|")


def parse(lines: list[str]) -> dict[str, dict[str, str]]:
    """Merge records by role.

    Stage-1/3 records (no ``svc=``) describe the role and merge first-wins.
    ``note=`` values are findings (they mark the row ``!``); ``info=`` values
    are facts worth showing that need no attention (device counts, a baked
    venv, a distance behind master that is only unmerged docs).
    Stage-2 records (``svc=``) each describe ONE process; a role can have
    several (a loaded-but-dead unit plus the hand-started process that
    really owns the port). The row is the live one; the others are named
    in a note, and no field is ever combined across two processes.
    """
    base: dict[str, dict[str, str]] = {}
    procs: dict[str, list[dict[str, str]]] = {}
    by_sha: dict[str, list[dict[str, str]]] = {}  # stage-3 distances, keyed by sha
    for line in lines:
        line = line.rstrip("\n")
        if not line:
            continue
        kv: dict[str, str] = {}
        for part in line.split("\t"):
            if "=" in part:
                k, v = part.split("=", 1)
                if k in ("note", "info") and kv.get(k):
                    kv[k] = kv[k] + "|" + v  # a record may carry several
                else:
                    kv[k] = v
        role = kv.get("role")
        if not role:
            continue
        if kv.get("svc"):
            procs.setdefault(role, []).append(kv)
            continue
        if kv.get("for_sha"):
            by_sha.setdefault(role, []).append(kv)
            continue
        rec = base.setdefault(role, {})
        for k, v in kv.items():
            if k == "note":
                _add_note(rec, v)
            elif k == "info":
                _add_note(rec, v, "infos")
            elif v and not rec.get(k):
                rec[k] = v
    merged: dict[str, dict[str, str]] = {}
    for role in list(base) + [r for r in procs if r not in base]:
        rec = dict(base.get(role, {}))
        plist = procs.get(role, [])
        if plist:
            live = [
                p
                for p in plist
                if p.get("state", "").startswith(("active/", "running"))
            ]
            chosen = (live or plist)[0]
            for k, v in chosen.items():
                if k == "note":
                    _add_note(rec, v)
                elif k == "info":
                    _add_note(rec, v, "infos")
                elif k == "state":
                    # The service's own verdict (stage 1) stays the verdict;
                    # the process state is kept beside it for the glyph.
                    rec["proc_state"] = v
                    rec.setdefault("state", v)
                elif v:
                    rec[k] = v  # the process's own facts win over role-level ones
            # Stage-3 distances attach to the sha this process actually has.
            for d in by_sha.get(role, []):
                # prefix match: --short=8 may return more digits when ambiguous
                if any(
                    x and x.startswith(d["for_sha"])
                    for x in (chosen.get("sha", ""), chosen.get("disk", ""))
                ):
                    for k in ("master_rel", "disk_master_rel"):
                        if d.get(k):
                            rec[k] = d[k]
            if len(plist) > 1:
                others = "; ".join(
                    f"{p.get('svc', '?')} {p.get('state', '?')}"
                    for p in plist
                    if p is not chosen
                )
                _add_note(rec, f"{len(plist)} processes for this role (also: {others})")
        else:
            for d in by_sha.get(role, []):
                for k in ("master_rel", "disk_master_rel"):
                    if d.get(k) and not rec.get(k):
                        rec[k] = d[k]
        merged[role] = rec
    return merged


def runs_as(rec: dict[str, str]) -> str:
    """Supervision label from stage-2 evidence, else the role's known shape."""
    managed = rec.get("managed", "")
    if managed == "UNMANAGED":
        return "unmanaged"
    if "(user unit)" in rec.get("svc", ""):
        return "user unit"
    if managed.startswith("systemd"):
        return "systemd"
    return rec.get("runs", "?")


def checkout(rec: dict[str, str]) -> str:
    """Clone @ branch sha, or the role's non-git provenance."""
    if rec.get("clone"):
        if rec.get("disk"):
            return f"{rec['clone']} DISK={rec['disk']} (HEAD {rec.get('sha', '')} {rec.get('branch', '')})"
        return f"{rec['clone']} {rec.get('branch', '')} {rec.get('sha', '')}".strip()
    return rec.get("checkout", "—")


def version(rec: dict[str, str]) -> str:
    """Self-reported version first (what is running), else the venv's."""
    ver = rec.get("version") or rec.get("installed") or rec.get("pyproject") or "?"
    pkg = rec.get("pkg", "")
    # Name the package when the role does not imply it (the Bluesky family).
    if (
        pkg
        and pkg not in ("tiled",)
        and rec.get("role")
        in ("Capture daemon", "Queueserver RE Manager", "Bluesky doc proxy")
    ):
        return f"{pkg} {ver}"
    return ver


def notes(rec: dict[str, str]) -> list[str]:
    """Everything that makes this row need attention, short form.

    Facts that need no action (a baked venv, a clone behind master, device
    counts) are :func:`infos`; they never mark a row ``!``.
    """
    out: list[str] = []
    if rec.get("managed") == "UNMANAGED":
        out.append("no systemd unit")
    if rec.get("worktree_of"):
        out.append(f"WORKTREE of {rec['worktree_of']} (not a clone)")
    if rec.get("disk"):
        out.append(f"disk ≠ HEAD ({rec.get('disk_date', '')})")
    staged, unstaged = rec.get("staged", "0"), rec.get("unstaged", "0")
    if staged not in ("", "0"):
        out.append(f"{staged} staged")
    if unstaged not in ("", "0"):
        out.append(f"{unstaged} unstaged")
    inst, proj = rec.get("installed"), rec.get("pyproject")
    if inst and proj and inst != proj:
        out.append(f"venv {inst} ≠ pyproject {proj}")
    if rec.get("stale"):
        out.append(
            "restart pending"
            if "baked" not in rec.get("stale", "")
            else "reinstall pending"
        )
    if rec.get("master_rel") and " ahead" in rec["master_rel"]:
        # Ahead of master = running unmerged code: a fact to act on. Behind
        # only = merges the host has not pulled; informational (see infos).
        out.append(rec["master_rel"].replace(" origin/master", " master"))
    if rec.get("disk_master_rel"):
        out.append(
            "disk " + rec["disk_master_rel"].replace(" origin/master", " master")
        )
    proc = rec.get("proc_state", "")
    if proc and not proc.startswith(("active/", "running")):
        out.append(f"unit {proc}")
    if rec.get("notes"):
        out.extend(n for n in rec["notes"].split("|") if n)
    return out


def infos(rec: dict[str, str]) -> list[str]:
    """Facts shown in the Notes column that do not mark the row ``!``."""
    out: list[str] = []
    if rec.get("baked"):
        out.append("baked venv")
    rel = rec.get("master_rel", "")
    if rel and rel != "= origin/master" and " ahead" not in rel:
        out.append(rel.replace(" origin/master", " master"))
    if rec.get("infos"):
        out.extend(n for n in rec["infos"].split("|") if n)
    return out


def glyph(rec: dict[str, str]) -> str:
    """✗ down, ! needs attention, ✓ clean.

    The stage-1 verdict (did the service answer?) decides ✗; a process
    that is not active/running (crash-looping, dead unit) can never be ✓.
    """
    state = rec.get("state", "")
    proc = rec.get("proc_state", "")
    if state.startswith(("down", "inactive", "failed")) or "DOWN" in state:
        return "✗"
    if state.startswith("absent"):
        return "·"
    if proc and not proc.startswith(("active/", "running")):
        return "!"
    return "!" if notes(rec) else "✓"


def render(merged: dict[str, dict[str, str]], width: int) -> str:
    """Box table sized to the terminal, columns wrapped, one service per row."""
    roles = [r for r in ROLE_ORDER if r in merged] + sorted(
        r for r in merged if r not in ROLE_ORDER
    )
    rows = []
    for role in roles:
        rec = merged[role]
        rec.setdefault("role", role)
        rows.append(
            [
                glyph(rec),
                role,
                runs_as(rec),
                checkout(rec),
                version(rec),
                "; ".join(notes(rec) + infos(rec)) or "—",
            ]
        )
    # Column widths: fixed-ish for the first five, the notes column absorbs the rest.
    fixed = [1, 12, 10, 24, 14]
    borders = 3 * len(HEADERS) + 1
    notes_w = max(12, width - sum(fixed) - borders)
    widths = fixed + [notes_w]

    def wrap_row(cells: list[str]) -> list[list[str]]:
        wrapped = [textwrap.wrap(c, w) or [""] for c, w in zip(cells, widths)]
        height = max(len(col) for col in wrapped)
        return [
            [col[i] if i < len(col) else "" for col in wrapped] for i in range(height)
        ]

    def line(left: str, mid: str, right: str) -> str:
        return left + mid.join("─" * (w + 2) for w in widths) + right

    def fmt(cells: list[str]) -> str:
        return "│" + "│".join(f" {c:<{w}} " for c, w in zip(cells, widths)) + "│"

    out = [line("┌", "┬", "┐"), fmt(HEADERS), line("├", "┼", "┤")]
    for i, row in enumerate(rows):
        for sub in wrap_row(row):
            out.append(fmt(sub))
        out.append(line("├", "┼", "┤") if i < len(rows) - 1 else line("└", "┴", "┘"))
    return "\n".join(out)


def main() -> int:
    """Stdin records -> table on stdout."""
    width = int(
        os.environ.get("COLUMNS") or shutil.get_terminal_size((100, 24)).columns
    )
    merged = parse(sys.stdin.readlines())
    if not merged:
        print("(no fleet records — nothing reachable, or stages 1-2 did not run)")
        return 0
    print(render(merged, max(72, width)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
