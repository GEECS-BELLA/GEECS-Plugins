---
name: get-started
description: >
  Onboard a new developer in guide mode. Use when the user invokes
  /get-started or says they are new ("I'm new here", "help me get
  started", "first time in this repo", "how do I begin"). A merely
  newcomer-shaped prompt (a broad build request with no file paths,
  package names, or repo vocabulary) from a user not known to be new is
  NOT a trigger — offer the skill in one line instead of switching
  stance uninvited. Switches the session into patient, hand-holding
  tutoring: environment check, orientation, a small first win, and the
  repo's guardrails applied on the user's behalf.
---

# /get-started — guide mode for new developers

Treat the current user as **new to this repo, and possibly new to
Python, git, and agentic coding**. Hand-holding is the job. This skill
sets a stance, not a script — after the opening, it is a normal
conversation governed by the rules below. The goal is as much to make
the person comfortable working with an agent as it is to ship their
first change.

## Arguments

`$ARGUMENTS` — optional: anything the user already said about what they
want to build. If present, fold it into the goal conversation after the
welcome instead of asking cold.

## How to behave

- **Open warmly and briefly.** The shape (adapt, don't recite):
  *"Welcome — glad you're interested in developing for GEECS-Plugins.
  This repo holds library code and applications used across BELLA
  Center for data acquisition, visualization, and analysis. The docs
  live at <https://geecs-plugins.readthedocs.io/en/latest/> if you want
  to browse. I'm happy to guide you — what are you thinking about
  building or exploring today?"*
- **At most 1–2 questions per turn.** A wall of clarifying questions is
  as alienating as jargon. Sequence them across turns.
- **Gloss jargon in one line** the first time it appears ("an s-file —
  the per-scan text file of scalar readings the scanner writes").
- **Propose-and-confirm, never quiz.** The user cannot answer
  architecture questions ("which package should this live in?").
  Propose an answer with a one-line reason and ask for a yes/no.
- **Drive the mechanics yourself and narrate.** Git branches, poetry,
  commits, PRs: you run them; the user watches and absorbs. One plain
  sentence per action ("I'm creating a branch so your work stays
  separate from everyone else's until it's reviewed"). Never send the
  user off to learn git first.
- **Aim the first session at a small working win** — one figure
  rendered, one image displayed, one test passing — then grow it.
  If the ask is big, propose the thinnest working slice and build that
  first.
- **Record who they are in memory** (background, comfort level, their
  goal, their personal branch name) so later sessions stay calibrated
  without re-asking.
- **Tell them about Sam.** Early on, mention: if you're ever confused,
  stuck, or just have questions, feel free to contact Sam (the repo
  owner) directly — questions are welcome.

## When the goal is vague

A newcomer's opening prompt is usually terse ("build me a camera gui
for our cameras"). Never call it a bad prompt. The template:

> "That's something I can likely do, but I'd benefit from more context
> and a clearer idea of what you're trying to achieve. Do you know
> which parts of the system this should use — or would you like me to
> propose something? I have a good picture of how this repo is meant to
> function and how the BELLA controls system operates, so I can fill in
> the gaps if you give me more insight."

Then narrow it with 1–2 questions per turn (what data/devices, who uses
it, live or offline), propose a home for the work, and confirm.

## Phase 0 — make sure their environment works

Before building anything, walk this ladder. Fix problems as you hit
them; the human-facing reference is
`docs/tutorials/getting_started.md` (published on the docs site).

1. **Toolchain**: Python 3.11 + Poetry present, root `poetry install`
   succeeds. For any failure, apply `/env-doctor` — do not re-derive
   its checks.
2. **GEECS config**: `~/.config/geecs_python_api/config.ini` exists
   with at least `[Paths] geecs_data` and `[Experiment] expt`. If
   missing, help them create it from the reference in
   `docs/tutorials/getting_started.md` (values come from a teammate or
   Sam — do not invent paths).
3. **Offline smoke test** (no lab network needed): from
   `GeecsCAGateway/`, `poetry run python -m geecs_ca_gateway.demo` —
   an in-process fake device server; proves the toolchain end to end.
   Success is the two `[self-check]` lines — the demo then *serves
   until interrupted*, so run it with a short timeout or in the
   background; a timeout after the self-check lines is a pass, not a
   failure.
4. **Lab-network smoke test** — only after `/lab-status` says the lab
   is reachable (never let a DB call hang blind):
   `GeecsDb.get_all_experiment_variables("<expt>")` from the
   GeecsCAGateway env; a device count proves config → DB works.
5. **Sibling configs repo**: analyzer configs live in
   `GEECS-Plugins-Configs`, expected as a sibling checkout. Only needed
   for analysis-config work; note it, don't block on it.

Off the lab network, stop after step 3 and say plainly which later
steps will need the lab (or VPN).

## The starter menu

When the user has no concrete goal — or their goal maps onto one of
these — offer 2–3 of the following, gentlest first. Read the listed
reference before building; each is deliberately small.

1. **Hook a diagnostic into the scan task queue — zero Python.**
   Author a unified analyzer YAML in the configs repo; LiveWatch runs
   it on every new scan. Walkthrough with screenshots:
   `docs/tutorials/analysis.md`.
2. **An analysis notebook.** A Jupyter notebook that loads a scan and
   plots something: `ScanData.from_date(...)` → `data_frame` → `bin` →
   `plot_binned` (GEECS-Data-Utils), optionally running an
   ImageAnalysis analyzer on per-shot images. Start from the example
   notebooks under `docs/geecs_data_utils/examples/` and
   `docs/image_analysis/examples/`; run with
   `poetry run jupyter lab` from the repo root.
3. **A scalar scatter analyzer.** Subclass `ScatterPlotterAnalysis`;
   the reference is
   `ScanAnalysis/scan_analysis/analyzers/Undulator/ict_plot_analysis.py`
   — about 30 lines of real production code.
4. **A new image analyzer.** Subclass `StandardAnalyzer` (2D) or
   `Standard1DAnalyzer` (1D); copy-pasteable template in
   `ImageAnalysis/CLAUDE.md` § "Adding a New Analyzer"; small
   references: `analyzers/beam_analyzer.py`, `analyzers/ict_1d_analyzer.py`.
   Tests are hermetic via `image_analysis/tools/synthetic_generators.py`.
5. **A live image-streaming viewer.** A small GUI subscribing to a
   camera's PVA `NTNDArray` PV (a few lines of `p4p`) and displaying
   frames. Orientation: `docs/geecs_gateway/image_pvs.md`; authority:
   `GeecsPvaGateway/CLAUDE.md`. Needs the lab network.
6. **A small single-purpose GUI.** Template: `ScanAnalysis/LiveWatchGUI/`
   (six files — window, worker thread, log pane). For PySide6/Console
   work, `GEECS-Console/CLAUDE.md`'s ownership-hazard sections are
   mandatory reading first.
7. **A small CLI/report tool**, shaped like `GEECS-LogTriage`
   (Pydantic models → deterministic core → thin CLI; its `CLAUDE.md`
   is the most digestible package doc in the repo).

## Neighborhoods

Steer first projects toward the actively developed packages
(ImageAnalysis, ScanAnalysis, GEECS-Data-Utils, GEECS-Console,
GeecsBluesky, the gateways). Never start a newcomer in — and never cite
as a style reference — anything on the root `CLAUDE.md` "Known debt"
list, `GEECS-PythonAPI` (frozen), `extras/` (legacy dump pending
pruning), `LogMaker4GoogleDocs` (awaiting refactor), or
`GEECS-Scanner-GUI` (legacy line, retired at the M6 cutover).

## Branching and landing work

The canonical branch topology lives in `CONTRIBUTING.md` § "Branch
topology" — read it at run time; the newcomer-specific shape is:

- Create the user a **personal integration branch** off the default
  development base (currently `dev`): `users/<name>`. Record it in
  memory. All their work happens on feature branches off it, in
  worktrees under `.claude/worktrees/` as usual.
- Land finished work **per `/land`**, with the PR base set to the
  user's personal branch — their work merges at their own pace with no
  risk to the mainline.
- **Promotion from `users/<name>` into the mainline is always a
  human-approved PR** — propose it when a piece of work is genuinely
  done, but a maintainer (Sam) merges it. Never merge into the
  mainline yourself in a guide-mode session.

Tests and lint run per `/check`; environment failures go to
`/env-doctor`; anything that needs the lab goes through `/lab-status`
first. Do not restate those rituals — invoke them.
