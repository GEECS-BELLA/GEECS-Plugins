# Sharing a day's data with external collaborators through Globus

*Planning note (see `Planning/README.md`). Delete this file in the PR that
lands the sharing runbook page under `docs/platform/` after the pilot has
run; anything still load-bearing moves to `GEECS-Data-Utils/CLAUDE.md`
(the staging tool's rules) or that runbook first.*

Drafted 2026-09-23 from a read of the data plane as it stands on master
(Tiled placement, the scan folder, the logbook mirror, the portal's
download seams) plus the Globus and Berkeley Lab documentation cited at
the end. Status: **direction proposed, nothing built, one institutional
inquiry to send.** Owner: Sam.

---

## Where we stand

**The need.** Collaborators who took data with us should be able to get
that data — a day, or a few days, of one experiment — from outside the
lab network, without accounts on the data share and without us running a
public service. Today the only thing that leaves the network is the
Google Doc log.

**The proposal.** Three pieces, in order of how much of each we own:

1. **A staging command we write** (`geecs-dataset`, in GEECS-Data-Utils):
   given an experiment and a day, it gathers the right files from the
   share into a collaboration folder, packs what must be packed, and
   writes a README and a manifest. Read-only over the data tree. This is
   the only code in the plan.
2. **A Globus collection someone provides** — read-only guest
   collections over that folder, one per shared dataset, granted to
   named collaborators. Which endpoint hosts it is the one open
   decision, and it is ScienceIT's to answer, not ours to engineer.
3. **A pilot** — one completed day, one collaborator, before anything
   more is built. No offline portal, no Tiled mirror, no UI.

**What collaborators get** is files and a Globus link: a file browser,
transfers to their laptop or their institution's storage, and (on a
server-class endpoint) HTTPS download links. Not our scan portal. The
README and the scalar tables are what make files enough.

**Simple is not simple in one place.** Everything about the copy is a
day of Python. The endpoint is an institutional question — a
subscription, a network path out of a private subnet, possibly a
machine — with a timeline we do not control. The inquiry goes out first,
and the copy tool is built while it is answered.

## Decisions

| Topic | Status | Where we landed |
|---|---|---|
| Copy into a staging folder, not share the NAS in place | decided | A guest collection rooted in the live data tree exposes every past and future day, reads over an SMB mount with stale listings and HDF5 locking trouble, and cannot pack the per-shot files. A staged copy is deliberate per day, packed, and disposable. Sharing the tree in place is the endgame only if a DTN with the share mounted materialises *and* sharing becomes routine. |
| Three tiers, summary by default | decided | `summary` (ScanInfo, s-files, `scan.log`, the `analysis/` tree, the day's logbook entries + attachments, run metadata, README, manifest — megabytes to a gigabyte), `stacks` (the finalized `<device>.h5` frame stacks — already one file per device per scan), `raw` (the LabVIEW per-shot files, packed one tar per device per scan). Scans and devices are selectable. |
| Pack the per-shot files; never ship them loose | decided | The measured scan is 21,407 files for 6.7 GB. Globus and the NAS both pay per file (the NAS reads small files at ~15 MB/s against a ~40 MB/s path cap; Globus recommends archives of a few GB and reports "endpoint too busy" on directories of many files). One uncompressed `tar` per device per scan, members in shot order — PNGs are already compressed, and this mirrors the one-file-per-device-per-scan direction the stacks already take. |
| The logbook comes from its markdown mirror, not its API | decided | `{experiment}/logbook/Y…/M…/D…/` on the share already has one markdown file per entry with YAML front matter and attachments copied beside it, links relative — an offline tree by construction. The exporter copies that day's subtree and skips entries whose front matter says `status: draft`. GEECS-Data-Utils never imports GeecsLogbook (the dependency points the other way). |
| Tiled contributes run metadata only | decided | The scalar table is the s-file, already on the share. From Tiled the exporter writes one `runs.json` per day — start and stop documents (preset, purpose text, scan variables, exit status) via `ScanCatalog` — and nothing else. Never a primary-stream read (an image run's frame is 14 GB; #836). Tiled unreachable → `runs.json` omitted with a note in the manifest, not a failure. |
| Read-only consumer of the data tree | decided | The tool never writes under `{experiment}/`: no `ScanPaths(read_mode=False)`, no `get_analysis_folder()` (it creates), no status YAML. Pinned by a test the way the folder-creation invariant is pinned elsewhere. `analysis_status/` and `.claim` files are excluded from every tier — queue state, not data. |
| Manifest + README, no per-file checksums | decided | `manifest.json` is a Pydantic model: experiment, days, scans, tiers, per-scan file counts and bytes per tier, what was skipped and why, tool version, the day's `runs.json` presence. Tar archives get a sha256 computed while they are written (free). Loose files are not hashed — Globus verifies checksums in transfer, and hashing a raw day at the NAS cap costs hours for nothing. |
| README is generated, with a fixed reader section | decided | Day facts from the manifest, then how to read what is there: the s-file layout, `ScanInfoScanNNN.ini`, the HDF5 stack layout (`/entry/data/data` + NDAttributes, open lock-free), the IMAQ PNG gotcha (`read_imaq_png_image`), the logbook front matter, and `pip install "geecs-data-utils @ git+https://github.com/GEECS-BELLA/GEECS-Plugins#subdirectory=GEECS-Data-Utils"`. Collaborators must not need us on the phone to open a stack. |
| Where it runs and where the copy lands | decided | On the host with the share mounted (192.168.6.14 today; the services box when it comes), by hand, as a CLI. The staging root is a facility value — `GEECS_DATASET_ROOT` in `site.env` → `config.ini`, never a literal in code — on local disk, not on the NAS (the endpoint reads it; the NAS cap must not sit behind the collaborator's transfer too). |
| Staged copies are disposable | decided | `geecs-dataset prune` deletes staged datasets older than a retention window or on request; the manifest records what was shared so the record outlives the copy. The tree can always be regenerated from the share. |
| The endpoint is IT's to run | decided | We do not stand up Globus Connect Server ourselves. The candidates (below) are ranked by how little we own. A headless Globus Connect Personal on our host is the one form we could run alone, and only as a pilot. |
| One person runs the command per dataset | decided | That is the approval step. No scheduler, no per-scan push, no agent verb. Nothing leaves the network unless someone typed the day. |
| Name the package `geecs_data_utils.dataset`, CLI `geecs-dataset` | tentative | "share" collides with the data share; "export" with `tiled_export` and the doc exporter. `inventory`, `stage`, `prune` subcommands. First `[tool.poetry.scripts]` entry in data-utils; the pattern exists in six other packages. |
| Manifest model lives in data-utils, not GEECS-Schemas | tentative | Data-utils has no intra-repo dependency and the manifest is read by nothing else in the repo. Move it to GEECS-Schemas only if a second consumer appears (the portal listing shared datasets would be one). |
| Automate guest-collection ACLs with the Globus SDK | deferred | Needs a registered Globus app and client credentials on our side — a second IT conversation. Setting a collaborator's read permission in the Globus web app takes a minute. Revisit after the third shared dataset. |
| A "Share this day" button in the portal | deferred | The portal already knows the day and the runs; the button is a form over `geecs-dataset stage`. Not before the pilot proves anyone wants a second dataset. |
| Google Drive as the transport | rejected for v1, kept as fallback | Same egress as the log today and no subscription question, but Drive is the wrong tool for gigabyte archives and thousands of files, and collaborators would download through a browser. If ScienceIT cannot offer any Globus route, `rclone` to the lab's Google Drive from the same staged tree is a two-line fallback, and the staging tool is unchanged. |

## The endpoint: three candidates

Verified against the Globus and Berkeley Lab pages cited below
(2026-09-23). The facts that decide it:

- **Guest collections (sharing) are a subscription feature** on both
  Globus Connect Server and Globus Connect Personal. Berkeley Lab runs
  managed collections (Lawrencium `lbnl#lrc`, Google Drive, S3, GCS) and
  a DTN programme through ScienceIT, so a subscription exists; whether a
  BELLA endpoint can sit under it is the question to ask.
- **Globus Connect Server needs inbound reachability**: TCP 443 from the
  Globus service's address blocks for the control channel (mandatory —
  "your endpoint cannot function" without it) and inbound 50000–51000
  for data, or it cannot receive from servers or talk to Personal
  endpoints at all. That is a Science DMZ machine with a routable
  address, which is exactly what ScienceIT's DTN service exists to
  provide (in production for the ALS and the Molecular Foundry).
- **Globus Connect Personal needs only outbound connections** (443,
  2223, 50000–51000 out, UDP for peer transfers). It runs headless on
  Linux. It can host guest collections only when covered by the
  subscription; it does not offer HTTPS download links.

| | A. Managed Globus Connect Personal on our host | B. Push to a store that already has a managed collection | C. A ScienceIT DTN with the staging root (or the share) mounted |
|---|---|---|---|
| What we run | one user-level daemon on the staging host | an outbound `rsync`/`scp` (or Globus CLI once A exists) per dataset | nothing; a mount |
| Network change | none (outbound only) | none (outbound ssh) | ScienceIT's: DMZ placement, firewall exceptions |
| IT ask | add Sam (or the endpoint) to the Lab's subscription | an allocation we may already have: NERSC Community File System (`gsharing` directories are designed for exactly this), a Lawrencium project space, or the Lab's Google Drive collection | a DTN consultation, plus the NAS team if the share itself is to be mounted |
| Collaborator experience | Globus web app + transfer; needs a Globus login (their institution, ORCID or Google) | same, on a big well-connected endpoint; NERSC/Lawrencium throughput | same, plus HTTPS "Get Link" downloads for people without an endpoint |
| Throughput | our host's uplink; adequate for a pilot | excellent | excellent; and no staging copy if the share is mounted |
| Side effect | none | an off-site copy of the staged data (a backup we do not have today) | the lab gets a DTN it will want for other things |
| Solo-maintainable | yes | yes | IT-run, which is the right owner |
| Time to first transfer | days after the subscription answer | days after an allocation confirms | weeks to months |

**Recommendation.** Ask ScienceIT for A and C in the same message; pilot
on whichever exists first. If BELLA already has a NERSC allocation, B is
the pilot that needs no ScienceIT answer at all, and its off-site copy is
worth having regardless. C is the endgame if sharing becomes routine or
the raw tier is what people actually want.

The inquiry, ready to send:

> We want to share selected BELLA experimental datasets (one day of one
> experiment at a time; a summary tier of tens of MB to a few GB, a raw
> tier of up to a few hundred GB) with named external collaborators as
> read-only Globus guest collections. The data lives on our NetApp share
> and is reachable from a Linux host on the BELLA control network, which
> reaches the internet outbound but is not routable inbound. Two
> questions: (1) Can that host run a Globus Connect Personal endpoint
> under the Lab's subscription so we can create guest collections from
> a staging folder on it? (2) Is there an existing subscription-managed
> collection with staging storage we could push to instead, or should
> we plan a Data Transfer Node with access to our share? We would pilot
> with one completed day and one collaborator before building anything.

## The staging tool

`geecs_data_utils.dataset`, a peer of `tiled_export` and `scan_paths`,
read-only over the tree it walks. Three subcommands:

```
geecs-dataset inventory Undulator 26_0921            # counts and bytes per tier, no writes
geecs-dataset stage Undulator 26_0921 --tier summary,stacks --scans 3-9 --out $GEECS_DATASET_ROOT
geecs-dataset prune --older-than 30d
```

Output layout, one directory per dataset, named so a collaborator can
tell datasets apart in a listing:

```
$GEECS_DATASET_ROOT/Undulator_26_0921/
  README.md
  manifest.json
  runs.json                            start/stop docs per scan, from Tiled (optional)
  logbook/                             that day's mirror subtree, drafts skipped
    0834-…md  Scan005/…md  Scan005/attachments/…
  scans/Scan003/
    ScanInfoScan003.ini  ScanDataScan003.txt  scan.log
    UC_ALineEBeam3.h5                  stacks tier (finalized only)
    UC_TC_Phosphor.tar                 raw tier: per-shot files, one tar per device
  analysis/Scan003/…   analysis/s003.txt
```

Rules the code keeps, each pinned by a test on a synthetic day:

- Never writes under `{experiment}/`; a monkeypatched `mkdir` under the
  data root fails the suite.
- Refuses a day that does not exist rather than creating anything.
- Skips an `.h5` without the `finalized` attribute and a scan whose
  folder is being written (a `scan.log` younger than a threshold), and
  says so in the manifest.
- Tar members are in shot order and the archive's sha256 in the
  manifest matches the file.
- Draft logbook entries are not copied; `analysis_status/` never is.
- Every facility value comes from `config.ini`; the tests run with a
  temporary one.

Sizing the effort with the measured numbers (26_0826 Scan006: 21,407
files, 6.7 GB; NAS small-file reads ~15 MB/s, large-file ~40 MB/s):

| Tier | Per scan | 50-scan day | Time at the NAS cap |
|---|---|---|---|
| summary | ~1–20 MB | ~0.1–1 GB | minutes |
| stacks (e.g. 10 plugin-backed cameras × 200 shots × ~2 MB) | ~4 GB | ~200 GB | ~1.5 h |
| raw (per-shot files, packed) | ~7 GB, 21k files | ~335 GB | ~6 h (file-count bound) |

A summary dataset is interactive. Stacks and raw are batch jobs; the
tool must be resumable (skip scans already staged and manifest-complete)
because the raw tier will run overnight on the shared host and the host
also serves the portal. Run it under `nice`, and never during a scan day
— it is the same NAS path the scanner writes through.

## Build order

0. **Send the inquiry** (above). In parallel, `geecs-dataset inventory`
   — read-only, prints the table above for a real day. Half a day; it
   also gives this plan the per-day numbers it is currently
   extrapolating from one scan.
1. **`stage` + `prune`**, manifest and README, the tests listed above.
   One to two agent-days. Version bump + CHANGELOG in data-utils. Lands
   whether or not IT has answered — a staged folder is useful on its own
   (a USB drive, a `rclone` to Drive).
2. **The pilot.** One completed day, summary + stacks tiers, one
   collaborator, on whichever endpoint exists first. Acceptance below.
   Delete the staged copy afterwards and confirm the manifest is enough
   of a record.
3. **Only after the pilot:** `docs/platform/data_sharing.md` (the
   runbook: who runs what, the endpoint, the retention rule) — and this
   note is deleted in that PR. The deferred rows above stay deferred
   until a second and third dataset ask for them.

## Acceptance

- The collaborator, with the README alone, opens an `.h5` stack, a
  packed device tar and an s-file, and matches a shot across the three.
- The transfer completes without an operator watching it; the Globus
  task report shows every file verified.
- Nothing was written under `{experiment}/` on the share during staging
  (`find -newer` against a marker file before and after).
- The staged dataset is deleted and `manifest.json` still says what was
  shared, with whom (a free-text field the operator fills), and when.
- The inventory of a real day matches the sizing table within a factor
  of two, or the table is corrected.

## Honest costs

- **The gate is institutional, not technical.** Weeks for a subscription
  answer, months for a DTN. The code is days. Do not let the code wait
  on the answer, and do not build a second copy tool while waiting.
- **Local disk on the endpoint host.** A raw day is a few hundred GB; the
  current host is a 16 GB reference box with the portal on it. A raw
  pilot waits for the services box (4 TB RAID1) or ships stacks only.
- **Hours on the NAS path** for anything beyond the summary tier, on the
  same path the scanner and the analysis queue use. Overnight, off scan
  days.
- **Collaborator friction.** Everyone needs a Globus login (free; most
  universities and labs are identity providers) and either Globus
  Connect Personal on a laptop or an institutional endpoint. Route C
  removes the endpoint requirement with HTTPS links; A does not.
- **What the record does not carry.** A staged tree is not a Tiled
  catalog and not a portal. Collaborators get files. If they need the
  interactive views, that is a separate conversation (an offline portal
  build over a staged tree is plausible and out of scope here).
- **Maintenance.** One module (~500 lines with tests), one runbook page,
  a `site.env` key. The endpoint's upkeep is IT's.

## Open questions for Sam

- Which collaborators and institutions first? (Do they have a Globus
  endpoint already — a university cluster, a national lab DTN?)
- Does BELLA or ATAP hold a NERSC allocation or a Lawrencium project?
  Either makes route B a pilot we can run this month.
- What is the PI's default for what leaves the lab: summary always, raw
  on request? The tool defaults to `summary`; the answer sets the
  runbook.
- Which cameras do collaborators actually analyse? It decides whether
  the `stacks` tier (plugin-backed cameras) is enough or the raw tier is
  the real ask.
- Retention of staged copies: delete on confirmation, or a fixed window?

## Sources (verified 2026-09-23)

- Globus, sharing requires a subscription (server and personal):
  https://docs.globus.org/faq/subscriptions/ ,
  https://docs.globus.org/faq/globus-connect-endpoints/ ,
  https://docs.globus.org/globus-connect-server/v5/reference/collection/create/
- Globus Connect Server v5.4 network requirements and what a restrictive
  firewall costs:
  https://docs.globus.org/globus-connect-server/v5/gcsv54-restricted-firewall-policy/impact-of-restricting-gcsv54-firewall-policy/
- Globus Connect Personal firewall configuration (outbound only):
  https://docs.globus.org/globus-connect-personal/firewall-configuration/
- HTTPS access to collections (server endpoints):
  https://docs.globus.org/globus-connect-server/v5.4/https-access-collections/
- Small-file guidance (archive to a few GB; "endpoint too busy"):
  https://docs.tacc.utexas.edu/datatransfer/globus/ ,
  https://docs.rc.fas.harvard.edu/kb/globus-file-transfer/
- Berkeley Lab ScienceIT: Globus at the Lab (managed collections, contacts)
  https://scienceit-docs.lbl.gov/data/globus/ ; the DTN service
  https://it.lbl.gov/service/scienceit/data-management-and-storage/data-transfer-node-with-globus/
- NERSC sharing with non-users via guest collections on the Community
  File System: https://docs.nersc.gov/filesystems/sharing/
- In-repo: `GeecsBluesky/TILED_SETUP.md` (Tiled placement),
  `GeecsLogbook/geecs_logbook/mirror.py` (the mirror tree),
  `GeecsPvaGateway/geecs_pva_gateway/file_plugin.py` (the stacks),
  `GEECS-Data-Utils/geecs_data_utils/tiled_catalog.py` (`ScanCatalog`).
