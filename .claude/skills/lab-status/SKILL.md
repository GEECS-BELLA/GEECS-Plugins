---
name: lab-status
description: >
  Probe lab-network and hardware reachability with bounded timeouts before
  doing anything that needs them. Use when a call hangs or you are about to
  make one ("timed out connecting to 192.168.6.14", a GeecsDb/MySQL call
  taking ~75 seconds, "Connection refused" from Tiled, aioca/CA connect
  timeouts, "GeecsTriggerTimeoutError: no shot within"), when the user asks
  "am I on the lab network / VPN?", before running a scan or hardware test,
  or at session start when work is network-dependent. Wraps
  scripts/lab_status.sh.
---

# /lab-status — what can this machine reach right now?

`scripts/lab_status.sh` owns the endpoints, ports, and timeouts (derived
from `~/.config/geecs_python_api/config.ini` — do not restate or hardcode
them here or anywhere else). This skill is about running it at the right
moment and acting on the answer.

## Tiers

- `scripts/lab_status.sh` — **tier 1, network only.** Pure TCP/HTTP/mount
  probes, ~2 s each, safe to run anywhere, anytime, unprompted.
- `scripts/lab_status.sh --hardware` — **tier 2, gateway liveness.**
  READ-ONLY Channel Access gets (heartbeat, device count) through the
  GeecsBluesky env. Reads only — it never writes a PV. Pass
  `--experiment NAME` if config.ini doesn't name one.

## The rules the probes enforce

1. **Never make an unbounded lab call to find out whether the lab is
   reachable.** A GeecsDb/MySQL call on a dead route blocks ~75 s; device
   waits block their full timeout. Run tier 1 first (2 s) whenever
   reachability is in doubt — especially at session start, since the user
   is often off-network.
2. **Network up ≠ hardware ready.** The gateway answering does not mean
   the trigger is firing: a scan can still die with
   `GeecsTriggerTimeoutError: no shot within N s` when the laser
   oscillator is off or the DG645 is in the wrong mode (external vs
   internal). Tier 2 tells you the gateway is alive; only the user can
   confirm beam/trigger state. Ask before burning scan numbers on retries.
3. **Authorization is separate from reachability.** Reads (CA monitors,
   DB queries, Tiled) are always fine once reachable. Anything that moves
   or fires hardware — `:SP` puts, scans of any kind — needs the user's
   explicit go for the session, regardless of what the probes say.

## Capability table (act on the probe result)

| Probe result | You can | You must not |
|---|---|---|
| network DOWN | hermetic tests, code, docs, mock-driven GUI work | call GeecsDb, connect CA, read Tiled — each hangs its full timeout |
| network UP, high rtt (VPN) | everything read-only; scans work but slowly (~5 s/shot free-run frame latency) | assume slowness is a code bug — check rtt first |
| network UP, gateway DOWN (tier 2) | DB + Tiled work; no device I/O | "fix" it from here — the gateway service needs attention on its host |
| gateway UP, devices_connected ≈ 0 | CA reads of gateway PVs | expect device data — the GEECS side is likely down |
| gateway UP, devices connected | reads, and (with user authorization) scans/puts | assume the trigger fires — verify beam state with the user first |

## Failure vocabulary → likely meaning

- `mysql.connector` hang / ~75 s stall → tier-1 DOWN case; you skipped the probe.
- Tiled `Connection refused` → service down on the box (network itself may be fine — check MySQL probe to distinguish).
- `GeecsTriggerTimeoutError: no shot within N s` mid-scan → hardware state (laser/DG645), not code; do not retry unattended.
- aioca connect timeout on one specific PV → that PV may not exist (ghost DB row, wrong name) — run the gateway DB audit (`geecs-ca-gateway-audit`), not this skill.
