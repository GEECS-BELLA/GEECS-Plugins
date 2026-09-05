#!/usr/bin/env bash
# lab_status.sh — bounded probes of lab-network and hardware reachability.
#
# The canonical host/port/timeout facts live in scripts/lib/lab_env.sh (the
# config.ini reader + endpoints this script and fleet_status.sh share; the
# /lab-status skill points there — do not restate them in prose). Every
# probe is bounded
# to seconds — the point is to replace the 75-second GeecsDb hang and the
# blind "is the trigger even firing?" scan attempt with one cheap command.
#
#   scripts/lab_status.sh                 # tier 1: network only (safe anywhere)
#   scripts/lab_status.sh --hardware      # + tier 2: read-only CA liveness
#   scripts/lab_status.sh --hardware --experiment Undulator
#
# Tier 1 is TCP/HTTP/filesystem — safe to run at any time, on or off
# network. Tier 2 performs READ-ONLY Channel Access gets (gateway heartbeat,
# device count) through the GeecsBluesky env; it never writes a PV.
#
# The DB is probed only through scripts/mysql_probe.py (a bounded, real
# handshake) — never a bare TCP connect of the MySQL port, which counts toward
# the server's host block (error 1129; #790). Why, and the remedy:
# docs/platform/fleet_map.md, the MySQL admonition.
set -u  # deliberately not -e: a failed probe is a *finding*, not an error

CA_TIMEOUT=3    # seconds per CA read (tier 2)
TCP_TIMEOUT=2   # seconds per port probe (lab_env.sh reads it)

HARDWARE=0
EXPERIMENT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --hardware) HARDWARE=1 ;;
        --experiment) shift; EXPERIMENT="${1:-}" ;;
        *) echo "usage: lab_status.sh [--hardware] [--experiment NAME]" >&2; exit 2 ;;
    esac
    shift
done

# config.ini reader, endpoints (LAB_HOST, TILED_PORT, DATA_ROOT, DB_PORT,
# CA_PORT), printers and the bounded port probe — shared with fleet_status.sh.
. "$(cd "$(dirname "$0")" && pwd)/lib/lab_env.sh"
[ -n "$EXPERIMENT" ] || EXPERIMENT="$(config_experiment)"

echo "== Tier 1: lab network (bounded, read-nothing) =="
if [ ! -f "$CONFIG" ]; then
    bad "config.ini missing ($CONFIG) — no endpoints known; this machine is not set up for lab access"
    exit 1
fi
if [ -z "$LAB_HOST" ]; then
    bad "[tiled] uri absent from config.ini — cannot derive the lab server host"
    exit 1
fi

NET_UP=1
# Handshake-completing probe (see header): rc 0 reachable, 3 reachable-but-
# blocked, 4 no connector available, 5 answered but handshake incomplete,
# 137 the probe itself was killed at the wall (a stalled credential lookup
# or connect — not a verdict), anything else = nothing answered. The probe's
# line names the host:port it actually probed (INI target or fallback).
db_line="$(mysql_probe "$LAB_HOST" "$DB_PORT")"
db_rc=$?
db_target="$(printf '%s' "$db_line" | awk '{print $2}')"
db_rest="$(printf '%s' "$db_line" | cut -d' ' -f3-)"
case $db_rc in
    0) ok "MySQL       $db_target ($db_rest)" ;;
    3) warn "MySQL       $db_target answers but has BLOCKED this address (MySQL 1129: too many aborted connects — a bare port probe somewhere on the VPN); GeecsDb calls fail fast until a DB admin runs FLUSH HOSTS" ;;
    4) skip "MySQL       $LAB_HOST:$DB_PORT not probed — ${db_line#no-connector }; a bare TCP probe is never the fallback" ;;
    5) warn "MySQL       $db_target $db_rest" ;;
    137) skip "MySQL       $LAB_HOST:$DB_PORT not probed — the probe was killed at its $(( TCP_TIMEOUT + 10 )) s wall (a stalled credential lookup on the data share, or a stalled connect); not a DB verdict" ;;
    *) bad "MySQL       ${db_target:-$LAB_HOST:$DB_PORT} — GeecsDb calls would hang ~75 s; do not make them"; NET_UP=0 ;;
esac
if port_open "$LAB_HOST" "$TILED_PORT"; then
    version="$(curl -s -m "$TCP_TIMEOUT" "http://$LAB_HOST:$TILED_PORT/api/v1/" | sed -nE 's/.*"library_version":"([^"]+)".*/\1/p')"
    ok "Tiled       $LAB_HOST:$TILED_PORT (v${version:-?})"
else
    bad "Tiled       $LAB_HOST:$TILED_PORT"
    NET_UP=0
fi
if port_open "$LAB_HOST" "$CA_PORT"; then
    ok "CA gateway  $LAB_HOST:$CA_PORT (TCP only — liveness needs --hardware)"
else
    bad "CA gateway  $LAB_HOST:$CA_PORT"
    NET_UP=0
fi
if [ -n "$DATA_ROOT" ]; then
    if [ -d "$DATA_ROOT" ]; then
        ok "Data mount  $DATA_ROOT"
    else
        bad "Data mount  $DATA_ROOT — scans could not claim folders / read data"
    fi
else
    skip "Data mount  (GEECS_DATA_LOCAL_BASE_PATH not in config.ini)"
fi
# Crude VPN-vs-lab hint: round-trip time to the lab server.
rtt="$(ping -c 1 -t 2 "$LAB_HOST" 2>/dev/null | sed -nE 's/.*time=([0-9.]+) ms.*/\1/p')"
if [ -n "$rtt" ]; then
    echo "  rtt ${rtt} ms  (rule of thumb: <5 ms on-site, more = VPN — expect ~5 s/shot free-run scans)"
fi

if [ "$HARDWARE" -eq 0 ]; then
    if [ "$NET_UP" -eq 1 ]; then
        echo "network: UP — add --hardware for read-only gateway/trigger liveness"
    else
        echo "network: DOWN or partial — hermetic work only (see the /lab-status skill for the capability table)"
    fi
    exit 0
fi

echo "== Tier 2: hardware liveness (READ-ONLY CA gets) =="
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BLUESKY_DIR="$REPO_ROOT/GeecsBluesky"
if [ -z "$EXPERIMENT" ]; then
    bad "no experiment name (pass --experiment NAME); cannot build gateway PV names"
    echo "role=CA gateway	state=down	note=no experiment name"
    exit 1
fi
if ! poetry -C "$BLUESKY_DIR" env info --path >/dev/null 2>&1; then
    bad "GeecsBluesky poetry env not installed — tier 2 needs its aioca (see /env-doctor)"
    echo "role=CA gateway	state=down	note=GeecsBluesky env not installed"
    exit 1
fi
EXPERIMENT="$EXPERIMENT" CA_TIMEOUT="$CA_TIMEOUT" poetry -C "$BLUESKY_DIR" run python - <<'PY'
import asyncio
import os

import geecs_bluesky.epics_env  # noqa: F401 — applies EPICS_CA_ADDR_LIST from config.ini
import aioca

experiment = os.environ["EXPERIMENT"]
timeout = float(os.environ["CA_TIMEOUT"])


async def read(pv):
    return await asyncio.wait_for(aioca.caget(pv), timeout=timeout)


async def main():
    # PV names come from the naming contract (lowercase components,
    # PV_CONTRACT.md §1) — never hand-assemble them here.
    from geecs_core.pv_naming import pv_name

    try:
        heartbeat = await read(pv_name(experiment, "CAGateway", "heartbeat"))
        connected = await read(pv_name(experiment, "CAGateway", "devices_connected"))
        version = await read(pv_name(experiment, "CAGateway", "version"))
    except Exception as exc:  # noqa: BLE001 — a failed probe is a finding
        print(f"  [DOWN] gateway PVs unreadable ({type(exc).__name__}: {exc})")
        print("         network may be up while the gateway service is not")
        print(f"role=CA gateway\tstate=down\tnote=PVs unreadable ({type(exc).__name__})")
        raise SystemExit(1)
    print(f"  [ OK ] gateway alive: heartbeat={int(heartbeat)}, "
          f"devices_connected={int(connected)}, version={version}")
    if int(connected) == 0:
        print("  [WARN] zero devices connected — GEECS side likely down")
    # The machine-readable form of the verdict, for fleet_status.sh (a stated
    # contract: one tab-separated key=value record, same shape as its own).
    note = "\tnote=zero devices connected" if int(connected) == 0 else ""
    print(f"role=CA gateway\tstate=ok\tversion={version}\tinfo={int(connected)} devices connected{note}")


asyncio.run(main())
PY
status=$?
if [ "$status" -eq 0 ]; then
    echo "hardware: gateway UP — note this does NOT prove the trigger is firing;"
    echo "a scan still needs the laser/DG645 state right (see the /lab-status skill)"
fi
exit "$status"
