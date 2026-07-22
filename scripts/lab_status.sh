#!/usr/bin/env bash
# lab_status.sh — bounded probes of lab-network and hardware reachability.
#
# This script owns the canonical host/port/timeout facts (the /lab-status
# skill points here; do not restate them in prose). Every probe is bounded
# to seconds — the point is to replace the 75-second GeecsDb hang and the
# blind "is the trigger even firing?" scan attempt with one cheap command.
#
#   scripts/lab_status.sh                 # tier 1: network only (safe anywhere)
#   scripts/lab_status.sh --hardware      # + tier 2: read-only CA liveness
#   scripts/lab_status.sh --hardware --experiment Undulator
#
# Tier 1 is pure TCP/HTTP/filesystem — safe to run at any time, on or off
# network. Tier 2 performs READ-ONLY Channel Access gets (gateway heartbeat,
# device count) through the GeecsBluesky env; it never writes a PV.
set -u  # deliberately not -e: a failed probe is a *finding*, not an error

CONFIG="$HOME/.config/geecs_python_api/config.ini"
TCP_TIMEOUT=2   # seconds per port probe
CA_TIMEOUT=3    # seconds per CA read (tier 2)

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

ini_get() {  # ini_get SECTION KEY — first match, trimmed
    awk -F'=' -v s="[$1]" -v k="$2" '
        $0 == s { insec = 1; next }
        /^\[/   { insec = 0 }
        insec && $1 ~ "^[ \t]*"k"[ \t]*$" { gsub(/^[ \t]+|[ \t\r]+$/, "", $2); print $2; exit }
    ' "$CONFIG" 2>/dev/null
}

# --- resolve endpoints from the shared config (one source of truth) --------
TILED_URI="$(ini_get tiled uri)"
LAB_HOST=""
TILED_PORT="8000"
if [ -n "$TILED_URI" ]; then
    LAB_HOST="$(printf '%s' "$TILED_URI" | sed -E 's|^[a-z]+://||; s|[:/].*$||')"
    p="$(printf '%s' "$TILED_URI" | sed -nE 's|^[a-z]+://[^:/]+:([0-9]+).*|\1|p')"
    if [ -n "$p" ]; then TILED_PORT="$p"; fi
fi
# The DB server, Tiled server, and CA gateway share one box (see
# GeecsCAGateway/DEPLOYMENT.md "one box" section) — derive all from tiled uri.
DB_PORT=3306
CA_PORT=5064
DATA_ROOT="$(ini_get Paths GEECS_DATA_LOCAL_BASE_PATH)"
if [ -z "$EXPERIMENT" ]; then EXPERIMENT="$(ini_get Experiment expt)"; fi
if [ -z "$EXPERIMENT" ]; then EXPERIMENT="$(ini_get Experiment exp_name)"; fi

ok()   { printf '  [ OK ] %s\n' "$1"; }
bad()  { printf '  [DOWN] %s\n' "$1"; }
skip() { printf '  [ -- ] %s\n' "$1"; }

port_open() {  # port_open HOST PORT — hard wall-clock bound, no exceptions
    # nc's -G/-w flags do not reliably bound a SYN into a blackholed route
    # (the half-up-VPN case this script exists for), so use bash /dev/tcp
    # with an explicit watchdog kill.
    ( exec 3<>"/dev/tcp/$1/$2" ) 2>/dev/null &
    local probe=$!
    ( sleep "$TCP_TIMEOUT"; kill -9 "$probe" 2>/dev/null ) &
    local watchdog=$!
    wait "$probe" 2>/dev/null
    local rc=$?
    kill "$watchdog" 2>/dev/null
    wait "$watchdog" 2>/dev/null
    return "$rc"
}

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
if port_open "$LAB_HOST" "$DB_PORT"; then
    ok "MySQL       $LAB_HOST:$DB_PORT"
else
    bad "MySQL       $LAB_HOST:$DB_PORT — GeecsDb calls would hang ~75 s; do not make them"
    NET_UP=0
fi
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
    exit 1
fi
if ! poetry -C "$BLUESKY_DIR" env info --path >/dev/null 2>&1; then
    bad "GeecsBluesky poetry env not installed — tier 2 needs its aioca (see /env-doctor)"
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
    from geecs_ca_gateway.pv_naming import pv_name

    prefix = pv_name(experiment, "CAGateway")
    try:
        heartbeat = await read(f"{prefix}:heartbeat")
        connected = await read(f"{prefix}:devices_connected")
        version = await read(f"{prefix}:version")
    except Exception as exc:  # noqa: BLE001 — a failed probe is a finding
        print(f"  [DOWN] gateway PVs unreadable ({type(exc).__name__}: {exc})")
        print("         network may be up while the gateway service is not")
        raise SystemExit(1)
    print(f"  [ OK ] gateway alive: heartbeat={int(heartbeat)}, "
          f"devices_connected={int(connected)}, version={version}")
    if int(connected) == 0:
        print("  [WARN] zero devices connected — GEECS side likely down")


asyncio.run(main())
PY
status=$?
if [ "$status" -eq 0 ]; then
    echo "hardware: gateway UP — note this does NOT prove the trigger is firing;"
    echo "a scan still needs the laser/DG645 state right (see the /lab-status skill)"
fi
exit "$status"
