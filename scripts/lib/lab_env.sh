#!/usr/bin/env bash
# lab_env.sh — the plumbing scripts/lab_status.sh and scripts/fleet_status.sh
# share: the client config.ini reader, the endpoints derived from it, the
# line printers, and (via net_probes.sh) the bounded probes. Sourced, never
# executed.
#
#   . "$(dirname "$0")/lib/lab_env.sh"
#
# Facility values come from ~/.config/geecs_python_api/config.ini (the one
# client contract); nothing lab-specific lives here beyond port numbers that
# the fleet map (docs/platform/fleet_map.md) treats as constants.

# GEECS_CONFIG_INI overrides the path (tests); never a bare CONFIG, which
# build tooling commonly exports.
CONFIG="${GEECS_CONFIG_INI:-$HOME/.config/geecs_python_api/config.ini}"
TCP_TIMEOUT="${TCP_TIMEOUT:-2}"   # seconds per port probe / HTTP get (net_probes.sh reads it)

ini_get() {  # ini_get SECTION KEY — first match, trimmed
    awk -F'=' -v s="[$1]" -v k="$2" '
        $0 == s { insec = 1; next }
        /^\[/   { insec = 0 }
        insec && $1 ~ "^[ \t]*"k"[ \t]*$" { gsub(/^[ \t]+|[ \t\r]+$/, "", $2); print $2; exit }
    ' "$CONFIG" 2>/dev/null
}

# --- endpoints from config.ini (never hardcode hosts in a script) ---------
# The DB server, Tiled server, and CA gateway share one box (GeecsCAGateway/
# DEPLOYMENT.md "one box") — the lab server host is derived from [tiled] uri.
TILED_URI="$(ini_get tiled uri)"
LAB_HOST="$(printf '%s' "$TILED_URI" | sed -E 's|^[a-z]+://||; s|[:/].*$||')"
TILED_PORT="$(printf '%s' "$TILED_URI" | sed -nE 's|^[a-z]+://[^:/]+:([0-9]+).*|\1|p')"
TILED_PORT="${TILED_PORT:-8000}"
WORKER_HOST="$(ini_get qserver host)"          # the queueserver worker ([qserver] host)
DATA_ROOT="$(ini_get Paths GEECS_DATA_LOCAL_BASE_PATH)"
DB_PORT=3306
CA_PORT=5064

config_experiment() {  # [Experiment] expt, else the legacy exp_name key
    local e; e="$(ini_get Experiment expt)"
    [ -n "$e" ] || e="$(ini_get Experiment exp_name)"
    printf '%s' "$e"
}

ok()   { printf '  [ OK ] %s\n' "$1"; }
bad()  { printf '  [DOWN] %s\n' "$1"; }
warn() { printf '  [WARN] %s\n' "$1"; }
skip() { printf '  [ -- ] %s\n' "$1"; }
info() { printf '         %s\n' "$1"; }

# bounded / port_open / mysql_probe — the reachability probes (port_open
# refuses the MySQL port; the DB is probed only by a completed handshake).
# shellcheck source=net_probes.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/net_probes.sh"
