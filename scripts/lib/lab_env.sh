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
# The DB server and the CA gateway share one box (GeecsCAGateway/DEPLOYMENT.md
# "one box"): LAB_HOST is the first [epics] ca_addr_list entry. Tiled has a
# host of its own ([tiled] uri) since the services moved off that box; a
# config with no ca_addr_list keeps the old one-box reading (the Tiled host).
url_host() { printf '%s' "$1" | sed -E 's|^[a-z]+://||; s|[:/].*$||'; }            # url_host URL — the host part
url_port() { printf '%s' "$1" | sed -nE 's|^[a-z]+://[^:/]+:([0-9]+).*|\1|p'; }    # url_port URL — the explicit port, or empty
TILED_URI="$(ini_get tiled uri)"
TILED_HOST="$(url_host "$TILED_URI")"
# The gateway is taken to be the FIRST entry (commas or spaces separate, as
# geecs_core.epics_env reads it); the probes print that they took it so.
CA_ADDR_FIRST="$(ini_get epics ca_addr_list | tr ',' ' ' | awk '{print $1}')"
LAB_HOST="${CA_ADDR_FIRST%%:*}"
# An explicit host:port entry names the gateway's CA port (EPICS WAN form).
CA_PORT=""
case "$CA_ADDR_FIRST" in *:*) CA_PORT="${CA_ADDR_FIRST##*:}" ;; esac
LAB_HOST="${LAB_HOST:-$TILED_HOST}"
TILED_PORT="$(url_port "$TILED_URI")"
TILED_PORT="${TILED_PORT:-8000}"
WORKER_HOST="$(ini_get qserver host)"          # the queueserver worker ([qserver] host)
DATA_ROOT="$(ini_get Paths GEECS_DATA_LOCAL_BASE_PATH)"
# The Archiver Appliance ([archiver] url, optional — a site without one has
# no key, and the probes print a skip row rather than a DOWN).
ARCHIVER_URL="$(ini_get archiver url)"
ARCHIVER_HOST="$(url_host "$ARCHIVER_URL")"
ARCHIVER_PORT="$(url_port "$ARCHIVER_URL")"
ARCHIVER_PORT="${ARCHIVER_PORT:-17665}"
archiver_version() {  # archiver_version HOST PORT — the appliance's version from its management API, or empty
    bounded "$TCP_TIMEOUT" curl -s -m "$TCP_TIMEOUT" "http://$1:$2/mgmt/bpl/getVersions" | sed -nE 's/.*"mgmt_version":"Archiver Appliance Version ([^"]+)".*/\1/p'
}
DB_PORT=3306
CA_PORT="${CA_PORT:-5064}"

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
