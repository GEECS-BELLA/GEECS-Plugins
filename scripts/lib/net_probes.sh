# net_probes.sh — the bounded reachability probes shared by scripts/lab_status.sh
# and scripts/fleet_status.sh. Source it; do not execute it.
#
#   bounded SECS CMD...       hard wall-clock bound around any command (macOS has no `timeout`)
#   port_open HOST PORT       bare TCP connect, bounded — REFUSES port 3306 (see below)
#   mysql_probe HOST PORT     handshake-completing MySQL probe; rc 0 ok / 1 down / 3 blocked / 4 no connector
#
# Why MySQL is special (#790): the server counts every connection that opens
# TCP to 3306 and drops without completing the handshake against
# max_connect_errors (default 100) and then refuses the host with error 1129
# until an admin runs FLUSH HOSTS — and it sees every VPN client as the VPN
# pool's ONE NAT address, so a bare /dev/tcp or nc probe in a watch loop
# blocked the DB for the whole pool (2026-09-04). A completed handshake — even
# a refused login — is not counted. So the DB probe is scripts/mysql_probe.py,
# and the bare probe here refuses that port outright. Callers may set
# TCP_TIMEOUT (seconds per probe, default 2).

MYSQL_PORT=3306
_NET_PROBES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_NET_PROBES_REPO="$(cd "$_NET_PROBES_DIR/../.." && pwd)"

bounded() {  # bounded SECS CMD... — hard wall-clock bound
    local secs="$1"; shift
    # Explicit stdin redirect: a background job in a non-interactive shell
    # otherwise gets /dev/null, which would starve `ssh bash -s` / `python -`.
    "$@" </dev/stdin &
    local job=$!
    # Watchdog detached from our stdout so a command substitution around
    # bounded() does not wait out the full timeout on the held pipe.
    ( sleep "$secs"; kill -9 "$job" 2>/dev/null ) >/dev/null 2>&1 </dev/null &
    local watchdog=$!
    wait "$job" 2>/dev/null
    local rc=$?
    kill "$watchdog" 2>/dev/null
    wait "$watchdog" 2>/dev/null
    return "$rc"
}

port_open() {  # port_open HOST PORT — bare TCP connect, no handshake; never for MySQL
    if [ "$2" = "$MYSQL_PORT" ]; then
        echo "port_open: refusing a bare TCP probe of MySQL port $2 — it counts toward the server's host block (error 1129); use mysql_probe (scripts/mysql_probe.py)" >&2
        return 2
    fi
    # nc's -G/-w flags do not reliably bound a SYN into a blackholed route
    # (the half-up-VPN case), so use bash /dev/tcp with an explicit watchdog.
    bounded "${TCP_TIMEOUT:-2}" bash -c "exec 3<>/dev/tcp/$1/$2" 2>/dev/null
}

_probe_python() {  # _probe_python MODULE — first interpreter that imports MODULE
    # GEECS_PROBE_PYTHON pins one (tests, a worktree without envs); otherwise
    # GEECS-Core owns the connector (and the credential lookup), GeecsBluesky
    # depends on it, the root docs env carries it transitively, and a PATH
    # python3 is the last resort. In-project .venv first (no poetry round
    # trip), then poetry's env path.
    local d p
    if [ -n "${GEECS_PROBE_PYTHON:-}" ]; then
        if "$GEECS_PROBE_PYTHON" -c "import $1" 2>/dev/null; then echo "$GEECS_PROBE_PYTHON"; return 0; fi
        echo "GEECS_PROBE_PYTHON=$GEECS_PROBE_PYTHON cannot import $1" >&2
        return 1
    fi
    for d in GEECS-Core GeecsBluesky .; do
        p="$_NET_PROBES_REPO/$d/.venv/bin/python"
        if [ ! -x "$p" ]; then
            p="$(poetry -C "$_NET_PROBES_REPO/$d" env info --path 2>/dev/null)/bin/python"
        fi
        if [ -x "$p" ] && "$p" -c "import $1" 2>/dev/null; then echo "$p"; return 0; fi
    done
    if command -v python3 >/dev/null 2>&1 && python3 -c "import $1" 2>/dev/null; then echo python3; return 0; fi
    return 1
}

mysql_probe() {  # mysql_probe HOST PORT — prints the probe's status line; rc per scripts/mysql_probe.py
    local py
    py="$(_probe_python mysql.connector)" || { echo "no-connector no interpreter with mysql-connector-python (GEECS-Core / GeecsBluesky poetry env; see /env-doctor)"; return 4; }
    # Bound = connect timeout + interpreter start-up and imports.
    bounded "$(( ${TCP_TIMEOUT:-2} + 10 ))" "$py" "$_NET_PROBES_DIR/../mysql_probe.py" --host "$1" --port "$2" --timeout "${TCP_TIMEOUT:-2}"
}
