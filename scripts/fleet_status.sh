#!/usr/bin/env bash
# fleet_status.sh — what code is each fleet service actually running?
#
# The fleet map (docs/platform/fleet_map.md) is the intended picture; this
# script is the observed one. During rapid development a host may run a
# feature-branch checkout, a checkout may have moved under a service that
# was never restarted, or a pyproject bump may never have been installed —
# none of which the doc can tell you. Every probe is bounded and read-only:
# it never writes a PV, never restarts a unit, never pulls a checkout (the
# one write is `git write-tree` in a clone with staged changes, which adds an
# unreferenced tree object to that clone's object store — harmless, gc'd).
#
#   scripts/fleet_status.sh                       # full picture (needs the lab)
#   scripts/fleet_status.sh --local-only          # just the local checkouts
#   scripts/fleet_status.sh --no-ssh              # service self-reports only
#   scripts/fleet_status.sh --ssh 192.168.6.14=<alias>   # ssh alias override
#   scripts/fleet_status.sh --experiment Undulator --no-fetch
#   scripts/fleet_status.sh --summary             # one box table + attention list
#   scripts/fleet_status.sh --watch 300           # dashboard pane: --summary every 300 s (--full for the log)
#
# Stages (each gated on the previous one):
#   0. reachability — scripts/lab_status.sh tier 1 (the same gate /lab-status
#      uses); network DOWN => stage 3 only, remote = UNKNOWN
#   1. self-reported versions over the services' own protocols: Tiled
#      (HTTP), Data Portal (/health), MCP (port), CA gateway (lab_status.sh
#      --hardware, read-only CA), PVA image fleet (read-only pvAccess gets)
#   2. host checkouts over ssh: every geecs-* systemd unit -> the clone it
#      runs from -> branch / sha / dirty / installed-vs-pyproject version /
#      "checkout moved after the service started"
#   3. local cross-reference: which local worktree/branch holds each
#      deployed sha, and how far each is from origin/master
#
# Endpoints come from ~/.config/geecs_python_api/config.ini (the one client
# contract) — the lab server from [tiled] uri, the worker from [qserver]
# host, ssh aliases from ~/.ssh/config by matching HostName. Nothing lab-
# specific is hardcoded here beyond the fleet map's port numbers.
set -u  # deliberately not -e: a failed probe is a *finding*, not an error

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$HOME/.config/geecs_python_api/config.ini"
PORTAL_PORT=8200       # fleet map: GEECS Data Portal
MCP_PORT=8100          # fleet map: GEECS-MCP HTTP mode
TCP_TIMEOUT=2          # seconds per port probe / HTTP get
SSH_TIMEOUT=25         # seconds per host (one ssh call per host)
PVA_TIMEOUT=2          # seconds per PVA get
FETCH_TIMEOUT=20       # seconds for the local `git fetch origin`
PVA_ROSTER="$REPO_ROOT/GeecsPvaGateway/deploy/fleet_status.bob"  # the checked-in fleet roster

EXPERIMENT=""
DO_SSH=1
DO_FETCH=1
LOCAL_ONLY=0
WATCH=0
SUMMARY=0
FULL=0
declare -a SSH_OVERRIDES=()
ORIG_ARGS=("$@")
while [ $# -gt 0 ]; do
    case "$1" in
        --watch) shift; WATCH="${1:-300}" ;;
        --summary) SUMMARY=1 ;;
        --full) FULL=1 ;;
        --experiment) shift; EXPERIMENT="${1:-}" ;;
        --no-ssh) DO_SSH=0 ;;
        --no-fetch) DO_FETCH=0 ;;
        --local-only) LOCAL_ONLY=1 ;;
        --ssh) shift; SSH_OVERRIDES+=("${1:-}") ;;
        *) echo "usage: fleet_status.sh [--experiment NAME] [--no-ssh] [--no-fetch] [--local-only] [--ssh IP=ALIAS]... [--summary|--full] [--watch SECS]" >&2; exit 2 ;;
    esac
    shift
done

# Dashboard mode: a persistent terminal pane (cmux/tmux) that reruns the
# one-shot probe on an interval. No daemon, no state — the same read-only
# run, redrawn. Ctrl-C ends it.
if [ "$WATCH" != "0" ]; then
    args=()
    for a in "${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}"; do
        case "$a" in --watch) skip_next=1; continue ;; esac
        if [ "${skip_next:-0}" = "1" ]; then skip_next=0; continue; fi
        args+=("$a")
    done
    [ "$FULL" -eq 0 ] && args+=("--summary")
    while :; do
        COLUMNS="$(tput cols 2>/dev/null || echo 100)" out="$("$0" "${args[@]+"${args[@]}"}" 2>&1)"
        clear
        echo "fleet status — $(date '+%F %H:%M:%S')  (every ${WATCH}s, Ctrl-C to stop)"
        echo
        printf '%s\n' "$out"
        sleep "$WATCH"
    done
fi

ini_get() {  # ini_get SECTION KEY — first match, trimmed
    awk -F'=' -v s="[$1]" -v k="$2" '
        $0 == s { insec = 1; next }
        /^\[/   { insec = 0 }
        insec && $1 ~ "^[ \t]*"k"[ \t]*$" { gsub(/^[ \t]+|[ \t\r]+$/, "", $2); print $2; exit }
    ' "$CONFIG" 2>/dev/null
}

ok()   { printf '  [ OK ] %s\n' "$1"; }
bad()  { printf '  [DOWN] %s\n' "$1"; }
warn() { printf '  [WARN] %s\n' "$1"; }
skip() { printf '  [ -- ] %s\n' "$1"; }
info() { printf '         %s\n' "$1"; }

bounded() {  # bounded SECS CMD... — hard wall-clock bound (macOS has no `timeout`)
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

port_open() {  # port_open HOST PORT
    bounded "$TCP_TIMEOUT" bash -c "exec 3<>/dev/tcp/$1/$2" 2>/dev/null
}

# Every probe also appends a tab-separated key=value record (with a role=)
# to REC_FILE; scripts/fleet_table.py renders those as the --summary table.
REC_FILE="$(mktemp -t fleet_rec.XXXXXX)"
LOG_FILE="$(mktemp -t fleet_log.XXXXXX)"
trap 'rm -f "$REC_FILE" "$LOG_FILE"' EXIT
rec() { printf '%s\n' "$1" >> "$REC_FILE"; }
if [ "$SUMMARY" -eq 1 ]; then
    exec 3>&1 1>"$LOG_FILE"   # the verbose log goes to a file; the table is printed at the end
fi

# --- endpoints from config.ini (never hardcode hosts here) -----------------
TILED_URI="$(ini_get tiled uri)"
LAB_HOST="$(printf '%s' "$TILED_URI" | sed -E 's|^[a-z]+://||; s|[:/].*$||')"
TILED_PORT="$(printf '%s' "$TILED_URI" | sed -nE 's|^[a-z]+://[^:/]+:([0-9]+).*|\1|p')"
TILED_PORT="${TILED_PORT:-8000}"
WORKER_HOST="$(ini_get qserver host)"
if [ -z "$EXPERIMENT" ]; then EXPERIMENT="$(ini_get Experiment expt)"; fi
if [ -z "$EXPERIMENT" ]; then EXPERIMENT="$(ini_get Experiment exp_name)"; fi

# Hosts to visit over ssh: the lab server and the worker, deduplicated
# (today they are one box; the fleet map says which services live where).
HOSTS=""
for h in "$LAB_HOST" "$WORKER_HOST"; do
    [ -n "$h" ] || continue
    case " $HOSTS " in *" $h "*) ;; *) HOSTS="$HOSTS $h" ;; esac
done

ssh_target() {  # ssh_target IP — --ssh override, else ~/.ssh/config alias whose HostName is IP, else IP
    local ip="$1" ov
    for ov in "${SSH_OVERRIDES[@]+"${SSH_OVERRIDES[@]}"}"; do
        case "$ov" in "$ip="*) printf '%s' "${ov#*=}"; return ;; esac
    done
    local alias
    alias="$(awk -v ip="$ip" '
        tolower($1) == "host" && NF == 2 { h = $2 }
        tolower($1) == "hostname" && $2 == ip && h != "" { print h; exit }
    ' "$HOME/.ssh/config" 2>/dev/null)"
    printf '%s' "${alias:-$ip}"
}

# Deployed shas found in stage 1/2, cross-referenced locally in stage 3.
# Format per line: "<role><TAB><sha><TAB><kind>" (kind: head = the clone's
# HEAD, disk = the commit the files on disk match when they differ).
DEPLOYED_SHAS=""
note_sha() { DEPLOYED_SHAS="$DEPLOYED_SHAS
$1	$2	$3"; }

# ===========================================================================
NET_UP=0
if [ "$LOCAL_ONLY" -eq 0 ]; then
    echo "== Stage 0: reachability (scripts/lab_status.sh, tier 1) =="
    lab_out="$("$REPO_ROOT/scripts/lab_status.sh" 2>&1)"
    printf '%s\n' "$lab_out" | grep -v '^== \|^network:'
    if printf '%s\n' "$lab_out" | grep -q '^network: UP'; then
        NET_UP=1
        echo "  network: UP"
    else
        echo "  network: DOWN or partial — remote fleet state is UNKNOWN from here (nothing below is a service verdict)"
    fi
    echo
fi

# ===========================================================================
if [ "$NET_UP" -eq 1 ]; then
    echo "== Stage 1: what each service says about itself (read-only) =="
    # Tiled — pip-installed, no checkout; its library version is the whole story.
    tv="$(bounded "$TCP_TIMEOUT" curl -s -m "$TCP_TIMEOUT" "http://$LAB_HOST:$TILED_PORT/api/v1/" | sed -nE 's/.*"library_version":"([^"]+)".*/\1/p')"
    if [ -n "$tv" ]; then ok "Tiled        $LAB_HOST:$TILED_PORT  tiled $tv"; rec "role=Tiled	state=ok	version=$tv	checkout=pip install"
    else bad "Tiled        $LAB_HOST:$TILED_PORT"; rec "role=Tiled	state=down	checkout=pip install"; fi

    # Data Portal — /health carries ok + catalog probe + installed version.
    PORTAL_HOST="${WORKER_HOST:-$LAB_HOST}"
    ph="$(bounded "$TCP_TIMEOUT" curl -s -m "$TCP_TIMEOUT" "http://$PORTAL_HOST:$PORTAL_PORT/health")"
    if [ -n "$ph" ]; then
        pv="$(printf '%s' "$ph" | sed -nE 's/.*"version": *"([^"]+)".*/\1/p')"
        pok="$(printf '%s' "$ph" | sed -nE 's/.*"ok": *(true|false).*/\1/p')"
        pcat="$(printf '%s' "$ph" | sed -nE 's/.*"catalog": *"([^"]*)".*/\1/p')"
        if [ "$pok" = "true" ]; then
            ok "Data Portal  $PORTAL_HOST:$PORTAL_PORT  geecs-data-portal ${pv:-?}  (catalog: ${pcat:-?})"
            rec "role=Data Portal	state=ok	version=${pv:-}"
        else
            warn "Data Portal  $PORTAL_HOST:$PORTAL_PORT  geecs-data-portal ${pv:-?}  up but catalog NOT ok (${pcat:-?})"
            rec "role=Data Portal	state=ok	version=${pv:-}	note=catalog not ok"
        fi
    else
        bad "Data Portal  $PORTAL_HOST:$PORTAL_PORT  (no /health answer)"
        rec "role=Data Portal	state=down"
    fi

    # Queueserver — ZMQ, no cheap self-report; port liveness (stage 2 finds the process).
    QS_HOST="${WORKER_HOST:-$LAB_HOST}"
    if port_open "$QS_HOST" 60615; then ok "Queueserver  $QS_HOST:60615  RE Manager control port listening"; rec "role=Queueserver RE Manager	state=ok"
    else bad "Queueserver  $QS_HOST:60615  RE Manager not listening"; rec "role=Queueserver RE Manager	state=down"; fi
    if port_open "$QS_HOST" 5568; then ok "Doc proxy    $QS_HOST:5568   document stream listening"; rec "role=Bluesky doc proxy	state=ok"
    else bad "Doc proxy    $QS_HOST:5568   not listening"; rec "role=Bluesky doc proxy	state=down"; fi

    # MCP HTTP mode — no version endpoint; port liveness only (stage 2 reads the venv).
    if port_open "${WORKER_HOST:-$LAB_HOST}" "$MCP_PORT"; then
        ok "GEECS-MCP    ${WORKER_HOST:-$LAB_HOST}:$MCP_PORT  listening (version via ssh below)"
        rec "role=GEECS-MCP	state=ok"
    else
        skip "GEECS-MCP    ${WORKER_HOST:-$LAB_HOST}:$MCP_PORT  not listening (HTTP mode is 'pending deploy' on the fleet map)"
        rec "role=GEECS-MCP	state=absent	note=not listening (pending deploy)"
    fi

    # CA gateway — reuse /lab-status tier 2 (read-only CA gets of heartbeat,
    # devices_connected, version). Contract: it prints "version=<str>".
    if [ -n "$EXPERIMENT" ]; then
        hw="$("$REPO_ROOT/scripts/lab_status.sh" --hardware --experiment "$EXPERIMENT" 2>&1 | sed -n '/Tier 2/,$p')"
        alive="$(printf '%s\n' "$hw" | grep -m1 'gateway alive')"
        if [ -n "$alive" ]; then
            ok "CA gateway   $LAB_HOST:5064  ${alive#*] }"
            gver="$(printf '%s' "$alive" | sed -nE 's/.*version=([^, ]+).*/\1/p')"
            gdev="$(printf '%s' "$alive" | sed -nE 's/.*devices_connected=([0-9]+).*/\1/p')"
            rec "role=CA gateway	state=ok	version=$gver	note=$gdev devices connected"
            printf '%s\n' "$hw" | grep '\[WARN\]' | sed 's/^ *\[WARN\] /  [WARN] CA gateway: /'
        else
            err="$(printf '%s\n' "$hw" | grep -m1 -E 'unreadable|not installed|no experiment' )"
            bad "CA gateway   $LAB_HOST:5064  ${err#*] }"
            rec "role=CA gateway	state=down"
        fi
    else
        skip "CA gateway   no experiment name (config.ini [Experiment] expt, or --experiment NAME)"
    fi

    # PVA image fleet — one gateway per camera server; roster = the checked-in
    # Phoebus fleet screen (the DB-driven roster is still owed). Read-only gets.
    if [ -f "$PVA_ROSTER" ]; then
        pvs="$(grep -o 'pv_name>pva://[^<]*:version' "$PVA_ROSTER" | sed 's|pv_name>pva://||' | sort -u)"
        if [ -n "$pvs" ] && poetry -C "$REPO_ROOT/GeecsPvaGateway" env info --path >/dev/null 2>&1; then
            PVA_PVS="$pvs" PVA_TIMEOUT="$PVA_TIMEOUT" REC_FILE="$REC_FILE" bounded 60 poetry -C "$REPO_ROOT/GeecsPvaGateway" run python - <<'PY'
import os
import re

pvs = os.environ["PVA_PVS"].split()
timeout = float(os.environ["PVA_TIMEOUT"])
# Host IPs are encoded in the PV name (dots -> underscores, PV_CONTRACT).
hosts = [re.sub(r"_", ".", pv.split(":")[2]) for pv in pvs]
# Unicast search to each camera server: UDP broadcast does not cross a VPN.
os.environ["EPICS_PVA_ADDR_LIST"] = " ".join(hosts)
os.environ["EPICS_PVA_AUTO_ADDR_LIST"] = "NO"
from p4p.client.thread import Context  # noqa: E402

versions = {}
down = []
with Context("pva", unwrap=False) as ctx:  # raw Values: str(NT wrapper) carries a timestamp
    for pv, host in zip(pvs, hosts):
        base = pv[: -len(":version")]
        try:
            ver = str(ctx.get(pv, timeout=timeout)["value"])
            beats = int(ctx.get(base + ":heartbeat", timeout=timeout)["value"])
        except Exception as exc:  # noqa: BLE001 — a failed probe is a finding
            print(f"  [DOWN] PVA gateway  {host:<15}  ({type(exc).__name__})")
            down.append(host)
            continue
        versions.setdefault(ver, []).append(host)
        print(f"  [ OK ] PVA gateway  {host:<15}  geecs-pva-gateway {ver}  heartbeat={beats}")
if len(versions) > 1:
    print("  [WARN] PVA fleet runs mixed versions — a rollout is incomplete or a box missed its pull-on-restart:")
    for ver, hs in sorted(versions.items()):
        print(f"         {ver}: {', '.join(hs)}")
n_ok = sum(len(hs) for hs in versions.values())
ver_txt = ", ".join(f"{v} ×{len(hs)}" for v, hs in sorted(versions.items())) or "?"
note = f"{n_ok} up" + (f", {len(down)} unreachable: {' '.join(down)}" if down else "")
if len(versions) > 1:
    note += "|MIXED versions"
state = "ok" if n_ok else "down"
with open(os.environ["REC_FILE"], "a") as fh:
    fh.write(f"role=PVA image gateways\tstate={state}\truns=NSSM\tcheckout=share clone\tversion={ver_txt}\tnote={note}\n")
PY
        else
            skip "PVA fleet    GeecsPvaGateway poetry env not installed (needs p4p; see /env-doctor)"
        fi
    else
        skip "PVA fleet    roster file missing ($PVA_ROSTER)"
    fi
    echo
fi

# ===========================================================================
# Stage 2 runs this on each host. Output: one "role=..." record per service
# (key=value pairs, tab-separated), which the local side formats. Services
# are discovered two ways and deduplicated by pid: systemd units (system AND
# user scope — a dev-deployed daemon often lives in `systemctl --user`) and
# the process that owns each fleet port (a queueserver started by hand in
# tmux has no unit at all — it must still show up, as UNMANAGED). Read-only:
# git queries, systemctl show, /proc reads, a venv python -c for the
# installed version. `poetry` is looked up in ~/.local/bin because plain ssh
# has no login PATH (fleet map bootstrap gotcha 4).
REMOTE_SNIPPET='
set -u
command -v systemctl >/dev/null 2>&1 || { echo "nosystemd"; exit 0; }
# fleet map ports -> role label (what a listener on that port is)
role_for_port() { case "$1" in
    5064) echo "CA gateway";; 8000) echo "Tiled";; 8200) echo "Data Portal";; 8100) echo "GEECS-MCP";;
    60615) echo "Queueserver RE Manager";; 5568) echo "Bluesky doc proxy";; *) echo "port $1";; esac; }
role_for_unit() { case "$1" in
    geecs-ca-gateway*) echo "CA gateway";; tiled*) echo "Tiled";; geecs-data-portal*) echo "Data Portal";;
    geecs-mcp*) echo "GEECS-MCP";; geecs-qserver*) echo "Queueserver RE Manager";; geecs-capture*) echo "Capture daemon";;
    *) echo "$1";; esac; }
FLEET_PORTS="5064 8000 8200 8100 60615 5568"
SEEN=" "
reflog_ts() {  # unix time HEAD last moved (checkout/pull/reset), from the reflog
    local f; f="$(git -C "$1" rev-parse --git-path logs/HEAD 2>/dev/null)"
    [ -f "$f" ] && tail -1 "$f" | sed -E "s/^[0-9a-f]+ [0-9a-f]+ .*> ([0-9]+) [-+][0-9]{4}\t.*/\1/"
}
pkgdir_up() {  # nearest pyproject.toml at or above $1, BELOW the clone root $2 (the root pyproject is the docs env, never a service)
    local d="$1"
    while [ -n "$d" ] && [ "$d" != "/" ] && [ "$d" != "$2" ]; do
        [ -f "$d/pyproject.toml" ] && { echo "$d"; return; }
        d="$(dirname "$d")"
    done
}
# Which distribution IS this process? From its own command line (-m module,
# a console script, or the poetry import_module shim), resolved by the
# interpreter that runs it — never by guessing from a directory. Prints
# dist / installed version / source dir (direct_url) / editable yes|no.
HELPER="$(mktemp)"; trap "rm -f $HELPER" EXIT
cat > "$HELPER" <<"PYH"
import importlib.metadata as m
import json
import os
import re
import sys

pid = sys.argv[1]
try:
    raw = open("/proc/%s/cmdline" % pid, "rb").read()
except OSError:
    sys.exit(0)
args = [a.decode(errors="replace") for a in raw.split(b"\0") if a]
mod = None
if "-m" in args and args.index("-m") + 1 < len(args):
    mod = args[args.index("-m") + 1]
elif "-c" in args and args.index("-c") + 1 < len(args):
    mm = re.search(r"import_module\(.([\w.]+)", args[args.index("-c") + 1])
    if mm:
        mod = mm.group(1)
dist = None
if mod:
    dist = (m.packages_distributions().get(mod.split(".")[0]) or [None])[0]
elif len(args) > 1 and not args[1].startswith("-"):
    name = os.path.basename(args[1])
    for ep in m.entry_points(group="console_scripts"):
        if ep.name == name:
            dist = ep.dist.name
            break
if not dist:
    sys.exit(0)
d = m.distribution(dist)
src = ""
editable = "no"
t = d.read_text("direct_url.json")
if t:
    j = json.loads(t)
    url = j.get("url", "")
    src = url[7:] if url.startswith("file://") else ""
    if j.get("dir_info", {}).get("editable"):
        editable = "yes"
print(dist)
print(d.version)
print(src)
print(editable)
PYH
emit() {  # emit ROLE NAME MANAGED STATE PID CWD PYEXE
    local role="$1" name="$2" managed="$3" state="$4" pid="$5" cwd="$6" pyexe="$7"
    local since="" since_ts="" clone="" pkgdir="" venv="" baked="" moved=""
    if [ -n "$pid" ] && [ "$pid" != "0" ] && [ -d "/proc/$pid" ]; then
        since="$(ps -o lstart= -p "$pid" 2>/dev/null | sed -E "s/^ +//")"
        [ -n "$since" ] && since_ts="$(date -d "$since" +%s 2>/dev/null)"
        [ -z "$cwd" ] && cwd="$(readlink "/proc/$pid/cwd" 2>/dev/null)"
        [ -z "$pyexe" ] && pyexe="$(tr "\0" "\n" < "/proc/$pid/cmdline" 2>/dev/null | head -1)"
    fi
    local rec="role=$role\tsvc=$name\tmanaged=$managed\tstate=$state\tsince=${since:-?}"
    [ -n "$cwd" ] && [ -d "$cwd" ] && clone="$(git -C "$cwd" rev-parse --show-toplevel 2>/dev/null)"
    case "$pyexe" in */bin/python*) venv="$(dirname "$(dirname "$pyexe")")";; esac
    local h_dist="" h_ver="" h_src="" h_edit=""
    if [ -n "$venv" ] && [ -x "$venv/bin/python" ] && [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
        { read -r h_dist; read -r h_ver; read -r h_src; read -r h_edit; } < <("$venv/bin/python" "$HELPER" "$pid" 2>/dev/null)
    fi
    if [ -n "$h_src" ] && [ -d "$h_src" ]; then
        # The process names its own repo package (editable or baked install);
        # ITS clone is the truth, even if the process was launched from
        # another checkout directory.
        pkgdir="$h_src"
        src_clone="$(git -C "$h_src" rev-parse --show-toplevel 2>/dev/null)"
        [ -n "$src_clone" ] && clone="$src_clone"
        if [ "$h_edit" = "no" ]; then baked="$venv"; rec="$rec\tbaked=${venv/#$HOME/\~}"; fi
    elif [ -n "$clone" ]; then
        # A third-party entry point (start-re-manager, the 0MQ proxy): the
        # deployable is the repo package the process was launched from.
        pkgdir="$(pkgdir_up "$cwd" "$clone")"
    fi
    if [ -n "$clone" ]; then
        local branch sha full cdate staged unstaged
        branch="$(git -C "$clone" rev-parse --abbrev-ref HEAD 2>/dev/null)"; [ "$branch" = "HEAD" ] && branch="(detached)"
        sha="$(git -C "$clone" rev-parse --short=8 HEAD 2>/dev/null)"; full="$(git -C "$clone" rev-parse HEAD 2>/dev/null)"
        cdate="$(git -C "$clone" log -1 --format=%cs 2>/dev/null)"
        staged="$(git -C "$clone" diff --cached --name-only 2>/dev/null | wc -l | tr -d " ")"
        unstaged="$(git -C "$clone" diff --name-only 2>/dev/null | wc -l | tr -d " ")"
        moved="$(reflog_ts "$clone")"
        rec="$rec\tclone=${clone/#$HOME/\~}\tbranch=$branch\tsha=$sha\tfull=$full\tcommit_date=$cdate\tstaged=$staged\tunstaged=$unstaged"
        if [ "$staged" != "0" ]; then
            # HEAD can lie: a ref advanced without a checkout (or a staged
            # rollback) leaves the files on disk — what actually runs — at
            # another commit. Name it if it is one (one git log, one awk).
            local idx_tree match
            idx_tree="$(git -C "$clone" write-tree 2>/dev/null)"
            match="$(git -C "$clone" log --all -400 --format="%H %cs %T" 2>/dev/null | awk -v t="$idx_tree" "\$3 == t {print \$1, \$2; exit}")"
            if [ -n "$match" ]; then
                rec="$rec\tdisk=${match:0:8}\tdisk_full=${match%% *}\tdisk_date=${match##* }"
            fi
        fi
        if [ -z "${baked:-}" ] && [ -n "$moved" ] && [ -n "$since_ts" ] && [ "$moved" -gt "$since_ts" ]; then
            rec="$rec\tstale=checkout moved $(date -d @"$moved" "+%F %H:%M") after the process started"
        fi
    fi
    if [ -n "$pkgdir" ] && [ -f "$pkgdir/pyproject.toml" ]; then
        local pname pver iver
        pname="$(sed -nE "s/^name *= *\"([^\"]+)\".*/\1/p" "$pkgdir/pyproject.toml" | head -1)"
        pver="$(sed -nE "s/^version *= *\"([^\"]+)\".*/\1/p" "$pkgdir/pyproject.toml" | head -1)"
        rec="$rec\tpkg=$pname\tpyproject=$pver"
        if [ -n "$venv" ] && [ -x "$venv/bin/python" ] && [ -n "$pname" ]; then
            iver="$("$venv/bin/python" -c "import importlib.metadata as m; print(m.version(\"$pname\"))" 2>/dev/null)"
            rec="$rec\tinstalled=${iver:-?}"
            if [ -n "${baked:-}" ]; then
                # Baked install: the code that runs is what was installed, so
                # compare the checkout move against the install, not the process.
                distinfo="$(ls -d "$venv"/lib/python*/site-packages/${pname//-/_}-*.dist-info 2>/dev/null | head -1)"
                inst_ts=""; [ -n "$distinfo" ] && inst_ts="$(stat -c %Y "$distinfo" 2>/dev/null)"
                if [ -n "$inst_ts" ] && [ -n "${moved:-}" ] && [ "$moved" -gt "$inst_ts" ]; then
                    rec="$rec\tstale=checkout moved $(date -d @"$moved" "+%F %H:%M") after the baked install ($(date -d @"$inst_ts" "+%F %H:%M")) — pip reinstall + restart pending"
                elif [ -n "$inst_ts" ] && [ -n "$since_ts" ] && [ "$inst_ts" -gt "$since_ts" ]; then
                    rec="$rec\tstale=reinstalled $(date -d @"$inst_ts" "+%F %H:%M") after the process started — restart pending"
                fi
            fi
        fi
    fi
    [ -n "$pyexe" ] && rec="$rec\tpyexe=${pyexe/#$HOME/\~}"
    printf "%b\n" "$rec"
}
# 1) systemd units, system and user scope
for scope in "" "--user"; do
    units="$(systemctl $scope list-units --all --plain --no-legend "geecs-*" "tiled.service" 2>/dev/null | awk "{print \$1}")"
    for u in $units; do
        active="$(systemctl $scope show -p ActiveState --value "$u")/$(systemctl $scope show -p SubState --value "$u")"
        pid="$(systemctl $scope show -p MainPID --value "$u")"
        wd="$(systemctl $scope show -p WorkingDirectory --value "$u")"
        label="$u"; [ -n "$scope" ] && label="$u (user unit)"
        [ "$pid" != "0" ] && SEEN="$SEEN$pid "
        emit "$(role_for_unit "$u")" "$label" "systemd" "$active" "$pid" "$wd" ""
    done
done
# 2) whoever owns each fleet port, if not already listed via its unit
for port in $FLEET_PORTS; do
    pid="$(ss -ltnpH "sport = :$port" 2>/dev/null | sed -nE "s/.*pid=([0-9]+).*/\1/p" | head -1)"
    [ -n "$pid" ] || continue
    case "$SEEN" in *" $pid "*) continue;; esac
    SEEN="$SEEN$pid "
    unit="$(sed -nE "s#.*/([^/]+\.service)\$#\1#p" "/proc/$pid/cgroup" 2>/dev/null | head -1)"
    if [ -n "$unit" ]; then managed="systemd ($unit)"; else managed="UNMANAGED"; fi
    emit "$(role_for_port "$port")" "$(role_for_port "$port") :$port" "$managed" "running pid $pid" "$pid" "" ""
done
[ "$SEEN" = " " ] && echo "nounits"
exit 0
'

fmt_host_records() {  # stdin: service records -> pretty lines; side effect: note_sha
    local line
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        case "$line" in
            nosystemd) skip "no systemd on this host"; continue ;;
            nounits) skip "no geecs-* / tiled units and nothing listening on the fleet ports"; continue ;;
            role=*) ;;
            *) info "ssh: $line"; continue ;;   # known-hosts notices, remote warnings — not records
        esac
        rec "$line"
        local role svc managed state since clone branch sha full cdate staged unstaged stale pkg pyproject installed baked pyexe disk disk_full disk_date
        role=""; svc=""; managed=""; state=""; since=""; clone=""; branch=""; sha=""; full=""; cdate=""; staged=""; unstaged=""; stale=""; pkg=""; pyproject=""; installed=""; baked=""; pyexe=""; disk=""; disk_full=""; disk_date=""
        local IFS=$'\t' kv
        for kv in $line; do
            case "$kv" in
                role=*) role="${kv#*=}" ;; svc=*) svc="${kv#*=}" ;; managed=*) managed="${kv#*=}" ;; state=*) state="${kv#*=}" ;; since=*) since="${kv#*=}" ;;
                clone=*) clone="${kv#*=}" ;; branch=*) branch="${kv#*=}" ;; sha=*) sha="${kv#*=}" ;;
                full=*) full="${kv#*=}" ;; commit_date=*) cdate="${kv#*=}" ;; staged=*) staged="${kv#*=}" ;; unstaged=*) unstaged="${kv#*=}" ;;
                stale=*) stale="${kv#*=}" ;; pkg=*) pkg="${kv#*=}" ;; pyproject=*) pyproject="${kv#*=}" ;;
                installed=*) installed="${kv#*=}" ;; baked=*) baked="${kv#*=}" ;; pyexe=*) pyexe="${kv#*=}" ;;
                disk=*) disk="${kv#*=}" ;; disk_full=*) disk_full="${kv#*=}" ;; disk_date=*) disk_date="${kv#*=}" ;;
            esac
        done
        unset IFS
        local tag="[ OK ]"
        case "$state" in active/*|running*) ;; *) tag="[DOWN]" ;; esac
        printf '  %s %-34s %-16s since %s\n' "$tag" "$svc" "$state" "${since:-?}"
        [ "$managed" = "UNMANAGED" ] && warn "$svc: no systemd unit owns this process (started by hand — tmux/nohup?); it will not survive a reboot or crash"
        if [ -n "$clone" ]; then
            local d=""
            [ "${staged:-0}" != "0" ] && d="$d  STAGED: $staged file(s)"
            [ "${unstaged:-0}" != "0" ] && d="$d  UNSTAGED: $unstaged modified file(s)"
            info "$clone @ $branch $sha ($cdate)$d"
            note_sha "${role:-$svc}" "$full" "head"
            if [ -n "$disk" ]; then
                warn "$svc: files on disk are commit $disk ($disk_date), not HEAD $sha — HEAD moved without a checkout, or a staged rollback is pending; what RUNS is $disk. Confirm which tree is intended before touching (skill: the STAGED/UNSTAGED row)"
                note_sha "${role:-$svc}" "$disk_full" "disk"
            fi
        elif [ -n "$pyexe" ]; then
            info "no git clone behind this process (interpreter $pyexe)"
        fi
        [ -n "$baked" ] && info "baked venv $baked (non-editable install — the clone can move without changing the running code)"
        if [ -n "$pkg" ]; then
            if [ -n "$installed" ] && [ "$installed" != "$pyproject" ]; then
                warn "$pkg: pyproject says $pyproject but the venv has $installed installed — poetry install / pip reinstall pending"
            else
                info "$pkg $pyproject${installed:+ (installed $installed)}"
            fi
        fi
        if [ -n "$stale" ]; then
            case "$stale" in *pending*) warn "$svc: $stale" ;; *) warn "$svc: $stale — the running process predates the code on disk (restart pending?)" ;; esac
        fi
    done
}

if [ "$NET_UP" -eq 1 ] && [ "$DO_SSH" -eq 1 ]; then
    echo "== Stage 2: host checkouts (ssh, read-only) =="
    if [ -z "$HOSTS" ]; then
        skip "no hosts derivable from config.ini ([tiled] uri / [qserver] host)"
    fi
    for host in $HOSTS; do
        target="$(ssh_target "$host")"
        echo "host $host (ssh $target)"
        out="$(printf '%s' "$REMOTE_SNIPPET" | bounded "$SSH_TIMEOUT" ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$target" bash -s 2>&1)"
        rc=$?
        if [ "$rc" -ne 0 ] && ! printf '%s' "$out" | grep -q '^role='; then
            bad "ssh $target failed (rc=$rc): $(printf '%s' "$out" | head -1)"
            rec "role=host $host	state=ok	runs=?	checkout=ssh failed	note=$(printf '%s' "$out" | head -1 | tr '\t' ' ')"
            info "no key-based access from here? pass --ssh $host=<alias> or run the stage-2 snippet on the host by hand"
            continue
        fi
        fmt_host_records <<< "$out"   # here-string, not a pipe: note_sha must run in this shell
    done
    echo
elif [ "$NET_UP" -eq 1 ]; then
    echo "== Stage 2: host checkouts — skipped (--no-ssh) =="
    echo
fi

# ===========================================================================
echo "== Stage 3: local checkouts and cross-reference =="
if [ "$DO_FETCH" -eq 1 ]; then
    if bounded "$FETCH_TIMEOUT" git -C "$REPO_ROOT" fetch --quiet origin 2>/dev/null; then
        info "origin fetched"
    else
        warn "git fetch origin failed/timed out — distances below use the last-fetched origin/master"
    fi
fi
master_sha="$(git -C "$REPO_ROOT" rev-parse --short=8 origin/master 2>/dev/null)"
info "origin/master = ${master_sha:-?}"
echo "  local worktrees:"
git -C "$REPO_ROOT" worktree list 2>/dev/null | while IFS= read -r wl; do
    wpath="${wl%% *}"; rest="${wl#"$wpath"}"
    case "$wpath" in
        "$REPO_ROOT") rel="(this checkout)" ;;
        "$REPO_ROOT"/*) rel="${wpath#"$REPO_ROOT"/}" ;;
        *) rel="$wpath" ;;
    esac
    dirty="$(git -C "$wpath" status --porcelain --untracked-files=no 2>/dev/null | wc -l | tr -d ' ')"
    printf '    %-48s %s%s\n' "$rel" "$(printf '%s' "$rest" | sed -E 's/^ +//')" "$([ "$dirty" != "0" ] && printf '  DIRTY:%s' "$dirty")"
done
if [ -n "$(printf '%s' "$DEPLOYED_SHAS" | tr -d '\n ')" ]; then
    echo "  deployed shas vs this repo:"
    printf '%s\n' "$DEPLOYED_SHAS" | while IFS=$'\t' read -r role full kind; do
        [ -n "$full" ] || continue
        short="${full:0:8}"
        label="$role"; [ "$kind" = "disk" ] && label="$role (files on disk)"
        if ! git -C "$REPO_ROOT" cat-file -e "$full^{commit}" 2>/dev/null; then
            warn "$label runs $short — commit unknown to this repo (unpushed on the host, or a branch this clone never fetched)"
            continue
        fi
        counts="$(git -C "$REPO_ROOT" rev-list --left-right --count "origin/master...$full" 2>/dev/null)"
        behind="${counts%%	*}"; ahead="${counts##*	}"
        # Name the sha: the remote branch it is the tip of, else the
        # branches that contain it (a merged commit), else nothing pushed.
        tips="$(git -C "$REPO_ROOT" branch -r --points-at "$full" 2>/dev/null | grep -v 'HEAD ->' | sed -E 's/^ +//' | paste -sd, -)"
        if [ -n "$tips" ]; then
            where="tip of $tips"
        else
            holders="$(git -C "$REPO_ROOT" branch -r --contains "$full" 2>/dev/null | grep -v 'HEAD ->' | sed -E 's/^ +//')"
            nh="$(printf '%s\n' "$holders" | grep -c .)"
            first="$(printf '%s\n' "$holders" | grep -m1 -E 'origin/master$' || printf '%s\n' "$holders" | head -1)"
            if [ "$nh" -eq 0 ]; then where="on no remote branch"
            elif [ "$nh" -eq 1 ]; then where="contained in $first"
            else where="contained in $first (+$((nh - 1)) more)"; fi
        fi
        local_wt="$(git -C "$REPO_ROOT" worktree list 2>/dev/null | awk -v s="${full:0:7}" -v root="$REPO_ROOT" '
            index($2, s) == 1 { p = $1; if (p == root) p = "(this checkout)"; else sub("^" root "/", "", p); print p }' | paste -sd, -)"
        if [ "$behind" = "0" ] && [ "$ahead" = "0" ]; then rel="= origin/master"
        elif [ "$ahead" = "0" ]; then rel="$behind behind origin/master"
        else rel="$ahead ahead, $behind behind origin/master"; fi
        info "$label: $short  $rel  ($where)${local_wt:+  local worktree: $local_wt}"
        # Keyed by sha so the table attaches the distance to the right process.
        if [ "$kind" = "disk" ]; then rec "role=$role	for_sha=$short	disk_master_rel=$rel"; else rec "role=$role	for_sha=$short	master_rel=$rel"; fi
    done
fi
[ "$NET_UP" -eq 0 ] && [ "$LOCAL_ONLY" -eq 0 ] && echo "remote: UNKNOWN (network down) — rerun on the lab network for the deployed picture"

if [ "$SUMMARY" -eq 1 ]; then
    exec 1>&3
    if [ "$NET_UP" -eq 1 ]; then
        python3 "$REPO_ROOT/scripts/fleet_table.py" < "$REC_FILE"
    else
        grep -E '^\s+\[(DOWN|WARN)\]' "$LOG_FILE" | head -6
        echo "  network: DOWN or partial — remote fleet state UNKNOWN (see --full)"
    fi
    echo
    attn="$(grep -E '^\s+\[(DOWN|WARN)\]' "$LOG_FILE" | sed -E 's/^ +//')"
    if [ -n "$attn" ]; then
        echo "Attention:"
        printf '%s\n' "$attn" | sed 's/^/  /'
    fi
    grep -E 'origin/master = ' "$LOG_FILE" | sed -E 's/^ +/  /'
    echo "  (full log: scripts/fleet_status.sh --full)"
fi
exit 0
