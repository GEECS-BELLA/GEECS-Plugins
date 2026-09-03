#!/usr/bin/env bash
# fleet_status.sh — what code is each fleet service actually running?
#
# The fleet map (docs/platform/fleet_map.md) is the intended picture; this
# script is the observed one. During rapid development a host may run a
# feature-branch checkout, a checkout may have moved under a service that
# was never restarted, or a pyproject bump may never have been installed —
# none of which the doc can tell you. Every probe is bounded and read-only:
# it never writes a PV, never restarts a unit, never pulls a checkout.
#
#   scripts/fleet_status.sh                       # full picture (needs the lab)
#   scripts/fleet_status.sh --local-only          # just the local checkouts
#   scripts/fleet_status.sh --no-ssh              # service self-reports only
#   scripts/fleet_status.sh --ssh 192.168.6.14=geecs-gw   # ssh alias override
#   scripts/fleet_status.sh --experiment Undulator --no-fetch
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
declare -a SSH_OVERRIDES=()
while [ $# -gt 0 ]; do
    case "$1" in
        --experiment) shift; EXPERIMENT="${1:-}" ;;
        --no-ssh) DO_SSH=0 ;;
        --no-fetch) DO_FETCH=0 ;;
        --local-only) LOCAL_ONLY=1 ;;
        --ssh) shift; SSH_OVERRIDES+=("${1:-}") ;;
        *) echo "usage: fleet_status.sh [--experiment NAME] [--no-ssh] [--no-fetch] [--local-only] [--ssh IP=ALIAS]..." >&2; exit 2 ;;
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
# Format per line: "<label> <sha>".
DEPLOYED_SHAS=""
note_sha() { DEPLOYED_SHAS="$DEPLOYED_SHAS
$1 $2"; }

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
    if [ -n "$tv" ]; then ok "Tiled        $LAB_HOST:$TILED_PORT  tiled $tv"; else bad "Tiled        $LAB_HOST:$TILED_PORT"; fi

    # Data Portal — /health carries ok + catalog probe + installed version.
    PORTAL_HOST="${WORKER_HOST:-$LAB_HOST}"
    ph="$(bounded "$TCP_TIMEOUT" curl -s -m "$TCP_TIMEOUT" "http://$PORTAL_HOST:$PORTAL_PORT/health")"
    if [ -n "$ph" ]; then
        pv="$(printf '%s' "$ph" | sed -nE 's/.*"version": *"([^"]+)".*/\1/p')"
        pok="$(printf '%s' "$ph" | sed -nE 's/.*"ok": *(true|false).*/\1/p')"
        pcat="$(printf '%s' "$ph" | sed -nE 's/.*"catalog": *"([^"]*)".*/\1/p')"
        if [ "$pok" = "true" ]; then
            ok "Data Portal  $PORTAL_HOST:$PORTAL_PORT  geecs-data-portal ${pv:-?}  (catalog: ${pcat:-?})"
        else
            warn "Data Portal  $PORTAL_HOST:$PORTAL_PORT  geecs-data-portal ${pv:-?}  up but catalog NOT ok (${pcat:-?})"
        fi
    else
        bad "Data Portal  $PORTAL_HOST:$PORTAL_PORT  (no /health answer)"
    fi

    # MCP HTTP mode — no version endpoint; port liveness only (stage 2 reads the venv).
    if port_open "${WORKER_HOST:-$LAB_HOST}" "$MCP_PORT"; then
        ok "GEECS-MCP    ${WORKER_HOST:-$LAB_HOST}:$MCP_PORT  listening (version via ssh below)"
    else
        skip "GEECS-MCP    ${WORKER_HOST:-$LAB_HOST}:$MCP_PORT  not listening (HTTP mode is 'pending deploy' on the fleet map)"
    fi

    # CA gateway — reuse /lab-status tier 2 (read-only CA gets of heartbeat,
    # devices_connected, version). Contract: it prints "version=<str>".
    if [ -n "$EXPERIMENT" ]; then
        hw="$("$REPO_ROOT/scripts/lab_status.sh" --hardware --experiment "$EXPERIMENT" 2>&1 | sed -n '/Tier 2/,$p')"
        alive="$(printf '%s\n' "$hw" | grep -m1 'gateway alive')"
        if [ -n "$alive" ]; then
            ok "CA gateway   $LAB_HOST:5064  ${alive#*] }"
            printf '%s\n' "$hw" | grep '\[WARN\]' | sed 's/^ *\[WARN\] /  [WARN] CA gateway: /'
        else
            err="$(printf '%s\n' "$hw" | grep -m1 -E 'unreadable|not installed|no experiment' )"
            bad "CA gateway   $LAB_HOST:5064  ${err#*] }"
        fi
    else
        skip "CA gateway   no experiment name (config.ini [Experiment] expt, or --experiment NAME)"
    fi

    # PVA image fleet — one gateway per camera server; roster = the checked-in
    # Phoebus fleet screen (the DB-driven roster is still owed). Read-only gets.
    if [ -f "$PVA_ROSTER" ]; then
        pvs="$(grep -o 'pv_name>pva://[^<]*:version' "$PVA_ROSTER" | sed 's|pv_name>pva://||' | sort -u)"
        if [ -n "$pvs" ] && poetry -C "$REPO_ROOT/GeecsPvaGateway" env info --path >/dev/null 2>&1; then
            PVA_PVS="$pvs" PVA_TIMEOUT="$PVA_TIMEOUT" bounded 60 poetry -C "$REPO_ROOT/GeecsPvaGateway" run python - <<'PY'
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
with Context("pva") as ctx:
    for pv, host in zip(pvs, hosts):
        base = pv[: -len(":version")]
        try:
            ver = str(ctx.get(pv, timeout=timeout))
            beats = int(ctx.get(base + ":heartbeat", timeout=timeout))
        except Exception as exc:  # noqa: BLE001 — a failed probe is a finding
            print(f"  [DOWN] PVA gateway  {host:<15}  ({type(exc).__name__})")
            continue
        versions.setdefault(ver, []).append(host)
        print(f"  [ OK ] PVA gateway  {host:<15}  geecs-pva-gateway {ver}  heartbeat={beats}")
if len(versions) > 1:
    print("  [WARN] PVA fleet runs mixed versions — a rollout is incomplete or a box missed its pull-on-restart:")
    for ver, hs in sorted(versions.items()):
        print(f"         {ver}: {', '.join(hs)}")
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
# Stage 2 runs this on each host. Output: one "unit=..." record per systemd
# unit (key=value pairs, tab-separated), which the local side formats.
# Read-only: git queries, systemctl show, a venv python -c for the installed
# version. `poetry` is looked up in ~/.local/bin because plain ssh has no
# login PATH (fleet map bootstrap gotcha 4).
REMOTE_SNIPPET='
set -u
POETRY="$HOME/.local/bin/poetry"; command -v "$POETRY" >/dev/null 2>&1 || POETRY="$(command -v poetry 2>/dev/null || true)"
command -v systemctl >/dev/null 2>&1 || { echo "nosystemd"; exit 0; }
units="$(systemctl list-units --all --plain --no-legend "geecs-*" "tiled.service" 2>/dev/null | awk "{print \$1}")"
[ -n "$units" ] || { echo "nounits"; exit 0; }
reflog_ts() {  # unix time HEAD last moved (checkout/pull/reset), from the reflog
    local f; f="$(git -C "$1" rev-parse --git-path logs/HEAD 2>/dev/null)"
    [ -f "$f" ] && tail -1 "$f" | sed -E "s/^[0-9a-f]+ [0-9a-f]+ .*> ([0-9]+) [-+][0-9]{4}\t.*/\1/"
}
for u in $units; do
    active="$(systemctl show -p ActiveState --value "$u")"; sub="$(systemctl show -p SubState --value "$u")"
    since="$(systemctl show -p ActiveEnterTimestamp --value "$u")"
    since_ts=""; [ -n "$since" ] && since_ts="$(date -d "$since" +%s 2>/dev/null)"
    workdir="$(systemctl show -p WorkingDirectory --value "$u")"
    exec_py="$(systemctl show -p ExecStart --value "$u" | sed -nE "s/.*path=([^ ;]+).*/\1/p")"
    rec="unit=$u\tactive=$active/$sub\tsince=${since:-?}"
    clone=""; pkgdir=""; venv=""
    if [ -n "$workdir" ] && [ -d "$workdir" ]; then
        clone="$(git -C "$workdir" rev-parse --show-toplevel 2>/dev/null)"; pkgdir="$workdir"
        [ -n "$clone" ] && [ -x "$POETRY" ] && venv="$(cd "$workdir" && "$POETRY" env info --path 2>/dev/null)"
    elif [ -n "$exec_py" ] && [ -x "$exec_py" ]; then
        # Baked venv (MCP pattern): the install records its source path.
        venv="$(dirname "$(dirname "$exec_py")")"
        src="$("$exec_py" -c "import importlib.metadata as m,json,sys
for d in m.distributions():
    t=d.read_text(\"direct_url.json\")
    if t and \"GEECS\" in t: print(json.loads(t)[\"url\"]); break" 2>/dev/null | sed "s|^file://||")"
        [ -n "$src" ] && [ -d "$src" ] && { pkgdir="$src"; clone="$(git -C "$src" rev-parse --show-toplevel 2>/dev/null)"; }
    fi
    if [ -n "$clone" ]; then
        branch="$(git -C "$clone" rev-parse --abbrev-ref HEAD 2>/dev/null)"; [ "$branch" = "HEAD" ] && branch="(detached)"
        sha="$(git -C "$clone" rev-parse --short=8 HEAD 2>/dev/null)"; full="$(git -C "$clone" rev-parse HEAD 2>/dev/null)"
        cdate="$(git -C "$clone" log -1 --format=%cs 2>/dev/null)"
        dirty="$(git -C "$clone" status --porcelain --untracked-files=no 2>/dev/null | wc -l | tr -d " ")"
        moved="$(reflog_ts "$clone")"
        rec="$rec\tclone=${clone/#$HOME/\~}\tbranch=$branch\tsha=$sha\tfull=$full\tcommit_date=$cdate\tdirty=$dirty"
        if [ -n "$moved" ] && [ -n "$since_ts" ] && [ "$active" = "active" ] && [ "$moved" -gt "$since_ts" ]; then
            rec="$rec\tstale=checkout moved $(date -d @"$moved" "+%F %H:%M") after the service started"
        fi
    fi
    if [ -n "$pkgdir" ] && [ -f "$pkgdir/pyproject.toml" ]; then
        pname="$(sed -nE "s/^name *= *\"([^\"]+)\".*/\1/p" "$pkgdir/pyproject.toml" | head -1)"
        pver="$(sed -nE "s/^version *= *\"([^\"]+)\".*/\1/p" "$pkgdir/pyproject.toml" | head -1)"
        rec="$rec\tpkg=$pname\tpyproject=$pver"
        if [ -n "$venv" ] && [ -x "$venv/bin/python" ] && [ -n "$pname" ]; then
            iver="$("$venv/bin/python" -c "import importlib.metadata as m; print(m.version(\"$pname\"))" 2>/dev/null)"
            rec="$rec\tinstalled=${iver:-?}"
        fi
    fi
    printf "%b\n" "$rec"
done
'

fmt_host_records() {  # stdin: unit records -> pretty lines; side effect: note_sha
    local line
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        case "$line" in
            nosystemd) skip "no systemd on this host"; continue ;;
            nounits) skip "no geecs-* / tiled units installed here"; continue ;;
        esac
        local unit active since clone branch sha full cdate dirty stale pkg pyproject installed
        unit=""; active=""; since=""; clone=""; branch=""; sha=""; full=""; cdate=""; dirty=""; stale=""; pkg=""; pyproject=""; installed=""
        local IFS=$'\t' kv
        for kv in $line; do
            case "$kv" in
                unit=*) unit="${kv#*=}" ;; active=*) active="${kv#*=}" ;; since=*) since="${kv#*=}" ;;
                clone=*) clone="${kv#*=}" ;; branch=*) branch="${kv#*=}" ;; sha=*) sha="${kv#*=}" ;;
                full=*) full="${kv#*=}" ;; commit_date=*) cdate="${kv#*=}" ;; dirty=*) dirty="${kv#*=}" ;;
                stale=*) stale="${kv#*=}" ;; pkg=*) pkg="${kv#*=}" ;; pyproject=*) pyproject="${kv#*=}" ;;
                installed=*) installed="${kv#*=}" ;;
            esac
        done
        unset IFS
        local tag="[ OK ]"
        case "$active" in active/*) ;; *) tag="[DOWN]" ;; esac
        printf '  %s %-26s %-16s since %s\n' "$tag" "$unit" "$active" "${since:-?}"
        if [ -n "$clone" ]; then
            local d=""; [ "${dirty:-0}" != "0" ] && d="  DIRTY: $dirty modified tracked file(s)"
            info "$clone @ $branch $sha ($cdate)$d"
            note_sha "$unit" "$full"
        fi
        if [ -n "$pkg" ]; then
            if [ -n "$installed" ] && [ "$installed" != "$pyproject" ]; then
                warn "$pkg: pyproject says $pyproject but the venv has $installed installed — poetry install pending"
            else
                info "$pkg $pyproject${installed:+ (installed $installed)}"
            fi
        fi
        [ -n "$stale" ] && warn "$unit: $stale — the running process predates the code on disk (restart pending?)"
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
        if [ "$rc" -ne 0 ] && ! printf '%s' "$out" | grep -q '^unit='; then
            bad "ssh $target failed (rc=$rc): $(printf '%s' "$out" | head -1)"
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
    printf '%s\n' "$DEPLOYED_SHAS" | while read -r label full; do
        [ -n "$full" ] || continue
        short="${full:0:8}"
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
    done
fi
[ "$NET_UP" -eq 0 ] && [ "$LOCAL_ONLY" -eq 0 ] && echo "remote: UNKNOWN (network down) — rerun on the lab network for the deployed picture"
exit 0
