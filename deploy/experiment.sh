#!/usr/bin/env bash
# experiment.sh — stand an experiment INSTANCE up, take it down, list them.
#
#   deploy/experiment.sh up   INSTANCE [--db-experiment NAME --address ADDR --broadcast ADDR]
#                                      [--ca-extra "HOST[:PORT] ..."] [--ref REF] [--dry-run]
#   deploy/experiment.sh down INSTANCE [--dry-run]
#   deploy/experiment.sh list
#
# An instance is the per-experiment service set beside the main experiment's
# units: geecs-ca-gateway@, geecs-qserver@ (+ -ready@), geecs-tiled-writer@,
# geecs-scanner@, geecs-logbook@ — one set per INSTANCE (a slug), each reading
# /etc/geecs/site.env and then /etc/geecs/experiments/INSTANCE.env.
#
# up    First time: writes the instance file from deploy/experiments.env.example
#       (--db-experiment, the address this host serves CA on and its
#       broadcast are required; every port follows from the first free SLOT,
#       see the example). Again: reuses the installed file as it is (and
#       refuses those flags). Then renders the instance units
#       (bootstrap_host.sh --experiment) — all six or it stops, nothing
#       installed — prints the device-overlap WARNING for the live
#       experiments (geecs_core.db.overlap, read-only), scaffolds <configs>/scanner_configs/experiments/<DB
#       experiment>/ when absent, installs + enables the units (sudo), and
#       prints the ports and the scanner URL.
# down  Stops and disables the instance's units. The instance file (its slot),
#       its state under /var/lib/geecs-experiments/INSTANCE/, its configs and
#       its data stay where they are; `up` again brings it back as it was.
# list  Every instance file, its experiment, slot, scanner port and unit states.
#
# Never touches the main experiment's units, site.env or config.ini. Run it
# as the service account (sudo for the install/enable steps). Knobs (from the
# environment): GEECS_SITE_ENV_PATH (default /etc/geecs/site.env),
# GEECS_EXPERIMENTS_DIR (default /etc/geecs/experiments).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=site_env_lib.sh
. "$REPO_ROOT/deploy/site_env_lib.sh"

usage() { sed -n '3,7p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
VERB="${1:-}"; [ -n "$VERB" ] && shift || usage
INSTANCE=""
if [ "$VERB" != "list" ]; then INSTANCE="${1:-}"; [ -n "$INSTANCE" ] && shift || usage; fi
DB_EXPERIMENT=""; ADDRESS=""; BROADCAST=""; CA_EXTRA=""; REF=""; DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --db-experiment) shift; DB_EXPERIMENT="${1:-}" ;;
        --address) shift; ADDRESS="${1:-}" ;;
        --broadcast) shift; BROADCAST="${1:-}" ;;
        --ca-extra) shift; CA_EXTRA="${1:-}" ;;
        --ref) shift; REF="${1:-}" ;;
        --dry-run) DRY=1 ;;
        *) usage ;;
    esac
    shift
done
case "$INSTANCE" in *[!A-Za-z0-9_-]*) echo "instance '$INSTANCE' is not a slug (letters, digits, '-', '_')" >&2; exit 2 ;; esac

SITE_ENV="${GEECS_SITE_ENV_PATH:-/etc/geecs/site.env}"
EXPERIMENTS_DIR="${GEECS_EXPERIMENTS_DIR:-/etc/geecs/experiments}"
[ -f "$SITE_ENV" ] || { echo "no site.env at $SITE_ENV" >&2; exit 2; }
run() { if [ "$DRY" -eq 1 ]; then echo "  [dry] $*"; else echo "  \$ $*"; "$@"; fi; }
say() { printf '\n== %s\n' "$1"; }

# The instance services' unit names (the bootstrap's instance families).
INSTANCE_UNITS="geecs-ca-gateway geecs-qserver geecs-qserver-ready geecs-tiled-writer geecs-scanner geecs-logbook"
units_for() { local u; for u in $INSTANCE_UNITS; do printf '%s@%s.service ' "$u" "$1"; done; }

env_value() {  # env_value FILE KEY — one key of an env file, parsed as systemd does
    # The key is unset first: the parent's site.env values must never stand in
    # for one the instance file lacks.
    ( unset "$2"; load_site_env "$1" >/dev/null 2>&1 || true; printf '%s' "${!2:-}" )
}

# The ports of slot N (deploy/experiments.env.example; slot 0 = the stock ports).
slot_port() {  # slot_port N BASE STEP
    echo $(( $2 + $1 * $3 ))
}

free_slot() {
    local n f used=" "
    for f in "$EXPERIMENTS_DIR"/*.env; do [ -f "$f" ] && used="$used$(env_value "$f" GEECS_INSTANCE_SLOT) "; done
    for n in 1 2 3 4 5 6 7 8 9; do case "$used" in *" $n "*) ;; *) echo "$n"; return 0 ;; esac; done
    echo "no free slot (1..9 all taken in $EXPERIMENTS_DIR)" >&2; return 1
}

render_instance_env() {  # render_instance_env SLOT > file — the example with every value filled
    local n="$1" ca control info din dout scanner logbook
    ca="$(slot_port "$n" 5064 100)"; control="$(slot_port "$n" 60615 100)"; info="$(slot_port "$n" 60625 100)"
    din="$(slot_port "$n" 5567 100)"; dout="$(slot_port "$n" 5568 100)"
    scanner="$(slot_port "$n" 8300 10)"; logbook="$(slot_port "$n" 8400 10)"
    local addr_list="$ADDRESS:$ca${CA_EXTRA:+ $CA_EXTRA}"
    # Values through ENVIRON, never -v: awk would read backslash escapes in -v.
    INST="$INSTANCE" SLOT="$n" EXPT="$DB_EXPERIMENT" CA="$ca" INTF="$ADDRESS" BCAST="$BROADCAST" \
    ADDRS="$addr_list" CONTROL="$control" INFO="$info" DIN="$din" DOUT="$dout" \
    SCANNER="$scanner" LOGBOOK="$logbook" awk '
        BEGIN {
            inst = ENVIRON["INST"]; slot = ENVIRON["SLOT"]; expt = ENVIRON["EXPT"]; ca = ENVIRON["CA"]
            intf = ENVIRON["INTF"]; bcast = ENVIRON["BCAST"]; addrs = ENVIRON["ADDRS"]
            control = ENVIRON["CONTROL"]; info = ENVIRON["INFO"]; din = ENVIRON["DIN"]; dout = ENVIRON["DOUT"]
            scanner = ENVIRON["SCANNER"]; logbook = ENVIRON["LOGBOOK"]
            v["GEECS_INSTANCE"] = inst; v["GEECS_INSTANCE_SLOT"] = slot
            v["GEECS_EXPERIMENT"] = expt; v["QS_EXPERIMENT"] = expt
            v["GEECS_CA_SERVER_PORT"] = ca
            v["EPICS_CAS_INTF_ADDR_LIST"] = intf; v["EPICS_CAS_BEACON_ADDR_LIST"] = bcast
            v["EPICS_CA_ADDR_LIST"] = "\"" addrs "\""; v["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
            v["QSERVER_ZMQ_CONTROL_ADDRESS_FOR_SERVER"] = "tcp://*:" control
            v["QSERVER_ZMQ_INFO_ADDRESS_FOR_SERVER"] = "tcp://*:" info
            v["QS_REDIS_PREFIX"] = "qs_" inst
            v["QS_DOC_PROXY_IN"] = din; v["QS_DOC_PROXY_OUT"] = dout
            v["QS_DOC_PUBLISH_ADDR"] = "localhost:" din
            v["QS_CONTROL_ADDR"] = "tcp://localhost:" control
            v["QS_CLIENT_CONTROL_ADDR"] = "tcp://localhost:" control
            v["QS_CLIENT_INFO_ADDR"] = "tcp://localhost:" info
            v["QS_CLIENT_DOC_ADDR"] = "localhost:" dout
            v["GEECS_SCANNER_PORT"] = scanner; v["GEECS_LOGBOOK_PORT"] = logbook
        }
        /^[A-Z_]+=/ { k = substr($0, 1, index($0, "=") - 1); if (k in v) { print k "=" v[k]; seen[k] = 1; next } }
        { print }
        END { for (k in v) if (!(k in seen)) { print "experiments.env.example lacks " k > "/dev/stderr"; exit 1 } }
    ' "$REPO_ROOT/deploy/experiments.env.example"
}

load_site_env "$SITE_ENV"
INSTANCE_ROOT="${GEECS_INSTANCE_CHECKOUT_ROOT:-$GEECS_CHECKOUT_ROOT}"
STAGE="$GEECS_CHECKOUT_ROOT/deploy-staging/instances"
MAIN_EXPERIMENT="$GEECS_EXPERIMENT"

# Two accounts, never mixed up: everything that builds or reads code — the
# bootstrap (clones, poetry envs, rendering), the overlap check, the configs
# scaffold — runs as the SERVICE account the units run as (its poetry cache,
# its config.ini); only installing into /etc and systemctl run as root.
# Run the verb as root (`sudo experiment.sh …`) or as an account with sudo.
SUDO="${GEECS_EXPERIMENT_SUDO:-sudo}"   # tests substitute a stub
SVC="$GEECS_SERVICE_USER"
if [ "$(id -u)" -eq 0 ]; then AS_ROOT=()
elif [ "$VERB" != "list" ] && ! $SUDO -n true 2>/dev/null; then
    echo "experiment.sh $VERB needs root: run it as 'sudo $0 …' (this account, $(id -un), has no sudo)" >&2; exit 2
else AS_ROOT=("$SUDO"); fi
if [ "$(id -un)" = "$SVC" ]; then AS_SVC=(); else AS_SVC=("$SUDO" -u "$SVC" -H); fi
as_svc() { "${AS_SVC[@]+"${AS_SVC[@]}"}" "$@"; }
as_root() { "${AS_ROOT[@]+"${AS_ROOT[@]}"}" "$@"; }
in_engine() {  # in_engine CMD... — CMD in the instance clone's GeecsBluesky dir, as the service account
    as_svc bash -c 'cd "$1" && shift && exec "$@"' in_engine "$ENGINE" "$@"
}

# The instance's own trees must be the service account's: a run under the
# wrong account (or an older verb) leaves them root-owned, and the bootstrap,
# running as the service account, could then neither stage nor install.
# Only trees no main unit runs from: the staging dir, and the instances'
# clone root when it is a root of its own.
claim_staging() {
    if [ -e "$STAGE" ] && [ "$(stat -c %U "$STAGE" 2>/dev/null || stat -f %Su "$STAGE")" != "$SVC" ]; then
        run as_root chown -R "$SVC:" "$STAGE"
    fi
}
claim_instance_root() {
    [ "$INSTANCE_ROOT" != "$GEECS_CHECKOUT_ROOT" ] || return 0
    if [ -e "$INSTANCE_ROOT" ] && [ -n "$(as_root find "$INSTANCE_ROOT" -maxdepth 3 ! -user "$SVC" -print -quit 2>/dev/null)" ]; then
        run as_root chown -R "$SVC:" "$INSTANCE_ROOT"
    fi
}

case "$VERB" in
list)
    printf '%-16s %-26s %-4s %-7s %s\n' INSTANCE EXPERIMENT SLOT SCANNER UNITS
    for f in "$EXPERIMENTS_DIR"/*.env; do
        [ -f "$f" ] || continue
        i="$(basename "$f" .env)"; states=""
        for u in $(units_for "$i"); do states="$states ${u%%@*}=$({ systemctl is-active "$u" 2>/dev/null || true; } | head -1)"; done
        printf '%-16s %-26s %-4s %-7s%s\n' "$i" "'$(env_value "$f" GEECS_EXPERIMENT)'" \
            "$(env_value "$f" GEECS_INSTANCE_SLOT)" "$(env_value "$f" GEECS_SCANNER_PORT)" "$states"
    done
    ;;
down)
    say "down '$INSTANCE' — units stopped and disabled; instance file, state, configs and data stay"
    # shellcheck disable=SC2046
    run as_root systemctl disable --now $(units_for "$INSTANCE")
    # Clear a failed state so the next `up` starts clean (the instance file,
    # its clones and their envs stay; `up` reinstalls the envs).
    run as_root systemctl reset-failed $(units_for "$INSTANCE") || true
    ;;
up)
    INSTALLED="$EXPERIMENTS_DIR/$INSTANCE.env"
    if [ -f "$INSTALLED" ]; then
        INSTANCE_FILE="$INSTALLED"
        if [ -n "$DB_EXPERIMENT$ADDRESS$BROADCAST$CA_EXTRA" ]; then
            echo "'$INSTANCE' already has an instance file ($INSTALLED): --db-experiment/--address/--broadcast/--ca-extra would be ignored — edit that file (as root) instead" >&2; exit 2
        fi
        say "up '$INSTANCE' — instance file exists ($INSTALLED), used as it is"
    else
        [ -n "$DB_EXPERIMENT" ] && [ -n "$ADDRESS" ] && [ -n "$BROADCAST" ] || {
            echo "a new instance needs --db-experiment NAME, --address (this host's CA serving address) and --broadcast" >&2; exit 2; }
        SLOT="$(free_slot)"
        INSTANCE_FILE="$STAGE/$INSTANCE.env"
        claim_staging
        render_instance_env "$SLOT" | as_svc bash -c 'mkdir -p "$1" && cat > "$2"' stage "$STAGE" "$INSTANCE_FILE"
        say "up '$INSTANCE' — new instance file (slot $SLOT) at $INSTANCE_FILE"
    fi
    # Validate in a subshell: the instance's values must not leak into ours.
    ( load_site_env "$INSTANCE_FILE"; check_instance_env "$INSTANCE" "$INSTANCE_FILE" )
    EXPT="$(env_value "$INSTANCE_FILE" GEECS_EXPERIMENT)"
    [ "$EXPT" != "$MAIN_EXPERIMENT" ] || { echo "'$EXPT' is the main experiment (site.env) — it keeps its own units" >&2; exit 2; }
    ENGINE="$INSTANCE_ROOT/qs-checkout/GeecsBluesky"
    claim_staging
    claim_instance_root

    say "units"
    # The bootstrap's own enable line names exactly the units it rendered
    # (dry run or not); its root-steps section is ours to run, not to print.
    BOOT_OUT="$(as_svc "$REPO_ROOT/deploy/bootstrap_host.sh" "$SITE_ENV" --experiment "$INSTANCE" --instance-env "$INSTANCE_FILE" \
        ${REF:+--ref "$REF"} $([ "$DRY" -eq 1 ] && echo --dry-run))"
    printf '%s\n' "$BOOT_OUT" | sed '/^== root steps/,$d'
    STAGED=()
    for u in $(printf '%s\n' "$BOOT_OUT" | sed -n 's/.*systemctl enable --now //p'); do STAGED+=("$u"); done
    # All six or none: a half instance (a manager without its scanner, a
    # scanner without its manager) is worse than none, and must not look done.
    if [ "${#STAGED[@]}" -ne "$(echo $INSTANCE_UNITS | wc -w)" ]; then
        echo "INCOMPLETE: $(( $(echo $INSTANCE_UNITS | wc -w) - ${#STAGED[@]} )) instance unit(s) not rendered (see the WARNINGs above) — nothing installed or enabled" >&2
        exit 1
    fi

    # Devices enabled in more than one live experiment: a WARNING, never a
    # refusal (fixing a device set is its owner's DB edit). Read-only; a DB
    # or code that cannot answer says "not checked", never "clean".
    say "device overlap (read-only, the live experiments)"
    # Live = the main experiment + every OTHER instance whose gateway or
    # manager runs (a `down` instance keeps its file but drives nothing).
    LIVE=("$MAIN_EXPERIMENT")
    for f in "$EXPERIMENTS_DIR"/*.env; do
        [ -f "$f" ] || continue
        i="$(basename "$f" .env)"; [ "$i" = "$INSTANCE" ] && continue
        if [ "${EXPERIMENT_ASSUME_ACTIVE:-0}" = "1" ] ||   # (tests: no systemd)
            systemctl is-active --quiet "geecs-ca-gateway@$i.service" "geecs-qserver@$i.service" 2>/dev/null; then
            LIVE+=("$(env_value "$f" GEECS_EXPERIMENT)")
        fi
    done
    LIVE+=("$EXPT")
    if [ "$DRY" -eq 1 ]; then printf '  [dry] python -m geecs_core.db.overlap'; printf " '%s'" "${LIVE[@]}"; echo
    else
        rc=0; OVERLAP="$(in_engine "$GEECS_POETRY" run python -m geecs_core.db.overlap "${LIVE[@]}" 2>&1)" || rc=$?
        printf '%s\n' "$OVERLAP" | sed 's/^/  /'
        [ "$rc" -eq 0 ] || echo "  WARNING: overlap NOT checked (exit $rc) — continuing; run python -m geecs_core.db.overlap by hand"
    fi

    # The configs folder, after the bootstrap (a fresh instance clone exists
    # now) and before the units start (they read it). Checked here first, so
    # an instance whose folder exists needs no particular code.
    say "configs ($EXPT)"
    EXPERIMENTS_ROOT="$GEECS_CONFIGS_ROOT/scanner_configs/experiments"
    if [ -d "$EXPERIMENTS_ROOT/$EXPT" ]; then echo "  $EXPERIMENTS_ROOT/$EXPT exists — left as it is"
    elif [ "$DRY" -eq 1 ]; then echo "  [dry] python -m geecs_bluesky.config_scaffold '$EXPT' --experiments-root '$EXPERIMENTS_ROOT'"
    else in_engine "$GEECS_POETRY" run python -m geecs_bluesky.config_scaffold "$EXPT" --experiments-root "$EXPERIMENTS_ROOT" | sed 's/^/  /'; fi

    [ "$INSTANCE_FILE" -ef "$INSTALLED" ] || run as_root install -D -m 0644 "$INSTANCE_FILE" "$INSTALLED"
    # The glob is expanded by root (the staging dir is the service account's).
    run as_root sh -c 'install -m 0644 "$1"/*@.service /etc/systemd/system/' install-units "$STAGE"
    run as_root systemctl daemon-reload
    # Restart, not just start: a re-run `up` must pick up reinstalled envs.
    run as_root systemctl enable "${STAGED[@]}"
    run as_root systemctl restart "${STAGED[@]}"

    say "instance '$INSTANCE' — experiment '$EXPT'"
    f="$INSTANCE_FILE"
    echo "  CA gateway   $(env_value "$f" EPICS_CAS_INTF_ADDR_LIST):$(env_value "$f" GEECS_CA_SERVER_PORT)   (clients add this host:port to their CA address list)"
    echo "  manager      $(env_value "$f" QSERVER_ZMQ_CONTROL_ADDRESS_FOR_SERVER) / $(env_value "$f" QSERVER_ZMQ_INFO_ADDRESS_FOR_SERVER); documents $(env_value "$f" QS_DOC_PROXY_OUT)"
    echo "  logbook      port $(env_value "$f" GEECS_LOGBOOK_PORT)"
    echo "  scanner      http://$(env_value "$f" EPICS_CAS_INTF_ADDR_LIST):$(env_value "$f" GEECS_SCANNER_PORT)/"
    ;;
*) usage ;;
esac
