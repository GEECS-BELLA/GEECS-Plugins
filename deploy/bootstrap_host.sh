#!/usr/bin/env bash
# bootstrap_host.sh — bring a service host to "every service has its clone,
# its env, its config, and a rendered unit" from ONE site.env. Idempotent
# and unprivileged: run it as the service account, as many times as you
# like; it never touches /etc, never restarts anything, and prints the
# root steps for a human at the end.
#
#   deploy/bootstrap_host.sh SITE_ENV [--ref REF] [--only svc,svc] [--dry-run] [--no-install]
#   deploy/bootstrap_host.sh SITE_ENV --experiment INSTANCE [--instance-env FILE] [--only ...] [--dry-run] [--no-install]
#
#   --ref REF      git ref to check out in each clone (default: master)
#   --only LIST    comma-separated subset of: gateway,portal,logbook,qserver,tiled-writer,mcp,scanner,archiver
#                  (re-stages only these: the staging dir's units are cleared first,
#                  so the printed install line covers exactly this run)
#   --no-install   clone/fetch only; skip poetry/pip installs
#   --dry-run      print what would happen
#   --experiment INSTANCE
#                  one experiment INSTANCE beside the main units: renders the
#                  instance templates (<unit>@.service) of gateway, logbook,
#                  qserver, tiled-writer and scanner and prints their enable
#                  lines for <unit>@INSTANCE. Existing clones are used AS
#                  THEY ARE (they run the main experiment's services: never
#                  fetched, moved or reinstalled here); an absent one is
#                  cloned at --ref and installed. Clones live under
#                  GEECS_INSTANCE_CHECKOUT_ROOT when site.env sets it (else
#                  the main root). config.ini is not touched.
#   --instance-env FILE
#                  the instance's values (default: the installed
#                  /etc/geecs/experiments/INSTANCE.env; keys in
#                  deploy/experiments.env.example)
#
# This is the fleet map's "fresh-host bootstrap" list made executable:
# per-service clones under GEECS_CHECKOUT_ROOT (one clone per service
# family — a pull for one service must never change the code under
# another running service), each poetry env installed INSIDE its package
# directory with that service's extras, the MCP venv baked non-editably
# from the worker's checkout, the service account's config.ini rendered
# from site.env if absent, and the unit files rendered into a staging
# directory. See docs/platform/site_profile.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SITE_ENV="${1:-}"; [ -n "$SITE_ENV" ] && shift || true
REF="master"; ONLY=""; DRY=0; INSTALL=1; INSTANCE=""; INSTANCE_ENV=""
while [ $# -gt 0 ]; do
    case "$1" in
        --ref) shift; REF="${1:-master}" ;;
        --only) shift; ONLY="${1:-}" ;;
        --dry-run) DRY=1 ;;
        --no-install) INSTALL=0 ;;
        --experiment) shift; INSTANCE="${1:-}"; [ -n "$INSTANCE" ] || { echo "--experiment needs an instance name" >&2; exit 2; } ;;
        --instance-env) shift; INSTANCE_ENV="${1:-}" ;;
        *) echo "usage: bootstrap_host.sh SITE_ENV [--ref REF] [--only LIST] [--dry-run] [--no-install] [--experiment INSTANCE [--instance-env FILE]]" >&2; exit 2 ;;
    esac
    shift
done
[ -n "$SITE_ENV" ] && [ -f "$SITE_ENV" ] || { echo "usage: bootstrap_host.sh SITE_ENV ..." >&2; exit 2; }
SITE_ENV="$(cd "$(dirname "$SITE_ENV")" && pwd)/$(basename "$SITE_ENV")"   # absolute: the printed sudo lines must work from any cwd

. "$REPO_ROOT/deploy/site_env_lib.sh"
load_site_env "$SITE_ENV"
check_site_env_consistency
require_site_keys GEECS_SERVICE_USER GEECS_SERVICE_HOME GEECS_CHECKOUT_ROOT GEECS_POETRY GEECS_REPO_URL \
    GEECS_TILED_URI GEECS_QSERVER_HOST GEECS_DATA_ROOT GEECS_PORTAL_MEMORY_HIGH GEECS_PORTAL_MEMORY_MAX
require_runtime_keys
# Read AFTER site.env is loaded (it may set the knob) — the same one
# render_units.sh honours, so the rendered EnvironmentFile= path and the
# printed install line always agree.
SITE_ENV_INSTALLED="${GEECS_SITE_ENV_PATH:-/etc/geecs/site.env}"

run() { if [ "$DRY" -eq 1 ]; then echo "  [dry] $*"; else echo "  \$ $*"; "$@"; fi; }
say() { printf '\n== %s\n' "$1"; }

# ---- the service table: name | clone dir | package dir | poetry extras ---
# One clone per service FAMILY (fleet map § one clone per service). The MCP
# server is baked from qs-checkout into its own venv (config-truth parity
# with the worker, without a shared working tree under a running service).
# The logbook shares portal-checkout (the two web viewers, restarted
# together after a pull) with its own poetry env inside GeecsLogbook/.
# The Tiled writer (geecs-tiled-writer, GeecsBluesky 0.103.0) is the
# consumer half of the engine's document spool: it runs from the WORKER's
# package and env (qs-checkout/GeecsBluesky — one spool line format, one
# env) and needs the `tiled` extra, which the worker's set already carries.
# Its extras are therefore the worker's set, deliberately: `poetry install
# --extras` REMOVES the optional packages of every extra it is not given, so
# a `--only tiled-writer` run installing "tiled" alone would strip ca /
# qserver / optimize from the env the running worker uses.
# The archiver (GeecsArchiver) is upstream's Archiver Appliance in its official
# container under a systemd unit: its clone holds the geecs-archiver CLI and
# the conf templates; the unit itself runs docker compose from /etc/geecs/archiver
# and never reads the clone. Its runtime prerequisite is Docker (checked below,
# the Redis pattern); its conf is rendered by the package's own render_conf.sh.
SERVICES="gateway portal logbook qserver tiled-writer mcp scanner archiver"
clone_of()   { case "$1" in gateway) echo "gateway-checkout";; portal|logbook) echo "portal-checkout";; qserver|tiled-writer|mcp|scanner) echo "qs-checkout";; archiver) echo "archiver-checkout";; esac; }
pkgdir_of()  { case "$1" in gateway) echo "GeecsCAGateway";; portal) echo "GEECS-DataPortal";; logbook) echo "GeecsLogbook";; qserver|tiled-writer) echo "GeecsBluesky";; mcp) echo "GEECS-MCP";; scanner) echo "GeecsScanner";; archiver) echo "GeecsArchiver";; esac; }
extras_of()  { case "$1" in gateway) echo "";; portal) echo "analysis";; logbook) echo "";; qserver|tiled-writer) echo "ca tiled qserver optimize";; mcp) echo "";; scanner) echo "";; archiver) echo "";; esac; }
# The queueserver is two units: the manager and the geecs-qserver-ready oneshot
# that opens its worker environment and asserts the plan list after every
# (re)start (#793) — enabled together, rendered from the same clone. The
# Tiled writer is its own service (own unit, own restart, no ordering
# against the manager): the spool directory is the only thing they share.
units_of()   { case "$1" in gateway) echo "geecs-ca-gateway";; portal) echo "geecs-data-portal";; logbook) echo "geecs-logbook";; qserver) echo "geecs-qserver geecs-qserver-ready";; tiled-writer) echo "geecs-tiled-writer";; mcp) echo "geecs-mcp";; scanner) echo "geecs-scanner";; archiver) echo "geecs-archiver";; esac; }
# ---- --experiment: one experiment instance, beside the main units ----------
# Additive by construction: it renders only the <unit>@.service templates,
# never re-renders or re-enables a main unit, and never moves, fetches or
# reinstalls a clone that exists (each one runs the main experiment's
# services — a pull there is that service's deploy). The instance's values
# come from its own file, which overrides site.env exactly as the units'
# second EnvironmentFile= does.
INSTANCE_SERVICES="gateway logbook qserver tiled-writer scanner"
instance_templates_of() { case "$1" in
    gateway) echo "GeecsCAGateway/deploy/geecs-ca-gateway@.service";;
    logbook) echo "GeecsLogbook/deploy/geecs-logbook@.service";;
    qserver) echo "GeecsBluesky/qserver/deploy/geecs-qserver@.service GeecsBluesky/qserver/deploy/geecs-qserver-ready@.service";;
    tiled-writer) echo "GeecsBluesky/qserver/deploy/geecs-tiled-writer@.service";;
    scanner) echo "GeecsScanner/deploy/geecs-scanner@.service";; esac; }
if [ -n "$INSTANCE" ]; then
    case "$INSTANCE" in *[!A-Za-z0-9_-]*) echo "--experiment: instance '$INSTANCE' is not a slug (letters, digits, '-', '_')" >&2; exit 2 ;; esac
    EXPERIMENTS_DIR="${GEECS_EXPERIMENTS_DIR:-/etc/geecs/experiments}"
    INSTANCE_ENV_INSTALLED="$EXPERIMENTS_DIR/$INSTANCE.env"
    INSTANCE_ENV="${INSTANCE_ENV:-$INSTANCE_ENV_INSTALLED}"
    [ -f "$INSTANCE_ENV" ] || { echo "no instance file: $INSTANCE_ENV (write it from deploy/experiments.env.example, or deploy/experiment.sh up)" >&2; exit 2; }
    INSTANCE_ENV="$(cd "$(dirname "$INSTANCE_ENV")" && pwd)/$(basename "$INSTANCE_ENV")"
    load_site_env "$INSTANCE_ENV"
    check_instance_env "$INSTANCE" "$INSTANCE_ENV"
    if [ -n "$ONLY" ]; then
        for requested in $(printf '%s' "$ONLY" | tr ',' ' '); do
            case " $INSTANCE_SERVICES " in *" $requested "*) ;; *) echo "bootstrap_host.sh: --only: '$requested' has no instance template (instance services: $INSTANCE_SERVICES)" >&2; exit 2 ;; esac
        done
    fi
    iwanted() { [ -z "$ONLY" ] && return 0; case ",$ONLY," in *",$1,"*) return 0;; *) return 1;; esac; }
    say "instance '$INSTANCE' — experiment '$GEECS_EXPERIMENT' (slot ${GEECS_INSTANCE_SLOT:-?}) — instance file $INSTANCE_ENV"
    STAGE="$GEECS_CHECKOUT_ROOT/deploy-staging/instances"
    # Where the instances' clones live: the main ones by default; a separate
    # root (GEECS_INSTANCE_CHECKOUT_ROOT in site.env) lets an instance run
    # code the main units' clones are not on, without a deploy of theirs.
    INSTANCE_ROOT="${GEECS_INSTANCE_CHECKOUT_ROOT:-$GEECS_CHECKOUT_ROOT}"
    echo "  clones under $INSTANCE_ROOT; a missing one is cloned at --ref $REF (the ref the instance templates and their code are on)"
    TEMPLATE_PATHS=(); ENABLE=(); NEW_CLONES=" "; INSTALLED_ENVS=" "
    for s in $INSTANCE_SERVICES; do
        iwanted "$s" || continue
        dir="$INSTANCE_ROOT/$(clone_of "$s")"
        # A clone this run created is ours to install into (every package dir
        # in it, once); one that already existed is used exactly as it is.
        case "$NEW_CLONES" in *" $dir "*) ;; *)
            if [ -e "$dir/.git" ]; then
                echo "  $s: $(clone_of "$s") exists ($(git -C "$dir" rev-parse --abbrev-ref HEAD) $(git -C "$dir" rev-parse --short=8 HEAD)) — used as it is"
            else
                run git clone --quiet --branch "$REF" "$GEECS_REPO_URL" "$dir"; NEW_CLONES="$NEW_CLONES$dir "
            fi ;;
        esac
        pkg="$dir/$(pkgdir_of "$s")"
        case "$NEW_CLONES" in *" $dir "*)
            if [ "$INSTALL" -eq 1 ]; then case "$INSTALLED_ENVS" in *" $pkg "*) ;; *)
                INSTALLED_ENVS="$INSTALLED_ENVS$pkg "; extras="$(extras_of "$s")"
                if [ "$DRY" -eq 1 ]; then echo "  [dry] (cd $pkg && poetry install${extras:+ --extras \"$extras\"})"
                elif [ -n "$extras" ]; then (cd "$pkg" && run "$GEECS_POETRY" install --quiet --extras "$extras")
                else (cd "$pkg" && run "$GEECS_POETRY" install --quiet); fi ;; esac
            fi ;;
        esac
        missing=""
        svc_paths=()
        for rel in $(instance_templates_of "$s"); do
            t="$dir/$rel"
            if [ "$DRY" -eq 1 ] && [ ! -e "$dir/.git" ]; then t="$REPO_ROOT/$rel"; fi   # a dry run cloned nothing: show this clone's template
            if [ -f "$t" ] && is_unit_template "$t"; then svc_paths+=("$t"); else missing="$missing $rel"; fi
        done
        if [ -n "$missing" ]; then
            echo "  $s: WARNING — $(clone_of "$s") lacks the instance template(s)$missing (it predates them); nothing rendered for $s — pulling that clone forward is a deploy of the main experiment's service, decided by a human"
            continue
        fi
        TEMPLATE_PATHS+=("${svc_paths[@]}")
        for rel in $(instance_templates_of "$s"); do u="$(basename "$rel" .service)"; ENABLE+=("${u}${INSTANCE}.service"); done
    done
    say "instance units (templates only, to $STAGE)"
    [ "$DRY" -eq 0 ] && [ -d "$STAGE" ] && rm -f "$STAGE"/*.service
    if [ "${#TEMPLATE_PATHS[@]}" -eq 0 ]; then echo "  nothing to render"
    elif [ "$DRY" -eq 1 ]; then echo "  [dry] render_units.sh $SITE_ENV $STAGE ${TEMPLATE_PATHS[*]}"
    else RENDER_QUIET=1 "$REPO_ROOT/deploy/render_units.sh" "$SITE_ENV" "$STAGE" "${TEMPLATE_PATHS[@]}" | sed 's/^/  /'; fi
    say "root steps (a human runs these; nothing above needed sudo)"
    if ! [ "$INSTANCE_ENV" -ef "$INSTANCE_ENV_INSTALLED" ]; then echo "  sudo install -D -m 0644 \"$INSTANCE_ENV\" \"$INSTANCE_ENV_INSTALLED\""; fi
    if [ "${#TEMPLATE_PATHS[@]}" -gt 0 ]; then
        echo "  sudo install -m 0644 \"$STAGE\"/*.service /etc/systemd/system/"
        echo "  sudo systemctl daemon-reload"
        echo "  sudo systemctl enable --now ${ENABLE[*]}"
    fi
    exit 0
fi

# The archiver is opt-in BY SITE: with no --only it is wanted only when
# site.env names an appliance (GEECS_ARCHIVER_HOST set — the PVA "empty =
# none" precedent), so a host without one keeps the smaller profile and a
# routine full rerun never demands Docker. `--only archiver` on such a site
# fails on the missing key, loudly, below.
wanted() {
    if [ -z "$ONLY" ]; then
        if [ "$1" = "archiver" ] && [ -z "${GEECS_ARCHIVER_HOST:-}" ]; then return 1; fi
        # geecs-mcp likewise: without the services' host it could only
        # answer service_unreachable, so a site without the key deploys none.
        if [ "$1" = "mcp" ] && [ -z "${GEECS_SERVICES_HOST:-}" ]; then return 1; fi
        return 0
    fi
    case ",$ONLY," in *",$1,"*) return 0;; *) return 1;; esac
}
# An --only name that is not a service used to select nothing and exit 0 —
# indistinguishable from success, so a stale runbook (or a retired role)
# looked like it had run. Name it instead.
if [ -n "$ONLY" ]; then
    for requested in $(printf '%s' "$ONLY" | tr ',' ' '); do
        case " $SERVICES " in
            *" $requested "*) ;;
            *) echo "bootstrap_host.sh: --only: unknown service '$requested'" >&2
               echo "  known services: $SERVICES" >&2
               exit 2 ;;
        esac
    done
fi

# The archiver's own site values (its reachable address, where the archive
# lives, the JVM heap): required when the archiver is wanted — render_conf.sh
# would refuse later, but a missing key must fail here, before anything runs.
if wanted archiver; then require_site_keys GEECS_ARCHIVER_HOST GEECS_ARCHIVER_DATA_ROOT GEECS_ARCHIVER_JAVA_OPTS; fi
# geecs-mcp reaches the scanner, portal and logbook at this host ([mcp] in config.ini).
if wanted mcp; then require_site_keys GEECS_SERVICES_HOST; fi

say "site '${GEECS_SITE:-?}' experiment '$GEECS_EXPERIMENT' — ref $REF — root $GEECS_CHECKOUT_ROOT"
if [ -z "$ONLY" ] && [ -z "${GEECS_ARCHIVER_HOST:-}" ]; then
    echo "  archiver: not wanted — site.env has no GEECS_ARCHIVER_HOST (this site runs no Archiver Appliance; set the three GEECS_ARCHIVER_* keys to deploy one)"
fi
if [ -z "$ONLY" ] && [ -z "${GEECS_SERVICES_HOST:-}" ]; then
    echo "  mcp: not wanted — site.env has no GEECS_SERVICES_HOST (set it to the host of the scanner, portal and logbook to deploy geecs-mcp)"
fi
if [ "$(id -un)" != "$GEECS_SERVICE_USER" ]; then
    echo "WARNING: running as $(id -un), site.env says the service account is $GEECS_SERVICE_USER;" >&2
    echo "         poetry keys venvs under the invoking user — units run as $GEECS_SERVICE_USER and will not find them." >&2
fi

say "prerequisites"
prereq_fail() { if [ "$DRY" -eq 1 ]; then echo "  WARNING (dry-run continues): $1"; else echo "$1" >&2; exit 1; fi; }
for tool in git python3.11; do command -v "$tool" >/dev/null || prereq_fail "missing: $tool"; done
[ -x "$GEECS_POETRY" ] || prereq_fail "GEECS_POETRY not executable: $GEECS_POETRY (install poetry as the service account)"
echo "  git $(git --version 2>/dev/null | awk '{print $3}'), $(python3.11 --version 2>/dev/null || echo 'python3.11 ?'), poetry $("$GEECS_POETRY" --version 2>/dev/null | awk '{print $NF}' | tr -d ')')"
[ -d "$GEECS_DATA_ROOT" ] && echo "  data share mounted at $GEECS_DATA_ROOT" || echo "  WARNING: data share not mounted at $GEECS_DATA_ROOT (services start without it, then fail loudly)"
[ -d "$GEECS_CONFIGS_ROOT" ] && echo "  configs repo at $GEECS_CONFIGS_ROOT" || echo "  WARNING: configs repo not at $GEECS_CONFIGS_ROOT"
# Redis is the state store the queueserver keeps its queue, history and
# permissions in. A missing package does NOT fail loudly: the unit treats
# redis-server.service as ordering only, and launch_re_manager.sh starts its
# own "redis-server --daemonize yes" whenever nothing answers on 6379 — so the
# omission silently becomes an UNSUPERVISED Redis that dies with the launcher,
# is absent after a reboot, and comes back empty (host finding 2026-09-06).
# Judge the packaged unit, not just a listener on the port.
REDIS_ROOT_STEP=""
if wanted qserver; then
    # systemctl PRINTS the state and still exits non-zero (is-enabled 1 for
    # disabled), so `$(cmd || echo default)` would capture both and split the
    # warning across two lines. Take the first line; default only when empty.
    # `|| true` inside the group is required, not decorative: this script runs
    # under `set -e -o pipefail`, so an is-enabled exit of 1 (disabled) or 127
    # (no systemd at all) would otherwise abort the whole bootstrap.
    redis_enabled="$({ systemctl is-enabled redis-server.service 2>/dev/null || true; } | head -1)"
    [ -n "$redis_enabled" ] || redis_enabled="absent"
    if [ "$redis_enabled" = "enabled" ]; then
        echo "  redis-server.service enabled (the queueserver state store)"
    else
        echo "  WARNING: redis-server.service is $redis_enabled — the queueserver launcher would start an UNSUPERVISED redis on 6379; the root steps below install the package"
        REDIS_ROOT_STEP="yes"
    fi
fi
# The archiver runs upstream's appliance image under `docker compose`, as
# the service account. Three things must hold before its unit can start:
# Docker Engine with the compose plugin installed, docker.service enabled,
# and the service account in the docker group. Judge each (the Redis
# pattern); the root steps install what is missing. Never Docker Desktop.
DOCKER_ROOT_STEP=""
if wanted archiver; then
    docker_bin="$(command -v docker || true)"
    compose_ok="no"; [ -n "$docker_bin" ] && docker compose version >/dev/null 2>&1 && compose_ok="yes"
    docker_enabled="$({ systemctl is-enabled docker.service 2>/dev/null || true; } | head -1)"
    [ -n "$docker_enabled" ] || docker_enabled="absent"
    if id -u "$GEECS_SERVICE_USER" >/dev/null 2>&1; then
        if id -nG "$GEECS_SERVICE_USER" 2>/dev/null | tr " " "\n" | grep -qx docker; then in_group="yes"; else in_group="no"; fi
    else in_group="unknown (no user $GEECS_SERVICE_USER on this machine)"; fi
    if [ -n "$docker_bin" ] && [ "$compose_ok" = "yes" ] && [ "$docker_enabled" = "enabled" ] && [ "$in_group" = "yes" ]; then
        echo "  docker ready for the archiver: $docker_bin, compose plugin, docker.service enabled, $GEECS_SERVICE_USER in the docker group"
    else
        echo "  WARNING: docker is not ready for the archiver (binary: ${docker_bin:-none}; compose plugin: $compose_ok; docker.service: $docker_enabled; $GEECS_SERVICE_USER in docker group: $in_group) — the root steps below install and enable it"
        DOCKER_ROOT_STEP="yes"
    fi
fi

say "clones (one per service family, at $REF)"
# Clone dirs that are NOT usable clones are recorded here; every later stage
# (environments, units, root steps) skips their services, so the run cannot
# end with a clean tail after a warning scrolled off the top.
SKIP_CLONES=" "
skipped_clone() { case "$SKIP_CLONES" in *" $1 "*) return 0;; *) return 1;; esac; }
for c in $(for s in $SERVICES; do wanted "$s" && clone_of "$s"; done | sort -u); do
    dir="$GEECS_CHECKOUT_ROOT/$c"
    if [ -e "$dir/.git" ]; then
        gitdir="$(git -C "$dir" rev-parse --git-dir 2>/dev/null || true)"
        common="$(git -C "$dir" rev-parse --git-common-dir 2>/dev/null || true)"
        top="$(git -C "$dir" rev-parse --show-toplevel 2>/dev/null || true)"
        if [ -z "$gitdir" ] || [ "$top" != "$(cd "$dir" && pwd -P)" ]; then
            # Unreadable .git, or git walked UP to an enclosing repository (a
            # stray .git under a git-managed home): not this directory's repo.
            echo "  $c: WARNING — .git present but git cannot read it as this directory's repository (dangling worktree, corrupt, or a stray .git inside another repo); skipped — fix by hand"
            SKIP_CLONES="$SKIP_CLONES$c "; continue
        fi
        case "$gitdir" in
            "$common"/worktrees/*)
                # A linked worktree shares its object store AND its branch refs
                # with another clone. If the same branch is checked out in both
                # (git allows that only via --ignore-other-worktrees/--force —
                # how the 2026-09-03 state arose), a pull in one moves the
                # other's HEAD without a checkout, and that service runs an older
                # tree than its HEAD claims. Not a clone of its own — see the
                # site profile page for the replacement recipe.
                echo "  $c: WARNING — linked worktree of $(cd "$dir" && cd "$common/.." && pwd), not a clone of its own; skipped (see the site profile page)"
                SKIP_CLONES="$SKIP_CLONES$c "; continue ;;
        esac
        echo "  $c: exists ($(git -C "$dir" rev-parse --abbrev-ref HEAD 2>/dev/null) $(git -C "$dir" rev-parse --short=8 HEAD 2>/dev/null)) — fetching, NOT moving HEAD (a pull is a deploy; do it per service, deliberately)"
        run git -C "$dir" fetch --quiet origin
    else
        run git clone --quiet --branch "$REF" "$GEECS_REPO_URL" "$dir"
    fi
done

say "environments (inside each package dir, with that service's extras)"
# Two services on one package dir with one extras set (the worker and the
# Tiled writer) share one env: install it once per run.
INSTALLED_ENVS=" "
for s in $SERVICES; do
    wanted "$s" || continue
    dir="$GEECS_CHECKOUT_ROOT/$(clone_of "$s")/$(pkgdir_of "$s")"
    skipped_clone "$(clone_of "$s")" && { echo "  $s: skipped — $(clone_of "$s") is not a usable clone (see the clones stage)"; continue; }
    [ "$INSTALL" -eq 1 ] || { echo "  $s: skipped (--no-install)"; continue; }
    if [ ! -d "$dir" ]; then
        echo "  $s: clone not present yet ($dir) — rerun after the clone step has run for real"; continue
    fi
    case "$INSTALLED_ENVS" in *" $dir "*) echo "  $s: shares the env installed above ($dir)"; continue ;; esac
    INSTALLED_ENVS="$INSTALLED_ENVS$dir "
    if [ "$s" = "mcp" ]; then
        venv="$GEECS_CHECKOUT_ROOT/geecs-mcp-venv"
        [ -d "$venv" ] || run python3.11 -m venv "$venv"
        # Non-editable by design: the running code is baked into the venv, so a
        # pull in qs-checkout never mutates code under the running MCP server.
        extras="$(extras_of mcp)"
        run "$venv/bin/pip" install --quiet --upgrade "$dir${extras:+[$extras]}"
        continue
    fi
    extras="$(extras_of "$s")"
    if [ -n "$extras" ]; then
        (cd "$dir" && run "$GEECS_POETRY" install --quiet --extras "$extras")
    else
        (cd "$dir" && run "$GEECS_POETRY" install --quiet)
    fi
done

say "config.ini for the service account (rendered from site.env only if absent)"
CFG="$GEECS_SERVICE_HOME/.config/geecs_python_api/config.ini"
render_config_ini() {
        cat <<EOF
# Rendered by deploy/bootstrap_host.sh from site.env ($(date +%F)). This is the
# client half of the site profile (docs/platform/site_profile.md); the key
# reference is docs/tutorials/getting_started.md. Edit freely.
[Paths]
geecs_data = $GEECS_DATA_ROOT
GEECS_DATA_LOCAL_BASE_PATH = $GEECS_DATA_ROOT
scanner_config_root_path = $GEECS_CONFIGS_ROOT
scan_analysis_configs_path = $GEECS_CONFIGS_ROOT/scan_analysis_configs

[Experiment]
expt = $GEECS_EXPERIMENT

[tiled]
uri = $GEECS_TILED_URI
# api_key = <ask the Tiled admin; never in site.env>

[epics]
ca_addr_list = $EPICS_CA_ADDR_LIST

[pva]
addr_list = ${GEECS_PVA_ADDR_LIST:-}
file_plugin_addr_list = ${GEECS_PVA_FILE_PLUGIN_ADDR_LIST:-}

[qserver]
host = $GEECS_QSERVER_HOST
doc_addr = $GEECS_QS_DOC_ADDR
EOF
        # The appliance's address for geecs-archiver (and the fleet probes), when
        # this site runs one. Port 17665 is a fleet constant, not a site value.
        if [ -n "${GEECS_ARCHIVER_HOST:-}" ]; then
            printf '\n[archiver]\nurl = http://%s:17665\n' "$GEECS_ARCHIVER_HOST"
        fi
        # The three services geecs-mcp is a client of, when this site names
        # their host. Ports are fleet constants (GEECS-MCP/DESIGN.md § 10).
        if [ -n "${GEECS_SERVICES_HOST:-}" ]; then render_mcp_section; fi
}
render_mcp_section() {
        printf '\n[mcp]\nscanner_url = http://%s:8300\nportal_url = http://%s:8200\nlogbook_url = http://%s:8400\n' \
            "$GEECS_SERVICES_HOST" "$GEECS_SERVICES_HOST" "$GEECS_SERVICES_HOST"
}
if [ -s "$CFG" ]; then   # -s: an empty placeholder file counts as absent
    # Never overwritten (it also holds the hand-entered Tiled api_key), but
    # site.env is the root: show where the existing file disagrees with the
    # rendered form so a stale client config is reconciled, not kept silently.
    echo "  exists: $CFG — left untouched; differences vs the form site.env would render (< existing, > rendered):"
    # Both sides normalised the way configparser reads them (comments and
    # blanks dropped, api_key hidden, `key=value` spacing and key case
    # unified) so only real value changes show. Relies on pipefail: the
    # `if` tests diff's status (1 = differences), not sed's.
    ini_norm() { grep -vE '^[[:space:]]*#|^[[:space:]]*$|api_key' | sed -E 's/^([^=]*[^[:space:]=])[[:space:]]*=[[:space:]]*/\1 = /' | awk -F' = ' '{ if (NF > 1) { k = tolower($1); sub(/^[^=]*= /, ""); print k " = " $0 } else print }'; }
    if diff <(ini_norm < "$CFG") <(render_config_ini | ini_norm) | sed 's/^/      /'; then
        echo "      (none — the file matches site.env)"
    fi
    # One additive migration: a site that gains an Archiver Appliance needs
    # [archiver] url in the EXISTING file, or geecs-archiver and the fleet
    # probes treat the appliance as unconfigured. Only a missing section is
    # appended; nothing already in the file is rewritten (the api_key rule).
    if [ -n "${GEECS_ARCHIVER_HOST:-}" ] && ! grep -qE '^[[:space:]]*\[archiver\][[:space:]]*$' "$CFG"; then
        if [ "$DRY" -eq 1 ]; then echo "  [dry] would append [archiver] url = http://$GEECS_ARCHIVER_HOST:17665 to $CFG"
        else printf '\n[archiver]\nurl = http://%s:17665\n' "$GEECS_ARCHIVER_HOST" >> "$CFG"; echo "  appended [archiver] url = http://$GEECS_ARCHIVER_HOST:17665 to $CFG (the only change made to an existing file)"; fi
    fi
    # The same additive migration for geecs-mcp's three service URLs. A file
    # that already has an [mcp] section (a pre-0.10 MCP host: client_identity,
    # max_shots) is not edited — the keys are printed to add by hand.
    if [ -n "${GEECS_SERVICES_HOST:-}" ]; then
        # Any [mcp] header counts as present (a trailing comment included):
        # a second section would be configparser's DuplicateSectionError.
        if ! grep -qE '^[[:space:]]*\[mcp\]' "$CFG"; then
            if [ "$DRY" -eq 1 ]; then echo "  [dry] would append [mcp] scanner_url/portal_url/logbook_url (http://$GEECS_SERVICES_HOST:8300/8200/8400) to $CFG"
            else render_mcp_section >> "$CFG"; echo "  appended [mcp] scanner_url/portal_url/logbook_url (http://$GEECS_SERVICES_HOST:8300/8200/8400) to $CFG"; fi
        else
            # The keys the existing [mcp] section holds, read inside it only.
            # configparser: section names are case-sensitive, keys are not,
            # and a key ends at its first '=' or ':'.
            have="$(awk '/^[[:space:]]*\[/ { insec = ($0 ~ /^[[:space:]]*\[mcp\]/) }
                         insec && /[=:]/ { k = $0; sub(/[=:].*/, "", k); gsub(/[[:space:]]/, "", k); print tolower(k) }' "$CFG")"
            missing=""
            for key in scanner_url portal_url logbook_url; do
                printf '%s\n' "$have" | grep -qx "$key" || missing="$missing $key"
            done
            if [ -n "$missing" ]; then
                echo "  NOTE: $CFG's [mcp] section lacks$missing — those verbs answer service_unreachable until these lines are added under [mcp] by hand:"
                for key in $missing; do render_mcp_section | grep "^$key = " | sed 's/^/      /'; done
            fi
        fi
    fi
elif [ "$DRY" -eq 1 ]; then echo "  [dry] would write $CFG:"; render_config_ini | sed 's/^/      /'
else
    # 0600: the operator adds the Tiled api_key to this file by hand.
    mkdir -p "$(dirname "$CFG")"; (umask 077; render_config_ini > "$CFG"); chmod 600 "$CFG"; echo "  wrote $CFG (mode 600; add [tiled] api_key by hand)"
fi

say "units (each rendered from ITS service's clone, to a staging dir)"
# A unit must match the code it will run: the gateway clone may sit pinned
# at an older, verified deploy while qs-checkout is at master, so the
# gateway's unit comes from the gateway's clone — never from whichever
# clone this script happens to run in (that clone is only the fallback).
STAGE="$GEECS_CHECKOUT_ROOT/deploy-staging"
templates_of() { case "$1" in
    gateway) echo "GeecsCAGateway/deploy/geecs-ca-gateway.service";; portal) echo "GEECS-DataPortal/deploy/geecs-data-portal.service";;
    logbook) echo "GeecsLogbook/deploy/geecs-logbook.service";;
    qserver) echo "GeecsBluesky/qserver/deploy/geecs-qserver.service GeecsBluesky/qserver/deploy/geecs-qserver-ready.service";;
    tiled-writer) echo "GeecsBluesky/qserver/deploy/geecs-tiled-writer.service";;
    mcp) echo "GEECS-MCP/deploy/geecs-mcp.service";;
    scanner) echo "GeecsScanner/deploy/geecs-scanner.service";;
    archiver) echo "GeecsArchiver/deploy/geecs-archiver.service";; esac; }
TEMPLATE_PATHS=()
# Services whose clone predates the templated units: no unit is rendered or
# enabled for them (pulling that clone forward is a deploy of that service,
# decided deliberately — never forced by a bootstrap rerun for another one).
SKIP_SERVICES=" "
skipped_service() { case "$SKIP_SERVICES" in *" $1 "*) return 0;; *) return 1;; esac; }
for s in $SERVICES; do
    wanted "$s" || continue
    skipped_clone "$(clone_of "$s")" && { echo "  $s: skipped — $(clone_of "$s") is not a usable clone"; continue; }
    # A service's units come from ITS clone, all of them or none: a clone that
    # predates one of the templates (e.g. the readiness oneshot, #793) is
    # skipped whole rather than half-rendered.
    svc_paths=()
    for rel in $(templates_of "$s"); do
        t="$GEECS_CHECKOUT_ROOT/$(clone_of "$s")/$rel"
        if [ -f "$t" ]; then
            if ! is_unit_template "$t"; then
                echo "  $s: WARNING — $(clone_of "$s") predates the templated units ($t: directives lack User=@SERVICE_USER@ / EnvironmentFile=@SITE_ENV@); no unit rendered or enabled — pull that clone forward (a deploy of $s), then rerun"
                SKIP_SERVICES="$SKIP_SERVICES$s "; break
            fi
            svc_paths+=("$t")
        elif [ -d "$GEECS_CHECKOUT_ROOT/$(clone_of "$s")" ]; then
            echo "  $s: WARNING — $(clone_of "$s") lacks $rel (predates that unit); no unit rendered or enabled for $s — pull that clone forward (a deploy of $s), then rerun"
            SKIP_SERVICES="$SKIP_SERVICES$s "; break
        else echo "  $s: clone absent — $(basename "$rel") from this clone ($REPO_ROOT)"; svc_paths+=("$REPO_ROOT/$rel"); fi
    done
    if ! skipped_service "$s" && [ "${#svc_paths[@]}" -gt 0 ]; then TEMPLATE_PATHS+=("${svc_paths[@]}"); fi
done
# The staging dir holds only THIS run's units: a stale .service from an
# earlier run (a service since skipped) must never ride along on the
# install line. cutover/other files in the dir are left alone.
[ "$DRY" -eq 0 ] && [ -d "$STAGE" ] && rm -f "$STAGE"/*.service
if [ "${#TEMPLATE_PATHS[@]}" -eq 0 ]; then echo "  nothing to render (every wanted service was skipped)"
elif [ "$DRY" -eq 1 ]; then echo "  [dry] render_units.sh $SITE_ENV $STAGE ${TEMPLATE_PATHS[*]}"
else RENDER_QUIET=1 "$REPO_ROOT/deploy/render_units.sh" "$SITE_ENV" "$STAGE" "${TEMPLATE_PATHS[@]}" | sed 's/^/  /'; fi

# The archiver's conf is not a unit: compose.yaml and appliances.xml carry
# the same kind of install-time holes, filled by the PACKAGE's renderer
# (render_units.sh refuses non-units by design) from the archiver's own
# clone, into $STAGE/archiver, beside its static files. Root installs the
# directory under /etc/geecs/archiver (the root steps below).
ARCHIVER_CONF=""
if wanted archiver; then
    say "archiver conf (compose.yaml + appliances.xml from site.env, beside the static conf, to the staging dir)"
    if skipped_clone archiver-checkout || skipped_service archiver; then
        echo "  skipped — archiver-checkout is not usable or predates the archiver (see above)"
    else
        rc="$GEECS_CHECKOUT_ROOT/archiver-checkout/GeecsArchiver/deploy/render_conf.sh"
        [ -x "$rc" ] || { echo "  clone absent — render_conf.sh from this clone ($REPO_ROOT)"; rc="$REPO_ROOT/GeecsArchiver/deploy/render_conf.sh"; }
        if [ "$DRY" -eq 1 ]; then echo "  [dry] $rc $SITE_ENV $STAGE/archiver"
        else RENDER_QUIET=1 "$rc" "$SITE_ENV" "$STAGE/archiver" | sed 's/^/  /'; fi
        ARCHIVER_CONF="yes"
    fi
fi

say "root steps (a human runs these; nothing above needed sudo)"
if [ -n "$REDIS_ROOT_STEP" ]; then
    # First: geecs-qserver.service is ordered After= this unit, and its
    # launcher's fallback is what we are here to prevent. Never carry a
    # dump.rdb across Redis major versions — see the qserver runbook.
    echo "  sudo apt-get update && sudo apt-get install -y redis-server   # queueserver state store; the package default (loopback only) is correct"
    echo "  echo 'vm.overcommit_memory = 1' | sudo tee /etc/sysctl.d/99-redis-overcommit.conf >/dev/null && sudo sysctl --system >/dev/null"
    echo "  sudo systemctl enable --now redis-server.service"
fi
if [ -n "$DOCKER_ROOT_STEP" ]; then
    # The archiver's runtime. The group line takes effect for the unit at its
    # next start (systemd reads /etc/group then); an interactive shell needs a
    # new login to use the docker CLI.
    echo "  sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2   # the archiver's runtime: Engine + compose plugin (never Docker Desktop)"
    echo "  sudo usermod -aG docker $GEECS_SERVICE_USER   # the unit runs docker compose as the service account"
    echo "  sudo systemctl enable --now docker.service"
fi
echo "  sudo install -D -m 0644 \"$SITE_ENV\" \"$SITE_ENV_INSTALLED\""
if [ "${#TEMPLATE_PATHS[@]}" -gt 0 ]; then
    echo "  sudo install -m 0644 \"$STAGE\"/*.service /etc/systemd/system/"
    echo "  sudo systemctl daemon-reload"
else
    echo "  # nothing staged — no units to install"
fi
if [ -n "$ARCHIVER_CONF" ]; then
    echo "  sudo install -d -m 0755 /etc/geecs/archiver && sudo install -m 0644 \"$STAGE/archiver\"/* /etc/geecs/archiver/   # the appliance's rendered conf (compose.yaml, appliances.xml, policies.py, ...)"
    echo "  sudo install -d -o $GEECS_SERVICE_USER -g $GEECS_SERVICE_USER -m 0750 \"$GEECS_ARCHIVER_DATA_ROOT/sts\" \"$GEECS_ARCHIVER_DATA_ROOT/lts\"   # the archive: short-term and long-term stores"
fi
for s in $SERVICES; do
    wanted "$s" || continue
    if skipped_clone "$(clone_of "$s")"; then echo "  # $(units_of "$s"): NOT enabled — $(clone_of "$s") is not a usable clone (see above)"
    elif skipped_service "$s"; then echo "  # $(units_of "$s"): NOT enabled — $(clone_of "$s") predates the templated units; pull it forward, then rerun (see above)"
    else echo "  sudo systemctl enable --now $(units_of "$s")"; fi
done
echo
if [ "$SKIP_SERVICES" != " " ]; then
    echo "WARNING: no unit rendered for:${SKIP_SERVICES}— their clone predates the templated units (see the units stage)." >&2
    echo
fi
if [ "$SKIP_CLONES" != " " ]; then
    # ${VAR} braces on purpose: a bare $VAR abutting a non-ASCII character is
    # parsed as part of the name by bash 3.2 under UTF-8 (unbound variable).
    echo "WARNING: skipped clone dir(s):${SKIP_CLONES}— not clones of their own (linked worktree or unreadable .git)." >&2
    echo "         Replace each in a maintenance window per docs/platform/site_profile.md, then rerun this script." >&2
    echo
fi
if [ -n "$REDIS_ROOT_STEP" ]; then
    echo "WARNING: redis-server.service is not enabled on this host — run the Redis root step above BEFORE" >&2
    echo "         enabling geecs-qserver, or its launcher will start an unsupervised Redis (see the qserver runbook)." >&2
    echo
fi
if [ -n "$DOCKER_ROOT_STEP" ]; then
    echo "WARNING: docker is not ready on this host — run the Docker root steps above BEFORE enabling geecs-archiver" >&2
    echo "         (the unit Requires=docker.service and runs compose as $GEECS_SERVICE_USER, who must be in the docker group)." >&2
    echo
fi
echo "Then: scripts/fleet_status.sh from any client — every row should read systemd / clean / matching versions."
if [ -n "$ARCHIVER_CONF" ]; then
    echo "Then, for the archiver: geecs-archiver onboard --experiment $GEECS_EXPERIMENT --dry-run (from the archiver clone's poetry env, or any client with [archiver] url) — GeecsArchiver/DEPLOYMENT.md § Onboarding."
fi
