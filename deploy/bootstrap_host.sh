#!/usr/bin/env bash
# bootstrap_host.sh — bring a service host to "every service has its clone,
# its env, its config, and a rendered unit" from ONE site.env. Idempotent
# and unprivileged: run it as the service account, as many times as you
# like; it never touches /etc, never restarts anything, and prints the
# root steps for a human at the end.
#
#   deploy/bootstrap_host.sh SITE_ENV [--ref REF] [--only svc,svc] [--dry-run] [--no-install]
#
#   --ref REF      git ref to check out in each clone (default: master)
#   --only LIST    comma-separated subset of: gateway,portal,qserver,capture,mcp
#   --no-install   clone/fetch only; skip poetry/pip installs
#   --dry-run      print what would happen
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
REF="master"; ONLY=""; DRY=0; INSTALL=1
while [ $# -gt 0 ]; do
    case "$1" in
        --ref) shift; REF="${1:-master}" ;;
        --only) shift; ONLY="${1:-}" ;;
        --dry-run) DRY=1 ;;
        --no-install) INSTALL=0 ;;
        *) echo "usage: bootstrap_host.sh SITE_ENV [--ref REF] [--only LIST] [--dry-run] [--no-install]" >&2; exit 2 ;;
    esac
    shift
done
[ -n "$SITE_ENV" ] && [ -f "$SITE_ENV" ] || { echo "usage: bootstrap_host.sh SITE_ENV ..." >&2; exit 2; }
SITE_ENV="$(cd "$(dirname "$SITE_ENV")" && pwd)/$(basename "$SITE_ENV")"   # absolute: the printed sudo lines must work from any cwd
SITE_ENV_INSTALLED="${GEECS_SITE_ENV_PATH:-/etc/geecs/site.env}"          # same knob render_units.sh honours

. "$REPO_ROOT/deploy/site_env_lib.sh"
load_site_env "$SITE_ENV"
check_site_env_consistency
require_site_keys GEECS_SERVICE_USER GEECS_SERVICE_HOME GEECS_CHECKOUT_ROOT GEECS_POETRY GEECS_REPO_URL \
    GEECS_EXPERIMENT GEECS_TILED_URI GEECS_QSERVER_HOST GEECS_QS_DOC_ADDR GEECS_DATA_ROOT GEECS_CONFIGS_ROOT EPICS_CA_ADDR_LIST

run() { if [ "$DRY" -eq 1 ]; then echo "  [dry] $*"; else echo "  \$ $*"; "$@"; fi; }
say() { printf '\n== %s\n' "$1"; }

# ---- the service table: name | clone dir | package dir | poetry extras ---
# One clone per service FAMILY (fleet map § one clone per service). The
# queueserver and capture daemon share qs-checkout by design (co-location
# and co-versioning are a requirement of the capture design); the MCP
# server is baked from qs-checkout into its own venv (config-truth parity
# with the worker, without a shared working tree under a running service).
SERVICES="gateway portal qserver capture mcp"
clone_of()   { case "$1" in gateway) echo "gateway-checkout";; portal) echo "portal-checkout";; qserver|capture|mcp) echo "qs-checkout";; esac; }
pkgdir_of()  { case "$1" in gateway) echo "GeecsCAGateway";; portal) echo "GEECS-DataPortal";; qserver|capture) echo "GeecsBluesky";; mcp) echo "GEECS-MCP";; esac; }
extras_of()  { case "$1" in gateway) echo "";; portal) echo "analysis";; qserver) echo "ca tiled qserver";; capture) echo "ca tiled qserver capture";; mcp) echo "analysis-run";; esac; }
unit_of()    { case "$1" in gateway) echo "geecs-ca-gateway";; portal) echo "geecs-data-portal";; qserver) echo "geecs-qserver";; capture) echo "geecs-capture";; mcp) echo "geecs-mcp";; esac; }
wanted()     { [ -z "$ONLY" ] || case ",$ONLY," in *",$1,"*) return 0;; *) return 1;; esac; }

say "site '${GEECS_SITE:-?}' experiment '$GEECS_EXPERIMENT' — ref $REF — root $GEECS_CHECKOUT_ROOT"
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

say "clones (one per service family, at $REF)"
for c in $(for s in $SERVICES; do wanted "$s" && clone_of "$s"; done | sort -u); do
    dir="$GEECS_CHECKOUT_ROOT/$c"
    if [ -d "$dir/.git" ]; then
        echo "  $c: exists ($(git -C "$dir" rev-parse --abbrev-ref HEAD) $(git -C "$dir" rev-parse --short=8 HEAD)) — fetching, NOT moving HEAD (a pull is a deploy; do it per service, deliberately)"
        run git -C "$dir" fetch --quiet origin
    else
        run git clone --quiet --branch "$REF" "$GEECS_REPO_URL" "$dir"
    fi
done

say "environments (inside each package dir, with that service's extras)"
for s in $SERVICES; do
    wanted "$s" || continue
    dir="$GEECS_CHECKOUT_ROOT/$(clone_of "$s")/$(pkgdir_of "$s")"
    [ "$INSTALL" -eq 1 ] || { echo "  $s: skipped (--no-install)"; continue; }
    if [ ! -d "$dir" ]; then
        echo "  $s: clone not present yet ($dir) — rerun after the clone step has run for real"; continue
    fi
    if [ "$s" = "mcp" ]; then
        venv="$GEECS_CHECKOUT_ROOT/geecs-mcp-venv"
        [ -d "$venv" ] || run python3.11 -m venv "$venv"
        # Non-editable by design: the running code is baked into the venv, so a
        # pull in qs-checkout never mutates code under the running MCP server.
        run "$venv/bin/pip" install --quiet --upgrade "$dir[$(extras_of mcp)]"
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
if [ -s "$CFG" ]; then   # -s: an empty placeholder file counts as absent
    echo "  exists: $CFG — left untouched (hand edits win; diff against the rendered form below if in doubt)"
else
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
image_analysis_configs_path = $GEECS_CONFIGS_ROOT/image_analysis_configs

[Experiment]
expt = $GEECS_EXPERIMENT

[tiled]
uri = $GEECS_TILED_URI
# api_key = <ask the Tiled admin; never in site.env>

[epics]
ca_addr_list = $EPICS_CA_ADDR_LIST

[qserver]
host = $GEECS_QSERVER_HOST
doc_addr = $GEECS_QS_DOC_ADDR
EOF
    }
    if [ "$DRY" -eq 1 ]; then echo "  [dry] would write $CFG:"; render_config_ini | sed 's/^/      /'
    else
        # 0600: the operator adds the Tiled api_key to this file by hand.
        mkdir -p "$(dirname "$CFG")"; (umask 077; render_config_ini > "$CFG"); chmod 600 "$CFG"; echo "  wrote $CFG (mode 600; add [tiled] api_key by hand)"
    fi
fi

say "units (each rendered from ITS service's clone, to a staging dir)"
# A unit must match the code it will run: the gateway clone may sit pinned
# at an older, verified deploy while qs-checkout is at master, so the
# gateway's unit comes from the gateway's clone — never from whichever
# clone this script happens to run in (that clone is only the fallback).
STAGE="$GEECS_CHECKOUT_ROOT/deploy-staging"
template_of() { case "$1" in
    gateway) echo "GeecsCAGateway/deploy/geecs-ca-gateway.service";; portal) echo "GEECS-DataPortal/deploy/geecs-data-portal.service";;
    qserver) echo "GeecsBluesky/qserver/deploy/geecs-qserver.service";; capture) echo "GeecsBluesky/capture/deploy/geecs-capture.service";;
    mcp) echo "GEECS-MCP/deploy/geecs-mcp.service";; esac; }
TEMPLATE_PATHS=()
for s in $SERVICES; do
    wanted "$s" || continue
    t="$GEECS_CHECKOUT_ROOT/$(clone_of "$s")/$(template_of "$s")"
    if [ -f "$t" ]; then TEMPLATE_PATHS+=("$t"); else echo "  $s: clone absent — template from this clone ($REPO_ROOT)"; TEMPLATE_PATHS+=("$REPO_ROOT/$(template_of "$s")"); fi
done
if [ "$DRY" -eq 1 ]; then echo "  [dry] render_units.sh $SITE_ENV $STAGE ${TEMPLATE_PATHS[*]}"
else RENDER_QUIET=1 "$REPO_ROOT/deploy/render_units.sh" "$SITE_ENV" "$STAGE" "${TEMPLATE_PATHS[@]}" | sed 's/^/  /'; fi

say "root steps (a human runs these; nothing above needed sudo)"
cat <<EOF
  sudo install -D -m 0644 "$SITE_ENV" "$SITE_ENV_INSTALLED"
  sudo install -m 0644 "$STAGE"/*.service /etc/systemd/system/
  sudo systemctl daemon-reload
EOF
for s in $SERVICES; do wanted "$s" && echo "  sudo systemctl enable --now $(unit_of "$s")"; done
echo
echo "Then: scripts/fleet_status.sh from any client — every row should read systemd / clean / matching versions."
