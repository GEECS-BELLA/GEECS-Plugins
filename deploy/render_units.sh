#!/usr/bin/env bash
# render_units.sh — fill the @PLACEHOLDER@ holes in every service unit
# template from a site.env, into a staging directory.
#
#   deploy/render_units.sh SITE_ENV OUT_DIR [TEMPLATE...]
#   deploy/render_units.sh /etc/geecs/site.env ~/deploy-staging
#
# With no TEMPLATE arguments every service template in THIS clone is
# rendered; bootstrap_host.sh passes each service's template from that
# service's own pinned clone instead, so a unit always matches the code it
# will run (fleet map: each clone sits at its service's last verified
# deploy). RENDER_QUIET=1 suppresses the install hint.
#
# Why a render step exists at all: systemd expands ${VAR} from
# EnvironmentFile= ONLY in command arguments — never in WorkingDirectory=,
# User=, Environment= lines, or the executable path. So paths and identity
# are filled here at install time (@CHECKOUT_ROOT@, @SERVICE_USER@,
# @SERVICE_HOME@, @POETRY@, @SITE_ENV@) and runtime values (experiment,
# EPICS addressing, TZ) reach the process through EnvironmentFile= at
# start. See docs/platform/site_profile.md.
#
# Unprivileged: writes only to OUT_DIR. Installing the result needs root —
# the script prints the exact sudo lines but never runs them.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SITE_ENV="${1:-}"
OUT_DIR="${2:-}"
if [ -z "$SITE_ENV" ] || [ -z "$OUT_DIR" ]; then
    echo "usage: render_units.sh SITE_ENV OUT_DIR [TEMPLATE...]" >&2
    exit 2
fi
[ -f "$SITE_ENV" ] || { echo "no such site.env: $SITE_ENV" >&2; exit 2; }
SITE_ENV="$(cd "$(dirname "$SITE_ENV")" && pwd)/$(basename "$SITE_ENV")"   # absolute: the printed sudo lines must work from any cwd
shift 2

# shellcheck source=site_env_lib.sh
. "$REPO_ROOT/deploy/site_env_lib.sh"
load_site_env "$SITE_ENV"
check_site_env_consistency
# Every key a template consumes: an unset ${VAR} in an ExecStart argument
# expands to an EMPTY argument (not to nothing), so a missing key must fail
# here, not on the host at 03:00.
require_site_keys GEECS_SERVICE_USER GEECS_SERVICE_HOME GEECS_CHECKOUT_ROOT GEECS_POETRY
require_runtime_keys

# The templates: one per service family. Default = this clone's copies;
# explicit paths (from bootstrap_host.sh) override.
TEMPLATES=(
    GeecsCAGateway/deploy/geecs-ca-gateway.service
    GEECS-DataPortal/deploy/geecs-data-portal.service
    GeecsBluesky/qserver/deploy/geecs-qserver.service
    GeecsBluesky/qserver/deploy/geecs-qserver-ready.service
    GeecsBluesky/capture/deploy/geecs-capture.service
    GEECS-MCP/deploy/geecs-mcp.service
)
if [ $# -gt 0 ]; then TEMPLATES=("$@"); fi

# Where the installed site.env will live — the rendered units point at it.
SITE_ENV_INSTALLED="${GEECS_SITE_ENV_PATH:-/etc/geecs/site.env}"

mkdir -p "$OUT_DIR"
for t in "${TEMPLATES[@]}"; do
    case "$t" in /*) src="$t" ;; *) src="$REPO_ROOT/$t" ;; esac
    [ -f "$src" ] || { echo "template missing: $t" >&2; exit 1; }
    # A template must BE a template (is_unit_template, site_env_lib.sh): a
    # unit file from before the site profile would otherwise pass through as
    # a "rendered" unit for an account that does not exist.
    if ! is_unit_template "$src"; then
        echo "not a site-profile template (directives lack User=@SERVICE_USER@ / EnvironmentFile=@SITE_ENV@): $src" >&2
        echo "  the clone it comes from predates the templated units — pull it forward (that is a deploy of that service), then re-render;" >&2
        echo "  bootstrap_host.sh --only <other services> renders the rest meanwhile" >&2
        exit 1
    fi
    dst="$OUT_DIR/$(basename "$t")"
    sed -e "s|@SERVICE_USER@|$GEECS_SERVICE_USER|g" \
        -e "s|@SERVICE_HOME@|$GEECS_SERVICE_HOME|g" \
        -e "s|@CHECKOUT_ROOT@|$GEECS_CHECKOUT_ROOT|g" \
        -e "s|@POETRY@|$GEECS_POETRY|g" \
        -e "s|@SITE_ENV@|$SITE_ENV_INSTALLED|g" \
        "$src" > "$dst"
    # Comments may mention @PLACEHOLDER@ by name; only directive lines count.
    if unit_directives "$dst" | grep -q '@[A-Z_]*@'; then
        echo "unfilled placeholder in $dst:" >&2; unit_directives "$dst" | grep -n '@[A-Z_]*@' >&2; exit 1
    fi
    echo "rendered $dst"
done

[ "${RENDER_QUIET:-0}" = "1" ] && exit 0
cat <<EOF

Rendered for site '${GEECS_SITE:-?}' / experiment '$GEECS_EXPERIMENT'.
Install (root; review the staged files first):

  sudo install -D -m 0644 "$SITE_ENV" "$SITE_ENV_INSTALLED"
  sudo install -m 0644 "$OUT_DIR"/*.service /etc/systemd/system/
  sudo systemctl daemon-reload
  # then, per service, once its clone + env exist (bootstrap_host.sh):
  #   sudo systemctl enable --now <unit>
EOF
