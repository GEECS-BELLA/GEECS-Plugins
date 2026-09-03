#!/usr/bin/env bash
# render_units.sh — fill the @PLACEHOLDER@ holes in every service unit
# template from a site.env, into a staging directory.
#
#   deploy/render_units.sh SITE_ENV OUT_DIR
#   deploy/render_units.sh /etc/geecs/site.env ~/deploy-staging
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
    echo "usage: render_units.sh SITE_ENV OUT_DIR" >&2
    exit 2
fi
[ -f "$SITE_ENV" ] || { echo "no such site.env: $SITE_ENV" >&2; exit 2; }

# shellcheck source=site.env.example
. "$REPO_ROOT/deploy/site_env_lib.sh"
load_site_env "$SITE_ENV"
require_site_keys GEECS_SERVICE_USER GEECS_SERVICE_HOME GEECS_CHECKOUT_ROOT GEECS_POETRY GEECS_EXPERIMENT

# The templates: one per service family. Paths relative to the repo root.
TEMPLATES=(
    GeecsCAGateway/deploy/geecs-ca-gateway.service
    GEECS-DataPortal/deploy/geecs-data-portal.service
    GeecsBluesky/qserver/deploy/geecs-qserver.service
    GeecsBluesky/capture/deploy/geecs-capture.service
    GEECS-MCP/deploy/geecs-mcp.service
)

# Where the installed site.env will live — the rendered units point at it.
SITE_ENV_INSTALLED="${GEECS_SITE_ENV_PATH:-/etc/geecs/site.env}"

mkdir -p "$OUT_DIR"
for t in "${TEMPLATES[@]}"; do
    src="$REPO_ROOT/$t"
    [ -f "$src" ] || { echo "template missing: $t" >&2; exit 1; }
    dst="$OUT_DIR/$(basename "$t")"
    sed -e "s|@SERVICE_USER@|$GEECS_SERVICE_USER|g" \
        -e "s|@SERVICE_HOME@|$GEECS_SERVICE_HOME|g" \
        -e "s|@CHECKOUT_ROOT@|$GEECS_CHECKOUT_ROOT|g" \
        -e "s|@POETRY@|$GEECS_POETRY|g" \
        -e "s|@SITE_ENV@|$SITE_ENV_INSTALLED|g" \
        "$src" > "$dst"
    # Comments may mention @PLACEHOLDER@ by name; only directive lines count.
    if grep -v '^[[:space:]]*#' "$dst" | grep -q '@[A-Z_]*@'; then
        echo "unfilled placeholder in $dst:" >&2; grep -v '^[[:space:]]*#' "$dst" | grep -n '@[A-Z_]*@' >&2; exit 1
    fi
    echo "rendered $dst"
done

cat <<EOF

Rendered for site '${GEECS_SITE:-?}' / experiment '$GEECS_EXPERIMENT'.
Install (root; review the staged files first):

  sudo install -D -m 0644 "$SITE_ENV" "$SITE_ENV_INSTALLED"
  sudo install -m 0644 "$OUT_DIR"/*.service /etc/systemd/system/
  sudo systemctl daemon-reload
  # then, per service, once its clone + env exist (bootstrap_host.sh):
  #   sudo systemctl enable --now <unit>
EOF
