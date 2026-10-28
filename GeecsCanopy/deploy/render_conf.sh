#!/usr/bin/env bash
# render_conf.sh - fill GeecsCanopy's compose template from a site.env and
# the image pin into a staging directory, ready for root to install under
# /etc/geecs/canopy.
#
#   GeecsCanopy/deploy/render_conf.sh SITE_ENV OUT_DIR
#
# The archiver's pattern: deploy/render_units.sh fills the unit, this fills
# the compose file's install-time holes (the pin from geecs_canopy/pin.py,
# this clone as the build context, the service account's ids). Runtime
# values stay ${VAR} for docker compose to read at `up`.
#
# Unprivileged: writes only to OUT_DIR; the root lines are printed, never run.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
SITE_ENV="${1:-}"
OUT_DIR="${2:-}"
if [ -z "$SITE_ENV" ] || [ -z "$OUT_DIR" ]; then
    echo "usage: render_conf.sh SITE_ENV OUT_DIR" >&2
    exit 2
fi
[ -f "$SITE_ENV" ] || { echo "no such site.env: $SITE_ENV" >&2; exit 2; }

# shellcheck source=../../deploy/site_env_lib.sh
. "$REPO_ROOT/deploy/site_env_lib.sh"
load_site_env "$SITE_ENV"
require_site_keys GEECS_SERVICE_USER GEECS_CANOPY_BIND GEECS_CANOPY_PORT \
    GEECS_CANOPY_DB GEECS_EXPERIMENT GEECS_CONFIGS_ROOT \
    EPICS_CA_ADDR_LIST EPICS_CA_AUTO_ADDR_LIST TZ
# The root path may be empty (served at /), but the key must be present.
[ -n "${GEECS_CANOPY_ROOT_PATH+x}" ] || { echo "site.env is missing GEECS_CANOPY_ROOT_PATH" >&2; exit 2; }
case "$GEECS_CANOPY_DB" in on|off) ;; *) echo "GEECS_CANOPY_DB is on or off, not '$GEECS_CANOPY_DB'" >&2; exit 2 ;; esac

pin() { sed -n "s/^$1 = \"\(.*\)\"$/\1/p" "$REPO_ROOT/GeecsCanopy/geecs_canopy/pin.py"; }
CANOPY_IMAGE="$(pin CANOPY_IMAGE)"; CANOPY_COMMIT="$(pin CANOPY_COMMIT)"
case "$CANOPY_IMAGE" in *@sha256:*) ;; *) echo "pin.py: CANOPY_IMAGE is not pinned by digest" >&2; exit 1 ;; esac

if id -u "$GEECS_SERVICE_USER" >/dev/null 2>&1; then
    SERVICE_UID="$(id -u "$GEECS_SERVICE_USER")"; SERVICE_GID="$(id -g "$GEECS_SERVICE_USER")"
else
    SERVICE_UID="$(id -u)"; SERVICE_GID="$(id -g)"
    echo "WARNING: no user '$GEECS_SERVICE_USER' here - rendered with uid:gid $SERVICE_UID:$SERVICE_GID (re-render on the service host)" >&2
fi

mkdir -p "$OUT_DIR"
dst="$OUT_DIR/compose.yaml"
sed -e "s|@CANOPY_IMAGE@|$CANOPY_IMAGE|g" \
    -e "s|@CANOPY_COMMIT@|$CANOPY_COMMIT|g" \
    -e "s|@BUILD_CONTEXT@|$REPO_ROOT|g" \
    -e "s|@SERVICE_UID@|$SERVICE_UID|g" \
    -e "s|@SERVICE_GID@|$SERVICE_GID|g" \
    "$HERE/compose.yaml.in" > "$dst"
if grep -vE '^[[:space:]]*#' "$dst" | grep -q '@[A-Z_]*@'; then
    echo "unfilled placeholder in $dst:" >&2; grep -vE '^[[:space:]]*#' "$dst" | grep -n '@[A-Z_]*@' >&2; exit 1
fi
echo "rendered $dst (Canopy $CANOPY_COMMIT)"

[ "${RENDER_QUIET:-0}" = "1" ] && exit 0
cat <<EOT

Install (root; review the staged file first; DEPLOYMENT.md has the full runbook):

  sudo install -d -m 0755 /etc/geecs/canopy
  sudo install -d -o $GEECS_SERVICE_USER -g $GEECS_SERVICE_USER -m 0750 /var/lib/geecs-canopy /var/backups/geecs-canopy
  sudo install -d -o $GEECS_SERVICE_USER -g $GEECS_SERVICE_USER -m 0700 /var/lib/geecs-canopy/pgdata
  sudo install -d -o 1000 -g $GEECS_SERVICE_USER -m 0750 /var/lib/geecs-canopy/module   # the journal: the image's appuser writes, the service account backs it up
  sudo install -m 0644 "$dst" /etc/geecs/canopy/compose.yaml
  sudo install -m 0600 -o $GEECS_SERVICE_USER GeecsCanopy/deploy/secrets.env.example /etc/geecs/canopy/secrets.env   # first install only, then fill it
  # the unit: deploy/render_units.sh SITE_ENV OUT GeecsCanopy/deploy/geecs-canopy.service
EOT
