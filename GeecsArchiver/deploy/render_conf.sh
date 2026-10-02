#!/usr/bin/env bash
# render_conf.sh - fill the archiver's conf templates from a site.env into a
# staging directory, beside the static conf files, ready for root to install
# under /etc/geecs/archiver.
#
#   GeecsArchiver/deploy/render_conf.sh SITE_ENV OUT_DIR
#   GeecsArchiver/deploy/render_conf.sh /etc/geecs/site.env ~/deploy-staging/archiver
#
# Why a second renderer: deploy/render_units.sh fills systemd units and,
# by design, refuses anything that is not one. The appliance's compose file
# and appliances.xml carry the same kind of install-time holes
# (@ARCHIVER_HOST@, @ARCHIVER_DATA_ROOT@, @SERVICE_UID@/@SERVICE_GID@), so
# they are filled here, from the same site.env, with the same loader.
# Runtime values (EPICS_CA_ADDR_LIST, TZ, GEECS_ARCHIVER_JAVA_OPTS) are not
# rendered: docker compose reads them from the unit's environment at `up`.
#
# Unprivileged: writes only to OUT_DIR. Installing the result needs root -
# the exact lines are printed, never run. RENDER_QUIET=1 suppresses them.
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
# The install-time keys this renderer fills, and the runtime keys the
# rendered compose file will read at `up` - an unset ${VAR} there would
# become an EMPTY value, so a missing key fails here, not on the host.
require_site_keys GEECS_SERVICE_USER GEECS_ARCHIVER_HOST GEECS_ARCHIVER_DATA_ROOT \
    GEECS_ARCHIVER_JAVA_OPTS EPICS_CA_ADDR_LIST EPICS_CA_AUTO_ADDR_LIST TZ

# The container runs Tomcat as the service account's ids. Off the service
# host (a laptop rendering for review) the account does not exist: fall
# back to the caller's ids and say so.
if id -u "$GEECS_SERVICE_USER" >/dev/null 2>&1; then
    SERVICE_UID="$(id -u "$GEECS_SERVICE_USER")"; SERVICE_GID="$(id -g "$GEECS_SERVICE_USER")"
else
    SERVICE_UID="$(id -u)"; SERVICE_GID="$(id -g)"
    echo "WARNING: no user '$GEECS_SERVICE_USER' on this machine - rendered with uid:gid $SERVICE_UID:$SERVICE_GID (re-render on the service host)" >&2
fi

mkdir -p "$OUT_DIR"
for t in compose.yaml appliances.xml; do
    src="$HERE/$t.in"; dst="$OUT_DIR/$t"
    [ -f "$src" ] || { echo "template missing: $src" >&2; exit 1; }
    sed -e "s|@ARCHIVER_HOST@|$GEECS_ARCHIVER_HOST|g" \
        -e "s|@ARCHIVER_DATA_ROOT@|$GEECS_ARCHIVER_DATA_ROOT|g" \
        -e "s|@SERVICE_UID@|$SERVICE_UID|g" \
        -e "s|@SERVICE_GID@|$SERVICE_GID|g" \
        "$src" > "$dst"
    # Comments may name a @PLACEHOLDER@; only non-comment lines count.
    if grep -vE '^[[:space:]]*#' "$dst" | grep -q '@[A-Z_]*@'; then
        echo "unfilled placeholder in $dst:" >&2
        grep -vE '^[[:space:]]*#' "$dst" | grep -n '@[A-Z_]*@' >&2
        exit 1
    fi
    echo "rendered $dst"
done
for f in server.xml context.xml policies.py archappl.properties; do
    install -m 0644 "$HERE/$f" "$OUT_DIR/$f"
    echo "copied   $OUT_DIR/$f"
done

[ "${RENDER_QUIET:-0}" = "1" ] && exit 0
cat <<EOF

Rendered for site '${GEECS_SITE:-?}' / experiment '${GEECS_EXPERIMENT:-?}', appliance at $GEECS_ARCHIVER_HOST:17665.
Install (root; review the staged files first):

  sudo install -d -m 0755 /etc/geecs/archiver
  sudo install -m 0644 "$OUT_DIR"/* /etc/geecs/archiver/
  sudo install -d -o $GEECS_SERVICE_USER -g $GEECS_SERVICE_USER -m 0750 "$GEECS_ARCHIVER_DATA_ROOT/sts" "$GEECS_ARCHIVER_DATA_ROOT/lts"
  # the unit: deploy/render_units.sh SITE_ENV OUT GeecsArchiver/deploy/geecs-archiver.service, then
  #   sudo install -m 0644 OUT/geecs-archiver.service /etc/systemd/system/ && sudo systemctl daemon-reload
  #   sudo systemctl enable --now geecs-archiver
EOF
