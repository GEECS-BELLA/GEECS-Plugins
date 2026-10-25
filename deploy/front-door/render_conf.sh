#!/usr/bin/env bash
# render_conf.sh - fill the front door's nginx config template from a
# site.env into a staging directory, beside the bundle's placeholder index,
# ready for root to install under /etc/geecs/front-door.
#
#   deploy/front-door/render_conf.sh SITE_ENV OUT_DIR
#   deploy/front-door/render_conf.sh /etc/geecs/site.env ~/deploy-staging/front-door
#
# Why a second renderer: deploy/render_units.sh fills systemd units and, by
# design, refuses anything that is not one (the archiver's render_conf.sh is
# the precedent). nginx reads no environment, so its two site values — the
# interface it listens on and the host the services run on — are filled
# here, from the same site.env, with the same loader.
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

# shellcheck source=../site_env_lib.sh
. "$REPO_ROOT/deploy/site_env_lib.sh"
load_site_env "$SITE_ENV"
require_site_keys GEECS_FRONT_DOOR_BIND GEECS_SERVICES_HOST

mkdir -p "$OUT_DIR"
src="$HERE/nginx.conf.in"; dst="$OUT_DIR/nginx.conf"
sed -e "s|@FRONT_DOOR_BIND@|$GEECS_FRONT_DOOR_BIND|g" \
    -e "s|@SERVICES_HOST@|$GEECS_SERVICES_HOST|g" \
    "$src" > "$dst"
# Comments may name a @PLACEHOLDER@; only non-comment lines count.
if grep -vE '^[[:space:]]*#' "$dst" | grep -q '@[A-Z_]*@'; then
    echo "unfilled placeholder in $dst:" >&2
    grep -vE '^[[:space:]]*#' "$dst" | grep -n '@[A-Z_]*@' >&2
    exit 1
fi
echo "rendered $dst"
install -m 0644 "$HERE/index.html" "$OUT_DIR/index.html"
echo "copied   $OUT_DIR/index.html"

[ "${RENDER_QUIET:-0}" = "1" ] && exit 0
cat <<EOF

Rendered for site '${GEECS_SITE:-?}': http://$GEECS_FRONT_DOOR_BIND/ in front of $GEECS_SERVICES_HOST.
Install (root; review the staged files first):

  sudo install -D -m 0644 "$OUT_DIR/nginx.conf" /etc/geecs/front-door/nginx.conf
  sudo install -d -m 0755 /srv/geecs-web
  [ -e /srv/geecs-web/index.html ] || sudo install -m 0644 "$OUT_DIR/index.html" /srv/geecs-web/
  # the unit: deploy/render_units.sh SITE_ENV OUT deploy/front-door/geecs-front-door.service, then
  #   sudo install -m 0644 OUT/geecs-front-door.service /etc/systemd/system/ && sudo systemctl daemon-reload
  #   sudo systemctl enable --now geecs-front-door
EOF
