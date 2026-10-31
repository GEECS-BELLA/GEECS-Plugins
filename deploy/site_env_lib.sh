# site_env_lib.sh — shared loader for site.env (sourced by render_units.sh
# and bootstrap_host.sh). Mirrors systemd's EnvironmentFile parsing so the
# scripts see exactly what the units will see: KEY=VALUE per line, lines
# starting with '#' or ';' ignored, optional surrounding quotes, NO shell
# expansion and NO trailing-comment stripping.

load_site_env() {  # load_site_env FILE — exports every key
    local file="$1" line key val
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in ''|'#'*|';'*) continue ;; esac
        line="$(printf '%s' "$line" | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')"
        [ -n "$line" ] || continue
        case "$line" in *=*) ;; *) echo "site.env: not KEY=VALUE: $line" >&2; return 2 ;; esac
        key="${line%%=*}"; val="${line#*=}"
        case "$val" in \"*\") val="${val#\"}"; val="${val%\"}" ;; \'*\') val="${val#\'}"; val="${val%\'}" ;; esac
        printf -v "$key" '%s' "$val"
        export "$key"
    done < "$file"
}

check_site_env_consistency() {  # the duplicated experiment key must agree with the primary
    if [ -n "${QS_EXPERIMENT:-}" ] && [ -n "${GEECS_EXPERIMENT:-}" ] && [ "$QS_EXPERIMENT" != "$GEECS_EXPERIMENT" ]; then
        echo "site.env: QS_EXPERIMENT ($QS_EXPERIMENT) != GEECS_EXPERIMENT ($GEECS_EXPERIMENT)" >&2; exit 2
    fi
}

# Runtime keys the unit templates rely on through EnvironmentFile= (no
# @PLACEHOLDER@, no ${VAR} — the services read these names directly). A
# profile missing one would start services on their built-in defaults
# (UTC, EPICS broadcast) instead of failing before deployment.
SITE_RUNTIME_KEYS="GEECS_EXPERIMENT QS_EXPERIMENT TZ EPICS_CA_ADDR_LIST EPICS_CA_AUTO_ADDR_LIST EPICS_CAS_INTF_ADDR_LIST EPICS_CAS_BEACON_ADDR_LIST GEECS_QS_DOC_ADDR GEECS_CONFIGS_ROOT"
require_runtime_keys() { require_site_keys $SITE_RUNTIME_KEYS; }

unit_directives() {  # unit_directives FILE — the non-comment, non-blank lines of a unit file
    grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$1"
}

# A unit file is a site-profile TEMPLATE only if its directives carry the
# holes: User=@SERVICE_USER@ and an EnvironmentFile=@SITE_ENV@ line. A file
# from before the site profile (a hand-edit copy with the generic account) has
# neither — and must never be installed as if it had been rendered (Phase 3
# live incident 2026-09-04: portal unit from a pre-#777 clone, status=217/USER).
# Comments are ignored: every template header mentions the placeholders in prose.
is_unit_template() {  # is_unit_template FILE
    unit_directives "$1" | grep -q '^User=@SERVICE_USER@$' && unit_directives "$1" | grep -q '^EnvironmentFile=@SITE_ENV@$'
}

require_site_keys() {  # require_site_keys KEY... — exit 2 naming the first missing one
    local k
    for k in "$@"; do
        [ -n "${!k:-}" ] || { echo "site.env is missing $k" >&2; exit 2; }
    done
}

# The keys an experiment INSTANCE file must carry (deploy/experiments.env.example):
# the instance units read them after site.env, so a missing one would start
# that instance on the MAIN experiment's value (its ports, its experiment).
INSTANCE_KEYS="GEECS_INSTANCE GEECS_EXPERIMENT QS_EXPERIMENT GEECS_CA_SERVER_PORT EPICS_CAS_INTF_ADDR_LIST EPICS_CAS_BEACON_ADDR_LIST EPICS_CA_ADDR_LIST EPICS_CA_AUTO_ADDR_LIST QSERVER_ZMQ_CONTROL_ADDRESS_FOR_SERVER QSERVER_ZMQ_INFO_ADDRESS_FOR_SERVER QS_REDIS_PREFIX QS_DOC_PROXY_IN QS_DOC_PROXY_OUT QS_DOC_PUBLISH_ADDR QS_CONTROL_ADDR QS_CLIENT_CONTROL_ADDR QS_CLIENT_INFO_ADDR QS_CLIENT_DOC_ADDR GEECS_SCANNER_PORT GEECS_LOGBOOK_PORT"

check_instance_env() {  # check_instance_env INSTANCE [FILE] — after load_site_env of the instance FILE
    local k INSTANCE_FILE_PATH="${2:?check_instance_env INSTANCE FILE}"
    for k in $INSTANCE_KEYS; do
        [ -n "${!k:-}" ] || { echo "instance file is missing $k" >&2; exit 2; }
        case "${!k}" in *@*@*) echo "instance file: $k still holds a placeholder (${!k})" >&2; exit 2 ;; esac
    done
    [ "$GEECS_INSTANCE" = "$1" ] || { echo "instance file: GEECS_INSTANCE ($GEECS_INSTANCE) != the instance '$1'" >&2; exit 2; }
    # Render-time keys are filled into the units from site.env alone; set here
    # they would steer the bootstrap but not the rendered units.
    for k in GEECS_CHECKOUT_ROOT GEECS_INSTANCE_CHECKOUT_ROOT GEECS_EXPERIMENTS_DIR GEECS_SITE_ENV_PATH GEECS_SERVICE_USER GEECS_SERVICE_HOME GEECS_POETRY; do
        if grep -q "^[[:space:]]*$k=" "$INSTANCE_FILE_PATH" 2>/dev/null; then echo "instance file: $k is a site.env key (render-time), not an instance one" >&2; exit 2; fi
    done
    [ -z "${EPICS_CA_SERVER_PORT:-}" ] || { echo "instance file: EPICS_CA_SERVER_PORT must not be set (it would re-port the worker's CA clients) — use GEECS_CA_SERVER_PORT" >&2; exit 2; }
    check_site_env_consistency
}
