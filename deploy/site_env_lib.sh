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
INSTANCE_KEYS="GEECS_INSTANCE GEECS_INSTANCE_SLOT GEECS_EXPERIMENT QS_EXPERIMENT GEECS_CA_SERVER_PORT EPICS_CAS_INTF_ADDR_LIST EPICS_CAS_BEACON_ADDR_LIST EPICS_CA_ADDR_LIST EPICS_CA_AUTO_ADDR_LIST QSERVER_ZMQ_CONTROL_ADDRESS_FOR_SERVER QSERVER_ZMQ_INFO_ADDRESS_FOR_SERVER QS_REDIS_PREFIX QS_DOC_PROXY_IN QS_DOC_PROXY_OUT QS_DOC_PUBLISH_ADDR QS_CONTROL_ADDR QS_CLIENT_CONTROL_ADDR QS_CLIENT_INFO_ADDR QS_CLIENT_DOC_ADDR GEECS_SCANNER_PORT GEECS_LOGBOOK_PORT"

# The directory of instance files (<Instance>.env). Resolved AFTER site.env
# is loaded — site.env may set it — by every script that reads or installs
# one, so they can never disagree.
experiments_dir() { printf '%s' "${GEECS_EXPERIMENTS_DIR:-/etc/geecs/experiments}"; }

check_instance_env() {  # check_instance_env INSTANCE FILE — validate FILE ALONE
    # In a subshell with every instance key unset first: a key the file
    # lacks must never pass on the value site.env (the MAIN experiment's)
    # already put in this environment.
    (
        for k in $INSTANCE_KEYS EPICS_CA_SERVER_PORT; do unset "$k"; done
        load_site_env "${2:?check_instance_env INSTANCE FILE}"
        _check_instance_env_loaded "$1" "$2"
    )
}

_check_instance_env_loaded() {  # the checks, over an environment holding the instance file's keys only
    local k INSTANCE_FILE_PATH="$2"
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
    # The clients' addresses must name THIS instance's manager: the readiness
    # oneshot and the scanner would otherwise assert / drive another one.
    local port="${QSERVER_ZMQ_CONTROL_ADDRESS_FOR_SERVER:-}"; port="${port##*:}"
    for k in QS_CONTROL_ADDR QS_CLIENT_CONTROL_ADDR; do
        local v="${!k:-}"
        [ "${v##*:}" = "$port" ] || { echo "instance file: $k ($v) is not this instance's manager (QSERVER_ZMQ_CONTROL_ADDRESS_FOR_SERVER port $port)" >&2; exit 2; }
    done
    [ -z "${EPICS_CA_SERVER_PORT:-}" ] || { echo "instance file: EPICS_CA_SERVER_PORT must not be set (it would re-port the worker's CA clients) — use GEECS_CA_SERVER_PORT" >&2; exit 2; }
    check_site_env_consistency
}

canonical_dir() {  # canonical_dir PATH — the physical path; for a path not yet created, its nearest existing ancestor's physical path + the rest
    local p="${1%/}" rest=""
    [ -n "$p" ] || p="/"
    while [ ! -d "$p" ] && [ "$p" != "/" ] && [ -n "$p" ]; do
        case "$p" in */*) ;; *) break ;; esac   # relative and gone: no progress possible
        rest="/${p##*/}$rest"; p="${p%/*}"; [ -n "$p" ] || p="/"
    done
    if [ -d "$p" ] && p="$(cd "$p" 2>/dev/null && pwd -P)"; then :; fi
    printf '%s%s' "${p%/}" "$rest"
}

# The directories bootstrap_host.sh clones for the MAIN experiment's services,
# under GEECS_CHECKOUT_ROOT (its clone_of table; a test pins the two equal).
MAIN_CLONE_DIRS="gateway-checkout portal-checkout qs-checkout archiver-checkout"

# Whether experiment instances have a clone root of their own.
#   1 (not separate) — GEECS_INSTANCE_CHECKOUT_ROOT unset: instances use the
#     main clones as they are.
#   0 (separate)     — any other root, e.g. <checkout root>/instances (a
#     sibling of the main clones: the documented layout).
#   exit 2           — a root that would let an instance's chown or
#     fast-forward reach main code: the checkout root itself (set
#     explicitly), one around it, one equal to, inside or around a main clone
#     directory, or the deploy-staging directory.
instance_root_is_separate() {
    local main inst clone c k
    # Absolute, and no . or .. components: a not-yet-created component could
    # otherwise hide a path that resolves into main code once it exists.
    for k in GEECS_CHECKOUT_ROOT GEECS_INSTANCE_CHECKOUT_ROOT; do
        [ -n "${!k:-}" ] || continue
        case "${!k}" in /*) ;; *) echo "$k (${!k}) must be an absolute path" >&2; exit 2 ;; esac
        case "/${!k}/" in */./*|*/../*) echo "$k (${!k}) must not contain . or .. components" >&2; exit 2 ;; esac
    done
    main="$(canonical_dir "$GEECS_CHECKOUT_ROOT")"
    inst="$(canonical_dir "${GEECS_INSTANCE_CHECKOUT_ROOT:-$GEECS_CHECKOUT_ROOT}")"
    if [ "$inst" = "$main" ]; then
        # Unset: instances share the main clones as they are (the design once
        # the arc is on master). SET to the checkout root itself — however
        # spelled — is a mistake that would chown / fast-forward main code.
        if [ -n "${GEECS_INSTANCE_CHECKOUT_ROOT:-}" ]; then
            echo "GEECS_INSTANCE_CHECKOUT_ROOT ($inst) is GEECS_CHECKOUT_ROOT itself — unset it to share the main clones, or name a directory of its own (e.g. <checkout root>/instances)" >&2; exit 2
        fi
        return 1
    fi
    if [ "$inst" = "$(canonical_dir "$main/deploy-staging")" ]; then
        echo "GEECS_INSTANCE_CHECKOUT_ROOT ($inst) is the deploy-staging directory" >&2; exit 2
    fi
    case "$main/" in "$inst"/*) echo "GEECS_INSTANCE_CHECKOUT_ROOT ($inst) contains GEECS_CHECKOUT_ROOT ($main)" >&2; exit 2 ;; esac
    for c in $MAIN_CLONE_DIRS; do
        clone="$(canonical_dir "$main/$c")"
        case "$inst/" in "$clone"/*) echo "GEECS_INSTANCE_CHECKOUT_ROOT ($inst) is or is inside the main clone $clone" >&2; exit 2 ;; esac
        case "$clone/" in "$inst"/*) echo "GEECS_INSTANCE_CHECKOUT_ROOT ($inst) contains the main clone $clone" >&2; exit 2 ;; esac
    done
    return 0
}
