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

require_site_keys() {  # require_site_keys KEY... — exit 2 naming the first missing one
    local k
    for k in "$@"; do
        [ -n "${!k:-}" ] || { echo "site.env is missing $k" >&2; exit 2; }
    done
}
