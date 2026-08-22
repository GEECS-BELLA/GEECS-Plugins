#!/usr/bin/env bash
set -euo pipefail

if ! command -v start-re-manager >/dev/null 2>&1; then
    echo "ERROR: start-re-manager is not on PATH." >&2
    echo "Install bluesky-queueserver or activate its environment." >&2
    exit 127
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PERMISSIONS_FILE="${SCRIPT_DIR}/user_group_permissions.yaml"
QS_STARTUP_DIR="${QS_STARTUP_DIR:-./startup}"
QS_REDIS_SERVER="${QS_REDIS_SERVER:-redis-server}"

port_is_answering() {
    (exec 3<>"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1
}

redis_is_answering() {
    port_is_answering 6379
}

if ! redis_is_answering; then
    if ! command -v "${QS_REDIS_SERVER}" >/dev/null 2>&1; then
        echo "ERROR: Redis is not answering on 127.0.0.1:6379." >&2
        echo "QS_REDIS_SERVER='${QS_REDIS_SERVER}' is not executable." >&2
        exit 127
    fi

    echo "Redis is not answering on 127.0.0.1:6379; starting ${QS_REDIS_SERVER}." >&2
    "${QS_REDIS_SERVER}" --bind 127.0.0.1 --port 6379 --daemonize yes

    for _ in {1..50}; do
        if redis_is_answering; then
            break
        fi
        sleep 0.1
    done

    if ! redis_is_answering; then
        echo "ERROR: Redis did not start or did not answer on 127.0.0.1:6379." >&2
        exit 1
    fi
fi

# Document stream (#648): the startup profile publishes bluesky documents to
# a bluesky-0MQ-proxy (in QS_DOC_PROXY_IN, out QS_DOC_PROXY_OUT); GUI clients
# subscribe to the out port for live progress. Stateless — restarting with
# the manager is fine. QS_DOC_PROXY=OFF skips it (pair with
# QS_DOC_PUBLISH_ADDR=OFF for the worker side).
QS_DOC_PROXY_IN="${QS_DOC_PROXY_IN:-5567}"
QS_DOC_PROXY_OUT="${QS_DOC_PROXY_OUT:-5568}"
# Case-insensitive OFF, matching the worker's QS_DOC_PUBLISH_ADDR check.
QS_DOC_PROXY_MODE="$(printf '%s' "${QS_DOC_PROXY:-ON}" | tr '[:lower:]' '[:upper:]')"
if [[ "${QS_DOC_PROXY_MODE}" != "OFF" ]] && ! port_is_answering "${QS_DOC_PROXY_IN}"; then
    if command -v bluesky-0MQ-proxy >/dev/null 2>&1; then
        echo "Starting bluesky-0MQ-proxy ${QS_DOC_PROXY_IN} -> ${QS_DOC_PROXY_OUT}." >&2
        # stderr stays attached (journal / terminal) so a bind failure —
        # e.g. the out port already taken — is diagnosable, not silent.
        bluesky-0MQ-proxy "${QS_DOC_PROXY_IN}" "${QS_DOC_PROXY_OUT}" >/dev/null &
    else
        echo "WARNING: bluesky-0MQ-proxy not on PATH; document stream disabled" >&2
        echo "(GUI live progress will be empty; set QS_DOC_PROXY=OFF to silence)." >&2
    fi
fi

# --keep-re: the startup profile defines RE = RunEngine({}) and the manager
# must keep it — without this pairing, `queue start` silently bounces items
# and only the manager log shows "Run Engine is not found in the RE Worker
# environment" (empirical, issue #636).
exec start-re-manager \
    --startup-dir "${QS_STARTUP_DIR}" \
    --user-group-permissions "${PERMISSIONS_FILE}" \
    --keep-re \
    --zmq-publish-console ON
