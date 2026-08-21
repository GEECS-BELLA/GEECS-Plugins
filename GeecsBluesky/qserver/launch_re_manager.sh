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

redis_is_answering() {
    (exec 3<>/dev/tcp/127.0.0.1/6379) >/dev/null 2>&1
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

exec start-re-manager \
    --startup-dir "${QS_STARTUP_DIR}" \
    --user-group-permissions "${PERMISSIONS_FILE}" \
    --zmq-publish-console ON
