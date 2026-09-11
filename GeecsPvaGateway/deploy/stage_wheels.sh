#!/usr/bin/env bash
# Stage the fleet's external wheels beside the share clone.
#
#   deploy/stage_wheels.sh "<share>/.../Active Version"
#
# Downloads the Windows/CPython-3.11 wheels for every pin in
# requirements-fleet.txt into <share>/pva-wheels (no dependencies: those were
# frozen at bootstrap), from any machine with PyPI reach.  A restart on any
# box then installs them offline (launch.bat, --no-index).  Re-run after
# editing requirements-fleet.txt, before pulling the share clone.
set -euo pipefail
share="${1:?usage: stage_wheels.sh <Active Version dir (parent of GEECS-Plugins)>}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dest="$share/pva-wheels"
mkdir -p "$dest"
python3 -m pip download --quiet --no-deps --only-binary=:all: \
    --platform win_amd64 --implementation cp --python-version 3.11 \
    -r "$here/requirements-fleet.txt" -d "$dest"
echo "staged in $dest:"
ls -1 "$dest"
