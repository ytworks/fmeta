#!/usr/bin/env bash
# Entry of the official image: activate the conda environment, then run the command.
# The image is run as the calling user (docker run --user), who has no account inside it:
# give that user a name and a writable home (matplotlib, OpenFF and AmberTools write caches).
set -e
if [[ -z "${USER:-}" ]]; then
    USER="$(id -un 2>/dev/null)" || USER=fmeta
    export USER
fi
if [[ ! -w "${HOME:-/}" ]]; then
    HOME="/tmp/home-$(id -u)"
    export HOME
    mkdir -p "$HOME"
fi
# shellcheck disable=SC1091
source /usr/local/bin/_activate_current_env.sh
exec "$@"
