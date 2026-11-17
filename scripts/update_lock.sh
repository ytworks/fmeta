#!/usr/bin/env bash
# Regenerate conda-linux-64.lock from environment.yml.
#
# The solve runs inside the base image named in the Dockerfile, so no conda is needed on the
# host. CONDA_OVERRIDE_CUDA tells the solver that a CUDA driver is present (a container without
# GPUs would otherwise get the CPU-only build of OpenMM). After a change of the lock: rebuild
# the image (scripts/in_docker.sh --build ...), rerun the tests and the GPU gates, and commit
# the lock together with the new evidence.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$(sed -n 's/^FROM \([^ ]*\).*$/\1/p' "$ROOT/Dockerfile" | head -n 1)"
CUDA="$(sed -n 's/^ *- cuda-version=\([0-9.]*\).*$/\1/p' "$ROOT/environment.yml")"
if [[ -z "$BASE" || -z "$CUDA" ]]; then
    echo "update_lock.sh: no base image in Dockerfile or no cuda-version in environment.yml" >&2
    exit 2
fi

docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e MAMBA_ROOT_PREFIX=/tmp/mamba \
    -e CONDA_OVERRIDE_CUDA="$CUDA" -v "$ROOT":/repo --entrypoint /bin/bash "$BASE" -c '
set -euo pipefail
micromamba create -y -q -p /tmp/env -f /repo/environment.yml
micromamba env export -p /tmp/env --explicit --md5 > /repo/conda-linux-64.lock.new
mv /repo/conda-linux-64.lock.new /repo/conda-linux-64.lock
micromamba list -p /tmp/env | grep -E "^ *(python|openmm|cuda-version|openff-toolkit|openmmforcefields|ambertools|rdkit|numpy) "
'
echo "wrote $ROOT/conda-linux-64.lock ($(grep -c '^https' "$ROOT/conda-linux-64.lock") packages)"
