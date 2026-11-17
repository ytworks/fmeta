#!/usr/bin/env bash
# Regenerate conda-linux-64.lock from environment.yml and docker/nvidia.pins.
#
# The solve runs inside the base image named in the Dockerfile, so no conda is needed on the
# host. CONDA_OVERRIDE_CUDA tells the solver that a CUDA driver is present (a container without
# GPUs would otherwise get the CPU-only build of OpenMM). The cuPotential plugin is not on
# conda-forge: the channel of its release is downloaded, checked against the pinned sha256 and
# used for the solve; in the lock the package is the file of that release, so the image build
# needs nothing but the lock. The compatibility libraries are not a conda package: their URL
# and sha256 become a comment line of the lock, which the Dockerfile reads.
#
# After a change of the lock: rebuild the image (scripts/in_docker.sh --build ...), rerun the
# tests and the GPU gates, and commit the lock together with the new evidence.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$(sed -n 's/^FROM \([^ ]*\).*$/\1/p' "$ROOT/Dockerfile" | head -n 1)"
CUDA="$(sed -n 's/^ *- cuda-version=\([0-9.]*\).*$/\1/p' "$ROOT/environment.yml")"
if [[ -z "$BASE" || -z "$CUDA" ]]; then
    echo "update_lock.sh: no base image in Dockerfile or no cuda-version in environment.yml" >&2
    exit 2
fi
# shellcheck disable=SC1091
source "$ROOT/docker/nvidia.pins"

docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e MAMBA_ROOT_PREFIX=/tmp/mamba \
    -e CONDA_OVERRIDE_CUDA="$CUDA" -e PLUGIN_CHANNEL_URL="$PLUGIN_CHANNEL_URL" \
    -e PLUGIN_CHANNEL_SHA256="$PLUGIN_CHANNEL_SHA256" -e CUDA_COMPAT_URL="$CUDA_COMPAT_URL" \
    -e CUDA_COMPAT_SHA256="$CUDA_COMPAT_SHA256" -v "$ROOT":/repo --entrypoint /bin/bash "$BASE" -c '
set -euo pipefail
# The base image has no download tool: a small environment of its own brings curl.
micromamba create -y -q -p /tmp/tools -c conda-forge curl
/tmp/tools/bin/curl -fsSL --retry 3 -o /tmp/channel.tar.gz "$PLUGIN_CHANNEL_URL"
echo "$PLUGIN_CHANNEL_SHA256  /tmp/channel.tar.gz" | sha256sum -c - > /dev/null
mkdir /tmp/nvidia
tar -xzf /tmp/channel.tar.gz -C /tmp/nvidia --strip-components=1 --no-same-owner
micromamba create -y -q -p /tmp/env -c file:///tmp/nvidia -f /repo/environment.yml
micromamba env export -p /tmp/env --explicit --md5 > /tmp/lock
# The plugin as the file of the release (same directory as the channel), not the local copy.
sed -i "s|^file:///tmp/nvidia/linux-64/|${PLUGIN_CHANNEL_URL%/*}/|" /tmp/lock
if grep -q "^file:" /tmp/lock || ! grep -q "^${PLUGIN_CHANNEL_URL%/*}/" /tmp/lock; then
    echo "update_lock.sh: the plugin line of the lock is not the file of the release" >&2
    exit 1
fi
sed -i "s|^@EXPLICIT\$|# cuda-compat: $CUDA_COMPAT_URL sha256=$CUDA_COMPAT_SHA256\n@EXPLICIT|" /tmp/lock
cp /tmp/lock /repo/conda-linux-64.lock.new
mv /repo/conda-linux-64.lock.new /repo/conda-linux-64.lock
micromamba list -p /tmp/env | grep -E "^ *(python|openmm|cuda-version|nv-openmm-plugin|openff-toolkit|openmmforcefields|ambertools|rdkit|numpy) "
'
echo "wrote $ROOT/conda-linux-64.lock ($(grep -c '^https' "$ROOT/conda-linux-64.lock") packages)"
