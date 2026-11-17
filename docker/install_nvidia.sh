#!/usr/bin/env bash
# Step of the image build (the Dockerfile runs it as root, after the environment of the lock is
# installed): the CUDA forward-compatibility libraries, the NVIDIA MPS programs of the same
# driver, and the licence texts of NVIDIA's software.
#
# With the compatibility libraries the CUDA 13.4 runtime of the image works on a host whose
# driver is older than R615. They go to /usr/local/cuda/compat, where the NVIDIA container
# runtime looks for them: it puts them on the loader path of a container only when they are
# newer than the driver of the host. A container that uses them is a client of the MPS daemon of
# their version only, so the image holds those MPS programs too (/opt/fmeta/mps/bin; the entry
# of the image puts them on the PATH exactly when the compatibility libraries are in use).
#
# None of this is a conda package: the lock names each file and its sha256 in a comment line
# (scripts/update_lock.sh writes them from docker/nvidia.pins), so the lock stays the one
# description of what the image holds.
set -euo pipefail

LOCK=/opt/fmeta/conda-linux-64.lock
LICENSES=/opt/fmeta/licenses
tmp="$(mktemp -d)"

fetch() {    # fetch NAME: download the file of the lock's "# NAME: URL sha256=SUM" line, check it, unpack it
    local line url sum
    line="$(sed -n "s/^# $1: //p" "$LOCK")"
    url="${line%% *}"
    sum="${line##* sha256=}"
    if [[ -z "$url" || "$sum" == "$line" ]]; then
        echo "install_nvidia.sh: no '# $1: URL sha256=SUM' line in $LOCK" >&2
        exit 1
    fi
    /opt/conda/bin/python -I - "$url" "$tmp/$1.deb" <<'PY'
import shutil
import sys
import urllib.request

request = urllib.request.Request(sys.argv[1], headers={"User-Agent": "fmeta-image-build"})
with urllib.request.urlopen(request, timeout=300) as response, open(sys.argv[2], "wb") as out:
    shutil.copyfileobj(response, out)
PY
    echo "$sum  $tmp/$1.deb" | sha256sum -c - > /dev/null
    dpkg-deb -x "$tmp/$1.deb" "$tmp/$1"
}

fetch cuda-compat
fetch cuda-mps
compat="$(dpkg-deb -f "$tmp/cuda-compat.deb" Version)"
mps="$(dpkg-deb -f "$tmp/cuda-mps.deb" Version)"
if [[ "$compat" != "$mps" ]]; then
    echo "install_nvidia.sh: the MPS programs ($mps) are not of the driver of the compatibility" \
         "libraries ($compat): docker/nvidia.pins" >&2
    exit 1
fi

mkdir -p /usr/local/cuda "$LICENSES/cuda-compat" "$LICENSES/libnvidia-compute"
mv "$tmp"/cuda-compat/usr/local/cuda-*/compat /usr/local/cuda/compat
cp "$tmp"/cuda-compat/usr/share/doc/cuda-compat-*/copyright "$LICENSES/cuda-compat/"
# The two MPS programs as NVIDIA ships them, and an empty directory of OpenCL vendors (the entry
# of the image points an MPS client there, see docker/entrypoint.sh).
mkdir -p /opt/fmeta/mps/bin /opt/fmeta/mps/opencl-vendors
install -m 755 "$tmp"/cuda-mps/usr/bin/nvidia-cuda-mps-control "$tmp"/cuda-mps/usr/bin/nvidia-cuda-mps-server \
    /opt/fmeta/mps/bin/
cp "$tmp"/cuda-mps/usr/share/doc/libnvidia-compute/copyright "$LICENSES/libnvidia-compute/"
# The licence and the notices of the cuPotential plugin, as its package installs them.
plugin="$(/opt/conda/bin/python -I -c \
    'import importlib.util as u; print(u.find_spec("nv_openmm_plugin").submodule_search_locations[0])')"
cp -r "$plugin/notices" "$LICENSES/nv-openmm-plugin"
rm -rf "$tmp"
