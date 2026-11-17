#!/usr/bin/env bash
# Step of the image build (the Dockerfile runs it as root, after the environment of the lock is
# installed): the CUDA forward-compatibility libraries and the licence texts of NVIDIA's software.
#
# With the compatibility libraries the CUDA 13.4 runtime of the image works on a host whose
# driver is older than R615. They are not a conda package: the lock names the file and its
# sha256 in a comment line (scripts/update_lock.sh writes it from docker/nvidia.pins), so the
# lock stays the one description of what the image holds. They go to /usr/local/cuda/compat,
# where the NVIDIA container runtime looks for them: it puts them on the loader path of a
# container only when they are newer than the driver of the host.
set -euo pipefail

LOCK=/opt/fmeta/conda-linux-64.lock
LICENSES=/opt/fmeta/licenses

line="$(sed -n 's/^# cuda-compat: //p' "$LOCK")"
url="${line%% *}"
sum="${line##* sha256=}"
if [[ -z "$url" || "$sum" == "$line" ]]; then
    echo "install_nvidia.sh: no '# cuda-compat: URL sha256=SUM' line in $LOCK" >&2
    exit 1
fi

tmp="$(mktemp -d)"
/opt/conda/bin/python -I - "$url" "$tmp/compat.deb" <<'EOF'
import shutil
import sys
import urllib.request

request = urllib.request.Request(sys.argv[1], headers={"User-Agent": "fmeta-image-build"})
with urllib.request.urlopen(request, timeout=300) as response, open(sys.argv[2], "wb") as out:
    shutil.copyfileobj(response, out)
EOF
echo "$sum  $tmp/compat.deb" | sha256sum -c - > /dev/null
dpkg-deb -x "$tmp/compat.deb" "$tmp/root"

mkdir -p /usr/local/cuda "$LICENSES/cuda-compat"
mv "$tmp"/root/usr/local/cuda-*/compat /usr/local/cuda/compat
cp "$tmp"/root/usr/share/doc/cuda-compat-*/copyright "$LICENSES/cuda-compat/"
# The licence and the notices of the cuPotential plugin, as its package installs them.
plugin="$(/opt/conda/bin/python -I -c \
    'import importlib.util as u; print(u.find_spec("nv_openmm_plugin").submodule_search_locations[0])')"
cp -r "$plugin/notices" "$LICENSES/nv-openmm-plugin"
rm -rf "$tmp"
