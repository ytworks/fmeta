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

# NVIDIA MPS. The daemon, its server and the CUDA driver library of every client must be of one
# driver version. A container loads the driver library of the host, or the one of the image's
# compatibility libraries when the driver of the host is older than the CUDA runtime (the NVIDIA
# container runtime then puts /usr/local/cuda/compat first on the loader path). In the second
# case the MPS programs are those of the image, of the driver of the compatibility libraries,
# instead of the ones mounted from the host. A client of that daemon must also not load the
# OpenCL library of the host: OpenMM initialises OpenCL when it is imported, and the older
# library then waits for the newer MPS server and never returns (FM-D13). Such a client has the
# CUDA platform and no OpenCL platform.
if /sbin/ldconfig -p 2> /dev/null | grep -m 1 -F 'libcuda.so.1 ' | grep -q -F ' /usr/local/cuda/compat/'; then
    export PATH="/opt/fmeta/mps/bin:$PATH"
    if [[ -S "${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}/control" ]]; then
        export OCL_ICD_VENDORS=/opt/fmeta/mps/opencl-vendors
    fi
fi
exec "$@"
