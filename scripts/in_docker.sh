#!/usr/bin/env bash
# Run a command inside the official fmeta image with this repository mounted at /work.
# Everything except CI is executed this way (.claude/rules/docker-execution.md).
#
#   scripts/in_docker.sh pytest
#   scripts/in_docker.sh --gpus 6,7 python scripts/gpu_gates.py --gpus 0,1
#   scripts/in_docker.sh --gpus 0,1,2,3 --detach --name run1 fmeta auto ... -o runs/run1
#
# Options (before the command):
#   --gpus LIST      host GPU numbers (comma separated) or `all`; no GPU when omitted.
#                    Inside the container the GPUs passed are numbered from 0 again.
#   --detach         start in the background; follow with `docker logs -f NAME`. The container
#                    is kept after it ends (logs, exit status).
#   --name NAME      container name. A finished container of that name that this script
#                    started from this repository is replaced, so the same command continues
#                    a stopped run; any other container of that name is left alone (error).
#   --mount SRC:DST  additional bind mount (repeatable)
#   --mps           make the processes clients of the NVIDIA MPS daemon that `scripts/mps.sh start`
#                    runs: for replicas that share a GPU (run.per_gpu above 1)
#   --image IMAGE    image to run (default: $FMETA_IMAGE, else fmeta:dev)
#   --build          build the image from the Dockerfile first
#
# The mounted source is the one that runs (PYTHONPATH=/work/src), as the calling user, so
# files written under the repository belong to that user and a clean commit stays clean.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${FMETA_IMAGE:-fmeta:dev}"
GPUS=""
DETACH=0
BUILD=0
NAME=""
MPS=0
MPS_NAME="${FMETA_MPS_NAME:-fmeta-mps}"
MPS_DIR="${FMETA_MPS_DIR:-/tmp/fmeta-mps-$(id -u)}"
MOUNTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --detach) DETACH=1; shift ;;
        --name) NAME="$2"; shift 2 ;;
        --mount) MOUNTS+=(-v "$2"); shift 2 ;;
        --mps) MPS=1; shift ;;
        --image) IMAGE="$2"; shift 2 ;;
        --build) BUILD=1; shift ;;
        --) shift; break ;;
        -*) echo "in_docker.sh: unknown option $1" >&2; exit 2 ;;
        *) break ;;
    esac
done

LOCK_SHA256="$(sha256sum "$ROOT/conda-linux-64.lock" | cut -d' ' -f1)"
if [[ $BUILD -eq 1 ]] || ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    docker build -t "$IMAGE" --build-arg LOCK_SHA256="$LOCK_SHA256" "$ROOT" >&2
fi
BUILT_FROM="$(docker image inspect --format '{{index .Config.Labels "fmeta.lock.sha256"}}' "$IMAGE")"
if [[ "$BUILT_FROM" != "$LOCK_SHA256" ]]; then
    echo "in_docker.sh: image $IMAGE was not built from the current conda-linux-64.lock;" \
         "rerun with --build" >&2
    exit 2
fi
if [[ $# -eq 0 ]]; then
    echo "in_docker.sh: no command given" >&2
    exit 2
fi

# Start in the directory that corresponds to the caller's, when that is inside the repository.
WORKDIR=/work
case "$PWD/" in
    "$ROOT"/*) WORKDIR="/work${PWD#"$ROOT"}" ;;
esac

ARGS=(--user "$(id -u):$(id -g)" -v "$ROOT":/work -w "$WORKDIR"
      -e PYTHONPATH=/work/src
      -e FMETA_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE")")
# The commit of this checkout and whether anything in it differs from that commit. Read here
# because git cannot answer inside the container when the checkout is a git worktree: its
# .git points to a directory that is not mounted. `fmeta benchmark` records both.
if CODE_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2> /dev/null)"; then
    CODE_DIRTY=0
    if [[ -n "$(git -C "$ROOT" status --porcelain 2> /dev/null)" ]]; then CODE_DIRTY=1; fi
    ARGS+=(-e FMETA_CODE_COMMIT="$CODE_COMMIT" -e FMETA_CODE_DIRTY="$CODE_DIRTY")
fi
if [[ "$GPUS" == all ]]; then
    ARGS+=(--gpus all)
elif [[ -n "$GPUS" ]]; then
    ARGS+=(--gpus "\"device=$GPUS\"")
fi
if [[ $MPS -eq 1 ]]; then
    if [[ "$(docker container inspect --format '{{.State.Status}}' "$MPS_NAME" 2> /dev/null || true)" != running ]]; then
        echo "in_docker.sh: --mps needs the MPS daemon: scripts/mps.sh start" >&2
        exit 2
    fi
    # A client reaches the daemon through its pipe directory and its shared memory.
    ARGS+=(--ipc "container:$MPS_NAME" -v "$MPS_DIR/pipe":/tmp/nvidia-mps)
fi
if [[ -n "$NAME" ]]; then
    FOUND="$(docker container inspect \
        --format '{{.State.Status}} {{index .Config.Labels "fmeta.repository"}}' "$NAME" 2> /dev/null || true)"
    if [[ "$FOUND" == "exited $ROOT" ]]; then
        docker rm "$NAME" > /dev/null
    elif [[ -n "$FOUND" ]]; then
        echo "in_docker.sh: a container named $NAME exists (${FOUND%% *}); stop it or use another name" >&2
        exit 2
    fi
    ARGS+=(--name "$NAME" --label "fmeta.repository=$ROOT")
fi
if [[ $DETACH -eq 1 ]]; then
    ARGS+=(--detach)
else
    ARGS+=(--rm)
    if [[ -t 0 ]]; then ARGS+=(--interactive); fi
    if [[ -t 0 && -t 1 ]]; then ARGS+=(--tty); fi
fi

exec docker run "${ARGS[@]}" ${MOUNTS[@]+"${MOUNTS[@]}"} --entrypoint /opt/conda/bin/tini "$IMAGE" \
    -- /usr/local/bin/fmeta-entry "$@"
