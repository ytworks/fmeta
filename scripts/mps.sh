#!/usr/bin/env bash
# The NVIDIA MPS daemon, in a container of the official fmeta image. With it several replicas
# share one GPU at the same time (run.per_gpu above 1) instead of taking turns on it.
#
#   scripts/mps.sh start [GPUS]   GPUS: host GPU numbers (comma separated) or `all` (default)
#   scripts/mps.sh status
#   scripts/mps.sh stop           ends every process that is still a client: stop the runs first
#
# Then: scripts/in_docker.sh --mps --gpus 0,1 fmeta run WORKDIR --gpus 0,1 --per-gpu 4
#
# Only containers started with --mps are clients; anything else keeps using the GPUs directly,
# also on the same GPU. The daemon and its clients run as the calling user. If the MPS server
# of a GPU fails, all its clients end: run them again and they continue from their checkpoints.
# The MPS programs are those of the driver whose CUDA library the containers load: the host's,
# or the image's own where the image's compatibility libraries are in use (a host driver older
# than R615; the entry of the image chooses, docker/entrypoint.sh). A client then has no OpenCL
# platform.
# The daemon speaks MPS protocol 3 (see `start` below); its log, with the lines of its servers,
# is `docker logs NAME`.
# FMETA_MPS_NAME (default fmeta-mps) names the container, FMETA_MPS_DIR (default
# /tmp/fmeta-mps-UID) holds its pipe and log directories.
set -euo pipefail

IMAGE="${FMETA_IMAGE:-fmeta:dev}"
NAME="${FMETA_MPS_NAME:-fmeta-mps}"
DIR="${FMETA_MPS_DIR:-/tmp/fmeta-mps-$(id -u)}"

case "${1:-}" in
    start)
        GPUS="${2:-all}"
        if [[ "$GPUS" != all ]]; then GPUS="\"device=$GPUS\""; fi
        mkdir -p "$DIR/pipe" "$DIR/log"
        # CUDA_DEVICE_ORDER: the daemon numbers the GPUs as nvidia-smi and the clients do.
        # --protocol 3: under the default protocol (2, "legacy") the control program of driver
        # R615 stops reading its log pipe at the first record of a client. The pipe then fills
        # with every client that starts, and when it is full (64 KiB, about 50 client starts)
        # the daemon and every process that writes to it block for good (FM-D13).
        docker run --detach --name "$NAME" --gpus "$GPUS" --ipc shareable \
            --user "$(id -u):$(id -g)" -v "$DIR/pipe":/tmp/nvidia-mps -v "$DIR/log":/var/log/nvidia-mps \
            -e CUDA_DEVICE_ORDER=PCI_BUS_ID --entrypoint /opt/conda/bin/tini "$IMAGE" \
            -- /usr/local/bin/fmeta-entry nvidia-cuda-mps-control -f --protocol 3 > /dev/null
        echo "MPS daemon $NAME started; its log: docker logs $NAME" ;;
    status)
        docker container inspect --format '{{.Name}} {{.State.Status}} since {{.State.StartedAt}}' "$NAME" ;;
    stop)
        docker stop "$NAME" > /dev/null
        docker rm "$NAME" > /dev/null
        echo "MPS daemon $NAME stopped" ;;
    *)
        echo "usage: scripts/mps.sh start [GPUS] | status | stop" >&2
        exit 2 ;;
esac
