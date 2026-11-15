# The official fmeta image: the environment of conda-linux-64.lock (OpenMM with CUDA and OpenCL,
# the OpenFF / GAFF generators, the test and lint tools) plus fmeta itself.
#
#   docker build -t fmeta:dev .            (scripts/in_docker.sh --build does this)
#   docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$PWD":/work fmeta:dev auto ...
#
# The base image is pinned by digest and the environment by the lock, so a rebuild gives the
# environment the GPU gates were recorded in. Change packages in environment.yml and regenerate
# the lock with scripts/update_lock.sh.
FROM mambaorg/micromamba:2.10.0-ubuntu24.04@sha256:c3897a06f87c1c13f038188b3665a8c4162da62eac5109ae6a939d1e290dcefa

LABEL org.opencontainers.image.source="https://github.com/ytworks/fmeta" \
      org.opencontainers.image.description="fmeta: funnel metadynamics on OpenMM (GPU)"

# The copy of the lock inside the image is how a process tells that it runs in the official
# environment (scripts/gpu_gates.py records evidence only when it matches the repository's lock).
COPY --chown=$MAMBA_USER:$MAMBA_USER conda-linux-64.lock /opt/fmeta/conda-linux-64.lock
RUN micromamba install -y -n base -f /opt/fmeta/conda-linux-64.lock \
    && micromamba clean --all --yes --force-pkgs-dirs

USER root
# OpenCL finds the NVIDIA driver through this file; the driver library itself is mounted by
# the NVIDIA container runtime. /work is where a run directory or the repository is mounted.
RUN mkdir -p /etc/OpenCL/vendors \
    && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd \
    && mkdir -p /work && chmod 1777 /work
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/fmeta-entry
USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml README.md /opt/fmeta/source/
COPY --chown=$MAMBA_USER:$MAMBA_USER src /opt/fmeta/source/src
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN python -m pip install --no-deps --no-build-isolation --no-cache-dir /opt/fmeta/source

ARG LOCK_SHA256=""
LABEL fmeta.lock.sha256="$LOCK_SHA256"

# GPU numbers inside the container follow nvidia-smi (CUDA would otherwise order by speed).
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_DEVICE_ORDER=PCI_BUS_ID
WORKDIR /work
# tini is the init process: it forwards the signals of `docker stop` and reaps children.
ENTRYPOINT ["/opt/conda/bin/tini", "--", "/usr/local/bin/fmeta-entry", "fmeta"]
CMD ["--help"]
