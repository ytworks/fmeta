# Docker execution rule

Owner rule (2026-11-15): everything except CI is executed and tested in Docker. Design: FM-D13
(`docs/design/engine.md`).

## The rule

- fmeta commands, simulations, `pytest`, `ruff`, the scripts in `scripts/`, the GPU gates and
  the benchmarks run in a container of the official image, through `scripts/in_docker.sh`.
- On the host: `git`, `docker`, file editing and read-only state checks (`nvidia-smi`,
  `docker logs`). No Python environment on the host (no conda, venv or `pip install`), and the
  project's Python is never run there.
- Exactly two exceptions:
  1. CI (`.github/workflows/`). Its `cpu` job installs with pip on the runner: that is what
     keeps the core usable without the conda-only packages.
  2. The SessionStart hook (`scripts/check_progress_size.py`, standard library only), which
     must also work in a session that has no Docker.
- A session without Docker does not fall back to the host: leave the verification to CI and
  report "not verified here".

## The image

- One image (`Dockerfile`): the environment of `conda-linux-64.lock` on a base image pinned by
  digest, plus fmeta. It holds the test and lint tools, so the image that is verified is the
  image that is distributed.
- Packages change in `environment.yml`; `scripts/update_lock.sh` regenerates the lock (in a
  container). After a change of the lock or the `Dockerfile`: rebuild, rerun the tests and the
  GPU gates, and commit the lock with the new evidence.
- GPU evidence (`bench/results/`) is written only inside the image, when the lock in the image
  is the lock of the repository; the record names the image, the lock and the GPUs.

## Usage

```bash
scripts/in_docker.sh pytest                                  # builds the image on first use
scripts/in_docker.sh --gpus 6,7 python scripts/gpu_gates.py --gpus 0,1
scripts/in_docker.sh --gpus 0,1,2,3 --detach --name run1 fmeta run runs/run1 --gpus 0,1,2,3
```

- The repository is mounted at `/work` and its source is the one that runs; the process is the
  calling user, so files it writes belong to that user.
- `--gpus` of the wrapper takes host GPU numbers and passes only those (none when omitted).
  Inside the container they are numbered from 0 again: `--gpus 6,7` outside is `0,1` inside.
- Long runs use `--detach`: follow with `docker logs -f NAME`, stop with `docker stop NAME`,
  continue by running the same command again.
