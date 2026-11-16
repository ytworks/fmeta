# GPU runbook

How to verify a change on a GPU machine. Everything here runs in a container of the official
image (`.claude/rules/docker-execution.md`, FM-D13); CI checks what needs no GPU (lint,
PROGRESS.md budget, evidence index, CPU tests, the image build). Criteria:
`.claude/rules/numerics-invariants.md`.

## Prerequisites

- An NVIDIA GPU with a working driver, and Docker with the NVIDIA Container Toolkit
  (`docker info` lists the `nvidia` runtime).
- The image. `scripts/in_docker.sh` builds it from `conda-linux-64.lock` on first use (about
  6 GB: the CUDA build of OpenMM, the ligand-parameter generators `openmmforcefields`,
  `openff-toolkit` and `ambertools`, and the test tools). It refuses an image that was built
  from another lock; `scripts/in_docker.sh --build ...` rebuilds.
- Two GPUs, and both free: `nvidia-smi --query-compute-apps=pid,process_name --format=csv` shows
  no other jobs on the GPUs you pass. The items `replicas_sequential` and `replicas_parallel`
  compare the memory in use on a GPU before and after a replica starts there: another process
  that frees memory on it in that moment fails them. With one GPU `replicas_parallel` is
  skipped, and nothing is recorded.

## GPU numbers

`--gpus` of the wrapper takes the numbers of the host (`nvidia-smi`) and passes only those GPUs.
Inside the container they are numbered from 0 again, and that is what fmeta and the gate script
see: `scripts/in_docker.sh --gpus 6,7 ... --gpus 0,1` runs on the host's GPUs 6 and 7. The
record stores the UUIDs, which identify the physical GPUs.

## Procedure

1. Commit the change (the gates record the commit). The record is flagged dirty when `src/`,
   `tests/`, `scripts/`, `examples/`, `docker/`, `pyproject.toml`, `environment.yml`,
   `conda-linux-64.lock`, `Dockerfile` or `.dockerignore` differ from it.
2. `scripts/in_docker.sh --gpus 6,7 python scripts/gpu_gates.py --gpus 0,1`:
   - `gpu-tests`: `pytest -m gpu`;
   - the items of `fmeta check` (`src/fmeta/check.py`, criteria in `CRITERIA` there): `platform`,
     `forces`, `toy_restart`, `toy_free_energy`, `pdb_sdf_openff`, `pdb_sdf_gaff`, `prepare`,
     `equilibrate`, `periodic_images`, `production`, `walkers`, `replicas_sequential`,
     `replicas_parallel`, `analysis`, `benchmark`, `membrane_build`, `membrane_run`.
   A skipped item is not a pass. The two `replicas_*` items (FM-D14) run real child processes
   and check their record `schedule.jsonl`, their hills and their `device.json`; they are
   skipped when `nvidia-smi` reports no memory in use for the GPU (then the device of a replica
   cannot be confirmed), and off a GPU platform.
   While iterating use `fmeta check --quick` or `scripts/gpu_gates.py --quick` (not recorded).
3. On ALL PASSED the script writes `bench/results/<date>-<commit>-gpu-gates.json` and updates
   `bench/results/index.json`. The record names the image, the hash of the lock and the UUIDs of
   the GPUs. Commit both files (CI validates the index), and update `PROGRESS.md` → Current
   Status if the gate status changed.
4. On failure: `gates_work/gpu-gates-not-recorded.json` lists every item with its detail, and
   `gates_work/<item>.traceback.txt` holds the traceback. Never relax a criterion to pass; a
   criterion changes only by a design amendment recorded in `docs/design/` and `PROGRESS.md`.

Evidence is written only inside the official image whose lock is the lock of the repository. A
run anywhere else (a conda environment on the host, an image built from an older lock) executes
the items and reports them, and records nothing.

## After a change of the environment

Packages change in `environment.yml`. Then:

```bash
scripts/update_lock.sh                           # solves the lock in a container
scripts/in_docker.sh --build pytest              # rebuilds the image and runs the CPU tests
scripts/in_docker.sh pytest -m slow
```

Commit the lock, and rerun the gates on that commit (the procedure above): the evidence of the
old lock does not cover the new image.

## Diagnostics

Wall time and the `benchmark` item (ns/day with and without the bias over the same stretch of
the test system, and the cost of a checkpoint) are recorded but never decide a verdict.
`scripts/in_docker.sh --gpus 6 fmeta doctor` prints the platforms, the optional packages and
the cost of a bias table update without running anything long.

When a run stops with "values that are not finite", the simulation broke down: nothing of the
broken step is in HILLS or COLVAR, and running the same command again continues from the last
checkpoint. If it happens again at the same place, look at the last COLVAR rows: a CV that
jumps by more than the funnel is long means the ligand and the protein were read in different
periodic images, which the item `periodic_images` exists to exclude.

## Production runs (reported, not gated)

Binding free energies of real systems (`examples/<name>/`) are long runs (40 ns per
replica by default) and are not analytic: they are reported, not gated. Run them on a clean commit, the
long stage as a detached container:

```bash
scripts/in_docker.sh --gpus 0 fmeta prepare examples/trypsin_benzamidine/fmeta.yaml -o runs/tb
scripts/in_docker.sh fmeta funnel runs/tb
scripts/in_docker.sh --gpus 0 fmeta equilibrate runs/tb
scripts/in_docker.sh --gpus 0,1,2,3 --detach --name tb fmeta run runs/tb --mode parallel
docker logs -f tb                                # `docker stop tb` stops the run
scripts/in_docker.sh fmeta status runs/tb        # state, time reached and device per replica
scripts/in_docker.sh fmeta analyze runs/tb --plot
```

`--mode parallel` puts one replica on each GPU of the container, and the next one on the GPU
that becomes free; `--gpus 0,1` names the GPUs instead (container numbers), and
`--mode sequential --gpus 0` runs the replicas one after another on one GPU (`run.mode` and
`run.gpus` in the configuration do the same). `docker stop` ends the replicas with the run:
the scheduler terminates its child processes before it returns, and `runs/tb/schedule.jsonl`
records what ran where. Running the same `fmeta run` command again continues from the last
checkpoints. Look at
`runs/tb/prepared/funnel.pdb` with the structure before spending GPU time. Record the outcome
(`summary.json`, the commit, the image, what converged and what did not) in the example's
README and in `PROGRESS.md`.

### Several replicas on one GPU (NVIDIA MPS)

One replica does not fill a large GPU. With the MPS daemon several replicas compute on it at
the same time (`run.per_gpu`, `--per-gpu`): on one RTX PRO 6000 Blackwell and 88k atoms, 4
replicas give 2.1 times and 6 replicas 2.5 times the total speed of one (FM-D14). Without the
daemon the replicas take turns and nothing is gained.

```bash
scripts/mps.sh start                             # the daemon, in a container of the image (all GPUs)
scripts/in_docker.sh --mps --gpus 0,1 --detach --name tb fmeta run runs/tb --gpus 0,1 --per-gpu 4
scripts/mps.sh status
scripts/mps.sh stop                              # ends every client: stop the runs first
```

Only containers started with `--mps` are clients of the daemon. If the MPS server of a GPU
fails, all its clients end; the same `fmeta run` command continues from the checkpoints.
