# GPU validation, run modes, official image, presets, examples, benchmarks

Status: In progress (PR #1)
Refs: roadmap R1–R4; FM-D13..FM-D17 (to be written as each milestone lands)

## Context

PR #1 built the framework on CPU; no GPU gate has been recorded and no real system has been
run (roadmap R1–R3). A machine with 8 GPUs is now available. Owner request (2026-11-15):
finish the development on the GPU machine and add

1. a mode that runs replicas concurrently on several GPUs and a mode that runs them one after
   another on one GPU;
2. an official, runnable Docker image;
3. a YAML file for the run settings (MD conditions, replicas, length) with official defaults
   suited to binding free energies;
4. one example each for a GPCR in a membrane and a kinase without one, with an experimental
   binding free energy;
5. benchmarks for GPCRs and kinases that have experimental values and FEP results, runnable so
   that the outcome can be compared.

Owner rule (same day): everything except CI is executed and tested in Docker, and the rule is
written down.

## Approach

One branch, small commits. Milestones:

| | Content | Done when |
|---|---|---|
| M0 | Docker rule, official image from a committed lock, wrapper, baseline, first GPU gate record (R1). FM-D13 | evidence taken inside the image is committed |
| M1 | Scheduler and run modes (item 1). FM-D14 | CPU tests and gates pass; `docker stop` leaves no child process |
| M2 | Handling of real input files. FM-D15 | characterisation tests, fixes, gates |
| M3 | Presets and official defaults (item 3). FM-D16 | preset tests and gates pass |
| M4 | Trypsin–benzamidine (R2); defaults settled | dG0 from the PDB entry, compared with experiment |
| M5 | TYK2 example, benchmark harness, TYK2 pilot (items 4, 5). FM-D17 | report for 3 ligands |
| M6 | A2A example (R3), A2A pilot (items 4, 5) | dG0 in a membrane; report for 3 ligands |
| M7 | Full benchmark runs, reports, documents, PR description | reports committed, CI green |

**M0 — Docker.** `.claude/rules/docker-execution.md` states the rule; `CLAUDE.md`, the GPU
runbook and the README follow it. One image (`Dockerfile`) is built from an explicit conda lock
generated inside a container and committed; it holds the test and lint tools, so the verified
image is the distributed one. `scripts/in_docker.sh` mounts the repository at `/work`, runs as
the host user and passes only the GPUs named with `--gpus`. Gate evidence records the image,
the lock hash and the GPU UUIDs, and is written only inside the official image. CI keeps the
`cpu` job and gains a `docker` job; publishing to GHCR is a separate workflow that only the
owner starts.

**M1 — Run modes.** `run.mode: auto | parallel | sequential` and `run.gpus: null | all | [..]`.
`parallel`: one replica per GPU, the next replica starts on the GPU that becomes free.
`sequential`: replicas in order on one GPU. `auto` keeps today's behaviour (parallel when
`gpus` is given). A new `src/fmeta/scheduler.py` runs jobs in child process groups, stops them
on SIGINT / SIGTERM and logs every assignment to `schedule.jsonl`; replicas always start as
subprocesses; a replica directory is locked while it runs. `fmeta status` shows progress.

**M2 — Real inputs.** Characterisation tests first (what OpenMM and PDBFixer do with chain
gaps, caps, existing hydrogens, disulfides, alternate locations), then: cap chain breaks with
ACE / NME instead of bonding across them, keep existing caps, carry protonation states of the
input over a hydrogen rebuild, restrain kept hetero components and treat them as obstacles,
`fmeta split --drop` for fusion partners, a non-confident membrane-plane check is a warning.

**M3 — Presets.** `src/fmeta/presets/soluble.yaml` and `membrane.yaml` list every key with its
unit; `fmeta init --preset` writes them; tests keep them equal to the dataclass defaults. The
ligand force-field version and charge method are pinned. `docs/configuration.md` is the
reference. Defaults are settled on trypsin–benzamidine (M4) from convergence diagnostics, before
any benchmark run, and never tuned against the benchmark's experimental values.

**M5–M7 — Examples and benchmarks.** Kinase example: TYK2 JH1, PDB 4GIH, its ligand 0X5
(`ejm_46` of the benchmark). GPCR example: adenosine A2A, PDB 5MZJ with theophylline, from the
OPM-oriented entry. Benchmarks: the TYK2 set (10 of 16 ligands) and the A2A set (8 of 17) of
Schrödinger's public binding-free-energy benchmark (MIT licence), which holds experimental and
FEP+ values per ligand. `fmeta benchmark prepare | run | status | report <set>` is
target-agnostic; a set is configuration in `examples/benchmarks/<set>/` (sources pinned by
commit and sha256, `reference.csv`); structure files are fetched, not committed. Reports
(`bench/reports/`) give dG0 per ligand with its confidence interval and convergence verdict
beside experiment and FEP, and MUE / RMSE / signed error / Pearson r / Kendall tau with
bootstrap intervals, raw and with the mean offset removed. Reports carry no verdict.
Three-ligand pilots come first; the full sets follow once throughput and convergence are
measured.

## Amendment (2026-11-16, owner)

The owner's decisions of 2026-11-16 change M4–M7; the detail is in FM-D14 and FM-D16
(amendments) and the Change Log of `PROGRESS.md`.

- Run length: 20 ns first, at most 40 ns. A scan of five faster bias settings on trypsin sets
  `run.length_ns` and the default bias by the rule of FM-D16 (second amendment), written before
  the scan. The 100 / 150 / 200 ns rule is withdrawn; `pytest -m slow` and checks on a CPU
  platform are not rerun.
- All 8 GPUs carry several replicas each with NVIDIA MPS (`run.per_gpu`). The GPU gates run
  once, on the commit of the final defaults.
- Done when, per item of the request: (1) the gates `replicas_parallel` and
  `replicas_sequential` pass on the final code and the trypsin example has run on one GPU in
  `sequential` mode; (2) the image of the final commit runs without the mounted source and the
  CI job `docker` passes; (3) the defaults from the scan are the same in the code, the presets,
  `docs/configuration.md` and the examples; (4) the three examples (trypsin, TYK2 ejm_46, A2A
  theophylline) run from their READMEs to dG0, written beside experiment; (5) both sets (10
  TYK2 and 8 A2A ligands) run with the official defaults and their reports are in
  `bench/reports/`. A replica that does not converge is reported as NOT CONVERGED; nothing is
  extended.

## Invariants

- No criterion in `.claude/rules/numerics-invariants.md` is relaxed. New gate items
  (`replicas_sequential`, `replicas_parallel`) add criteria.
- Changes to the runner (M1), system building (M2, M3) and platform handling (M0, M1) pass the
  GPU gates on a clean commit before the next milestone builds on them.
- Force-field parameters are never invented: caps and ions use the Amber templates.
- `src/fmeta/` stays target-agnostic; targets and benchmark sets are configuration under
  `examples/`.
- Convergence criteria are not changed to make a strong binder count as converged; such runs
  are reported as NOT CONVERGED.

## Verification

All commands run through `scripts/in_docker.sh`.

- CPU: `ruff check .`, `python scripts/check_progress_size.py`,
  `python scripts/verify_bench_results.py`, `pytest`, `pytest -m slow`.
- GPU: `python scripts/gpu_gates.py` ALL PASSED on a clean commit after every gated change;
  evidence in `bench/results/`.
- Run modes: 5 replicas on 5 GPUs at once; the same on one GPU in order; `docker stop` leaves
  no process and the same command resumes from the checkpoints.
- Image: a run without the source mounted works; output files belong to the host user.
- Examples: each README followed from the first command to dG0; result and experiment recorded.
- Benchmarks: `fmeta benchmark report` output committed for both sets.
- Not gated: binding free energies of real systems are reported with their diagnostics.
