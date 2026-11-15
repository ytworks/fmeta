# Progress

Cross-session memory. Rules: `.claude/rules/progress-log.md`.
Older Change Log entries: `docs/progress/archive/change-log.md`.

## Product

**Goal** (owner request 2026-11-15; roadmap `docs/roadmap.md`): funnel metadynamics on OpenMM
that runs on GPUs, end to end, from two files.

- Inputs: a protein PDB and a ligand SDF in its bound pose. Everything else is derived:
  protein clean-up and hydrogens, ligand parameters (OpenFF / GAFF), funnel, box.
- Soluble proteins in a rhombic dodecahedron; membrane proteins (GPCRs) in a bilayer built by
  `Modeller.addMembrane`, with the funnel pointing into water.
- Engine: well-tempered metadynamics on the funnel CVs (projection `s`, extent `rho`) with
  OpenMM custom forces; replicas for error bars, shared-bias walkers for speed; exact restart.
- Result: standard binding free energy (plateau mean, 95 % confidence interval over replicas)
  with convergence diagnostics that say NOT CONVERGED or "no binding" when that is the case.
- OpenMM only: no input from other MD programs.
- Planned (owner request 2026-11-15, roadmap R7–R10): replicas side by side on several GPUs or
  one after another on one GPU; official presets; a kinase and a GPCR example; benchmarks
  against experiment and FEP.
- Hardware: an 8-GPU machine (since 2026-11-15). Everything except CI runs in the official
  Docker image (FM-D13).

## Current Status

Snapshot date: 2026-11-16. Decisions: `docs/design/engine.md` (FM-D1..D14 Accepted; FM-D2, D4,
D5, D6, D7, D11 and D12 amended since, see the Change Log). Everything below is on PR #1 (not merged).
Plan in progress: `docs/plans/2026-11-15-gpu-validation-modes-benchmarks.md`.

| Milestone | Status |
|---|---|
| M0 Scaffold | Complete |
| M1 CVs and funnel walls | Complete; equal to numpy references and finite differences |
| M2 Bias engine, HILLS / COLVAR, restart | Complete; table equal to `openmm.app.Metadynamics`, restart byte-identical |
| M3 Analysis and the solvable toy system | Complete; toy dG0 within 0.5 kJ/mol of exact (4 × 40 ns) |
| M4 Automatic funnel, validation | Complete on synthetic pockets (direction ≤ 10°) |
| M5 System building, equilibration, workflow, CLI | Complete; PDB + SDF alone, with automatic clean-up |
| M6 Membrane | Complete on CPU: real `addMembrane` build (POPC, 108 lipids) and a short run; membrane-plane fit on synthetic helix bundles only |
| M7 Replicas, walkers, figures, `fmeta check` | Complete on CPU |
| Harness: CPU CI + GPU gates | Introduced 2026-11-15 (PR #1) |
| Official image, Docker-only execution (FM-D13) | In use on the GPU machine; the CI job `docker` and the publish workflow have never run |
| GPU plan M0: first gate record (R1) | Complete: all 16 gates on 2c4ad4e; the fmeta inside the image runs without the mounted source (quick check on CUDA, not evidence) |
| GPU plan M1: run modes and the scheduler (R7, FM-D14) | Complete: all 18 gates on 8d1f10a; `docker stop` and continuation checked on the trypsin run |
| GPU plan M2, M3, M5 infrastructure: real inputs, presets, benchmark commands (R8–R10) | In progress on separate branches; not merged, not gated |
| GPU plan M4: trypsin-benzamidine (R2) | Running: 5 replicas × 500 ns from PDB entry 3PTB on GPUs 0–4 (container `fmeta-tb`, commit 8d1f10a), started 2026-11-15 23:17 UTC, about 18 h at 685 ns/day. No result yet |
| GPU plan M5: kinase example (R10) | Running: TYK2 with ejm_46 from PDB entry 4GIH (`examples/tyk2_ejm46`, 88476 atoms), replica 0 on GPU 5 (container `fmeta-tyk2`, commit 8d1f10a) since 2026-11-15 23:30 UTC; the other replicas follow when GPUs 0–4 are free. No result yet |
| GPU plan M6, M7: GPCR example, benchmark runs (R3, R9, R10) | Not started |

- GPU gates: **all 18 passed** (`gpu-tests` and the 17 items of `fmeta check`) on clean 8d1f10a
  (CUDA, mixed precision, 2 GPUs, in the image):
  `bench/results/2026-11-15-8d1f10a89-gpu-gates.json` (dated in UTC). It covers the run modes
  (FM-D14): `replicas_sequential` and `replicas_parallel` ran real child processes on the GPUs.
- Other checks, in the image (lock d061e45a…): `ruff check .` passes and `pytest` 102 passed on
  the code of FM-D14. Operation on a CPU platform is not guaranteed (owner, 2026-11-16,
  FM-D12): `pytest -m slow` (6 passed) and `fmeta check --platform CPU` were last run on the
  source of 2c4ad4e and are not rerun.
- CI: green on 171220a. The commits after it were pushed on 2026-11-16; the result of CI on
  them, with the first run of the job `docker`, is not checked here.
- Not verified yet:
  - a binding free energy of a real protein-ligand system (the first run is in progress);
    the membrane-plane estimate on a real membrane protein;
  - reading HILLS with PLUMED itself;
  - the publish workflow (never run).
- Known gaps: ligand protonation is taken from the SDF as drawn (R5); unresolved loops are not
  modelled; non-protein components are removed unless listed in `system.keep_hetero`; a
  directory prepared before 2026-11-16 keeps periodic funnel walls in `prepared/system.xml` and
  must be prepared again; the gate `toy_free_energy` passes with 0.04 kJ/mol to spare, and
  about 1 run in 30 of its 10 ns misses the criterion on every platform (FM-D12), so a change
  of the toy trajectory (another GPU, another OpenMM) can fail it without a defect.
- Blockers: none. Pushes from the GPU machine work since 2026-11-16 (the `gh` login of the
  machine as the git credential; the owner added the `workflow` scope).

## Change Log

### 2026-11-16 — [progress] Kinase example added and started: TYK2 with ejm_46 from PDB entry 4GIH (R10)
- What: `examples/tyk2_ejm46/` (README with the experimental Ki of 4.8 nM, −11.3 kcal/mol, and the steps; a configuration that only names the two input files). On gated commit 8d1f10a in the image, `fmeta split`, `prepare`, `funnel` and `equilibrate` passed without a change: 0X5 and 119 waters removed, 1 heavy atom rebuilt, no chain break, funnel placed automatically (path to bulk 2.65 nm, bound pose at s = 1.20 nm), 88476 atoms, wall energy 0.000 kJ/mol in equilibration. Run started with `--mode sequential` on GPU 5 (container `fmeta-tyk2`): replica 0 at about 570 ns/day. It moves to the parallel mode when GPUs 0–4 are free.
- Not verified yet: the free energy; whether a ligand this tight rebinds within 500 ns (the convergence criterion of 3 rebinding events is unchanged).
- Refs: roadmap R10, `examples/tyk2_ejm46/README.md`.

### 2026-11-16 — [progress] First run from a real PDB entry started (R2); stop and continuation of a parallel run verified
- What: trypsin-benzamidine from entry 3PTB, on gated commit 8d1f10a in the image. `fmeta split`, `prepare`, `funnel` and `equilibrate` passed without a change: BEN, CA and 62 waters removed, 6 disulfide bonds in the topology, no chain break, funnel placed automatically (path to bulk 2.30 nm), 72434 atoms, wall energy 0.000 kJ/mol in equilibration. `fmeta run --mode parallel` on GPUs 0–4: 5 replicas on 5 GPUs at once, 660–700 ns/day each (`fmeta status`, `nvidia-smi`). `docker stop` returned within 1 s, left no process on the GPUs and recorded every replica as stopped in `schedule.jsonl`; the same command continued all five from their checkpoints (HILLS without a repeated or missing hill).
- Not verified yet: the free energy itself (500 ns per replica, about 18 h).
- Refs: roadmap R2 and R7, FM-D14, `examples/trypsin_benzamidine/README.md`.

### 2026-11-16 — [progress] GPU gates passed on the run modes (FM-D14): M1 of the GPU plan gated
- What: `scripts/gpu_gates.py` in the official image on two GPUs (host 6 and 7), clean 8d1f10a: `gpu-tests` and all 17 items passed, none skipped (234 s). `replicas_sequential`: 2 replicas one after another on one GPU. `replicas_parallel`: 3 replicas on 2 GPUs, the third on the GPU that became free. Each replica raised the memory in use on its GPU by 595–596 MiB, and `schedule.jsonl` and `device.json` agree with the assignment. Other values: forces CV 3.8e-07 nm; `periodic_images` CVs 8.4e-07 nm, wall energy 1.3e-04 kJ/mol; toy 10 ns reweighted −2.99, plateau −3.46 (exact −4.95 kJ/mol); 1371 ns/day plain, 1312 with the bias (24062 atoms); membrane 35483 atoms, 812 ns/day.
- Not verified yet: `docker stop` of a parallel run on GPUs and its continuation.
- Refs: FM-D14, roadmap R7, `bench/results/2026-11-15-8d1f10a89-gpu-gates.json`, `README.md`.

### 2026-11-16 — [amendment] FM-D12 (owner): operation on a CPU platform is not guaranteed and not confirmed
- What: from FM-D14 on, the pipeline and the run modes are verified by the GPU gates alone. `fmeta check --platform CPU` and the pipeline runs of `pytest -m slow` are not rerun as verification (last run on the source of 2c4ad4e), and a new feature gets no test that runs dynamics on a CPU platform. The default `pytest` stays the test of logic and numerics. Unchanged until the owner decides: item 3 of the test policy in `CLAUDE.md` (the default `pytest` passes on a CPU-only machine) and the `cpu` job of CI.
- Why: owner instruction of 2026-11-16: operation on a CPU need not be guaranteed, and the time spent on confirming it is wasted. Production runs on GPUs only.
- Refs: FM-D12 (amendment), FM-D14.

### 2026-11-16 — [design] FM-D14: run modes `auto` / `parallel` / `sequential`, and one scheduler for the child processes of a run
- What: `run.mode` and `--mode`; `run.gpus` is a list or `all`. `src/fmeta/scheduler.py` starts every replica and walker as a child process in its own process group, gives out the devices in list order, ends the children on SIGINT / SIGTERM (kill after 5 s or at a second signal) and writes every decision to `schedule.jsonl`, which `verify_schedule` checks. A work directory and a replica directory are held with `flock`; `device.json` records the device a run asked for, the device the platform reports and the memory in use before and after; `fmeta status` lists state, time reached and device per replica; preparation and equilibration use the first device. New gate items `replicas_sequential` and `replicas_parallel` (a record needs two GPUs). `ruff` passes; `pytest` 102 passed (21 new, on the order of the scheduler's decisions). GPU gates not run yet on this code.
- Why: owner request of 2026-11-15 for both modes. The code before handled no signal, so a stopped run or container could leave replicas on the GPUs, and the gates ran replicas in the calling process, so the launch of the children was outside them. `auto` keeps the behaviour before (one device unless `gpus` is given). Not added: a key for several replicas per GPU (effect not measured; a key cannot be withdrawn later).
- Refs: FM-D14; roadmap R7; `docs/gpu-runbook.md`; `src/fmeta/scheduler.py`, `src/fmeta/workflow.py`, `src/fmeta/engine/platform.py`, `src/fmeta/check.py`, `src/fmeta/cli.py`; `tests/test_scheduler.py`.
