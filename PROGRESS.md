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
| GPU plan M1: run modes and the scheduler (R7, FM-D14) | Implemented; `pytest` 102 passed; GPU gates (now 18) not run yet on this code |
| GPU plan M2–M7: real inputs, presets, examples, benchmarks (R2, R3, R8–R10) | Not started |

- GPU gates: **all 16 passed** on clean 2c4ad4e (CUDA, mixed precision, 2 GPUs, in the image):
  `bench/results/2026-11-15-2c4ad4e50-gpu-gates.json` (dated in UTC). The run modes (FM-D14)
  change gated paths (runner, platform handling) and add the items `replicas_sequential` and
  `replicas_parallel`: not run yet on that code.
- Other checks, in the image (lock d061e45a…): `ruff check .` passes and `pytest` 102 passed on
  the code of FM-D14. Operation on a CPU platform is not guaranteed (owner, 2026-11-16,
  FM-D12): `pytest -m slow` (6 passed) and `fmeta check --platform CPU` were last run on the
  source of 2c4ad4e and are not rerun.
- CI: green on 171220a. The commits up to c5fc36c were pushed on 2026-11-16; the result of CI
  on them, with the first run of the job `docker`, is not checked here.
- Not verified yet:
  - a binding free energy of a real protein-ligand system; input from a real PDB entry;
    the membrane-plane estimate on a real membrane protein;
  - `docker stop` of a parallel run on GPUs and its continuation (FM-D14);
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

### 2026-11-16 — [amendment] FM-D12 (owner): operation on a CPU platform is not guaranteed and not confirmed
- What: from FM-D14 on, the pipeline and the run modes are verified by the GPU gates alone. `fmeta check --platform CPU` and the pipeline runs of `pytest -m slow` are not rerun as verification (last run on the source of 2c4ad4e), and a new feature gets no test that runs dynamics on a CPU platform. The default `pytest` stays the test of logic and numerics. Unchanged until the owner decides: item 3 of the test policy in `CLAUDE.md` (the default `pytest` passes on a CPU-only machine) and the `cpu` job of CI.
- Why: owner instruction of 2026-11-16: operation on a CPU need not be guaranteed, and the time spent on confirming it is wasted. Production runs on GPUs only.
- Refs: FM-D12 (amendment), FM-D14.

### 2026-11-16 — [design] FM-D14: run modes `auto` / `parallel` / `sequential`, and one scheduler for the child processes of a run
- What: `run.mode` and `--mode`; `run.gpus` is a list or `all`. `src/fmeta/scheduler.py` starts every replica and walker as a child process in its own process group, gives out the devices in list order, ends the children on SIGINT / SIGTERM (kill after 5 s or at a second signal) and writes every decision to `schedule.jsonl`, which `verify_schedule` checks. A work directory and a replica directory are held with `flock`; `device.json` records the device a run asked for, the device the platform reports and the memory in use before and after; `fmeta status` lists state, time reached and device per replica; preparation and equilibration use the first device. New gate items `replicas_sequential` and `replicas_parallel` (a record needs two GPUs). `ruff` passes; `pytest` 102 passed (21 new, on the order of the scheduler's decisions). GPU gates not run yet on this code.
- Why: owner request of 2026-11-15 for both modes. The code before handled no signal, so a stopped run or container could leave replicas on the GPUs, and the gates ran replicas in the calling process, so the launch of the children was outside them. `auto` keeps the behaviour before (one device unless `gpus` is given). Not added: a key for several replicas per GPU (effect not measured; a key cannot be withdrawn later).
- Refs: FM-D14; roadmap R7; `docs/gpu-runbook.md`; `src/fmeta/scheduler.py`, `src/fmeta/workflow.py`, `src/fmeta/engine/platform.py`, `src/fmeta/check.py`, `src/fmeta/cli.py`; `tests/test_scheduler.py`.

### 2026-11-16 — [amendment] FM-D12: the scatter of the toy gate is on record, criteria unchanged; GPU plan M0 complete
- What: the 10 ns run of the gate `toy_free_energy` was repeated with 48 other seeds, and a 40 ns run with 12, on CUDA mixed, CUDA double and Reference (diagnostic in the image, not evidence). Error of the reweighted estimate at 10 ns, mean ± sd: +0.48 ± 0.83, +0.51 ± 0.83, +0.46 ± 0.83 kJ/mol; beyond the criterion of 2.0 in 2, 2 and 1 of 48 runs; the plateau estimate never beyond 3.0 (largest 2.62). At 40 ns (36 runs): reweighted +0.12 ± 0.33, plateau +0.40 ± 0.50. The gate's seed gave the same numbers in both full gate runs. M0 closed with the fmeta inside the image, without the mounted source (image b08f577e…, built from b19deff): `fmeta doctor` found CUDA; `fmeta check --quick` on CUDA 15 passed, 0 failed, 0 skipped; all 116 files it wrote belong to the caller (quick mode on a GPU shared with the diagnostic: not evidence).
- Why: the gate passed with 0.04 kJ/mol to spare and 12 seeds (entry below) suggested an offset of mixed precision. 48 seeds show none: the platforms agree, and the positive mean belongs to a 10 ns run. About 1 trajectory in 30 misses the criterion with a correct implementation, so a failure after a change of the toy trajectory is compared with these numbers first. The tolerance is not widened for a run; another form of the criterion would be a new amendment (open proposal: the mean over several seeds).
- Refs: FM-D12 (table); `README.md` (検証状況); `src/fmeta/check.py` (`check_toy_free_energy`).

### 2026-11-16 — [progress] First GPU gate record (R1): all 16 gates passed on 2c4ad4e
- What: `scripts/gpu_gates.py` in the official image (image 6f68772d…, lock d061e45a…) on two RTX PRO 6000 Blackwell (host GPUs 6 and 7), CUDA, mixed precision, clean 2c4ad4e, 211 s: `pytest -m gpu` 2 passed and the 15 items of `fmeta check` passed. Measured: forces against Reference CV 3.8e-07 nm, energy 7.2e-06, force 5.6e-06; `periodic_images` CVs 2.3e-06 nm, wall energy 5.1e-03 kJ/mol; toy 10 ns reweighted 1.96 and plateau 1.49 kJ/mol from exact (criteria 2.0 and 3.0), 24 rebinding events; 24062 atoms at 1343 ns/day plain and 1293 with the bias; membrane (35500 atoms) 803 ns/day. Because the toy margin is 0.04 kJ/mol, the same 10 ns run was repeated with 12 seeds per platform (diagnostic, not evidence). Error of the reweighted estimate, mean ± sd and largest: CUDA mixed +0.96 ± 0.64, 1.96; CUDA double +0.61 ± 0.52, 1.18; Reference +0.44 ± 0.59, 1.52 kJ/mol. The gate's seed is the worst of the 12 on CUDA mixed. The criterion is unchanged. `pytest` 81 passed, `pytest -m slow` 6 passed on the same source.
- Refs: `bench/results/2026-11-15-2c4ad4e50-gpu-gates.json` (dated in UTC), `bench/results/index.json`; `docs/roadmap.md` (R1 done); `README.md` (検証状況).

### 2026-11-16 — [amendment] FM-D7: after dynamics the ligand is judged by the funnel walls, not atom by atom
- What: the first full gate run (clean 81efa0d, CUDA, 2 GPUs, 193 s) gave 14 passed, 1 failed (`membrane_run`); nothing was recorded. The stage `equilibrate` had stopped with "ligand atoms lie 0.02 nm outside the funnel wall". Now `validate_funnel(..., sample_kT=...)` judges a structure from dynamics by the wall energy of the ligand centre: an error above 10 kT (`WALL_BREACH_KT`); atoms beyond the wall and a centre within 0.2 nm of the lower wall are warnings. The checks of the input pose at preparation and the gate criteria are unchanged. `pytest` 81 passed; the two new tests fail on the code before. Gates not rerun yet.
- Why: measured on the failed run. The benzene test ligand had diffused 0.37 nm in 70 ps (protein CA RMSD 0.08 nm); its centre was 0.11 nm inside the wall and one atom 0.017 nm outside; the walls did 0.088 kJ/mol on average (0.035 kT). The walls act on the centroid, so the atom rule on one snapshot was decided by diffusion, not by the funnel. Rejected: a wider funnel or a shorter equilibration in the gate (a user's run would still stop); warnings only (a ligand the walls do not hold must stop the pipeline).
- Refs: FM-D7; `.claude/rules/numerics-invariants.md`; `src/fmeta/funnel/validate.py`, `src/fmeta/workflow.py`; `tests/test_autofunnel.py`, `tests/test_system.py`.

### 2026-11-16 — [amendment] FM-D2, FM-D5, FM-D6: CV forces without periodic boundaries; gate `periodic_images` redefined; runs stop before recording NaN
- What: the forces of the CVs, the walls and the bias no longer use periodic boundaries (the `periodic` argument is gone); the box rule is unchanged. Before: 5 of 40 biased GPU runs of 800 ps on the test system of `fmeta check` ended in NaN. After: the 5 failing seeds pass and so do 40 new runs; started with a corner of the cell inside the complex, the complex changed cell 57 times in 200 ps with the CVs within 8.9e-07 nm of numpy. The gate `periodic_images` now runs dynamics from two placements and probes the upper wall along the narrowest direction of the box (CV tolerance 2e-3 nm unchanged; wall energy within 0.1 kJ/mol): it fails on the periodic forces (1.2e6 kJ/mol), which the old static gate passed. `forces` loses its seventh position, a periodic image of another one. A run now stops with an error before a value that is not finite reaches HILLS, COLVAR or the grid. `fmeta check --quick` on CUDA: 15 passed, 0 failed, 0 skipped (not evidence).
- Why: two facts from the first GPU runs. OpenMM's periodic wrap of a displacement is the nearest image only within half of `a.x`, `b.y` and `c.z`, which is 0.354 L along z in the rhombic dodecahedron, so the CVs jumped (`s` from 2.75 to −4.00 nm in one step) in a box that satisfies FM-D6. And a context holds the ligand and the protein as one molecule, because the centroid forces join them, so a platform moves them between cells together. Rejected: periodic CVs in a cube (41% more volume).
- Refs: FM-D2, FM-D5, FM-D6; `.claude/rules/numerics-invariants.md`; `src/fmeta/funnel/forces.py`, `src/fmeta/check.py`, `src/fmeta/engine/runner.py`; `tests/test_engine.py`.

### 2026-11-16 — [amendment] FM-D4 (roadmap R4): the bias table holds spline coefficients; two force forms by platform
- What: fmeta fits the B-spline coefficients of the bias itself after every hill, and the energy expression evaluates the bicubic spline from a `Discrete2DFunction`. The bias is the same function as before (test against a `Continuous2DFunction`: energy 1e-10, forces 1e-8). Measured on the test system of `fmeta check` (24062 atoms, CUDA, mixed precision, 4 fs, one hill per ps): 73.4% of the speed of plain MD before, 96–97% after (1.84 ms per hill). GPU platforms get a `CustomCentroidBondForce` plus a zero-energy force that reads the CVs outside the integration; Reference and CPU get one `CustomCVForce`; a context of the other kind is refused. `benchmark` compares the same stretch with and without the bias and reports the cost of a checkpoint.
- Why: the old form was below the 80% the plan set for starting R4. Two forms because of three measurements on OpenMM 8.6.1: a centroid force with a table and an energy parameter derivative returns zero on CUDA / OpenCL without an error; a `CustomCVForce` in every step costs about 20% on a GPU; a table update of a centroid force takes 0.7–0.9 s on Reference / CPU.
- Refs: FM-D4; `docs/roadmap.md` (R4 done); `src/fmeta/engine/bias.py`, `src/fmeta/engine/grid.py`, `src/fmeta/check.py`; `tests/test_engine.py`, `tests/test_platform.py`.

### 2026-11-15 — [amendment] FM-D2: the forces compute the CVs in the angle form
- What: `s = d cos θ`, `rho = sqrt((d sin θ)² + eps²)` from the distance ligand–p0 and the angle ligand–p0–p1, instead of three distances; the numpy reference uses the dot and the cross product of the same two vectors. Gate item `forces` on CUDA (mixed precision), worst CV deviation from Reference: 3.6e-05 nm before (criterion 2e-5 nm: failed), 3.8e-07 nm after. The criterion is unchanged.
- Why: the first run of the item on a GPU. Near the axis `dL0² − s²` is a difference of nearly equal numbers and loses its digits in single precision.
- Refs: FM-D2; `src/fmeta/funnel/forces.py`, `src/fmeta/funnel/geometry.py`; `tests/test_forces.py`.

### 2026-11-15 — [design] FM-D13: everything except CI runs in one official image built from a lock
- What: `Dockerfile` (base pinned by digest, the environment of `conda-linux-64.lock`, fmeta, the test and lint tools), `scripts/in_docker.sh` (repository mounted, calling user, only the GPUs named), `scripts/update_lock.sh`, rule `.claude/rules/docker-execution.md`; `CLAUDE.md`, the GPU runbook and the README give Docker commands. Gate evidence is written only in the image whose lock is the repository's, and names the image, the lock and the GPU UUIDs. Roadmap: R7–R10 added (run modes, real inputs, presets, examples and benchmarks). The CI job `docker` and a publish workflow (tags and manual runs only) follow in the next commit.
- Why: owner rule of 2026-11-15 (execution and tests run on Docker, except CI). `environment.yml` pins nothing, so evidence taken in a conda environment does not describe what a later user gets.
- Refs: FM-D13; `docs/plans/2026-11-15-gpu-validation-modes-benchmarks.md`; a4b91eb, fb6bc95, 383b5ec; PR #1.

### 2026-11-15 — [amendment] FM-D11: fmeta names the ligand force field (`openff-2.2.1`, `gaff-2.11`)
- What: when the settings name no ligand force field, fmeta uses `openff-2.2.1` or `gaff-2.11`, with AM1-BCC charges. Before this change, in the official image (the first run with the real ligand packages): `pytest` 74 passed and `pytest -m slow` 6 passed, but `fmeta check --platform CPU --quick` gave 13 passed, 2 failed (`pdb_sdf_openff`, `pdb_sdf_gaff`). With this change: `pytest` 76 passed; `fmeta check --platform CPU --quick` 15 passed, 0 failed, 0 skipped.
- Why: the generators of `openmmforcefields` 0.16 have no default force field, so a PDB and an SDF alone failed; a default taken from the installed packages would change the parameters with the environment.
- Refs: FM-D11; `src/fmeta/system/ligand.py`; `tests/test_inputs.py` (new: `test_default_settings_work_with_the_real_generators`, which fails for both generators on the code before; the two tests that hide the ligand packages also hide the submodule it imports).
