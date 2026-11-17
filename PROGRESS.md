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
- Result: standard binding free energy by the published procedure: from the accumulated bias
  over the second half of a run, with a block error. The replicas together are the measurement:
  mean and 95 % interval, diagnostics per replica, a "no binding" statement, no convergence
  verdict.
- OpenMM only: no input from other MD programs.
- Owner request 2026-11-15 (roadmap R7–R10). In: replicas side by side on several GPUs or one
  after another on one GPU (R7); official presets (R9); real inputs (R8); a kinase and a GPCR
  example with results beside experiment (R3, R10); a kinase and a GPCR benchmark set, run and
  reported beside experiment and FEP+ (R10).
- Hardware: an 8-GPU machine (since 2026-11-15). Everything except CI runs in the official
  Docker image (FM-D13).

## Current Status

Snapshot date: 2026-11-17. Decisions: `docs/design/engine.md` (FM-D1..D18 Accepted;
FM-D2, D4, D5, D6, D7, D9, D10, D11, D12, D13, D14, D15, D16 and D17 amended since, see the Change Log). `main` is the work of PR #1 (merged; final code d4041f9; plan
`docs/plans/2026-11-15-gpu-validation-modes-benchmarks.md`: every run ended, every check this
machine can run has run, three review rounds fixed). In progress on branch
`claude/cupotential-backend` (draft PR #3): NVIDIA cuPotential as the default back end (owner
request 2026-11-17; plan `docs/plans/2026-11-17-cupotential-backend.md`, M0 to M4). M0 and M1
are done, NVIDIA MPS included; M2 is being measured; the owner: everything is developed in
that PR.

| Milestone | Status |
|---|---|
| M0 Scaffold | Complete |
| M1 CVs and funnel walls | Complete; equal to numpy references and finite differences |
| M2 Bias engine, HILLS / COLVAR, restart | Complete; table equal to `openmm.app.Metadynamics`, restart byte-identical |
| M3 Analysis and the solvable toy system | Complete; toy dG0 within 0.5 kJ/mol of exact (4 × 40 ns, plateau mean). Headline since 2026-11-16 (FM-D9, third amendment): the mean over the second half of the run of dG0(t) from the bias, with a block error; on the toy mean error +0.32 ± 0.15 and +0.41 ± 0.15 kJ/mol (wells of 20 and 40 kJ/mol, 8 × 20 ns each; cross-check from the samples after the first 2 ns: −0.03 ± 0.15 and −0.11 ± 0.08), no well: 8 of 8 reported as no binding |
| M4 Automatic funnel, validation | Complete on synthetic pockets (direction ≤ 10°) |
| M5 System building, equilibration, workflow, CLI | Complete; PDB + SDF alone, with automatic clean-up |
| M6 Membrane | Complete on CPU: real `addMembrane` build (POPC, 108 lipids) and a short run; membrane-plane fit on synthetic helix bundles only |
| M7 Replicas, walkers, figures, `fmeta check` | Complete on CPU |
| Harness: CPU CI + GPU gates | Introduced 2026-11-15 (PR #1) |
| Official image, Docker-only execution (FM-D13) | In use on the GPU machine. The image built from 783ecff runs without the mounted source: `fmeta check --quick` on CUDA passed (17 items) and the 232 files and directories it wrote belong to the calling user (a check, not evidence; the same on 47e84f7). The same on 384538e (the code of c59ec99): 17 passed, 232 entries of the calling user, `fmeta` imported from the image. The same on 4f976cb, the final code with the review fixes: 17 passed, 234 entries of the calling user, `fmeta` imported from the image. The same on 7c731c5 (second review round) and on d4041f9, the final code with the fix of the third review round: 17 passed, 234 entries of the calling user each time; only documents and evidence changed after d4041f9. The CI job `docker` passed on 4f976cb, 5e03c02, 97bcd70 and d30d1e6; the publish workflow has never run |
| cuPotential plan M0: the image with the plugin (FM-D13, two amendments of 2026-11-17) | On the branch, not on `main`. Image 648480d3… of lock ee32ef34…: OpenMM 8.4.0, CUDA 13.4, `nv-openmm-plugin` 0.3.5, the MPS programs of driver 615.71.10; on this host (driver 595.99.02) a container loads the compatibility `libcuda` 615.71.10 and takes the MPS programs of the image. In it: `ruff check .` passes, `pytest` 193 passed; NVIDIA MPS works: 4 replicas at once on one GPU as clients of the daemon of `scripts/mps.sh`, with each back end (0.5 ns, exit 0); an MPS client has no OpenCL platform. In the image of the former lock 0b6c344f… (the same packages, without the MPS programs): `pytest -m slow` (7 passed), `openmm.testInstallation` (4 platforms); directories of OpenMM 8.6.1 load and run; a Context with the plugin steps on the rhombic dodecahedron and on a membrane box with the funnel forces and the bias; the parameters of the TYK2 ligand are identical to the old image (OpenFF and GAFF); GPU gates with stock OpenMM: all 18 passed on abd4548. In the image of the new lock: `pytest -m slow` 7 passed, and the GPU gates with stock OpenMM: all 18 passed on 565b359 |
| cuPotential plan M1: the back end as an option (FM-D18) | On the branch. `run.backend` (`openmm` by default, `cupotential`, `auto`), `src/fmeta/engine/backend.py`; a run keeps its back end, recorded in `run.yaml` and `device.json`; one gate record per back end, the one of cuPotential without speeds. With `run.backend: cupotential` (force and integrator of the plugin) the soluble test system (24062 atoms, 2 replicas on 2 GPUs) and the membrane test system (37401 atoms, built in the old image) ran, went on, were stopped by `docker stop` and went on to 1.5 ns (1503 lines in every HILLS); `--backend openmm` on them is refused (exit 1, no file changed). `ruff check .` passes, `pytest` 193 passed (the plugin is not used by the default tests). Not run on this code with the plugin: the GPU gates (they run on the final code, for both back ends). M2: in progress, on the TYK2 and A2A examples built in the new image (88479 and 221068 atoms); M3, M4: not started |
| GPU plan M0: first gate record (R1) | Complete: all 16 gates on 2c4ad4e; the fmeta inside the image runs without the mounted source (quick check on CUDA, not evidence) |
| GPU plan M1: run modes and the scheduler (R7, FM-D14) | Complete: all 18 gates on 8d1f10a; `docker stop` and continuation checked on the trypsin run |
| GPU plan M3: official presets and the configuration reference (R9, FM-D16) | Merged: `fmeta init --preset`, `preset:` in a configuration, `docs/configuration.md`. GPU gates passed on 47e84f7 |
| GPU plan M2: real inputs (R8, FM-D15) | Merged, both parts. Membrane part: `fmeta split --drop`, kept components are receptor, orientation warning, membrane box 8% above the box rule. Protein part: chain gaps capped (NME / ACE), caps and protonation states of the input kept, kept hetero in their own chain, `info.protein_input` in `prepared/report.json`. `pytest` 128 passed. GPU gates passed on 47e84f7; the real `addMembrane` test (`-m slow`) not rerun. On real files (GPU, 200 ps of free equilibration, zero wall energy): 3PTB, 4GIH, the prepared TYK2 protein of the Schrödinger set; protein model alone on both A2A files |
| GPU plan M5 infrastructure: `fmeta benchmark` (R10, FM-D17) | Merged: `fmeta benchmark prepare`, `run`, `status`, `report`; sets `examples/benchmarks/tyk2` (10 ligands) and `a2a` (8 ligands) with experimental and FEP+ values; reports to `bench/reports/` without a verdict, checked by `scripts/verify_bench_results.py`. Tested on a synthetic set with a local source. On the merged code both real sets download and build: TYK2 pilot 86–91k atoms (eeb5302), caps and histidine states of the FEP+ input kept; A2A pilot 116–117k atoms (c41c08b; 228 POPC, box 12.74 × 6.69 × 13.9–14.1 nm), superposed at 0.112 nm, HIS 264 / 278 doubly protonated as in the file, gap capped, sodium kept (set READMEs). The pilot runs and the whole sets are under M7. GPU gates passed on 47e84f7 |
| GPU plan M4: trypsin-benzamidine (R2), the run length and the scan of the bias | Defaults from the scan by its rule (FM-D16): the setting `sig2` (`metad.sigma_s` 0.05 and `sigma_rho` 0.1 nm; bias factor 10 and height 1.5 kJ/mol stay) and `run.length_ns` 40, with 2 of 5 replicas passing the convergence check of that time. Read without a verdict, with the headline of the third amendment of FM-D9: dG0 −28.3 kJ/mol, 95% interval −66.9 to 10.4, replicas −78.2 to −4.6 (experiment −26.8 to −30.5, not used). The six settings of the scan give −14.9 to −29.1 kJ/mol at 40 ns with this headline; with the estimate from the samples, the headline when the scan was read, −2.9 to −28.0: the value depends on the bias setting, and two mean-force estimators over the pooled replicas did not reduce that (spread 25.1 kJ/mol against 32.8 and 42.0; FM-D16, third amendment). R2 is reworded: the value with its interval. The example ran again by its README steps with the defaults, one replica after another on one GPU (47e84f7, 5 × 40 ns, exit 0, `docker stop` and the same command twice): dG0 −44.04 kJ/mol, 95% interval −80.51 to −7.57, replicas −85.3 to −10.9; the ten replicas of both runs −36.16 (95% −57.42 to −14.90). R2 is closed as reworded |
| GPU plan M5: kinase example (R10) | TYK2 with ejm_46 from PDB entry 4GIH (`examples/tyk2_ejm46`, 88476 atoms). Run from its README with the defaults (5 × 40 ns, 47e84f7, the five replicas on one GPU with MPS), headline of the third amendment of FM-D9: dG0 −14.85 kJ/mol (−3.55 kcal/mol), 95% interval −26.60 to −3.11, replicas −29.00 to −3.97, rebinding events 5, 0, 1, 1, 1; experiment −47.3 kJ/mol, outside the interval. Reported as measured in the README |
| GPU plan M6: GPCR example (R3) | `examples/a2a_theophylline` (OPM file of 5MZJ): builds without a workaround (221110 atoms, gap LEU208–GLU219 capped, sodium ion kept, 4 disulfide bonds, funnel into the extracellular water, no error in the validation); equilibrated with the 2 ns free stage; membrane production works (5 replicas on one GPU with MPS, about 506 ns/day in total). That run (former default bias) was stopped at about 11 ns when the defaults changed. Run again from its README with the defaults (5 × 40 ns, 47e84f7, 221101 atoms, the five replicas on one GPU with MPS at 106–107 ns/day each, exit 0, one `docker stop` and the same command): dG0 −18.64 kJ/mol (−4.45 kcal/mol), 95% interval −33.31 to −3.97, replicas −33.60 to −4.60, rebinding events 2, 7, 1, 3, 9; the measurement on the simulated construct is −33.0 kJ/mol, all measurements −33.0 to −27.0, inside the interval at its lower end. The site is closed for a 0.3 nm probe in the crystal structure; theophylline left and came back in every replica. R3 is closed. Reported as measured in the README |
| GPU plan M7: benchmark runs (R10) | Pilots (3 ligands per set, 5 × 20 ns, former default bias, commits eeb5302 and f2ea653): the TYK2 pilot is complete (15 runs) and its report is committed as a pilot (`bench/reports/2026-11-16-45055cc99-tyk2`: dG0 −18.96, −11.14 and −23.71 kJ/mol for ejm_44, ejm_31 and ejm_46 against −31.05, −39.92 and −47.32 from experiment; 95% intervals 19 to 62 kJ/mol wide); the A2A pilot is complete (15 runs) and its report is committed as a pilot (`bench/reports/2026-11-16-80bf2209b-a2a`: dG0 −52.81, −65.99 and −90.40 kJ/mol for 4q, 4d and 4e D against −37.18, −43.95 and −50.52 from experiment; 95% intervals 29 to 76 kJ/mol wide; 9 of the 15 replicas left the site once in the 20 ns, 1 of them came back, 6 never left). Both pilot reports hold the headline of the first amendment of FM-D9 (from the samples; `bench/reports/README.md`). The whole sets ran with the defaults from 47e84f7 in fresh work directories (TYK2 10 ligands, A2A 8 ligands, 5 × 40 ns, 90 replicas, every one to its end, both exit 0); reports by the code of 7386d28: `bench/reports/2026-11-16-7386d2881-tyk2` and `-a2a`. Agreement with experiment in kcal/mol, fmeta / FEP+. TYK2: MUE 3.60 / 0.37, offset removed 1.81 / 0.35, Pearson r 0.15 / 0.96; every ligand above experiment (+1.0 to +6.9); experiment inside the 95% interval for 5 of 10; 47 of 50 replicas with a return to the site. A2A: MUE 4.34 / 0.74, offset removed 3.49 / 0.79, Pearson r 0.57 / 0.79; 6 of 8 below experiment (4d −12.0, 4e D −9.3); experiment inside the interval for 7 of 8; 25 of 40 replicas without a return. Intervals 4.3 to 16.4 (TYK2) and 7.0 to 28.9 kcal/mol (A2A) wide against experimental spans of 4.3 and 3.2. R10 is closed. Reported as measured (set READMEs, `README.md`) |

- GPU gates: **all 18 passed** (`gpu-tests` and the 17 items of `fmeta check`) on clean 565b359,
  the branch: stock OpenMM 8.4.0 (back end `openmm`) in the image of the present lock
  ee32ef34… (CUDA 13.4, mixed precision, 2 GPUs, host driver 595.99.02 with the compatibility
  `libcuda` 615.71.10; dated in UTC): `bench/results/2026-11-17-565b3592d-gpu-gates.json`.
  No gate record of the cuPotential back end yet (it is taken on the final code).
  `toy_free_energy`: from the samples −4.76, from the bias −4.34, against the exact −4.95
  kJ/mol, criteria 2.0 and 3.0. Speed (1481 ns/day plain, 1406 with the bias, 932 membrane)
  decides nothing. Before it: abd4548 (lock 0b6c344f…, the same packages without the MPS
  programs); on `main` (lock d061e45a…, OpenMM 8.6.1): d4041f9.
- Review of PR #1: three rounds, 9 findings, each fixed in its own commit (6fa7cd2, a4b09b1,
  5792001, 3e652e0, ab8f906, 4f976cb; 97bcd70, 7c731c5; d4041f9; FM-D5, FM-D9, FM-D10, FM-D11,
  FM-D15). All 9 threads answered and resolved.
- Function checks on real systems (code of 783ecff, GPU, short runs whose values mean nothing):
  every function of PR #1 that the gates do not reach has run once; table in `README.md`
  ("機能ごとの動作確認"). One defect found and fixed in c59ec99.
- Other checks, in the image (lock d061e45a…): `ruff check .` passes and `pytest` 175 passed on
  d4041f9; the two slow walker tests (`-m slow tests/test_walkers.py`) last ran on 4f976cb.
  Operation on a CPU platform is not guaranteed (owner, 2026-11-16, FM-D12): `pytest -m slow`
  (6 passed) and `fmeta check --platform CPU` last ran on the source of 2c4ad4e.
- CI: green on d4041f9, the final code of `main`, and on f4a1423 of PR #3 (run 44016334155).
  **Red on cc789b3 and 565b359 of PR #3** (runs 44029977738 and 44033031759, read with `gh` on
  2026-11-17): in the `docker` job `test_toy_binding_free_energy_end_to_end` (20 ns, seed 12)
  gives −5.955 kJ/mol from the samples, 1.0015 from the exact value against its tolerance of
  1.0; both `cpu` jobs pass. The same code and packages give −5.295 on the GPU machine: the
  trajectory of the fixed seed depends on the CPU. Not fixed in this commit.
- Not verified yet:
  - a binding free energy of a real protein-ligand system with a narrow interval (trypsin with
    the defaults: 95% intervals −66.9 to 10.4 and −80.5 to −7.6 kJ/mol in two runs, and the
    value depends on the bias setting; TYK2 with ejm_46: −26.6 to −3.1, 32 kJ/mol above
    experiment; A2A with theophylline: −33.3 to −4.0, 14 kJ/mol above the measurement on the
    construct; the 18 ligands of the benchmark sets: intervals 18 to 121 kJ/mol wide);
  - settings chosen for accuracy (walkers in the defaults, run length, bias): deferred by the
    owner until everything is shown to work; the accuracy of walkers on trypsin is not measured;
  - reading HILLS with PLUMED itself; the publish workflow (never run).
- Known gaps: ligand protonation is taken from the SDF as drawn (R5); unresolved loops are not
  modelled; non-protein components are removed unless listed in `system.keep_hetero`; a
  directory prepared before 2026-11-16 keeps periodic funnel walls in `prepared/system.xml` and
  must be prepared again; the gate `toy_free_energy` had 0.04 kJ/mol to spare with the former
  reweighting (about 1 run in 30 of its 10 ns missed the criterion on every platform, FM-D12);
  with the estimate from the samples its error is 0.19 kJ/mol for the gate's seed in the
  present image (1.00 in the image of `main`), and the scatter of that estimate over seeds at
  10 ns is not measured.
- GPUs (owner, 2026-11-16: all 8 are for this task): the measurement of M2 of the cuPotential
  plan runs on them (short runs, the MPS daemon is up). Speed of stock OpenMM in the image of
  `main` (second wave, 47e84f7, 2026-11-16):
  88k atoms 606 ns/day alone, 1283 / 1541 / 1620 per GPU with 4 / 6 / 8 replicas under MPS;
  the A2A set about 1015 per GPU with 6, the A2A example about 506 with 5.
- Blockers: CI of PR #3 is red (the toy test above). NVIDIA MPS works on this host (driver 595.99.02, below the R615 of CUDA 13.4)
  with the MPS programs of the image (FM-D13, second amendment of 2026-11-17); the host driver
  is not raised (owner, 2026-11-17). With stock OpenMM in the new image, 4 replicas on one GPU
  under MPS: 925 to 934 ns/day each (soluble test system, 24062 atoms).

## Change Log

### 2026-11-17 — [progress] GPU gates and the slow tests pass in the image with the MPS programs, on stock OpenMM (565b359); CI is red on a toy test
- What: `scripts/gpu_gates.py` on clean 565b359 (2 GPUs, image 648480d3… of lock ee32ef34…, host driver 595.99.02, `libcuda` 615.71.10 loaded, back end `openmm`, ended 06:46 UTC): all 18 passed. Forces against Reference: CV 3.8e-07 nm, energy 7.2e-06, force 7.1e-06; `periodic_images`: CV 1.2e-06 nm, wall energy 1.4e-10 kJ/mol; `toy_free_energy` −4.76 (samples) and −4.34 (bias) against −4.95 kJ/mol, the values of abd4548. The membrane test system has 37422 atoms. Speed of stock OpenMM (a diagnostic): 1481 ns/day plain, 1406 with the bias, membrane 932 (abd4548: 1413, 1410, 967). `pytest -m slow` in the same image: 7 passed. This closes "Not run in the new image" of the entry below. CI is red on cc789b3 and 565b359 (`docker` job; both `cpu` jobs pass): `test_toy_binding_free_energy_end_to_end` (20 ns, seed 12) gives −5.955 kJ/mol from the samples, 1.0015 from the exact value against a tolerance of 1.0; the same code and packages give −5.295 on the GPU machine, so the trajectory of the fixed seed depends on the CPU. Not fixed in this commit.
- Why: FM-D13: after a change of the lock the tests and the gates are taken again in the new image (owner, 2026-11-17: if the MPS programs work, they go into the image and the gates are taken again).
- Refs: `bench/results/2026-11-17-565b3592d-gpu-gates.json`, `bench/results/index.json`, `README.md`; commit 565b359; CI runs 44029977738, 44033031759.

### 2026-11-17 — [amendment] FM-D13: the image holds the MPS programs of the driver of its compatibility libraries; NVIDIA MPS works on the R595 host
- What: `docker/nvidia.pins` names NVIDIA's `libnvidia-compute` 615.71.10 with its sha256, the lock holds it as a second comment line (`scripts/update_lock.sh --pins`: no new solve, the packages are unchanged; lock 0b6c344f… → ee32ef34…), and the build puts `nvidia-cuda-mps-control` and `nvidia-cuda-mps-server` into `/opt/fmeta/mps/bin` with the licence text and refuses a version other than that of the compatibility libraries. The entry of the image puts them first on `PATH` exactly when the loader resolves `libcuda.so.1` to the compatibility libraries, and hides NVIDIA's OpenCL library from an MPS client (under that daemon the OpenCL library of the host does not return, and `import openmm` waited for ever): such a client has no OpenCL platform. `scripts/mps.sh start` starts the daemon through the entry. Verified in image 648480d3… on this host (driver 595.99.02): `PATH` and platforms with a GPU, without one and as a client; 4 replicas at once on one GPU as MPS clients, with the cuPotential back end and with stock OpenMM (0.5 ns, 503 lines in every HILLS, exit 0; stock 925 to 934 ns/day per replica at 24062 atoms); `ruff`, `pytest` 193 passed. Not run in the new image: `pytest -m slow`, the GPU gates (they follow on this commit). No test covers MPS: CI has no GPU.
- Why: owner, 2026-11-17: the driver of the host is not raised for now, and the MPS programs of R615 may be tried in a container. Several replicas per GPU are where the throughput is (FM-D14); the daemon of the host with a client on the compatibility library, and a client on the library of the host, both fail (the FM-D13 entry below). This replaces the stop condition "the host driver must be raised" of the plan.
- Refs: FM-D13, second amendment of 2026-11-17 (`docs/design/engine.md`); `docker/nvidia.pins`, `docker/install_nvidia.sh`, `docker/entrypoint.sh`, `scripts/mps.sh`, `scripts/update_lock.sh`, `conda-linux-64.lock`, `Dockerfile`, `.claude/rules/docker-execution.md`, `docs/gpu-runbook.md`, `README.md`.

### 2026-11-17 — [design] FM-D18: `run.backend`, NVIDIA's cuPotential plugin as the back end of a run (an option; the default stays `openmm`)
- What: M1 of the cuPotential plan. `src/fmeta/engine/backend.py` replaces, in memory and just before the Context of a run is made, OpenMM's `NonbondedForce` and `LangevinMiddleIntegrator` by the classes of the plugin; the funnel forces, `prepared/system.xml`, the build, the equilibration and the analysis stay OpenMM's own. `cupotential` under a condition the plugin does not support (platform, precision, PME, tolerance 5e-4, automatic mesh, box of 3 cutoffs, the plugin loads) is an error before a file is touched; `auto` gives OpenMM there with the reason on record. A run keeps the back end it began with (`run.yaml`, `device.json`, `fmeta status`); another one by name is refused before a device or a file is touched, with walkers before a rollback. The hash of a benchmark configuration with `openmm` is the one of the reports of 7386d28. Gate records: one per back end; the one of cuPotential holds no speed and no duration, and CI rejects one that does. Verified: `ruff`, `pytest` 193 passed, `verify_bench_results.py` (9 records); on a GPU both test systems with `run.backend: cupotential`: run, continue, `docker stop`, continue to 1.5 ns, refusal of `--backend openmm`. Not run: `pytest -m slow`, GPU gates on this code, CI on it.
- Why: owner request of 2026-11-17 (cuPotential as the default). An option first: which parts of the plugin are used and how accurate they are is measured before the default changes (M2, M3). The licence of the plugin forbids publishing benchmark results without NVIDIA's permission, hence the record without speeds.
- Refs: FM-D18 (`docs/design/engine.md`), `docs/plans/2026-11-17-cupotential-backend.md`; `src/fmeta/engine/backend.py`, `src/fmeta/workflow.py`, `src/fmeta/engine/runner.py`, `src/fmeta/check.py`, `src/fmeta/benchmark.py`, `scripts/gpu_gates.py`, `scripts/verify_bench_results.py`, `docs/configuration.md`, `tests/test_backend.py`, `tests/test_walkers.py`, `tests/test_benchmark.py`, `tests/test_harness.py`.

### 2026-11-17 — [progress] GPU gates pass in the image with the cuPotential plugin, on stock OpenMM 8.4.0 (abd4548)
- What: `scripts/gpu_gates.py` on clean abd4548 (2 GPUs, in the image of lock 0b6c344f…, host driver 595.99.02, `libcuda` 615.71.10 loaded, 04:10 to 04:15 UTC): all 18 passed. No item uses the plugin: this is the record of stock OpenMM 8.4.0 on CUDA 13.4 that the change of the lock requires. Forces against Reference: CV 3.8e-07 nm, energy 7.2e-06, force 7.1e-06; `periodic_images`: CV 1.2e-06 nm, wall energy 5.6e-04 kJ/mol; `toy_free_energy` −4.76 (samples) and −4.34 (bias) against −4.95 kJ/mol (d4041f9 in the old image: −3.95 and −3.46). The membrane test system has 37419 atoms (old image: 37401). Speed of stock OpenMM (a diagnostic): 1413 ns/day plain, 1410 with the bias, membrane 967 (d4041f9, old image: 1284, 1295, 927). The record names the driver, the `libcuda`, the CUDA runtime (13.4) and the plugin (0.3.5). Not run: CI on the new lock (it runs for the pull request).
- Why: the change gate and FM-D13: after a change of the lock the tests and the gates are taken again in the new image. With this, M0 of the cuPotential plan is complete except for NVIDIA MPS (Blockers).
- Refs: `bench/results/2026-11-17-abd4548c9-gpu-gates.json`, `bench/results/index.json`, `README.md`; commit abd4548.

### 2026-11-17 — [amendment] FM-D13: the image holds NVIDIA's cuPotential plugin, on OpenMM 8.4.0 and CUDA 13.4; NVIDIA MPS does not work with it on the R595 host
- What: M0 of the cuPotential plan, on branch `claude/cupotential-backend`; no code under `src/fmeta/` changed. `environment.yml`: `openmm=8.4.*`, `cuda-version=13.4`, `nv-openmm-plugin=0.3.5`; `docker/nvidia.pins` names the release of the plugin and the `.deb` of the CUDA compatibility libraries (615.71.10) with their sha256, and the lock names both (the plugin as a package line, the libraries as a comment line that `docker/install_nvidia.sh` reads), so the hash of the lock still decides what an official image is. NVIDIA's licence texts are in the image (`/opt/fmeta/licenses`). A gate record names the host driver, the loaded `libcuda`, the CUDA runtime and the plugin. With OpenMM the ligand tools went from `openmmforcefields` 0.16.0 and AmberTools 26.0 to 0.15.1 and 24.8: the parameters of the TYK2 ligand ejm_46 are identical in both images (charges, sigma, epsilon: largest difference 0; bonded terms equal; OpenFF and GAFF). In the new image (f7d97cf1…, lock 0b6c344f…, host driver 595.99.02, `libcuda` 615.71.10 loaded): `ruff check .` passes, `pytest` 175 passed, `pytest -m slow` 7 passed, `openmm.testInstallation` computes forces on 4 platforms, directories of OpenMM 8.6.1 load and run, and a Context with the plugin steps on the rhombic dodecahedron (24062 atoms) and on a membrane box (37401 atoms) with the funnel forces and the bias, energies finite, 1.3e-5 and 1.1e-5 (relative) from stock PME at the starting state. Failed: a container as an MPS client (`cuInit` does not return with the compatibility `libcuda`; with the host's `libcuda` the kernels of CUDA 13.4 do not load, `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`). Not run: GPU gates in the new image; CI (`docker` job) on the new lock.
- Why: owner request of 2026-11-17 (cuPotential as the default back end); the plugin is built only for OpenMM 8.4, CUDA 13.4 and Python 3.12, and one image stays the rule. Speeds measured with the plugin are kept out of tracked files (owner, 2026-11-17: the licence forbids their disclosure without NVIDIA's permission). The MPS failure is the first stop condition of the plan: several replicas per GPU gave the best throughput so far (1620 against 606 ns/day at 88k atoms, old image).
- Refs: FM-D13 amendment (`docs/design/engine.md`), `docs/plans/2026-11-17-cupotential-backend.md`; `environment.yml`, `docker/nvidia.pins`, `docker/install_nvidia.sh`, `scripts/update_lock.sh`, `conda-linux-64.lock`, `Dockerfile`, `scripts/gpu_gates.py`, `tests/test_harness.py`, `.claude/rules/docker-execution.md`, `docs/gpu-runbook.md`, `README.md`.
