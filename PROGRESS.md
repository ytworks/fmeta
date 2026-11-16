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

Snapshot date: 2026-11-16. Decisions: `docs/design/engine.md` (FM-D1..D17 Accepted;
FM-D2, D4, D5, D6, D7, D9, D11, D12, D13, D14, D16 and D17 amended since, see the Change Log). Everything below is on PR #1 (not merged).
Plan: `docs/plans/2026-11-15-gpu-validation-modes-benchmarks.md`, with the run lengths, the
scan and the GPU use as changed by the Change Log entries of 2026-11-16. Every run of it has
ended and every check of it that this machine can run has run. Open: the result of CI.

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
| Official image, Docker-only execution (FM-D13) | In use on the GPU machine. The image built from 783ecff runs without the mounted source: `fmeta check --quick` on CUDA passed (17 items) and the 232 files and directories it wrote belong to the calling user (a check, not evidence; the same on 47e84f7). The same on 384538e, the final code: 17 passed, 232 entries of the calling user, `fmeta` imported from the image; only `README.md` and `PROGRESS.md` changed after it. The result of the CI job `docker` is not checked here; the publish workflow has never run |
| GPU plan M0: first gate record (R1) | Complete: all 16 gates on 2c4ad4e; the fmeta inside the image runs without the mounted source (quick check on CUDA, not evidence) |
| GPU plan M1: run modes and the scheduler (R7, FM-D14) | Complete: all 18 gates on 8d1f10a; `docker stop` and continuation checked on the trypsin run |
| GPU plan M3: official presets and the configuration reference (R9, FM-D16) | Merged: `fmeta init --preset`, `preset:` in a configuration, `docs/configuration.md`. GPU gates passed on 47e84f7 |
| GPU plan M2: real inputs (R8, FM-D15) | Merged, both parts. Membrane part: `fmeta split --drop`, kept components are receptor, orientation warning, membrane box 8% above the box rule. Protein part: chain gaps capped (NME / ACE), caps and protonation states of the input kept, kept hetero in their own chain, `info.protein_input` in `prepared/report.json`. `pytest` 128 passed. GPU gates passed on 47e84f7; the real `addMembrane` test (`-m slow`) not rerun. On real files (GPU, 200 ps of free equilibration, zero wall energy): 3PTB, 4GIH, the prepared TYK2 protein of the Schrödinger set; protein model alone on both A2A files |
| GPU plan M5 infrastructure: `fmeta benchmark` (R10, FM-D17) | Merged: `fmeta benchmark prepare`, `run`, `status`, `report`; sets `examples/benchmarks/tyk2` (10 ligands) and `a2a` (8 ligands) with experimental and FEP+ values; reports to `bench/reports/` without a verdict, checked by `scripts/verify_bench_results.py`. Tested on a synthetic set with a local source. On the merged code both real sets download and build: TYK2 pilot 86–91k atoms (eeb5302), caps and histidine states of the FEP+ input kept; A2A pilot 116–117k atoms (c41c08b; 228 POPC, box 12.74 × 6.69 × 13.9–14.1 nm), superposed at 0.112 nm, HIS 264 / 278 doubly protonated as in the file, gap capped, sodium kept (set READMEs). The pilot runs and the whole sets are under M7. GPU gates passed on 47e84f7 |
| GPU plan M4: trypsin-benzamidine (R2), the run length and the scan of the bias | Defaults from the scan by its rule (FM-D16): the setting `sig2` (`metad.sigma_s` 0.05 and `sigma_rho` 0.1 nm; bias factor 10 and height 1.5 kJ/mol stay) and `run.length_ns` 40, with 2 of 5 replicas passing the convergence check of that time. Read without a verdict, with the headline of the third amendment of FM-D9: dG0 −28.3 kJ/mol, 95% interval −66.9 to 10.4, replicas −78.2 to −4.6 (experiment −26.8 to −30.5, not used). The six settings of the scan give −14.9 to −29.1 kJ/mol at 40 ns with this headline; with the estimate from the samples, the headline when the scan was read, −2.9 to −28.0: the value depends on the bias setting, and two mean-force estimators over the pooled replicas did not reduce that (spread 25.1 kJ/mol against 32.8 and 42.0; FM-D16, third amendment). R2 is reworded: the value with its interval. The example ran again by its README steps with the defaults, one replica after another on one GPU (47e84f7, 5 × 40 ns, exit 0, `docker stop` and the same command twice): dG0 −44.04 kJ/mol, 95% interval −80.51 to −7.57, replicas −85.3 to −10.9; the ten replicas of both runs −36.16 (95% −57.42 to −14.90). R2 is closed as reworded |
| GPU plan M5: kinase example (R10) | TYK2 with ejm_46 from PDB entry 4GIH (`examples/tyk2_ejm46`, 88476 atoms). Run from its README with the defaults (5 × 40 ns, 47e84f7, the five replicas on one GPU with MPS), headline of the third amendment of FM-D9: dG0 −14.85 kJ/mol (−3.55 kcal/mol), 95% interval −26.60 to −3.11, replicas −29.00 to −3.97, rebinding events 5, 0, 1, 1, 1; experiment −47.3 kJ/mol, outside the interval. Reported as measured in the README |
| GPU plan M6: GPCR example (R3) | `examples/a2a_theophylline` (OPM file of 5MZJ): builds without a workaround (221110 atoms, gap LEU208–GLU219 capped, sodium ion kept, 4 disulfide bonds, funnel into the extracellular water, no error in the validation); equilibrated with the 2 ns free stage; membrane production works (5 replicas on one GPU with MPS, about 506 ns/day in total). That run (former default bias) was stopped at about 11 ns when the defaults changed. Run again from its README with the defaults (5 × 40 ns, 47e84f7, 221101 atoms, the five replicas on one GPU with MPS at 106–107 ns/day each, exit 0, one `docker stop` and the same command): dG0 −18.64 kJ/mol (−4.45 kcal/mol), 95% interval −33.31 to −3.97, replicas −33.60 to −4.60, rebinding events 2, 7, 1, 3, 9; the measurement on the simulated construct is −33.0 kJ/mol, all measurements −33.0 to −27.0, inside the interval at its lower end. The site is closed for a 0.3 nm probe in the crystal structure; theophylline left and came back in every replica. R3 is closed. Reported as measured in the README |
| GPU plan M7: benchmark runs (R10) | Pilots (3 ligands per set, 5 × 20 ns, former default bias, commits eeb5302 and f2ea653): the TYK2 pilot is complete (15 runs) and its report is committed as a pilot (`bench/reports/2026-11-16-45055cc99-tyk2`: dG0 −18.96, −11.14 and −23.71 kJ/mol for ejm_44, ejm_31 and ejm_46 against −31.05, −39.92 and −47.32 from experiment; 95% intervals 19 to 62 kJ/mol wide); the A2A pilot is complete (15 runs) and its report is committed as a pilot (`bench/reports/2026-11-16-80bf2209b-a2a`: dG0 −52.81, −65.99 and −90.40 kJ/mol for 4q, 4d and 4e D against −37.18, −43.95 and −50.52 from experiment; 95% intervals 29 to 76 kJ/mol wide; 9 of the 15 replicas left the site once in the 20 ns, 1 of them came back, 6 never left). Both pilot reports hold the headline of the first amendment of FM-D9 (from the samples; `bench/reports/README.md`). The whole sets ran with the defaults from 47e84f7 in fresh work directories (TYK2 10 ligands, A2A 8 ligands, 5 × 40 ns, 90 replicas, every one to its end, both exit 0); reports by the code of 7386d28: `bench/reports/2026-11-16-7386d2881-tyk2` and `-a2a`. Agreement with experiment in kcal/mol, fmeta / FEP+. TYK2: MUE 3.60 / 0.37, offset removed 1.81 / 0.35, Pearson r 0.15 / 0.96; every ligand above experiment (+1.0 to +6.9); experiment inside the 95% interval for 5 of 10; 47 of 50 replicas with a return to the site. A2A: MUE 4.34 / 0.74, offset removed 3.49 / 0.79, Pearson r 0.57 / 0.79; 6 of 8 below experiment (4d −12.0, 4e D −9.3); experiment inside the interval for 7 of 8; 25 of 40 replicas without a return. Intervals 4.3 to 16.4 (TYK2) and 7.0 to 28.9 kcal/mol (A2A) wide against experimental spans of 4.3 and 3.2. R10 is closed. Reported as measured (set READMEs, `README.md`) |

- GPU gates: **all 18 passed** (`gpu-tests` and the 17 items of `fmeta check`) on clean 7386d28,
  the final code: it is the code of c59ec99 (the fix of the cross-check from the samples, which
  `toy_free_energy` reads); only documents changed after it (CUDA, mixed precision, 2 GPUs, in
  the image): `bench/results/2026-11-16-7386d2881-gpu-gates.json` (dated in UTC).
  `toy_free_energy`: from the samples −3.95, from the bias −3.46, against the exact −4.95 kJ/mol,
  criteria 2.0 and 3.0. `replicas_sequential` and `replicas_parallel` ran real child processes
  on the GPUs (FM-D14). Speed values (1351 ns/day plain, 1304 with the bias, 925 membrane)
  decide nothing. The record before it: d4bbcc6, the code before that fix.
- Function checks on real systems (code of 783ecff, GPU, short runs whose values mean nothing):
  every function of this development that the gates do not reach has run once: walkers on
  trypsin with `docker stop` and the same command, extension of a finished run, `fmeta auto`
  from a YAML, `fmeta init`, the input routes (`--complex`, two files, GAFF, `--membrane`),
  `fmeta split --drop`, `fmeta doctor`, `fmeta analyze --plot` / `--discard-ns`,
  `fmeta benchmark status` / `report` with the current analysis. Table: `README.md`
  ("機能ごとの動作確認"). The sequential trypsin run has ended. Its analysis showed one defect,
  fixed in c59ec99: the cross-check from the samples was missing for a replica whose weights
  span more than the digits of a float (the headline from the bias is not affected).
- Other checks, in the image (lock d061e45a…): `ruff check .` passes and `pytest` 161 passed on
  c59ec99. The image check without the mounted source passed on 384538e, the same code (the
  image row above). Operation on a CPU platform is not guaranteed (owner, 2026-11-16,
  FM-D12): `pytest -m slow` (6 passed) and `fmeta check --platform CPU` were last run on the
  source of 2c4ad4e and are not rerun.
- CI: green on 171220a. The commits after it were pushed on 2026-11-16; the result of CI on
  them, with the first run of the job `docker`, is not checked here.
- Not verified yet:
  - a binding free energy of a real protein-ligand system with a narrow interval (trypsin with
    the defaults: 95% intervals −66.9 to 10.4 and −80.5 to −7.6 kJ/mol in two runs, and the
    value depends on the bias setting;
    TYK2 with ejm_46: −26.6 to −3.1 kJ/mol, 32 kJ/mol above experiment; A2A with
    theophylline: −33.3 to −4.0 kJ/mol, 14 kJ/mol above the measurement on the construct;
    the 18 ligands of the benchmark sets: intervals 18 to 121 kJ/mol wide);
  - settings chosen for accuracy (walkers in the defaults, run length, bias): deferred by the
    owner until everything is shown to work; walkers operate on trypsin, their accuracy is not
    measured;
  - reading HILLS with PLUMED itself;
  - the publish workflow (never run).
- Known gaps: ligand protonation is taken from the SDF as drawn (R5); unresolved loops are not
  modelled; non-protein components are removed unless listed in `system.keep_hetero`; a
  directory prepared before 2026-11-16 keeps periodic funnel walls in `prepared/system.xml` and
  must be prepared again; the gate `toy_free_energy` had 0.04 kJ/mol to spare with
  the former reweighting, where about 1 run in 30 of its 10 ns missed the criterion on every
  platform (FM-D12); with the estimate from the samples (FM-D9, first amendment) its error is
  1.00 kJ/mol for the gate's seed, and the scatter of that estimate over seeds at 10 ns is not
  measured.
- GPUs (owner, 2026-11-16: all 8 are for this task): no run is in progress, the GPUs are free
  and the MPS daemon is stopped. The second wave ran from 47e84f7 with the defaults, every GPU
  with several replicas under NVIDIA MPS (`run.per_gpu`, FM-D14 amendment), from 06:41 to
  18:54 UTC on 2026-11-16: the three examples (ended at 10:08, 13:43 and 16:11 UTC), the TYK2
  set (18:54 UTC) and the A2A set (18:51 UTC); 105 replicas of 40 ns, every command exit 0.
- Speed (measured: 88k atoms 606 ns/day alone and 1283 / 1541 / 1620 ns/day per GPU with
  4 / 6 / 8 replicas under MPS; A2A set system about 1015 ns/day per GPU with 6 replicas, A2A
  example about 506 with 5): both full sets and the three examples at 5 × 40 ns took 12 h
  13 min on 8 GPUs, with the function checks in between.
- Blockers: none. Pushes from the GPU machine work since 2026-11-16 (the `gh` login of the
  machine as the git credential; the owner added the `workflow` scope).

## Change Log

### 2026-11-16 — [progress] The image of the final code runs without the mounted source (384538e)
- What: the image built from clean 384538e (the code of c59ec99; documents and evidence since) passed `fmeta check --quick` on CUDA with an empty directory as `/work` (2 GPUs, 19:16 to 19:18 UTC): 17 passed, 0 failed, 0 skipped; the 232 files and directories it wrote belong to the calling user; `fmeta` is imported from the image (`/opt/conda/lib/python3.12/site-packages/fmeta`). A check, not gate evidence. This closes "the image check on the final commit" of the entries on the gates and on the benchmark sets. Every check of the plan that this machine can run has now run. Not verified here: the result of CI (`cpu`, `docker`); the publish workflow (started by the owner); `pytest -m slow` (owner: no CPU verification).
- Why: item 2 of the owner request (an official, runnable Docker image): the image that is distributed must work without the repository, on the code that the gates verified.
- Refs: `Dockerfile`, `conda-linux-64.lock` (sha256 d061e45a…), `README.md`; FM-D13.

### 2026-11-16 — [progress] Both benchmark sets ran to the end with the defaults; reports beside experiment and FEP+; R10 closed
- What: `fmeta benchmark run` ended with exit 0 for both sets on 47e84f7 (TYK2, 10 ligands in water, 06:41 to 18:54 UTC; A2A, 8 ligands in POPC, 06:48 to 18:51 UTC; 5 × 40 ns per ligand, 90 replicas, every one to its end; the TYK2 set was stopped at 12:58 UTC for the function checks and continued with the same command). `fmeta benchmark report` by the code of 7386d28 on a clean tree wrote `bench/reports/2026-11-16-7386d2881-tyk2` and `-a2a`. Agreement with experiment in kcal/mol, fmeta / FEP+. TYK2: MUE 3.60 / 0.37, RMSE 4.12 / 0.45, mean signed error +3.60 / +0.19, MUE with the offset removed 1.81 / 0.35, Pearson r 0.15 / 0.96. A2A: MUE 4.34 / 0.74, RMSE 5.79 / 0.98, mean signed error −3.74 / +0.23, offset removed 3.49 / 0.79, Pearson r 0.57 / 0.79. Experiment inside the 95% interval for 5 of 10 and 7 of 8 ligands; intervals 4.3 to 16.4 and 7.0 to 28.9 kcal/mol wide against experimental spans of 4.3 and 3.2: the order of the ligands is not resolved. Returns to the site: 47 of 50 TYK2 replicas (153 rebinding events), 15 of 40 A2A replicas (23 left once and stayed out, 2 did not reach the unbound region). Nothing was tuned for a set; no verdict. `scripts/verify_bench_results.py` passes. The MPS daemon is stopped. Roadmap R10 moved to done. Not verified here: the result of CI; the image check on the final commit.
- Why: item 5 of the owner request: benchmarks with experimental and FEP values, run and compared. The values are reported as measured; the settings for accuracy are deferred by the owner until everything is shown to work.
- Refs: `bench/reports/2026-11-16-7386d2881-tyk2.json`, `.md`, `bench/reports/2026-11-16-7386d2881-a2a.json`, `.md`, `bench/reports/index.json`, `bench/reports/README.md`, the set READMEs, `README.md`, `docs/roadmap.md` (R10); FM-D17, FM-D9.

### 2026-11-16 — [progress] GPU gates pass on the final code (7386d28, the code of c59ec99)
- What: `scripts/gpu_gates.py` on clean 7386d28 (2 GPUs, in the image, 18:52 to 18:57 UTC): all 18 passed. Under the gated paths only two example READMEs changed between c59ec99 and 7386d28. Forces against Reference: CV 3.8e-07 nm, energy 7.2e-06, force 7.1e-06; `periodic_images`: CV 8.5e-07 nm, wall energy 8.0e-05 kJ/mol; `toy_free_energy` −3.95 (samples) and −3.46 (bias) against −4.95 kJ/mol, the values of d4bbcc6: the fix of c59ec99 changes nothing where the weights stay within the digits of a float. Speed (a diagnostic): 1351 ns/day plain, 1304 with the bias (3.6%), membrane 925. This closes "the gates on the code with that fix are not run yet" of the entry on the trypsin run. Not verified here: the result of CI; the image check on this code.
- Why: the change gate. c59ec99 changed the analysis that `toy_free_energy` reads, and the plan runs the gates once on the final code.
- Refs: `bench/results/2026-11-16-7386d2881-gpu-gates.json`, `bench/results/index.json`, `README.md`.

### 2026-11-16 — [progress] A2A with theophylline by its README steps (membrane, 221101 atoms): dG0 −18.64 kJ/mol; R3 closed
- What: `examples/a2a_theophylline` ran from its README on 47e84f7 with the defaults (5 × 40 ns; the five replicas on one GPU with MPS, 106–107 ns/day each, 9 h 5 min, exit 0; one `docker stop` after 35 min and the same `fmeta run` command, every replica 40000 hills). Analysis with the code of c59ec99: dG0 −18.64 kJ/mol (−4.45 kcal/mol), 95% interval −33.31 to −3.97, replicas −27.40, −4.60, −33.60, −11.17, −16.41; rebinding events 2, 7, 1, 3, 9; first exit (s above 3.0 nm) at 0.8 to 1.9 ns. Measurement on the simulated construct −33.0 kJ/mol (SPR), all measurements −33.0 to −27.0: inside the interval at its lower end, the value 14.4 kJ/mol above. One replica (−4.60) is within 1 kT of the empty-funnel value (−2.18) and is noted as no binding for that replica; the set is not. Equilibrated box 12.57 × 13.19 × 12.99 nm, 0.685 nm² per lipid, zero wall energy. With this, the three examples have run from their READMEs to dG0. Roadmap R3 moved to done.
- Why: item 4 of the owner request (a GPCR example with an experimental value) and the open question of the preparation: the site is closed for a 0.3 nm probe in the crystal structure, and the ligand still left and came back along the funnel in every replica.
- Refs: `examples/a2a_theophylline/README.md`, `README.md`, `docs/roadmap.md` (R3, R10); FM-D9, FM-D15. Not committed: the run directory (`runs/`, ignored).

### 2026-11-16 — [progress] Trypsin by its README steps, one GPU, sequential: dG0 −44.04 kJ/mol; the cross-check from the samples lost bins of low weight (fixed, c59ec99)
- What: `examples/trypsin_benzamidine` ran from its README on 47e84f7 with `--mode sequential` on one GPU (5 × 40 ns, 703–725 ns/day, exit 0; `docker stop` and the same command twice, every replica 40000 hills): dG0 −44.04 kJ/mol (−10.53 kcal/mol), 95% interval −80.51 to −7.57, replicas −60.16, −39.03, −10.94, −85.27, −24.80; rebinding events 2, 3, 6, 3, 4. With the scan run of the same settings (−28.28): ten replicas −36.16, 95% −57.42 to −14.90; experiment −26.8 to −30.5. Roadmap R2 is closed as reworded. Defect found by this run: replica 3 had "no estimate from the samples" although 3837 samples lay in the unbound region. `reweighted_profile` summed the weights with `numpy.histogram` (differences of a cumulative sum), which returns 0 for a bin more than about 16 digits below the sum before it; the log-weights spanned 53.6. Each bin is now summed relative to its own largest weight; the new test failed before the fix. After it: replica 3 −75.35 from the samples, replica 0 of the scan run −87.04 (was none), every other number of both trypsin runs and of the TYK2 example unchanged. `ruff` passes, `pytest` 161 passed. Not run on this code: GPU gates (`toy_free_energy` reads this function) and the image check; both once more on the final code.
- Why: the headline of a replica comes from the bias and was not affected; the cross-check is a diagnostic that must not be missing for a numerical reason. Correction to the entry "Every function of this development has operated": "No defect was found; no code changed" held at that time only.
- Refs: c59ec99 (`src/fmeta/analysis/fes.py`, `tests/test_analysis.py`), `examples/trypsin_benzamidine/README.md`, `README.md`, `docs/roadmap.md` (R2).

### 2026-11-16 — [progress] Every function of this development has operated on the GPU: checks on real systems beside the gates (783ecff)
- What: the functions the gates do not reach ran once each on the code of 783ecff, in the image. Trypsin (3PTB): 4 shared-bias walkers × 1 ns on one GPU with MPS (424–433 ns/day each, 1000 hills each); `docker stop` at 550–560 hills and the same command: every walker rolled back to the checkpoint at 500, the first 500 HILLS lines identical to the copies at the stop; `fmeta auto` from a YAML with `run: {length_ns, replicas, mode, gpus}` (2 replicas side by side on 2 GPUs, exit 0); `fmeta run --length-ns 1.5` on that finished run (first 1003 HILLS lines identical, 1503 at the end); `fmeta analyze --plot` and `--discard-ns`; `fmeta prepare` by `--complex`, by two files without a YAML and with GAFF (72431, 72431, 72428 atoms). A2A: `--membrane POPC` on the command line (221123 atoms, no error in the validation); `fmeta split --drop A:1001-1106` on the RCSB entry 5MZJ (90 residues dropped). `fmeta init` for both presets; `fmeta doctor`; `fmeta benchmark status` and `report` with the current analysis on a copy of the running TYK2 set (written under the ignored `runs/`, not a report of the set); a directory prepared with another configuration is refused. The image built from 783ecff passed `fmeta check --quick` on CUDA without the mounted source (17 items; 232 files and directories, all of the calling user). No defect was found; no code changed. Not verified: `pytest -m slow` (owner: no CPU verification), the result of CI, the publish workflow; the fifth replica of the sequential trypsin run is in progress.
- Why: the owner's definition of 2026-11-16: "works" is the operation of every function of this development; the benchmark sets continue on the GPUs but do not decide it.
- Refs: `README.md` ("機能ごとの動作確認"), `bench/results/2026-11-16-d4bbcc67f-gpu-gates.json` (the gated functions).

### 2026-11-16 — [progress] GPU gates pass on the code with the published measurement procedure (d4bbcc6)
- What: owner clarification of 2026-11-16: "works" means every function of this development operates, not that every benchmark run ends. The function checks therefore come before the benchmark sets; the TYK2 set was stopped at 12:58 UTC (12 of 50 replicas finished, restartable from its checkpoints) and its GPUs went to the checks. First of them: `scripts/gpu_gates.py` on clean d4bbcc6 (2 GPUs, in the image): all 18 passed. Forces against Reference: CV 3.8e-07 nm, energy 7.2e-06, force 7.1e-06; `periodic_images`: CV 8.4e-07 nm, wall energy 2.5e-04 kJ/mol; `toy_free_energy` −3.95 and −3.46 against −4.95 kJ/mol. This closes "not run yet on the code after 47e84f7" (analysis without a verdict, one math-library thread per replica, headline from the bias). Not verified here: the result of CI.
- Why: the gates are the check of the gated functions (forces, bias, restart, walkers, both run modes, analysis, membrane build and run) on the current code; the functions they do not reach are checked on real systems next.
- Refs: `bench/results/2026-11-16-d4bbcc67f-gpu-gates.json`, `bench/results/index.json`, `README.md`.

### 2026-11-16 — [amendment] The measurement follows the published procedure: dG0 of a replica from the bias, with a block error (FM-D9, third amendment)
- What: owner instruction of 2026-11-16: measure by the formal method; that everything works comes before settings for accuracy. The headline `dG` of a replica is the mean over the second half of the run of dG0(t) from the accumulated bias (the first half is rejected); new key `dG_error`, its standard error from 5 blocks. The two estimates from the samples are cross-checks at the unchanged 4 kJ/mol; `dG_source` is gone. No threshold and no test or gate criterion changed. Toy, 8 × 20 ns: error +0.32 ± 0.15 (RMSE 0.52) and +0.41 ± 0.15 kJ/mol (RMSE 0.58) for wells of 20 and 40 kJ/mol, `dG_error` 0.34 and 0.35 against a scatter between seeds of 0.43; no well: 8 of 8 no binding. Same runs, new headline: trypsin −28.28 kJ/mol (95% −66.93 to 10.37; was −27.35), TYK2 with ejm_46 −14.85 (95% −26.60 to −3.11; was −10.05). `ruff` passes, `pytest` 160 passed. Not run: GPU gates on this code (once, on the final code), `pytest -m slow`. Correction: replica 0 of the trypsin run with the default bias left the site once, at 27.5 ns, and did not come back; "one never left the site" in the entry "Verdict of the bias scan" was wrong.
- Why: the owner's instruction. The published method (Limongelli, Bonomi, Parrinello 2013) reads the value from the surface that the bias builds, after a rejected first part, with a block error; the estimate from the samples is not part of it and had no value where the samples do not reach the unbound region (9 of 15 A2A pilot replicas), so one set held headlines of two kinds. Deferred until operation is shown: walkers in the defaults, run length, bias settings; no walkers test was started.
- Refs: FM-D9 (third amendment, `docs/design/engine.md`), `.claude/rules/numerics-invariants.md`, `src/fmeta/analysis/`, `src/fmeta/workflow.py`, `tests/test_analysis.py`, `README.md`, the trypsin and TYK2 example READMEs; supersedes the headline of the entry "dG0 from the samples after the first 2 ns".

### 2026-11-16 — [progress] Report of the A2A pilot committed (3 ligands, 5 × 20 ns, former hill widths)
- What: the A2A pilot finished (15 runs of 20 ns, membrane systems of 116–117k atoms, commits c41c08b and f2ea653) and `fmeta benchmark report` wrote `bench/reports/2026-11-16-80bf2209b-a2a` with the analysis of 80bf220: dG0 −52.81, −65.99 and −90.40 kJ/mol (−12.62, −15.77, −21.61 kcal/mol) for 4q, 4d and 4e D against −37.18, −43.95 and −50.52 from experiment; 95% intervals 29 to 76 kJ/mol wide; rebinding events 0, 1 and 0. In 20 ns 9 of the 15 replicas left the site once (4q 5, 4d 3, 4e D 1), 1 of them came back and 6 never left: each surface holds one passage out at most. Named as a pilot in `bench/reports/README.md`: not a result of the official defaults, nothing selected from it. `scripts/verify_bench_results.py` passes.
- Why: the pilots exist to exercise the membrane production path and `fmeta benchmark report` on real data before the whole sets end; both worked without a defect on this set.
- Refs: `bench/reports/2026-11-16-80bf2209b-a2a.json`, `.md`, `bench/reports/index.json`, `bench/reports/README.md`; roadmap R10; FM-D17.
