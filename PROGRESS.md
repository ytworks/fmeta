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
  after another on one GPU (R7); official presets (R9). Planned: real inputs (R8); a kinase and
  a GPCR example, benchmarks against experiment and FEP (R10).
- Hardware: an 8-GPU machine (since 2026-11-15). Everything except CI runs in the official
  Docker image (FM-D13).

## Current Status

Snapshot date: 2026-11-16. Decisions: `docs/design/engine.md` (FM-D1..D17 Accepted;
FM-D2, D4, D5, D6, D7, D9, D11, D12, D13, D14, D16 and D17 amended since, see the Change Log). Everything below is on PR #1 (not merged).
Plan in progress: `docs/plans/2026-11-15-gpu-validation-modes-benchmarks.md`, with the run
lengths, the scan and the GPU use as changed by the Change Log entries of 2026-11-16.

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
| Official image, Docker-only execution (FM-D13) | In use on the GPU machine. The image built from 783ecff runs without the mounted source: `fmeta check --quick` on CUDA passed (17 items) and the 232 files and directories it wrote belong to the calling user (a check, not evidence; the same on 47e84f7). The result of the CI job `docker` is not checked here; the publish workflow has never run |
| GPU plan M0: first gate record (R1) | Complete: all 16 gates on 2c4ad4e; the fmeta inside the image runs without the mounted source (quick check on CUDA, not evidence) |
| GPU plan M1: run modes and the scheduler (R7, FM-D14) | Complete: all 18 gates on 8d1f10a; `docker stop` and continuation checked on the trypsin run |
| GPU plan M3: official presets and the configuration reference (R9, FM-D16) | Merged: `fmeta init --preset`, `preset:` in a configuration, `docs/configuration.md`. GPU gates passed on 47e84f7 |
| GPU plan M2: real inputs (R8, FM-D15) | Merged, both parts. Membrane part: `fmeta split --drop`, kept components are receptor, orientation warning, membrane box 8% above the box rule. Protein part: chain gaps capped (NME / ACE), caps and protonation states of the input kept, kept hetero in their own chain, `info.protein_input` in `prepared/report.json`. `pytest` 128 passed. GPU gates passed on 47e84f7; the real `addMembrane` test (`-m slow`) not rerun. On real files (GPU, 200 ps of free equilibration, zero wall energy): 3PTB, 4GIH, the prepared TYK2 protein of the Schrödinger set; protein model alone on both A2A files |
| GPU plan M5 infrastructure: `fmeta benchmark` (R10, FM-D17) | Merged: `fmeta benchmark prepare`, `run`, `status`, `report`; sets `examples/benchmarks/tyk2` (10 ligands) and `a2a` (8 ligands) with experimental and FEP+ values; reports to `bench/reports/` without a verdict, checked by `scripts/verify_bench_results.py`. Tested on a synthetic set with a local source. On the merged code both real sets download and build: TYK2 pilot 86–91k atoms (eeb5302), caps and histidine states of the FEP+ input kept; A2A pilot 116–117k atoms (c41c08b; 228 POPC, box 12.74 × 6.69 × 13.9–14.1 nm), superposed at 0.112 nm, HIS 264 / 278 doubly protonated as in the file, gap capped, sodium kept (set READMEs). The pilot runs are under M7. GPU gates passed on 47e84f7. Not done: a finished run of a set |
| GPU plan M4: trypsin-benzamidine (R2), the run length and the scan of the bias | Defaults from the scan by its rule (FM-D16): the setting `sig2` (`metad.sigma_s` 0.05 and `sigma_rho` 0.1 nm; bias factor 10 and height 1.5 kJ/mol stay) and `run.length_ns` 40, with 2 of 5 replicas passing the convergence check of that time. Read without a verdict, with the headline of the third amendment of FM-D9: dG0 −28.3 kJ/mol, 95% interval −66.9 to 10.4, replicas −78.2 to −4.6 (experiment −26.8 to −30.5, not used). The six settings of the scan give −14.9 to −29.1 kJ/mol at 40 ns with this headline; with the estimate from the samples, the headline when the scan was read, −2.9 to −28.0: the value depends on the bias setting, and two mean-force estimators over the pooled replicas did not reduce that (spread 25.1 kJ/mol against 32.8 and 42.0; FM-D16, third amendment). R2 is reworded: the value with its interval. The example ran again by its README steps with the defaults, one replica after another on one GPU (47e84f7, 5 × 40 ns, exit 0, `docker stop` and the same command twice): dG0 −44.04 kJ/mol, 95% interval −80.51 to −7.57, replicas −85.3 to −10.9; the ten replicas of both runs −36.16 (95% −57.42 to −14.90). R2 is closed as reworded |
| GPU plan M5: kinase example (R10) | TYK2 with ejm_46 from PDB entry 4GIH (`examples/tyk2_ejm46`, 88476 atoms). Run from its README with the defaults (5 × 40 ns, 47e84f7, the five replicas on one GPU with MPS), headline of the third amendment of FM-D9: dG0 −14.85 kJ/mol (−3.55 kcal/mol), 95% interval −26.60 to −3.11, replicas −29.00 to −3.97, rebinding events 5, 0, 1, 1, 1; experiment −47.3 kJ/mol, outside the interval. Reported as measured in the README |
| GPU plan M6: GPCR example (R3) | `examples/a2a_theophylline` (OPM file of 5MZJ): builds without a workaround (221110 atoms, gap LEU208–GLU219 capped, sodium ion kept, 4 disulfide bonds, funnel into the extracellular water, no error in the validation); equilibrated with the 2 ns free stage; membrane production works (5 replicas on one GPU with MPS, about 506 ns/day in total). That run (former default bias) was stopped at about 11 ns when the defaults changed. It is run again from its README with the defaults of this commit (5 × 40 ns). No result yet |
| GPU plan M7: benchmark runs (R10) | Pilots (3 ligands per set, 5 × 20 ns, former default bias, commits eeb5302 and f2ea653): the TYK2 pilot is complete (15 runs) and its report is committed as a pilot (`bench/reports/2026-11-16-45055cc99-tyk2`: dG0 −18.96, −11.14 and −23.71 kJ/mol for ejm_44, ejm_31 and ejm_46 against −31.05, −39.92 and −47.32 from experiment; 95% intervals 19 to 62 kJ/mol wide); the A2A pilot is complete (15 runs) and its report is committed as a pilot (`bench/reports/2026-11-16-80bf2209b-a2a`: dG0 −52.81, −65.99 and −90.40 kJ/mol for 4q, 4d and 4e D against −37.18, −43.95 and −50.52 from experiment; 95% intervals 29 to 76 kJ/mol wide; 9 of the 15 replicas left the site once in the 20 ns, 1 of them came back, 6 never left). Both pilot reports hold the headline of the first amendment of FM-D9 (from the samples; `bench/reports/README.md`). The whole sets (TYK2 10 ligands, A2A 8 ligands, 5 × 40 ns) run from 47e84f7 in fresh work directories. No report of a whole set yet |

- GPU gates: **all 18 passed** (`gpu-tests` and the 17 items of `fmeta check`) on clean d4bbcc6,
  the code with the analysis of the third amendment of FM-D9, without a convergence verdict and
  with one math-library thread per replica (CUDA, mixed precision, 2 GPUs, in the image):
  `bench/results/2026-11-16-d4bbcc67f-gpu-gates.json` (dated in UTC). `toy_free_energy`: from the
  samples −3.95, from the bias −3.46, against the exact −4.95 kJ/mol, criteria 2.0 and 3.0.
  `replicas_sequential` and `replicas_parallel` ran real child processes on the GPUs (FM-D14).
  Speed values (1255 ns/day plain, 1219 with the bias, 759 membrane) were taken while the other
  six GPUs ran production; they decide nothing. The analysis changed after this record
  (c59ec99, the sum of the weighted histogram behind the estimate from the samples, which
  `toy_free_energy` reads): the gates on the code with that fix are **not run yet**; they run
  once more on the final code.
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
  c59ec99. The image check without the mounted source is not repeated on that code yet. Operation on a CPU platform is not guaranteed (owner, 2026-11-16,
  FM-D12): `pytest -m slow` (6 passed) and `fmeta check --platform CPU` were last run on the
  source of 2c4ad4e and are not rerun.
- CI: green on 171220a. The commits after it were pushed on 2026-11-16; the result of CI on
  them, with the first run of the job `docker`, is not checked here.
- Not verified yet:
  - a binding free energy of a real protein-ligand system with a narrow interval (trypsin with
    the defaults: 95% intervals −66.9 to 10.4 and −80.5 to −7.6 kJ/mol in two runs, and the
    value depends on the bias setting;
    TYK2 with ejm_46: −26.6 to −3.1 kJ/mol, 32 kJ/mol above experiment);
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
- GPUs (owner, 2026-11-16: all 8 are for this task): every GPU carries several replicas with
  NVIDIA MPS (`run.per_gpu`, FM-D14 amendment). Second wave, started at 06:41 UTC on 2026-11-16
  from 47e84f7 with the defaults and continued from its checkpoints at 07:41 UTC with one
  math-library thread per replica: the three examples on GPUs 2, 3 and 4 (trypsin one replica
  after another), the TYK2 set on GPUs 0 and 5, the A2A set on GPUs 6 and 7; the A2A pilot
  finishes on GPU 1. The TYK2 example ended at 10:08 UTC; its GPU 3 went to the TYK2 set.
  The A2A pilot ended at 12:04 UTC; its GPU 1 is with the A2A set. Since 13:17 UTC, after the
  function checks: the TYK2 set on GPUs 0, 3 and 5 with 13 replicas each (every remaining
  replica in flight; about 1600 ns/day per GPU), the A2A set on GPUs 1, 6 and 7. The trypsin
  run ended at 13:43 UTC; since 13:55 UTC the A2A set has GPUs 1, 2, 6 and 7 with 7 replicas
  each (its 28 remaining replicas in flight, about 150 ns/day each).
- Time estimate (measured: 88k atoms 606 ns/day alone and 1283 / 1541 / 1620 ns/day per GPU with
  4 / 6 / 8 replicas under MPS; A2A set system about 1015 ns/day per GPU with 6 replicas, A2A
  example about 506 with 5): both full sets and the three examples at 5 × 40 ns are about
  3.7 GPU-days, 11 to 13 hours on 8 GPUs.
- Blockers: none. Pushes from the GPU machine work since 2026-11-16 (the `gh` login of the
  machine as the git credential; the owner added the `workflow` scope).

## Change Log

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

### 2026-11-16 — [progress] TYK2 example with the defaults: dG0 −10.05 kJ/mol against −47.3 from experiment; the note on dG(t) states its change, not a missing plateau
- What: `examples/tyk2_ejm46` ran from its README on 47e84f7 (5 × 40 ns, five replicas on one GPU with MPS, about 300 ns/day each): dG0 −10.05 kJ/mol (−2.40 kcal/mol), 95% interval −24.57 to 4.47, replicas −4.65, −9.51, −6.80, 0.58, −29.89; rebinding events 5, 0, 1, 1, 1; every replica left the site within 5.1 ns. The experimental value (−47.3 kJ/mol) is outside the interval; the README reports it as measured. The diagnostic "dG still drifting (… over the plateau window)" now reads "dG(t) from the bias changed by … over the last 50% of the run"; the plot label "plateau mean" reads "mean of the last half". No number, threshold or key changed. `ruff` passes, `pytest` 160 passed. Not run: GPU gates on this code (once, on the final code).
- Why: owner instruction of 2026-11-16: the calculation builds a free-energy surface, so dG(t) from the bias does not level off within a run; a note must not present that as a defect. The bias fills the surface for as long as the run lasts and the difference between the site and the unbound region moves with it (TYK2: −84 kJ/mol at 2 ns, −14 at 40 ns in the mean of five).
- Refs: `examples/tyk2_ejm46/README.md`, `src/fmeta/analysis/__init__.py`, `src/fmeta/analysis/plots.py`, `examples/trypsin_benzamidine/README.md`; FM-D9 (second amendment); roadmap R10.

### 2026-11-16 — [amendment] No convergence verdict: the set of replicas is the measurement (value and interval); the checks of a replica are diagnostics
- What: owner instruction of 2026-11-16: a replica of this length is not a simulation that converges; change the way of thinking. `result.json` and `summary.json` lose `converged` and `problems`; `fmeta analyze` prints the mean of the replicas with its 95% interval and the notes of the unchanged checks as `diagnostics`; `no_binding` stays for a replica and for the value of the set. Benchmark reports have one set of statistics over every ligand with a value. Three estimators for the set were compared by a rule written first (toy: all within 0.80 kJ/mol at 4 × 10 ns; spread between the six bias settings on trypsin at 40 ns: 25.1 kJ/mol for the mean of per-replica estimates, 32.8 and 42.0 for two mean-force estimators): the estimate does not change. The defaults stay. The scheduler starts each replica with one math-library thread (load average 1250 to 46; 3 to 30% faster). `ruff` passes, `pytest` 160 passed. Not run: GPU gates on this code (once, on the final code).
- Why: the owner's instruction, and the scan: 0 to 17 rebinding events per replica in 40 ns, replicas of one setting between −1.7 and −78 kJ/mol, no setting with five replicas free of notes. A verdict that every run fails says nothing; the interval says how much is open. No estimator removed the dependence on the bias setting: it is a limit of the sampling at 40 ns.
- Refs: FM-D9 (second amendment), FM-D16 (third amendment), FM-D17 and FM-D14 amendments (`docs/design/engine.md`); `src/fmeta/analysis/__init__.py`, `src/fmeta/workflow.py`, `src/fmeta/benchmark.py`, `src/fmeta/scheduler.py`; `.claude/rules/numerics-invariants.md`; supersedes the NOT CONVERGED wording of the entry "Verdict of the bias scan".

### 2026-11-16 — [progress] GPU gates pass on the code with the defaults of the scan verdict; the image runs without the source; second wave started
- What: `scripts/gpu_gates.py` on clean 47e84f7 (2 GPUs, in the image): all 18 passed. `toy_free_energy` with the estimate of the FM-D9 amendment: −3.95 (plateau −3.46) against the exact −4.95 kJ/mol; forces against Reference: CV 3.8e-07 nm, energy 7.2e-06, force 7.1e-06; `periodic_images`: CV 1.3e-06 nm, wall energy 8.2e-06 kJ/mol. This closes "GPU gates on this code: not run yet" for the presets (R9), the real inputs (R8) and `fmeta benchmark`. The image built from 47e84f7 passed `fmeta check --quick` on CUDA without the mounted source, and all 232 files and directories it wrote belong to the calling user (a check, not evidence). The second wave runs from 47e84f7 with the defaults (5 × 40 ns): the three examples by their README steps, trypsin with `--mode sequential` on one GPU, and both whole benchmark sets. Not verified here: the result of CI.
- Why: the gates run once, on the code that holds the final defaults. Their speed values were taken while the image check used the same two GPUs (1270 and 1022 ns/day; alone on 8d1f10a: 1371 and 1312) and decide nothing. The gates must run in the main checkout: inside a git worktree the script cannot read the commit in the container.
- Refs: `bench/results/2026-11-16-47e84f72b-gpu-gates.json`, `bench/results/index.json`, `README.md`, `docs/roadmap.md` (R8, R9), `docs/gpu-runbook.md`.

### 2026-11-16 — [amendment] Verdict of the bias scan: hill widths 0.05 / 0.1 nm and 40 ns per replica are the defaults (2 of 5 replicas converge on trypsin)
- What: the scan of FM-D16 ran to 40 ns (5 settings × 5 replicas). No setting has five replicas without a convergence problem at 10, 15, 20, 30 or 40 ns, with the analysis of the FM-D9 amendment or the one before it. By the second point of the rule the setting with the most passing replicas at 40 ns becomes the default: `sig2`, `metad.sigma_s` 0.05 and `sigma_rho` 0.1 nm (were 0.025 / 0.05), 2 of 5; `run.length_ns` 40 (was 500); bias factor 10 and height 1.5 kJ/mol stay. Others at 40 ns: `bf20` 1 of 5, the rest and the former default 0. Code, presets, `docs/configuration.md`, `README.md` and the example texts follow. `sig2` has the widest scatter of all: dG0 −27.4 kJ/mol, SEM 14.9, replicas −78.2, −1.7, −44.3, −3.4, −9.1; one never left the site. Gate item `toy_free_energy` alone on CUDA with the new estimate: −3.95 and −3.46 against −4.95 kJ/mol (criteria 2.0 and 3.0). `ruff` passes, `pytest` 160 passed. Not run: the GPU gates on this commit (next, with the second wave).
- Why: the rule was written before the scan and is applied as written: it counts passing replicas, not scatter, and never the distance to experiment. The owner's limit is 40 ns, so roadmap R2 ("five replicas to convergence") is not met; a run with the defaults can end as NOT CONVERGED on such a system and can be extended exactly (FM-D5).
- Refs: FM-D16 (result of the scan), `docs/configuration.md`, `src/fmeta/engine/grid.py`, `src/fmeta/config.py`, `src/fmeta/presets/`.

### 2026-11-16 — [amendment] dG0 from the samples after the first 2 ns (FM-D9); the plateau mean becomes a cross-check
- What: owner instruction of 2026-11-16: the first 1 to 2 ns change strongly, use only what comes after. `fmeta analyze` now reports as dG0 the estimate from the samples after the first 2 ns (`--discard-ns`; second half of a run under 4 ns): slices of 250 ps, each sample weighted with the bias of its slice, slices combined with the MBAR equations (`sliced_log_weights`). Cross-checks at the unchanged 4 kJ/mol: the same estimate from the second half alone (new) and the plateau mean of dG0(t) from the bias (the former headline); the former reweighting with `c(t)` is removed. No threshold changed. Toy, 8 × 20 ns: error −0.03 ± 0.15 (RMSE 0.40; plateau 0.52) and −0.11 ± 0.08 kJ/mol (RMSE 0.25; plateau 0.58) for wells of 20 and 40 kJ/mol; no well: 8 of 8 reported as no binding. Trypsin, default bias: 0 of 5 converged up to 60 ns, 1 of 5 at 75 ns (old analysis: 2 of 5); scan settings: 0 of 5 at 10 to 30 ns under both analyses. `ruff` passes, `pytest` 160 passed. Not run: GPU gates on this code.
- Why: the owner's instruction. At the start the ligand sits in the site while the bias fills it, which is not a sample of the equilibrium the estimate assumes. The second-half check keeps the power of the removed one: on trypsin the new headline and the plateau mean agree while the second half alone is 11 to 45 kJ/mol higher in every replica.
- Refs: FM-D9 (amendment), `.claude/rules/numerics-invariants.md`, `src/fmeta/analysis/`, `tests/test_analysis.py`, FM-D16 (table of the trypsin run).
