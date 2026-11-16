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
- Owner request 2026-11-15 (roadmap R7–R10). In: replicas side by side on several GPUs or one
  after another on one GPU (R7); official presets (R9). Planned: real inputs (R8); a kinase and
  a GPCR example, benchmarks against experiment and FEP (R10).
- Hardware: an 8-GPU machine (since 2026-11-15). Everything except CI runs in the official
  Docker image (FM-D13).

## Current Status

Snapshot date: 2026-11-16. Decisions: `docs/design/engine.md` (FM-D1..D16 Accepted, of FM-D15 the
membrane part; FM-D2, D4, D5, D6, D7, D11 and D12 amended since, see the Change Log). Everything below is on PR #1 (not merged).
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
| GPU plan M3: official presets and the configuration reference (R9, FM-D16) | Merged: `fmeta init --preset`, `preset:` in a configuration, `docs/configuration.md`. GPU gates not run on this code yet |
| GPU plan M2: real inputs (R8, FM-D15) | Membrane part merged: `fmeta split --drop`, kept components are receptor, orientation warning, membrane box 8% above the box rule; `pytest` 113 passed. GPU gates not run on this code yet, and the real `addMembrane` test (`-m slow`) not rerun. Protein part (chain gaps, caps, protonation states) in progress on a branch |
| GPU plan M5 infrastructure: `fmeta benchmark` (R10, FM-D17) | In progress on a branch; not merged, not gated |
| GPU plan M4: trypsin-benzamidine (R2) and the scan of the defaults | Running: 5 replicas × 500 ns from PDB entry 3PTB on GPUs 0–4 (container `fmeta-tb`, commit 8d1f10a), started 2026-11-15 23:17 UTC, 33 ns each at 2026-11-16 00:29 UTC (660 ns/day). Scan (protocol and decision rule in `examples/trypsin_benzamidine/README.md`, written before the runs): `dt2` running, 3 × 200 ns at 2 fs on GPUs 5–7 (container `fmeta-scan-dt2`) since 00:22 UTC; `pace2` and `srho03` prepared, not started. No result yet |
| GPU plan M5: kinase example (R10) | Paused: TYK2 with ejm_46 from PDB entry 4GIH (`examples/tyk2_ejm46`, 88476 atoms, commit 8d1f10a). Replica 0 ran on GPU 5 from 2026-11-15 23:30 UTC and was stopped at 2026-11-16 00:22 UTC for the scan; all replicas continue from the checkpoints when GPUs are free. No result yet |
| GPU plan M6: GPCR example (R3) | Draft: `examples/a2a_theophylline` (OPM file of 5MZJ, 221143 atoms, 484 POPC). Builds, places the funnel into the extracellular water and passes 1 ns of equilibration with a placeholder input (`TER` records added by hand); as `fmeta split` writes it, `prepare` stops (kept ion read into the protein chain). No metadynamics yet |
| GPU plan M7: benchmark runs (R10) | Not started |

- GPU gates: **all 18 passed** (`gpu-tests` and the 17 items of `fmeta check`) on clean 8d1f10a
  (CUDA, mixed precision, 2 GPUs, in the image):
  `bench/results/2026-11-15-8d1f10a89-gpu-gates.json` (dated in UTC). It covers the run modes
  (FM-D14): `replicas_sequential` and `replicas_parallel` ran real child processes on the GPUs.
- Other checks, in the image (lock d061e45a…): `ruff check .` passes and `pytest` 110 passed on
  the code of FM-D16 with the fix of dG(t). Operation on a CPU platform is not guaranteed (owner, 2026-11-16,
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
- GPUs (plan decision changed 2026-11-16, see the Change Log): 0–4 trypsin, 5–7 the scan. Gates
  run on 6–7 with the scan container stopped for the run and restarted with the same command.
- Time estimate from the measured speeds (72k atoms 660, 88k atoms 570 ns/day at 4 fs; the A2A
  box, 221k atoms, 167 ns/day at 2 fs in equilibration, 4 fs not measured): the TYK2 pilot
  (3 ligands × 5 × 500 ns) is about 13 GPU-days. Everything planned is about 115 GPU-days, two
  weeks on 8 GPUs: TYK2 44, A2A 67 if 4 fs gives twice the speed of 2 fs, trypsin with the scan 7.
- Blockers: the A2A example waits for the protein part of FM-D15 (a kept ion that follows the
  chain without `TER`, caps at the loop gap). Nothing else. Pushes from the GPU machine work
  since 2026-11-16 (the `gh` login of the machine as the git credential; the owner added the
  `workflow` scope).

## Change Log

### 2026-11-16 — [amendment] Plan decision on the GPUs changed: the scan also runs on GPUs 6–7, TYK2 paused
- What: the plan kept GPUs 0–5 for production and 6–7 for gates and development. Now GPUs 5–7 run the three 2 fs replicas of the scan (`fmeta-scan-dt2`, started 00:22 UTC, 3 × 200 ns, 360–390 ns/day on an unshared GPU); for a gate run that container is stopped and afterwards started again with the same command. The TYK2 run on GPU 5 was stopped at 00:22 UTC (replica 0 near 25 ns) and continues from its checkpoints later. The scan protocol and its decision rule were committed (2cf99bd) before any scan run started; `pace2` and `srho03` follow when GPUs are free.
- Why: three facts. A full gate run takes 234 s, so GPUs 6–7 stood idle nearly all the time. `docker stop` and the exact continuation of a run are verified (trypsin, 2026-11-16). The scan decides the defaults before a benchmark may start, so it sets the date of everything after it, and the owner asked for the shortest time.
- Refs: `docs/plans/2026-11-15-gpu-validation-modes-benchmarks.md` (GPU allocation), `examples/trypsin_benzamidine/README.md` ("Scan of the defaults"), FM-D14, FM-D16.

### 2026-11-16 — [design] FM-D15, membrane part: parts dropped first, kept components are receptor, orientation warning, box height
- What: `fmeta split --drop CHAIN[:FIRST-LAST]`; a component kept with `system.keep_hetero` is class `other` (restrained, funnel obstacle, in the box rule; `kept_atoms` in `meta.json`); an orientation that could not be checked is a warning, not "ok"; a membrane box is built 8% above the box rule (the rule itself unchanged). `examples/a2a_theophylline` (draft) replaces `examples/gpcr_membrane`: first real A2A build from the OPM file of 5MZJ, 221143 atoms, 484 POPC, funnel exit 16° from +z into the extracellular water, 1 ns equilibration passes (wall energy 0.000 kJ/mol), about 167 ns/day at 2 fs. Experimental values of theophylline from ChEMBL records: −7.9 to −6.5 kcal/mol, Kd on the crystal construct −7.89. `ruff` passes, `pytest` 113 passed; each regression test failed before its fix. Not run: GPU gates, `-m slow`.
- Why: with 3% the A2A box lost 3.5% of its height under NPT and `equilibrate` stopped on `box_cv`; a kept Na+ was neither restrained nor an obstacle; a check that checked nothing read as a pass. Open: a kept ion inside the protein chain stops `prepare`, caps at chain gaps, protonation states of a prepared file (protein part of FM-D15, on a branch); 8% rests on two runs of 1 ns.
- Refs: FM-D15 (`docs/design/engine.md`), roadmap R8 and R3, `examples/a2a_theophylline/`, `src/fmeta/system/`, c0981f7 407c7cb f06d9c1 99af428 bd08b67 2d061d9.

### 2026-11-16 — [progress] Fixed: dG(t) of a run with an uneven number of hills had only a few points
- What: `analyze_run` took dG(t) where a multiple of `len(hills) // 20` met a multiple of `len(hills) // 200`. With a hill count that is a multiple of 200 that is 20 points (unchanged by the fix). With any other count, as in a run analysed before its end, it was 1 to 7 points, so the plateau mean and its drift came from 1 to 3 values. The stride of dG(t) is now a multiple of the stride of c(t). Found on the trypsin run at 20 ns (plateau std 0.00 on four of five replicas; after the fix every replica reports its drift, 1 to 4 rebinding events each, not converged, as expected at 4% of the length). The regression test fails on the code before (7 points at 650 hills). `ruff` passes, `pytest` 110 passed.
- Why: the headline is the plateau mean of dG(t), and the benchmark reports are made while runs are in progress.
- Refs: `src/fmeta/analysis/__init__.py`, `tests/test_analysis.py`.

### 2026-11-16 — [design] FM-D16: official defaults as packaged presets, one reference of the keys
- What: presets `soluble` / `membrane` in the package, written by `fmeta init --preset`; a configuration may hold only its difference (`preset:`), and the one saved in a work directory is resolved and names the ligand force field. Examples reduced to that. `docs/configuration.md`: every key with unit, default and origin. Ligand pins kept (openff-2.2.1 / gaff-2.11, AM1-BCC by `sqm`; charges identical to pre-assigned `am1bcc` for benzamidinium and theophylline). No default changed. `ruff` passes, `pytest` 109 passed. GPU gates not run on this code yet.
- Why: owner request of 2026-11-15 for a configuration YAML with official defaults. One copy of the defaults, held equal by tests; every value has a stated origin. Values that differ from the literature read (hills every 1 ps against 2 ps, `sigma_rho` 0.05 against 0.03 nm, 4 fs against 2 fs, wall constants) are listed for the scan on trypsin-benzamidine and are never tuned on a benchmark.
- Refs: FM-D16, FM-D11, roadmap R9, `docs/configuration.md`, `src/fmeta/presets/`, `tests/test_config.py`. Also in this merge: FM-D14 now holds the measurement of several runs on one GPU (at most 9% more in total, so no key).

### 2026-11-16 — [progress] Kinase example added and started: TYK2 with ejm_46 from PDB entry 4GIH (R10)
- What: `examples/tyk2_ejm46/` (README with the experimental Ki of 4.8 nM, −11.3 kcal/mol, and the steps; a configuration that only names the two input files). On gated commit 8d1f10a in the image, `fmeta split`, `prepare`, `funnel` and `equilibrate` passed without a change: 0X5 and 119 waters removed, 1 heavy atom rebuilt, no chain break, funnel placed automatically (path to bulk 2.65 nm, bound pose at s = 1.20 nm), 88476 atoms, wall energy 0.000 kJ/mol in equilibration. Run started with `--mode sequential` on GPU 5 (container `fmeta-tyk2`): replica 0 at about 570 ns/day. It moves to the parallel mode when GPUs 0–4 are free.
- Not verified yet: the free energy; whether a ligand this tight rebinds within 500 ns (the convergence criterion of 3 rebinding events is unchanged).
- Refs: roadmap R10, `examples/tyk2_ejm46/README.md`.
