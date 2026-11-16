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

Snapshot date: 2026-11-16. Decisions: `docs/design/engine.md` (FM-D1..D17 Accepted;
FM-D2, D4, D5, D6, D7, D11, D12 and D16 amended since, see the Change Log). Everything below is on PR #1 (not merged).
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
| GPU plan M2: real inputs (R8, FM-D15) | Merged, both parts. Membrane part: `fmeta split --drop`, kept components are receptor, orientation warning, membrane box 8% above the box rule. Protein part: chain gaps capped (NME / ACE), caps and protonation states of the input kept, kept hetero in their own chain, `info.protein_input` in `prepared/report.json`. `pytest` 128 passed. GPU gates not run on this code yet, and the real `addMembrane` test (`-m slow`) not rerun. On real files (GPU, 200 ps of free equilibration, zero wall energy): 3PTB, 4GIH, the prepared TYK2 protein of the Schrödinger set; protein model alone on both A2A files |
| GPU plan M5 infrastructure: `fmeta benchmark` (R10, FM-D17) | Merged: `fmeta benchmark prepare`, `run`, `status`, `report`; sets `examples/benchmarks/tyk2` (10 ligands) and `a2a` (8 ligands) with experimental and FEP+ values; reports to `bench/reports/` without a verdict, checked by `scripts/verify_bench_results.py`. Tested on a synthetic set with a local source. On the merged code (eeb5302) both real sets download and build: TYK2 pilot 86–91k atoms, caps and histidine states of the FEP+ input kept; A2A `4q` superposed at 0.112 nm, HIS 264 / 278 doubly protonated as in the file, gap capped, sodium kept (set READMEs). TYK2 pilot equilibrating on GPUs 5–6 since 01:26 UTC (container `fmeta-bench-tyk2`). Not done: a run of a set, GPU gates on this code |
| GPU plan M4: trypsin-benzamidine (R2), the run length and the scan of the defaults | Running: 5 replicas from PDB entry 3PTB on GPUs 0–4 (container `fmeta-tb`, commit 8d1f10a), started 2026-11-15 23:17 UTC, 41–44 ns each at 2026-11-16 00:53 UTC (660 ns/day). Analysed at 40 ns: not converged (3–5 rebinding events per replica, drift +12 to +22 kJ/mol). The default length of 500 ns is withdrawn (owner); the run ends at the length its convergence gives: 100, 150 or 200 ns (rule in `examples/trypsin_benzamidine/README.md`, FM-D16 amendment). Scan, compared at that length: `dt2` running at 2 fs on GPUs 5–7 (container `fmeta-scan-dt2`) since 00:22 UTC; `pace2` and `srho03` prepared, run only if the rule can still change a default. No result yet |
| GPU plan M5: kinase example (R10) | Paused: TYK2 with ejm_46 from PDB entry 4GIH (`examples/tyk2_ejm46`, 88476 atoms, commit 8d1f10a). Replica 0 ran on GPU 5 from 2026-11-15 23:30 UTC and was stopped at 2026-11-16 00:22 UTC for the scan; all replicas continue from the checkpoints when GPUs are free. No result yet |
| GPU plan M6: GPCR example (R3) | Draft: `examples/a2a_theophylline` (OPM file of 5MZJ). On b54292f the input as `fmeta split` writes it builds without a workaround: 221110 atoms, gap LEU208–GLU219 capped (NME / ACE), sodium ion kept, 4 disulfide bonds, funnel into the extracellular water, no error in the validation. Equilibration at 300 K with 10 ns of free stage running on GPU 7 since 01:08 UTC (container `fmeta-a2a-eq`). The README of the example still describes the placeholder input. No metadynamics yet |
| GPU plan M7: benchmark runs (R10) | Not started |

- GPU gates: **all 18 passed** (`gpu-tests` and the 17 items of `fmeta check`) on clean 8d1f10a
  (CUDA, mixed precision, 2 GPUs, in the image):
  `bench/results/2026-11-15-8d1f10a89-gpu-gates.json` (dated in UTC). It covers the run modes
  (FM-D14): `replicas_sequential` and `replicas_parallel` ran real child processes on the GPUs.
- Other checks, in the image (lock d061e45a…): `ruff check .` passes and `pytest` 157 passed on
  the merge of FM-D17. Operation on a CPU platform is not guaranteed (owner, 2026-11-16,
  FM-D12): `pytest -m slow` (6 passed) and `fmeta check --platform CPU` were last run on the
  source of 2c4ad4e and are not rerun.
- CI: green on 171220a. The commits after it were pushed on 2026-11-16; the result of CI on
  them, with the first run of the job `docker`, is not checked here.
- Not verified yet:
  - a binding free energy of a real protein-ligand system (the first run is in progress);
  - reading HILLS with PLUMED itself;
  - the publish workflow (never run).
- Known gaps: ligand protonation is taken from the SDF as drawn (R5); unresolved loops are not
  modelled; non-protein components are removed unless listed in `system.keep_hetero`; a
  directory prepared before 2026-11-16 keeps periodic funnel walls in `prepared/system.xml` and
  must be prepared again; the gate `toy_free_energy` passes with 0.04 kJ/mol to spare, and
  about 1 run in 30 of its 10 ns misses the criterion on every platform (FM-D12), so a change
  of the toy trajectory (another GPU, another OpenMM) can fail it without a defect.
- GPUs (plan decision changed 2026-11-16, see the Change Log): 0–4 trypsin, 5–7 the scan, shared
  with the equilibrations of the A2A example, the TYK2 pilot and the lipid patch work. Gates run
  on 6–7 with those containers stopped for the run and restarted with the same commands.
- Time estimate from the measured speeds (72k atoms 660, 88k atoms 570 ns/day at 4 fs; the A2A
  box, 221k atoms, 167 ns/day at 2 fs in equilibration, 4 fs not measured), for 5 replicas per
  ligand, 10 TYK2 and 9 A2A ligands, A2A at twice its 2 fs speed: about 22 GPU-days at 100 ns
  per replica (TYK2 9, A2A 13), 45 at 200 ns. It was about 115 at 500 ns.
- Blockers: none. Pushes from the GPU machine work since 2026-11-16 (the `gh` login of the
  machine as the git credential; the owner added the `workflow` scope).

## Change Log

### 2026-11-16 — [design] FM-D17: benchmark sets as configuration, one scheduler call, a report without a verdict
- What: `fmeta benchmark prepare | run | status | report SET -o WORKDIR` (`src/fmeta/benchmark.py`, target-agnostic). A set is a directory of `examples/benchmarks/`: `set.yaml` (source at a fixed commit, sha256 per file, ligands, pilot, differences from a preset, a structure to superpose on for a membrane set) and `reference.csv` (experiment, FEP, absolute FEP, each with its source). Sets `tyk2` (10 ligands) and `a2a` (8 ligands, superposed on the OPM file of 5OLZ) from the Schrödinger public benchmark; no structure file is stored. Each ligand is an ordinary work directory; all stages of all ligands go through one scheduler call, replica 0 of every ligand first. A report (`bench/reports/`) gives dG0 per ligand with experiment and FEP, MUE, RMSE, mean signed error, Pearson r and Kendall tau with bootstrap intervals, MUE and RMSE also after removing the mean error, for all ligands and for the converged ones; it names commit, image and lock and has no verdict. `ruff` passes, `pytest` 157 passed (synthetic set, local source). On the branch, with the input handling of 8d1f10a: the three TYK2 pilot ligands built (86–91k atoms); A2A superposed on 297 CA atoms, RMSD 0.112 nm, `4q` built (110352 atoms). Not run: a real set on the merged code, GPU gates.
- Why: owner request of 2026-11-15 for benchmarks with experimental and FEP values that can be run and compared. FEP+ values are relative ones shifted onto the experimental mean, so the comparison without the mean error is the fair one; a report is a measurement, not a gate, so that nothing is tuned to pass it.
- Refs: FM-D17 (`docs/design/engine.md`), roadmap R10, `examples/benchmarks/`, `bench/reports/README.md`, `tests/test_benchmark.py`, 3bb4477 8c6267b 6586b26 81fe11f dbec83e 434f589 73f7633.

### 2026-11-16 — [design] FM-D15, protein part: gaps capped, caps and protonation states of the input kept
- What: `read_protein` builds the model from what the file states (`system/protein.py`): a chain gap (C-N > 0.2 nm, with or without TER) is split and capped with NME / ACE, a gap of one residue is refused, caps of the input stay, protonation states are read from hydrogens or residue names and hold through a rebuild, kept hetero go to their own chain; `prepared/report.json` (`info.protein_input`) reports gaps, caps, altlocs, states, disulfides. `build_complex` returns the facts as a fourth value. `ruff` passes, `pytest` 128 passed on the merge; each regression test failed before its fix. Real files on a GPU: 3PTB, 4GIH, the prepared TYK2 protein of the Schrödinger set (UNK caps kept, states from its hydrogens). Not run: GPU gates, `-m slow`, the A2A example through `prepare`.
- Why: OpenMM bonded LEU208 to GLU219 over 1.15 nm, caps became charged ends and HIP / HIE became HID, each without a message: each changes the charge or the shape at the pocket.
- Refs: FM-D15 (`docs/design/engine.md`, protein part), roadmap R8, `tests/test_inputs.py`, f0db6b7 84199f2 7dd1516 c39403a.

### 2026-11-16 — [amendment] Default run length: 500 ns withdrawn (owner); the new value comes from the trypsin run by a rule written first
- What: the plan and FM-D16 had 5 replicas × 500 ns as the default. The owner decided on 2026-11-16 that 500 ns is too long. The new `run.length_ns` is the shortest of 100, 150 and 200 ns at which all five replicas of the official trypsin-benzamidine run pass the convergence criteria; 200 ns if they pass at none, stated as not converged. Rule committed with the replicas at 41–44 ns. Five replicas stay; no criterion changes. The scan compares at that length; `pace2` and `srho03` run only if fewer than half of the baseline replicas pass there (otherwise their condition cannot be met). Code and presets hold 500 until the rule has given the value. The two chain scripts that queued the old runs were stopped.
- Why: the owner's decision. Two measurements with the default bias say that transitions are frequent: trypsin 3–5 rebinding events per replica within 40 ns; TYK2 with ejm_46 (4.8 nM) left the site within 5 ns and was back within 20 ns. 500 ns came from the literature, not from a measurement here.
- Refs: FM-D16 (amendment), `examples/trypsin_benzamidine/README.md` ("Run length", amendment of the scan), `docs/plans/2026-11-15-gpu-validation-modes-benchmarks.md` (defaults, compute table).

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
