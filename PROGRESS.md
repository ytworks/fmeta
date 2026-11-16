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
- Result: standard binding free energy from the samples after the first 2 ns. The replicas
  together are the measurement: mean and 95 % interval, diagnostics per replica, a "no binding"
  statement, no convergence verdict.
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
| M3 Analysis and the solvable toy system | Complete; toy dG0 within 0.5 kJ/mol of exact (4 × 40 ns, plateau mean). Headline since 2026-11-16 (FM-D9 amendment): from the samples after the first 2 ns; on the toy mean error −0.03 ± 0.15 and −0.11 ± 0.08 kJ/mol (wells of 20 and 40 kJ/mol, 8 × 20 ns each), no well: 8 of 8 reported as no binding |
| M4 Automatic funnel, validation | Complete on synthetic pockets (direction ≤ 10°) |
| M5 System building, equilibration, workflow, CLI | Complete; PDB + SDF alone, with automatic clean-up |
| M6 Membrane | Complete on CPU: real `addMembrane` build (POPC, 108 lipids) and a short run; membrane-plane fit on synthetic helix bundles only |
| M7 Replicas, walkers, figures, `fmeta check` | Complete on CPU |
| Harness: CPU CI + GPU gates | Introduced 2026-11-15 (PR #1) |
| Official image, Docker-only execution (FM-D13) | In use on the GPU machine. The image built from 47e84f7 runs without the mounted source: `fmeta check --quick` on CUDA passed (17 items) and the 232 files and directories it wrote belong to the calling user (a check, not evidence). The result of the CI job `docker` is not checked here; the publish workflow has never run |
| GPU plan M0: first gate record (R1) | Complete: all 16 gates on 2c4ad4e; the fmeta inside the image runs without the mounted source (quick check on CUDA, not evidence) |
| GPU plan M1: run modes and the scheduler (R7, FM-D14) | Complete: all 18 gates on 8d1f10a; `docker stop` and continuation checked on the trypsin run |
| GPU plan M3: official presets and the configuration reference (R9, FM-D16) | Merged: `fmeta init --preset`, `preset:` in a configuration, `docs/configuration.md`. GPU gates passed on 47e84f7 |
| GPU plan M2: real inputs (R8, FM-D15) | Merged, both parts. Membrane part: `fmeta split --drop`, kept components are receptor, orientation warning, membrane box 8% above the box rule. Protein part: chain gaps capped (NME / ACE), caps and protonation states of the input kept, kept hetero in their own chain, `info.protein_input` in `prepared/report.json`. `pytest` 128 passed. GPU gates passed on 47e84f7; the real `addMembrane` test (`-m slow`) not rerun. On real files (GPU, 200 ps of free equilibration, zero wall energy): 3PTB, 4GIH, the prepared TYK2 protein of the Schrödinger set; protein model alone on both A2A files |
| GPU plan M5 infrastructure: `fmeta benchmark` (R10, FM-D17) | Merged: `fmeta benchmark prepare`, `run`, `status`, `report`; sets `examples/benchmarks/tyk2` (10 ligands) and `a2a` (8 ligands) with experimental and FEP+ values; reports to `bench/reports/` without a verdict, checked by `scripts/verify_bench_results.py`. Tested on a synthetic set with a local source. On the merged code both real sets download and build: TYK2 pilot 86–91k atoms (eeb5302), caps and histidine states of the FEP+ input kept; A2A pilot 116–117k atoms (c41c08b; 228 POPC, box 12.74 × 6.69 × 13.9–14.1 nm), superposed at 0.112 nm, HIS 264 / 278 doubly protonated as in the file, gap capped, sodium kept (set READMEs). The pilot runs are under M7. GPU gates passed on 47e84f7. Not done: a finished run of a set |
| GPU plan M4: trypsin-benzamidine (R2), the run length and the scan of the bias | Defaults from the scan by its rule (FM-D16): the setting `sig2` (`metad.sigma_s` 0.05 and `sigma_rho` 0.1 nm; bias factor 10 and height 1.5 kJ/mol stay) and `run.length_ns` 40, with 2 of 5 replicas passing the convergence check of that time. Read without a verdict since the second amendment of FM-D9: dG0 −27.4 kJ/mol, 95% interval −68.7 to 14.0, replicas −78.2 to −1.7 (experiment −26.8 to −30.5, not used). The six settings of the scan give −2.9 to −28.0 kJ/mol at 40 ns: the value depends on the bias setting, and two mean-force estimators over the pooled replicas did not reduce that (spread 25.1 kJ/mol against 32.8 and 42.0; FM-D16, third amendment). R2 is reworded: the value with its interval. The example runs again with the defaults, one replica after another on one GPU: in progress |
| GPU plan M5: kinase example (R10) | TYK2 with ejm_46 from PDB entry 4GIH (`examples/tyk2_ejm46`, 88476 atoms). Run from its README with the defaults (5 × 40 ns, 47e84f7, the five replicas on one GPU with MPS): dG0 −10.05 kJ/mol (−2.40 kcal/mol), 95% interval −24.57 to 4.47, replicas −29.89 to 0.58, rebinding events 5, 0, 1, 1, 1; experiment −47.3 kJ/mol, outside the interval. Reported as measured in the README |
| GPU plan M6: GPCR example (R3) | `examples/a2a_theophylline` (OPM file of 5MZJ): builds without a workaround (221110 atoms, gap LEU208–GLU219 capped, sodium ion kept, 4 disulfide bonds, funnel into the extracellular water, no error in the validation); equilibrated with the 2 ns free stage; membrane production works (5 replicas on one GPU with MPS, about 506 ns/day in total). That run (former default bias) was stopped at about 11 ns when the defaults changed. It is run again from its README with the defaults of this commit (5 × 40 ns). No result yet |
| GPU plan M7: benchmark runs (R10) | Pilots (3 ligands per set, 5 × 20 ns, former default bias, commits eeb5302 and f2ea653): the TYK2 pilot is complete (15 runs) and its report is committed as a pilot (`bench/reports/2026-11-16-45055cc99-tyk2`: dG0 −18.96, −11.14 and −23.71 kJ/mol for ejm_44, ejm_31 and ejm_46 against −31.05, −39.92 and −47.32 from experiment; 95% intervals 19 to 62 kJ/mol wide); the A2A pilot runs on GPU 1, no report yet. The whole sets (TYK2 10 ligands, A2A 8 ligands, 5 × 40 ns) run from 47e84f7 in fresh work directories. No report of a whole set yet |

- GPU gates: **all 18 passed** (`gpu-tests` and the 17 items of `fmeta check`) on clean 47e84f7,
  the code with the defaults of the scan verdict (CUDA, mixed precision, 2 GPUs, in the image):
  `bench/results/2026-11-16-47e84f72b-gpu-gates.json` (dated in UTC). `toy_free_energy`: −3.95
  (plateau −3.46) against the exact −4.95 kJ/mol, criteria 2.0 and 3.0. `replicas_sequential`
  and `replicas_parallel` ran real child processes on the GPUs (FM-D14). Its speed values were
  taken while the image check used the same two GPUs; the record of 8d1f10a has them alone.
  Not run yet on the code after 47e84f7 (analysis without a verdict, one math-library thread
  per replica in the scheduler): once more, on the final code.
- Other checks, in the image (lock d061e45a…): `ruff check .` passes and `pytest` 160 passed on
  the code with the reworded diagnostic of dG(t). Operation on a CPU platform is not guaranteed (owner, 2026-11-16,
  FM-D12): `pytest -m slow` (6 passed) and `fmeta check --platform CPU` were last run on the
  source of 2c4ad4e and are not rerun.
- CI: green on 171220a. The commits after it were pushed on 2026-11-16; the result of CI on
  them, with the first run of the job `docker`, is not checked here.
- Not verified yet:
  - a binding free energy of a real protein-ligand system with a narrow interval (trypsin with
    the defaults: 95% interval −68.7 to 14.0 kJ/mol, and the value depends on the bias setting;
    TYK2 with ejm_46: −24.6 to 4.5 kJ/mol, 37 kJ/mol above experiment);
  - reading HILLS with PLUMED itself;
  - the publish workflow (never run).
- Known gaps: ligand protonation is taken from the SDF as drawn (R5); unresolved loops are not
  modelled; non-protein components are removed unless listed in `system.keep_hetero`; a
  directory prepared before 2026-11-16 keeps periodic funnel walls in `prepared/system.xml` and
  must be prepared again; the gate `toy_free_energy` had 0.04 kJ/mol to spare with
  the former reweighting, where about 1 run in 30 of its 10 ns missed the criterion on every
  platform (FM-D12); with the estimate of the FM-D9 amendment its error is 1.00 kJ/mol for the
  gate's seed, and the scatter of that estimate over seeds at 10 ns is not measured.
- GPUs (owner, 2026-11-16: all 8 are for this task): every GPU carries several replicas with
  NVIDIA MPS (`run.per_gpu`, FM-D14 amendment). Second wave, started at 06:41 UTC on 2026-11-16
  from 47e84f7 with the defaults and continued from its checkpoints at 07:41 UTC with one
  math-library thread per replica: the three examples on GPUs 2, 3 and 4 (trypsin one replica
  after another), the TYK2 set on GPUs 0 and 5, the A2A set on GPUs 6 and 7; the A2A pilot
  finishes on GPU 1. The TYK2 example ended at 10:08 UTC; its GPU 3 went to the TYK2 set.
- Time estimate (measured: 88k atoms 606 ns/day alone and 1283 / 1541 / 1620 ns/day per GPU with
  4 / 6 / 8 replicas under MPS; A2A set system about 1015 ns/day per GPU with 6 replicas, A2A
  example about 506 with 5): both full sets and the three examples at 5 × 40 ns are about
  3.7 GPU-days, 11 to 13 hours on 8 GPUs.
- Blockers: none. Pushes from the GPU machine work since 2026-11-16 (the `gh` login of the
  machine as the git credential; the owner added the `workflow` scope).

## Change Log

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

### 2026-11-16 — [amendment] 20 ns first and a scan of faster bias settings; several replicas per GPU with MPS; 2 ns membrane free stage
- What: owner decisions of 2026-11-16: 100 to 200 ns per replica is still too long, run 20 ns first and judge, shorten the equilibration, all 8 GPUs are for this task. The 100 / 150 / 200 ns rule is withdrawn; the trypsin run was stopped at 75–81 ns (0 of 5 replicas converged at 20 ns, 2 of 5 at 75 ns). New rule, written before the scan: five faster bias settings on trypsin (bias factor 20, height 3.0, sigma × 2 and combinations), 5 replicas each, analysed at 10, 15, 20, 30 and 40 ns; the first length at which a setting has all five replicas converged gives `run.length_ns` and the default bias; at most 40 ns. `run.per_gpu` / `--per-gpu` put several replicas on one GPU (NVIDIA MPS in a container: `scripts/mps.sh`, `scripts/in_docker.sh --mps`): 1283 / 1541 / 1620 ns/day per GPU with 4 / 6 / 8 replicas of 88k atoms, against 606 alone. Membrane free stage 2 ns (was 10), with `free_blocks`. `ruff` passes, `pytest` 160 passed. Not run: GPU gates on this code (once, on the final defaults); `--mps` of the wrapper by an actual call.
- Why: the owner's decisions. With the default bias 20 ns is not enough on trypsin, so a shorter run needs a faster bias; it is selected on trypsin by the convergence diagnostics only, never on a benchmark. Five replicas and every criterion stay.
- Refs: FM-D14 (amendment), FM-D16 (second amendment), FM-D13, `docs/configuration.md`, `tests/test_scheduler.py`, `tests/test_system.py`.

### 2026-11-16 — [progress] A2A benchmark set at the preset's lipid padding: 116k atoms instead of 231k
- What: `examples/benchmarks/a2a/set.yaml` no longer sets `system.padding: 1.5`; the preset's 1.2 nm holds. The three pilot ligands built on c41c08b: 115625, 117494 and 116949 atoms, 228 POPC, box 12.74 × 6.69 × 13.9–14.1 nm, no error in the validation, unbound ligand 2.43–2.54 nm from the nearest periodic image. `examples/a2a_theophylline` keeps 1.5 nm: in its frame (OPM file of 5MZJ) the membrane is 2 × 2 patches with either value (221110 and 221176 atoms).
- Why: `Modeller.addMembrane` builds from whole lipid patches of 6.37 × 6.69 nm, so the size moves in steps: for `4q`, 1.5 nm gave 2 × 2 patches (231482 atoms) and 1.2 nm gives 2 × 1. Both pass the box rule, no ligand had been equilibrated, and no rule asks for the larger value, which doubled the cost of the set.
- Refs: c41c08b, `examples/benchmarks/a2a/README.md` ("Configuration"), FM-D17.

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
