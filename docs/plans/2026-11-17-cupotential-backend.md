# cuPotential as the default back end

Status: In progress
Refs: FM-D18 (to be written in M3); amendments of FM-D4, FM-D5, FM-D13, FM-D14

## Context

Owner request (2026-11-17): make fmeta faster with NVIDIA cuPotential and make it the default.

cuPotential is not another engine. It is a plugin of OpenMM's CUDA platform
(`nv_openmm_plugin` 0.3.5, early access) with replacements for `NonbondedForce`,
`LangevinMiddleIntegrator` and the isotropic `MonteCarloBarostat`. FM-D1 (OpenMM is the only
engine) stays as it is.

Facts that shape the plan:

| Fact | Consequence |
|---|---|
| The plugin needs OpenMM `>=8.4,<8.5`, CUDA 13.4 and Python 3.12; the image has OpenMM 8.6.1 and CUDA 12.9 | The official image goes **down** to OpenMM 8.4.0; tests and GPU gates are taken again in the new image (FM-D13) |
| `openmmforcefields 0.16.0` on conda-forge needs `openmm >=8.5.1`; 0.15.1 needs `ambertools <25` | The ligand tools go down with it. The force-field names are fixed by fmeta (`openff-2.2.1`, `gaff-2.11`); the difference in charges is measured in M0 |
| The host driver is R595; CUDA 13.4 needs R615 or the forward-compatibility libraries `cuda-compat-13-4` | The compatibility libraries go into the image. Their use together with MPS is not documented: M0 tries it |
| The plugin's whole-step CUDA graph is off when a System has fmeta's `CustomCentroidBondForce` / `CustomCVForce` | The speed-up NVIDIA publishes (one simulation per GPU) does not carry over as it stands |
| Today's best throughput of an 88k-atom system is 8 replicas per GPU under MPS (1620 ns/day per GPU; 606 for one replica alone) | **Whether the plugin gains under MPS decides the outcome.** It is measured, not assumed |
| There is no replacement for `MonteCarloMembraneBarostat` | A membrane system is "plugin force (+ integrator) with the stock membrane barostat" |
| The plugin force works on CUDA in single / mixed precision with PME, tolerance 5e-4 and a box of at least 3 cutoffs; it is not bit-identical to stock PME | Where it cannot be used a run goes on stock OpenMM and records why; a new gate measures its accuracy |
| The toy system has no PME force and two massless particles | The toy gates do not exercise the plugin force: a check on a real system is needed |

Owner decisions (2026-11-17):

1. Speeds measured with cuPotential are **not written to public files** (clause 2.12 of
   NVIDIA's licence). Tracked files, commit messages and PR text hold stock OpenMM speeds and,
   for cuPotential, gate verdicts and accuracy only.
2. The image with `libcupo` **is published** to ghcr, with NVIDIA's licence text in the image
   and a statement in the README.

## Approach

### Design

- New key `run.backend`: `auto` (the default after M4) | `cupotential` | `openmm`; option
  `--backend` of `fmeta run`, `fmeta check` and `scripts/gpu_gates.py`.
  - `auto`: cuPotential when the platform is CUDA in single or mixed precision, the plugin can
    be imported and the System meets the conditions (one PME `NonbondedForce`, tolerance 5e-4,
    automatic mesh, box size). Otherwise the run uses stock OpenMM and writes a **named
    reason** to the log, `run.yaml` and `device.json`. A plugin error without a name stops the
    run: it never falls back silently.
  - `cupotential`: an error before a file is touched when the conditions are not met.
- New module `src/fmeta/engine/backend.py`: the decision (`resolve`, a pure function), the
  check of the conditions, reading the record of a run, and `build` (conversion of the System,
  the integrator, the record). Every place that makes a Simulation goes through `build`:
  `workflow._run`, `check_pbc`, `check_benchmark`.
- The conversion happens in memory right before the Context is made. `prepared/system.xml`
  stays a stock System; equilibration and the internals of `addMembrane` stay on stock OpenMM.
- The composition (which of force / integrator / barostat comes from the plugin) is one per
  class of system (soluble, membrane): **of the compositions that pass every gate, the one
  with the highest total ns/day per GPU**. For a class where that is below today's value,
  `auto` resolves to stock OpenMM. The rule and the chosen composition are FM-D18. Speed never
  decides a test or a gate.
- One run and one work directory keep the back end they started with.
  - It is recorded in `run.yaml`; a `run.yaml` without the entry means stock OpenMM. On a
    restart `auto` follows the record; for a new replica `auto` follows the existing
    `rep*/run.yaml` of the work directory.
  - An explicit value that contradicts the record is refused in `_run` before the platform is
    chosen (`check_same_run` runs after the Simulation exists and does not judge an entry that
    one side lacks). With walkers the back end joins the conditions of `replica_jobs` and is
    refused before the rollback.
  - The `system` conditions of `run.yaml` (particles, barostat) come from the stock System
    before the conversion.
- Found by reading the code, fixed with the conversion:
  - The default soluble box is a rhombic dodecahedron: every check of the plugin uses that box
    and the membrane box.
  - `_run` makes the Context with the box of the build and then loads the state: the
    equilibrated box is set on the System before the Context exists.
  - With `run.seed` 0 the back end draws the seed itself.
  - `exclude_probe` is called before the Context exists.
- `configuration_hash` of the benchmarks includes the back end; `openmm` gives the hash from
  before the key existed. Sets in progress get `backend: openmm` in `set.yaml`. Reports name
  the back end of each replica; a mixture is one diagnostic line, never a reason to drop a
  replica.
- Speeds: `Item` gets a separate field for performance values (today ns/day is inside the
  `detail` text). For a cuPotential record `make_record` leaves the field out; the numbers
  stay in the untracked `gates_work/` and `bench/private/`. `scripts/verify_bench_results.py`
  (CI) refuses a cuPotential record that holds performance values. One option of
  `gpu_gates.py` is the switch for the day NVIDIA's permission exists.
- One gate record per back end: stock OpenMM keeps today's file name, cuPotential is
  `<date>-<commit9>-cupotential-gpu-gates.json`; record and index get `backend` (absent means
  `openmm`).
- Image: the plugin is one line of the lock (the `.conda` of the release, URL and hash). URL
  and sha256 of the compatibility libraries are comment lines of the lock, read by the
  `Dockerfile`: the existing "official image" check (sha256 of the lock inside the image)
  then covers both. The libraries are used only when the host driver is older than the CUDA
  runtime of the image. `environment()` of the gate record also names the plugin version, the
  CUDA runtime, the host driver and whether the compatibility libraries are in use.

### Milestones

| | Content | Done when |
|---|---|---|
| M0 image | trial solve; `environment.yml` (`openmm=8.4.*`, `cuda-version=13.4`, `nv-openmm-plugin=0.3.5`), `scripts/update_lock.sh`, the lock, `Dockerfile` (compatibility libraries, licence texts in `/opt/fmeta/licenses/`), `gpu_gates.environment()`. No change in `src/fmeta/` | the new image passes `ruff`, `pytest`, `pytest -m slow` and the 18 GPU gates on stock OpenMM, record committed; a plugin Context steps on the dodecahedron and the membrane box, alone and as 4 MPS clients; the charges of the TYK2 ligand are compared between the old and the new image and the difference is in the amendment of FM-D13; a prepared directory written with 8.6.1 loads (else the examples are prepared again) |
| M1 back end (opt-in) | `backend.py`; `run.backend` (default `openmm` at this point) with both presets and `docs/configuration.md` in one commit; `_run`, `check_pbc`, `check_benchmark` go through `build`; records in `run.yaml`, `device.json`, `fmeta status`; restart and walker rules; hash; separation of the records | `run.backend: cupotential` runs production on the soluble and the membrane test system and continues after a stop; the default `pytest` passes without the plugin; default behaviour is unchanged |
| M2 measurement | TYK2 example (88k atoms) and A2A example (221k atoms, membrane): stock OpenMM and each composition (force; force + integrator; force + integrator + barostat; the first two for the membrane) with 1 / 4 / 6 / 8 replicas per GPU (MPS), as short real `fmeta run`s with the bias, on the 8 GPUs side by side | the table is in `bench/private/` and the composition of each class is chosen by the rule; the stock OpenMM values of the new image go to the public record |
| M3 verification | `fmeta check` item `backend`; verification of the integrator (toy system, or NPT); the run items assert the recorded back end. Criteria from measured values, written to FM-D18 | the chosen composition passes the new and every existing item |
| M4 default | `RunConfig.backend = "auto"`, presets, `docs/configuration.md`, README (third-party software; publishing speeds needs NVIDIA's permission), runbook, examples; FM-D18, amendments of FM-D4 / D5 / D13 / D14, the three rule files, `PROGRESS.md`, `docs/roadmap.md` | the GPU gates pass for both back ends on the final code and both records are committed; CI is green; the TYK2 and A2A examples run with the defaults as the README says and `device.json` names cuPotential |

The two parts of the request: "faster" is met when M2 finds a composition above today's best
value and it passes M3; "the default" is M4.

Full GPU gate runs: M0 (required after a change of the lock) and the final code. In between,
`pytest` and the checks that a change touches.

### Stop and report (the only two reasons to stop between milestones)

- A CUDA Context or MPS does not work with the compatibility libraries: the host driver must
  then go to R615 or newer, which is the owner's action.
- No composition is above today's best throughput (soluble, 88k atoms: 1620 ns/day per GPU;
  A2A example: about 506). The default is then not switched and the numbers are reported to
  the owner.

## Invariants

- No criterion is relaxed. The new criteria (accuracy of the back end) are recorded in FM-D18
  and `PROGRESS.md` with their reason.
- CV forces (CVs, walls, bias) stay stock OpenMM forces and never use periodic boundaries.
- The criteria of `periodic_images`, `production`, `walkers`, `replicas_*` and `membrane_run`
  do not change.
- The default `pytest` passes on a CPU-only machine without the plugin (lazy import). The
  `cpu` job of CI runs a newer OpenMM from pip, the image 8.4.0: both pass.
- No force-field parameter is invented; `src/fmeta/` stays target-agnostic.

## Verification

Everything through `scripts/in_docker.sh`.

- Each commit: `ruff check .`, `python scripts/check_progress_size.py`,
  `python scripts/verify_bench_results.py`, `pytest`.
- Unit tests (default selection, no plugin) and the regression each one catches:
  - the resolution table: `auto` choosing cuPotential on Reference, in double precision or
    without the plugin; `cupotential` running on stock OpenMM instead of failing;
  - the conditions: a tolerance of 1e-5, an explicit mesh or a box under 3 cutoffs accepted;
  - the conversion (with a stand-in for the plugin): another force removed, `CMMotionRemover`
    lost, force groups changed;
  - restart: a back end other than the recorded one accepted, or a file changed before the
    refusal (including the rollback of walkers);
  - the hash: `openmm` differing from a literal value of before the key, `cupotential` equal;
  - the harness: the records of two back ends in one file; a performance value in a
    cuPotential record.
- Items of `fmeta check` (both box shapes):
  - `platform`: a real step of the plugin with PME.
  - `backend`: (1) `auto` resolves to cuPotential on CUDA; (2) the plugin force and stock PME
    at 5e-4, each against a CUDA double-precision reference of the same System with a tight
    tolerance, also for coordinates with molecules shifted by lattice vectors; the criterion
    comes from the measurements and is written to FM-D18, bounded above by the numbers of
    `forces` (energy 1e-4, force 2e-3): beyond them cuPotential does not become the default;
    (3) CVs, wall and bias energy agree between the back ends within the criteria of `forces`.
  - integrator and barostat: `toy_free_energy` with the plugin integrator at the same 2.0 /
    3.0 kJ/mol. If massless particles cannot be stepped, the reason goes to FM-D18 and a
    fixed-seed NPT run compares the means of temperature, density and potential energy with
    stock OpenMM (tolerances from the measured seed-to-seed scatter of stock runs).
  - the run items (`production`, `walkers`, `replicas_*`, `membrane_run`) assert that the back
    end in the record of each run is the one asked for and that the restart used the binary
    checkpoint, so that an `auto` that fell back silently cannot pass.
- GPU gates: `scripts/in_docker.sh --gpus 6,7 python scripts/gpu_gates.py --gpus 0,1`, and on
  the final code once more with `--backend openmm`; records and index entries committed in
  `bench/results/`.
- Examples: `examples/tyk2_ejm46` and `examples/a2a_theophylline` run short with the default
  settings; `fmeta status` and `fmeta analyze` work on them.

## Risks

- Early-access software: clause 4 of the licence (not tested for production use, may be
  withdrawn) and clause 9 (delete on termination). `run.backend: openmm` stays a working path
  with its own gate record; without the plugin `auto` resolves to stock OpenMM with a reason.
- The step down of OpenMM and the ligand tools: the measurements of FM-D4 are from 8.6.1; if
  the ligand charges change, results of the new image are another parameterisation than the
  benchmark reports made so far.
- Compatibility libraries with MPS: the daemon uses the host's libraries (R595), a client the
  ones of CUDA 13.4. The libraries cannot be used with drivers R620 and newer (and are not
  needed from R615 on).
- No gate measures dG0 on a real system (replicas scatter by tens of kJ/mol). The evidence for
  the back end is the accuracy of forces and energies, the check of the integrator and the
  existing gates with unchanged criteria.
- The toy gate misses about 1 time in 30 by scatter (FM-D12); the new image draws other random
  numbers.
- Clause 2.12 also names "regression … data relating to the Software". Accuracy values stay in
  the public record as decided; the switch that removes them sits with the one for speeds.
- The image grows by about 1.2 GB (plugin 0.6, compatibility libraries 0.5): disk and time of
  `image.yml`.
- fmeta has no licence file of its own; the notices in the image cover third-party software
  only.
