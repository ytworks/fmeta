# Roadmap

Product: funnel metadynamics that runs end to end from two files — a protein PDB and a ligand
SDF in its bound pose — to a standard binding free energy with an error estimate, on one to
many GPUs, for soluble and membrane proteins. Decisions: `docs/design/engine.md`.

## Done (2026-11-15, PR #1)

CVs and funnel walls as OpenMM custom forces; own bias engine with clamped lookup, mirrored
hills, HILLS / COLVAR and exact restart; automatic funnel placement and validation; system
building from PDB + SDF with automatic clean-up, in water or in a bilayer built by
`Modeller.addMembrane`; membrane-plane estimate for orientation; staged equilibration;
replicas and shared-bias walkers; analysis (dG0, dG0(t), reweighting, empty-funnel reference,
figures); the `fmeta` command line; `fmeta check` and the GPU gates script; CPU CI.
Verified on CPU only (see `PROGRESS.md`).

## Done since (PR #1)

- **The official image** (2026-11-15, FM-D13). One image built from a lock; everything except
  CI runs in it through `scripts/in_docker.sh`.
- **R4 — Throughput** (2026-11-16, FM-D4 amendment). With the table refit by OpenMM at every
  hill the bias ran at 73.4% of the speed of plain MD on the GPU. With the spline coefficients
  fitted by fmeta it runs at 96–97% (24062 atoms, one hill per ps).
- **R1 — First GPU gate record** (2026-11-16). `scripts/gpu_gates.py` in the official image on
  two GPUs: all 16 gates passed on 2c4ad4e
  (`bench/results/2026-11-15-2c4ad4e50-gpu-gates.json`, dated in UTC). Closes: GPU platform,
  OpenFF / GAFF route. The first GPU runs changed FM-D2, D4, D5, D6, D7 and D11 on the way.
- **R7 — Run modes** (2026-11-16, FM-D14). Replicas side by side on several GPUs or one after
  another on one GPU (`run.mode`, `run.gpus`, `--mode`); one scheduler that starts the replicas
  as child processes, records the assignment in `schedule.jsonl` and stops the children on a
  signal; `fmeta status`. All 18 gates passed on 8d1f10a
  (`bench/results/2026-11-15-8d1f10a89-gpu-gates.json`). `docker stop` of a parallel run on five
  GPUs left no process behind, and the same command continued every replica from its checkpoint
  (trypsin-benzamidine, 2026-11-16).

## Next

Plan for R2, R3 and R7–R10: `docs/plans/2026-11-15-gpu-validation-modes-benchmarks.md`.

- **R2 — Real-system validation.** Trypsin-benzamidine (`examples/trypsin_benzamidine`): five
  replicas to convergence, compared with experiment and published funnel-metadynamics values.
  Reported, not gated.
- **R3 — Membrane validation.** A GPCR with a weak ligand (adenosine A2A with theophylline,
  5MZJ): funnel placement and membrane-plane estimate on a real structure against its OPM
  orientation; then a production run.
- **R5 — Ligand charge states.** Choose the protonation state from pKa rather than taking the
  SDF as drawn.
- **R6 — More membrane compositions.** Mixed bilayers and cholesterol through custom patches.
- **R8 — Real inputs.** Chain breaks capped; caps and protonation states of a prepared
  structure kept; kept non-protein components restrained and seen by the funnel placement.
- **R9 — Official presets.** `soluble` and `membrane` presets that hold every key with its
  default; a configuration reference.
- **R10 — Examples and benchmarks.** A kinase (TYK2, 4GIH) and a GPCR (A2A, 5MZJ) example with
  experimental values; `fmeta benchmark` on a TYK2 and an A2A set that have experimental and
  FEP values; reports in `bench/reports/`.

## Not planned

Input from other MD programs (owner decision, 2026-11-15). A PLUMED back end.
