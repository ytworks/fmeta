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
- **R9 — Official presets** (2026-11-16, FM-D16). The official defaults are the defaults of the
  code; the presets `soluble` and `membrane` of the package show every key with its default and
  unit (`fmeta init --preset`); a configuration may hold only its difference from a preset;
  `docs/configuration.md` gives the origin of every value. Tests hold the three together. The
  defaults that differ from the literature are listed there and are the questions of the scan
  on trypsin-benzamidine (R2). GPU gates on this code: not run yet.
- **R8 — Real inputs, membrane part** (2026-11-16, FM-D15). `fmeta split --drop` removes a fusion
  partner before anything is derived; a component kept with `system.keep_hetero` is part of the
  receptor (restrained, an obstacle for the funnel, in the box rule); an orientation that could
  not be checked is a warning; a membrane box is built 8% above the box rule. GPU gates on this
  code: not run yet.
- **R8 — Real inputs, protein part** (2026-11-16, FM-D15). A chain break (C–N above 0.2 nm,
  with or without `TER`) is capped with NME and ACE and no loop is built; caps of the input are
  kept; protonation states are read from the hydrogens or the residue names of the input and
  carried through the rebuild of the hydrogens; a kept component that follows the chain without
  `TER` gets its own chain; `prepared/report.json` lists gaps, caps, states, disulfides and
  alternate locations. Built and equilibrated on the GPU from 3PTB, 4GIH and the prepared TYK2
  protein of the Schrödinger set. GPU gates on this code: not run yet.

## Next

Plan for R2, R3 and R7–R10: `docs/plans/2026-11-15-gpu-validation-modes-benchmarks.md`.

- **R2 — Real-system validation.** Trypsin-benzamidine (`examples/trypsin_benzamidine`): five
  replicas to convergence, compared with experiment and published funnel-metadynamics values.
  Reported, not gated.
- **R3 — Membrane validation.** A GPCR with a weak ligand (adenosine A2A with theophylline,
  5MZJ, `examples/a2a_theophylline`). Done on the OPM file: the build from the input as
  `fmeta split` writes it (221110 atoms; the gap LEU208–GLU219 capped, the sodium ion kept), the
  funnel into the extracellular water, 1 ns of equilibration, and the membrane-plane estimate
  against the OPM orientation (14° and 0.2 nm off). Open: 10 ns of equilibration (running), the
  production run.
- **R5 — Ligand charge states.** Choose the protonation state from pKa rather than taking the
  SDF as drawn.
- **R6 — More membrane compositions.** Mixed bilayers and cholesterol through custom patches.
- **R10 — Examples and benchmarks.** A kinase (TYK2, 4GIH) and a GPCR (A2A, 5MZJ) example with
  experimental values. In (2026-11-16, FM-D17): `fmeta benchmark prepare | run | status | report`
  and the sets `examples/benchmarks/tyk2` (10 ligands) and `a2a` (8 ligands) with experimental
  and FEP+ values; a report goes to `bench/reports/` and holds no verdict. Open: the examples'
  results, the pilot runs (3 ligands per set), the whole sets, the reports.

## Not planned

Input from other MD programs (owner decision, 2026-11-15). A PLUMED back end.
