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
  on trypsin-benzamidine (R2). All 18 gates passed on 47e84f7, the code with the defaults of
  the scan verdict (`bench/results/2026-11-16-47e84f72b-gpu-gates.json`).
- **R8 — Real inputs, membrane part** (2026-11-16, FM-D15). `fmeta split --drop` removes a fusion
  partner before anything is derived; a component kept with `system.keep_hetero` is part of the
  receptor (restrained, an obstacle for the funnel, in the box rule); an orientation that could
  not be checked is a warning; a membrane box is built 8% above the box rule. Gates passed on
  47e84f7 (same record).
- **R8 — Real inputs, protein part** (2026-11-16, FM-D15). A chain break (C–N above 0.2 nm,
  with or without `TER`) is capped with NME and ACE and no loop is built; caps of the input are
  kept; protonation states are read from the hydrogens or the residue names of the input and
  carried through the rebuild of the hydrogens; a kept component that follows the chain without
  `TER` gets its own chain; `prepared/report.json` lists gaps, caps, states, disulfides and
  alternate locations. Built and equilibrated on the GPU from 3PTB, 4GIH and the prepared TYK2
  protein of the Schrödinger set. Gates passed on 47e84f7 (same record).
- **R2 — Real-system validation** (2026-11-16, as reworded that day). Trypsin-benzamidine
  (`examples/trypsin_benzamidine`): five replicas of 40 ns with the official defaults, their
  value with its interval beside experiment and the published funnel-metadynamics value.
  Reported, not gated. Two runs with the values of the defaults (the value of a replica from
  the bias, FM-D9, third amendment): the scan run −28.3 kJ/mol (95% interval −66.9 to 10.4) and
  the run by the steps of the README on one GPU with `--mode sequential` −44.0 kJ/mol (−80.5
  to −7.6); their ten replicas together −36.2 kJ/mol (−57.4 to −14.9). Experiment: −26.8 to
  −30.5 kJ/mol; published funnel metadynamics: −35.6 ± 2.9 kJ/mol. R2 read "five replicas to
  convergence" before: a replica of 40 ns does not converge on its own, and the set of replicas
  is the measurement (FM-D9, second amendment). The value depends on the bias setting (FM-D16,
  third amendment). Not reached: a value with a narrow interval on a real system; the settings
  for it are deferred by the owner until everything is shown to work.
- **R3 — Membrane validation** (2026-11-16). A GPCR with a weak ligand: adenosine A2A with
  theophylline in POPC from the OPM file of 5MZJ (`examples/a2a_theophylline`). The steps of
  its README ran from the download to dG0 on 47e84f7: the build from the input as `fmeta split`
  writes it (221101 atoms; the gap LEU208–GLU219 capped, the sodium ion kept), the funnel into
  the extracellular water, the equilibration with 2 ns free, five replicas of 40 ns with the
  official defaults. dG0 −18.6 kJ/mol (95% interval −33.3 to −4.0); the measurement on the
  simulated construct is −33.0 kJ/mol, all measurements −33.0 to −27.0. The site is closed for
  a 0.3 nm probe in the crystal structure, and theophylline left and came back along the funnel
  in every replica (2, 7, 1, 3, 9 rebinding events). The membrane-plane estimate is 14° and
  0.2 nm off the OPM orientation. Reported, not gated. Not reached: a narrow interval (as R2).
- **R10 — Examples and benchmarks** (2026-11-16, FM-D17). A kinase example (TYK2 with ejm_46,
  4GIH) and a GPCR example (A2A with theophylline, 5MZJ; R3) with experimental values, each run
  from its README to dG0. `fmeta benchmark prepare | run | status | report` and the sets
  `examples/benchmarks/tyk2` (10 ligands, in water) and `a2a` (8 ligands, in POPC) with
  experimental and FEP+ values; a report goes to `bench/reports/` and holds no verdict. Both
  whole sets ran with the official defaults on 47e84f7: 5 replicas of 40 ns per ligand, 90
  replicas, every one to its end, nothing tuned for a set. Reports:
  `bench/reports/2026-11-16-7386d2881-tyk2` and `-a2a`. Agreement with experiment in kcal/mol,
  fmeta / FEP+: MUE 3.60 / 0.37 (TYK2) and 4.34 / 0.74 (A2A); with the mean offset removed
  1.81 / 0.35 and 3.49 / 0.79; Pearson r 0.15 / 0.96 and 0.57 / 0.79. The 95% intervals are 4.3
  to 16.4 (TYK2) and 7.0 to 28.9 kcal/mol (A2A) wide; the experimental values span 4.3 and 3.2.
  Reported, not gated. Not reached: narrow intervals and the order of the ligands (as R2); in
  the A2A set the ligand did not come back to the site within 40 ns in 25 of the 40 replicas.
  Plan: `docs/plans/2026-11-15-gpu-validation-modes-benchmarks.md`.
- **R11 — The cuPotential back end** (2026-11-17, FM-D18, PR #3). On the owner's request the
  runs use NVIDIA's cuPotential plugin for OpenMM (`nv-openmm-plugin` 0.3.5, early access) by
  default: `run.backend: auto` gives the plugin where its conditions hold (CUDA in single or
  mixed precision, PME as the plugin takes it) and OpenMM's own forces and integrator with a
  recorded reason otherwise; `--backend openmm` stays a complete, gated path. The plugin takes
  the nonbonded force, the integrator and the barostat of a soluble system, and the force and
  the integrator of a membrane system; the CV forces stay OpenMM's. A run directory keeps the
  back end it began with. The image went to OpenMM 8.4.0 and CUDA 13.4 for it and holds the
  plugin, NVIDIA's compatibility libraries and the MPS programs under NVIDIA's licences.
  `fmeta check` has 19 items; `backend` and `membrane_backend` measure the accuracy of both
  back ends on the two test systems. The gates of the final code (1c09775) passed with each
  back end: `bench/results/2026-11-17-1c0977590-cupotential-gpu-gates.json` and
  `bench/results/2026-11-17-1c0977590-gpu-gates.json`. The TYK2 example ran by the steps of its
  README with the default (5 replicas of 1 ns; every `device.json` names cuPotential). Speeds
  measured with the plugin are in no tracked file until NVIDIA permits it (owner, 2026-11-17);
  the parts were chosen from them by the rule of FM-D18. Plan:
  `docs/plans/2026-11-17-cupotential-backend.md`.

## Next

- **R5 — Ligand charge states.** Choose the protonation state from pKa rather than taking the
  SDF as drawn.
- **R6 — More membrane compositions.** Mixed bilayers and cholesterol through custom patches.

## Not planned

Input from other MD programs (owner decision, 2026-11-15). A PLUMED back end.
