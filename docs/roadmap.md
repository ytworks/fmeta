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

## Next

- **R1 — First GPU gate record.** Run `scripts/gpu_gates.py` on a GPU machine with the conda
  environment; commit the evidence. Closes: GPU platform, OpenFF / GAFF route.
- **R2 — Real-system validation.** Trypsin-benzamidine (`examples/trypsin_benzamidine`): five
  replicas to convergence, compared with experiment and published funnel-metadynamics values.
  Reported, not gated.
- **R3 — Membrane validation.** A GPCR with a weak ligand (`examples/gpcr_membrane`): funnel
  placement and membrane-plane estimate on a real structure against its OPM orientation; then a
  production run.
- **R4 — Throughput.** Measure the bias overhead on the GPU (`benchmark` gate item); if the
  spline refit at every hill costs more than a few percent, reduce it (coarser grid in `s`,
  or a different table form).
- **R5 — Ligand charge states.** Choose the protonation state from pKa rather than taking the
  SDF as drawn.
- **R6 — More membrane compositions.** Mixed bilayers and cholesterol through custom patches.

## Not planned

Input from other MD programs (owner decision, 2026-11-15). A PLUMED back end.
