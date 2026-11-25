# Benchmark set: TYK2 kinase

Ten ligands of the TYK2 series of the "JACS set", a kinase in water (no membrane).

```bash
scripts/in_docker.sh fmeta benchmark prepare examples/benchmarks/tyk2 -o runs/bench-tyk2
scripts/in_docker.sh --gpus 0,1,2,3 --detach --name bench-tyk2 \
    fmeta benchmark run examples/benchmarks/tyk2 -o runs/bench-tyk2 --gpus 0,1,2,3
scripts/in_docker.sh fmeta benchmark status examples/benchmarks/tyk2 -o runs/bench-tyk2
scripts/in_docker.sh fmeta benchmark report examples/benchmarks/tyk2 -o runs/bench-tyk2
```

`prepare` takes the three pilot ligands (`ejm_44`, `ejm_31`, `ejm_46`: the weakest, a middle
and a strong binder); `--ligands all` adds the other seven to the same work directory.

With the defaults a new work directory runs 5 replicas of 100 ns per ligand (FM-D16, fourth
amendment, 2026-11-25). The result below was made with 40 ns, the default of then:
`fmeta benchmark prepare ... --length-ns 40` repeats that plan. A work directory keeps the
length its ligands were prepared with.

## Source and licence

Structures and reference values come from the Schrödinger
[public binding free energy benchmark](https://github.com/schrodinger/public_binding_free_energy_benchmark),
commit `a65d86b782a678c75388566ce00446e45283ee63` (`main` and tag `v2.0` on 2026-11-16),
MIT licence (Copyright (c) 2023 Gregory A. Ross). No file of it is stored here: `set.yaml`
names each file at that commit with its sha256, and `fmeta benchmark prepare` downloads and
verifies them.

| File | Content |
|---|---|
| `fep_benchmark_inputs/structure_inputs/jacs_set/tyk2_protein.pdb` | chain A, residues 890-1177, with hydrogens; from PDB 4GIH |
| `fep_benchmark_inputs/structure_inputs/jacs_set/tyk2_ligands.sdf` | 16 ligands in their bound poses, all neutral |
| `21_4_results/ligand_predictions/jacs_set/tyk2_out.csv` | experimental and FEP+ values of the 16 ligands |

The termini of the protein file are capped, and the caps are written as two `HETATM` residues
named `UNK` with an empty chain (`UNK 890`: an acetyl group; `UNK 1177`: an N-methylamide).

## Reference values (`reference.csv`)

The values are those of `tyk2_out.csv`, rounded to four decimals (the file holds
single-precision numbers); the same experimental values are in the `r_exp_dg` field of the
SDF records.

- **Experiment.** The source gives a free energy per ligand in kcal/mol, not the measured Ki,
  and not the temperature of the measurement: `exp_ki_nm` and `exp_temperature_k` are empty,
  and the value is used as given. `exp_doi` is the reference the benchmark names for the
  series (Wang et al., J. Am. Chem. Soc. 137, 2695 (2015), doi:10.1021/ja512751q); the papers
  of the measurements were not read for this set.
- **FEP.** FEP+ predictions of Ross et al., Commun. Chem. 6, 222 (2023),
  doi:10.1038/s42004-023-01019-9. They are relative free energies (`fep_kind: relative`):
  `21_4_results/README.md` of the repository says "the CSV files contained here have predicted
  _absolute_ binding free energies that are known up to an unknown additive constant. The
  additive constant was set such that the mean of the predicted absolute binding free energies
  equals the mean of experimental binding free energies." The shift was made over all 16
  ligands (means in the file: experiment -9.7462, prediction -9.7456 kcal/mol). On the ten
  ligands of this set the two means differ, and an absolute method such as funnel
  metadynamics has no such shift at all: a report therefore gives the MUE and the RMSE a second time,
  after the mean error on the same ligands is removed, and that is the fair comparison
  with FEP+.
- **Absolute FEP** (`afe_*`): empty. Khalak et al., Chem. Sci. 12, 13958 (2021),
  doi:10.1039/D1SC03472C (CC BY-NC 3.0) calculated absolute binding free energies for TYK2,
  but its per-ligand values could not be read from a primary source while this set was
  written (the article page refused the download), so none is entered.

The whole series as its source reports it: 16 ligands, 24 perturbations, pairwise RMSE 0.69
and MUE 0.55 kcal/mol, R² 0.86, Kendall tau 0.81.

## Configuration

`config` in `set.yaml` is empty: the defaults of fmeta, at 300 K. The standard state of every
value is 1 mol/L.

The back end of the runs is the default as well (`run.backend: auto`, FM-D18 in
`docs/design/engine.md`): in the official image a new work directory runs with NVIDIA's
cuPotential plugin of OpenMM, and a report names the back end of its replicas. The result below
was made with OpenMM's own forces and integrator, before `run.backend` existed. A work
directory of that time has another configuration than the set asks for today, and
`fmeta benchmark` refuses to go on in it; with `config: {run: {backend: openmm}}` in `set.yaml`
the set is the one it was prepared for.

## The protein file as fmeta reads it

Measured on 2026-11-16 with commit eeb5302 (`fmeta benchmark prepare`, then
`fmeta benchmark run --until prepare` for the three pilot ligands; `prepare.log` and
`prepared/report.json` of each ligand).

| Point | Result |
|---|---|
| Caps | the two `UNK` residues are read as the caps they are and kept: acetyl before THR 890, N-methylamide after GLN 1177 |
| Histidines | read from the hydrogens of the file: HIS 907 doubly protonated; HIS 958, 993, 1018, 1052 and 1173 with the hydrogen on ND1; HIS 893, 940, 956, 1015, 1021, 1097 and 1146 with it on NE2 |
| Hydrogens | those of the file are kept; 7 are added |
| Systems | `ejm_44` 86 105 atoms, `ejm_31` 91 416, `ejm_46` 85 854; no error in the validation. `ejm_46` has one warning: the pose is only 0.20 nm inside the funnel wall |

The protein model is therefore that of the FEP+ input; force field, water and ions are fmeta's.

## Result

The whole set with the defaults (`config` empty): 10 ligands, 5 replicas of 40 ns each, every
replica to its end. The simulations ran from commit 47e84f7 on 2026-11-16 (06:41 to 18:54 UTC,
beside the A2A set and the examples on 8 GPUs, several replicas per GPU with NVIDIA MPS).
Report: `bench/reports/2026-11-16-7386d2881-tyk2.md`, written by the code of 7386d28 (the dG0
of a replica from the accumulated bias; FM-D9, third amendment). Nothing was tuned for the set,
and the report holds no verdict. Values in kcal/mol:

| Ligand | dG0 fmeta | 95% interval | Experiment | FEP+ | fmeta − experiment | Rebinding events |
|---|---|---|---|---|---|---|
| ejm_44 | −6.43 | −13.54 to 0.68 | −7.42 | −6.48 | +0.99 | 18 |
| ejm_49 | −6.07 | −10.23 to −1.91 | −7.75 | −8.05 | +1.68 | 15 |
| ejm_43 | −7.01 | −12.98 to −1.04 | −8.26 | −8.30 | +1.25 | 18 |
| ejm_48 | −3.17 | −6.15 to −0.19 | −9.00 | −8.86 | +5.83 | 16 |
| ejm_31 | −7.73 | −15.94 to 0.48 | −9.54 | −9.91 | +1.81 | 17 |
| ejm_42 | −2.87 | −6.34 to 0.60 | −9.78 | −9.95 | +6.91 | 23 |
| ejm_54 | −6.11 | −9.35 to −2.87 | −10.53 | −10.38 | +4.42 | 15 |
| jmc_28 | −7.73 | −10.74 to −4.71 | −10.98 | −10.34 | +3.25 | 6 |
| ejm_46 | −6.10 | −8.23 to −3.96 | −11.31 | −10.82 | +5.21 | 10 |
| jmc_23 | −7.02 | −13.18 to −0.87 | −11.70 | −11.27 | +4.68 | 15 |

Agreement with experiment over the 10 ligands (in brackets the 2.5 to 97.5 percentile of a
bootstrap over the ligands):

| Measure | fmeta | FEP+ |
|---|---|---|
| MUE (kcal/mol) | 3.60 [2.37, 4.82] | 0.37 [0.22, 0.53] |
| RMSE (kcal/mol) | 4.12 [2.83, 5.16] | 0.45 [0.27, 0.61] |
| Mean signed error (kcal/mol) | +3.60 [2.37, 4.82] | +0.19 [−0.05, 0.45] |
| MUE, offset removed (kcal/mol) | 1.81 [1.00, 2.23] | 0.35 [0.19, 0.46] |
| RMSE, offset removed (kcal/mol) | 1.99 [1.27, 2.38] | 0.41 [0.24, 0.51] |
| Pearson r | 0.15 [−0.29, 0.59] | 0.96 [0.93, 0.99] |
| Kendall tau | 0.11 [−0.33, 0.52] | 0.96 [0.78, 1.00] |

What the numbers say:

- Every value of fmeta is above experiment (weaker binding), by 1.0 to 6.9 kcal/mol. The values
  of fmeta lie between −2.9 and −7.7 kcal/mol, the experimental ones between −7.4 and −11.7.
- The experimental value is inside the 95% interval for 5 of the 10 ligands (ejm_44, ejm_49,
  ejm_43, ejm_31, jmc_23). The intervals are 4.3 to 16.4 kcal/mol wide (18 to 69 kJ/mol); the
  experimental values of the set span 4.3 kcal/mol. The order of the ligands is not resolved
  at this length and number of replicas.
- The FEP+ values are relative free energies placed on the mean of experiment by their source,
  so their MUE holds no error of the absolute scale. On the rows with the offset removed and on
  the correlations, which compare like with like, FEP+ is closer to experiment than fmeta.
- The ligand left the site and came back in 47 of the 50 replicas (153 rebinding events in
  all); in the other three it left and did not return within 40 ns.
- ejm_46 is also the example `examples/tyk2_ejm46` (protein model from 4GIH, other seeds):
  −6.10 kcal/mol here (−25.50 kJ/mol, 95% interval −34.45 to −16.56) against −3.55 there
  (−14.85 kJ/mol, −26.60 to −3.11). The two intervals overlap.

## What is not settled

- The temperature of the measurements is not in the source; a report compares the values as
  given, without a correction to 300 K.
- A value with a narrow interval. The settings for accuracy (walkers, run length, bias) are
  not chosen yet: the owner deferred them until every function is shown to work.
