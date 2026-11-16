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
  metadynamics has no such shift at all: a report therefore gives every measure a second time
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

## What is not settled

- The temperature of the measurements is not in the source; a report compares the values as
  given, without a correction to 300 K.
- fmeta's reading of the protein file (the `UNK` caps, the histidine states) decides what is
  simulated: check `prepare.log` and `prepared/report.json` of a ligand before a long run.
