# TYK2 kinase domain with ejm_46 (PDB entry 4GIH)

A soluble kinase with a tight neutral inhibitor: the JH1 (kinase) domain of human TYK2 with
2,6-dichloro-N-{2-[(cyclopropylcarbonyl)amino]pyridin-4-yl}benzamide (ligand 0X5 of the entry,
23 heavy atoms, compound 46 of the paper, `ejm_46` in the free-energy benchmark sets). The
same ligand is part of the TYK2 benchmark (`examples/benchmarks/tyk2`).

| Quantity | Value | Source |
|---|---|---|
| Experimental Ki | 4.8 nM | Liang et al., Eur. J. Med. Chem. 67, 175 (2013), doi:10.1016/j.ejmech.2013.03.070, as listed for the entry by BindingDB |
| Experimental binding free energy | −11.3 kcal/mol | RT ln Ki (−11.35 at 298.15 K; the benchmark tables list −11.31) |
| fmeta | −3.55 kcal/mol, 95% interval −6.36 to −0.74 | 5 replicas of 40 ns with the defaults; see "Result" below |

The eight-GPU study uses the same production and analysis protocol for this kinase and
the A2A example. Reweighting PR #4 at 80 ns per walker gave **−10.55 kcal/mol**
(95% interval −17.02 to −4.08). The 300 K reference is −11.42 kcal/mol, giving a signed error
of +0.87 kcal/mol. The interval half-width of 6.47 kcal/mol exceeds the 0.96 kcal/mol target. The four groups
had 2, 0, 1 and 1 contact-confirmed returns. The frozen region definition for this preparation
is `paper-regions.json`. [Comparison and reproduction commands](../../docs/plans/2026-11-18-paper-binding.md)

The crystal structure is 2.0 Å, one chain, residues 890 to 1177 of the kinase domain with the
engineered mutations C936A, Q969A, E971A, K972A, D1023N and C1142A. The missing residues are at
the two ends only, so the chain has no break. The mutations are kept as they are in the entry.
Which construct the Ki was measured on has not been checked here.

## Inputs

The structure files are not part of the repository. Everything runs in the official image
(`scripts/in_docker.sh`, from the root of the repository):

```bash
mkdir -p runs/tyk2_dl runs/tyk2_in
scripts/in_docker.sh python -I -c "import urllib.request; urllib.request.urlretrieve('https://files.rcsb.org/download/4GIH.pdb', '/work/runs/tyk2_dl/4gih.pdb')"
scripts/in_docker.sh fmeta split runs/tyk2_dl/4gih.pdb --resname 0X5 --smiles "Clc1cccc(Cl)c1C(=O)Nc2ccnc(NC(=O)C3CC3)c2" -o runs/tyk2_in
cp runs/tyk2_dl/4gih.pdb runs/tyk2_in/4gih.pdb
mv runs/tyk2_in/ligand.sdf runs/tyk2_in/ejm_46.sdf
cp examples/tyk2_ejm46/fmeta.yaml runs/tyk2_in/fmeta.yaml
```

`fmeta split` cuts the ligand out of the entry and gives it bond orders and hydrogens from the
SMILES (the neutral form). The protein file is the entry as downloaded: `fmeta prepare` removes
the waters and the crystal ligand, rebuilds the missing side-chain atom and adds the hydrogens.

## Run

```bash
scripts/in_docker.sh --gpus 0 fmeta prepare runs/tyk2_in/fmeta.yaml -o runs/tyk2
scripts/in_docker.sh fmeta funnel runs/tyk2                # look at runs/tyk2/prepared/funnel.pdb
scripts/in_docker.sh --gpus 0 fmeta equilibrate runs/tyk2
scripts/in_docker.sh --gpus 0,1,2,3,4 --detach --name fmeta-tyk2 fmeta run runs/tyk2 --mode parallel
scripts/in_docker.sh fmeta status runs/tyk2                # time reached, ns/day and GPU per replica
scripts/in_docker.sh fmeta analyze runs/tyk2 --plot        # after the run, or during it
```

`docker stop fmeta-tyk2` stops the run; the same `fmeta run` command continues it from the last
checkpoints. With one GPU use `--gpus 0` and `--mode sequential`: the replicas then run one
after another.

In the official image the production run is made with NVIDIA's cuPotential plugin of OpenMM:
`run.backend` is `auto` by default (FM-D18, `docs/design/engine.md`). `fmeta status` names the
back end of every replica; `run: {backend: openmm}` in `fmeta.yaml` gives OpenMM's own forces
and integrator.

## What the preparation gives (measured on 2026-11-15 UTC, commit 8d1f10a)

- Removed from the entry: 0X5 × 1, HOH × 119. One missing heavy atom rebuilt, 2307 hydrogens
  added at pH 7. Six atoms of the entry have two alternate locations.
- Pose: 22 of the 23 ligand heavy atoms are within 0.45 nm of the protein.
- Funnel: found by the automatic placement (open path to bulk 2.65 nm long, axis within 0.0° of
  the exit direction); lower wall at s = 0.70 nm, upper wall at s = 4.65 nm, width 0.6 nm; the
  bound pose is at s = 1.20 nm. The unbound region is 1.05 nm clear of the solute.
- Box: rhombic dodecahedron with an image distance of 10.92 nm, 88476 atoms (OpenFF ligand
  parameters, 0.15 M NaCl).
- Equilibration: mean funnel wall energy during the free stage 0.000 kJ/mol.

## Result

Measured on 2026-11-16 (UTC): the steps above with the official defaults (5 replicas of 40 ns)
on commit 47e84f7 in the official image. One difference from the `fmeta run` line above: the
five replicas ran at the same time on one GPU with NVIDIA MPS
(`scripts/in_docker.sh --mps --gpus 3 ... fmeta run runs/tyk2 --gpus 0 --per-gpu 5`), at about
300 ns/day each. The numbers are those of `fmeta analyze runs/tyk2 --plot` with the analysis of
the third amendment of FM-D9 (`docs/design/engine.md`): the value of a replica is the mean over
the second half of the run of dG(t) from the accumulated bias, with a block error.

The run was made with OpenMM's own forces and integrator: commit 47e84f7 is from before
`run.backend` existed, and today that is `run.backend: openmm`. A run of the full length with
the cuPotential back end has not been made.

| Quantity | fmeta | Experiment |
|---|---|---|
| dG0 | −14.85 kJ/mol (−3.55 kcal/mol) | −47.3 kJ/mol (−11.31 kcal/mol) |
| 95% interval of the five replicas | −26.60 to −3.11 kJ/mol (standard error 4.23) | |
| Each replica, with its block error | −18.48 ± 2.04, −12.51 ± 3.58, −3.97 ± 2.82, −10.31 ± 3.07, −29.00 ± 7.13 kJ/mol | |
| From the samples after the first 2 ns (cross-check) | −4.65, −9.51, −6.80, 0.58, −29.89 kJ/mol | |
| Rebinding events per replica | 5, 0, 1, 1, 1 | |
| Replicas with a note in the diagnostics | 5 of 5 | |

The value is 32 kJ/mol (7.8 kcal/mol) above the experimental one, and the experimental value is
outside the interval. It is reported as measured: nothing was tuned on this system, and the
five replicas together are the measurement (FM-D9, second amendment).

What the diagnostics of `fmeta analyze` say:

- Four of the five replicas have fewer than 3 rebinding events. In every replica the bias
  pushed the ligand out early (`COLVAR`: s above 3.0 nm for the first time at 1.0 to 5.1 ns);
  in those four it then came back from the unbound region to the site once or not at all.
- In four of the five replicas an estimate from the samples (after the first 2 ns, or of the
  second half) differs from the value by 6.6 to 20.6 kJ/mol.
- In every replica dG(t) from the bias changed by 12 to 43 kJ/mol over the last half of the
  run. The bias fills the free-energy surface for as long as the run lasts, and the difference
  between the site and the unbound region moves with it: from −84 kJ/mol at 2 ns to −14 kJ/mol
  at 40 ns in the mean of the five (`runs/tyk2/summary.png`).

The run can be extended exactly (`fmeta run runs/tyk2 --length-ns 80`); that was not done,
because nothing runs longer than 40 ns by the owner's decision of 2026-11-16.
