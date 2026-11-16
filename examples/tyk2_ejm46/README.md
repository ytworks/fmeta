# TYK2 kinase domain with ejm_46 (PDB entry 4GIH)

A soluble kinase with a tight neutral inhibitor: the JH1 (kinase) domain of human TYK2 with
2,6-dichloro-N-{2-[(cyclopropylcarbonyl)amino]pyridin-4-yl}benzamide (ligand 0X5 of the entry,
23 heavy atoms, compound 46 of the paper, `ejm_46` in the free-energy benchmark sets). The
same ligand is part of the TYK2 benchmark (`examples/benchmarks/tyk2`).

| Quantity | Value | Source |
|---|---|---|
| Experimental Ki | 4.8 nM | Liang et al., Eur. J. Med. Chem. 67, 175 (2013), doi:10.1016/j.ejmech.2013.03.070, as listed for the entry by BindingDB |
| Experimental binding free energy | −11.3 kcal/mol | RT ln Ki (−11.35 at 298.15 K; the benchmark tables list −11.31) |
| fmeta | not available yet | see "Result" below |

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

Not available yet. The run with the official defaults (5 replicas of 40 ns) was started on
2026-11-16 (UTC) on commit 47e84f7 in the official image, the five replicas on one GPU with
NVIDIA MPS. This section will hold the standard binding free energy with its 95% interval over
the replicas, the rebinding counts and the diagnostics, whatever they turn out to be.

What to expect: a replica of 40 ns does not converge on its own; the five together are the
measurement (FM-D9, second amendment, `docs/design/engine.md`). For a ligand that binds this
tightly the replicas may have few rebinding events and differ widely, and the interval then
says so.
