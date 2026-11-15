# Trypsin-benzamidine (PDB entry 3PTB)

The classic test case of funnel metadynamics: a small charged ligand (benzamidinium, +1 e) in
the S1 pocket of bovine beta-trypsin. No membrane.

| Quantity | Value | Source |
|---|---|---|
| Experimental binding free energy | −6.4 to −7.3 kcal/mol | the measurements quoted by Limongelli et al. (their references 40 and 41) |
| Funnel metadynamics, published | −8.5 ± 0.7 kcal/mol | Limongelli, Bonomi, Parrinello, PNAS 110, 6358 (2013), doi:10.1073/pnas.1303186110 |
| fmeta | not available yet | see "Result" below |

## Inputs

The structure files are not part of the repository. Everything runs in the official image
(`scripts/in_docker.sh`, from the root of the repository):

```bash
mkdir -p runs/tb_dl runs/tb_in
scripts/in_docker.sh python -I -c "import urllib.request; urllib.request.urlretrieve('https://files.rcsb.org/download/3PTB.pdb', '/work/runs/tb_dl/3ptb.pdb')"
scripts/in_docker.sh fmeta split runs/tb_dl/3ptb.pdb --resname BEN --smiles "NC(=[NH2+])c1ccccc1" -o runs/tb_in
cp runs/tb_dl/3ptb.pdb runs/tb_in/3ptb.pdb
mv runs/tb_in/ligand.sdf runs/tb_in/benzamidine.sdf
cp examples/trypsin_benzamidine/fmeta.yaml runs/tb_in/fmeta.yaml
```

`fmeta split` cuts the ligand out of the entry and gives it the hydrogens of the charged
(amidinium) form from the SMILES. The protein file is the entry as downloaded: `fmeta prepare`
removes the waters, the calcium ion and the crystal ligand and adds the hydrogens.

## Run

```bash
scripts/in_docker.sh --gpus 0 fmeta prepare runs/tb_in/fmeta.yaml -o runs/tb
scripts/in_docker.sh fmeta funnel runs/tb                  # look at runs/tb/prepared/funnel.pdb
scripts/in_docker.sh --gpus 0 fmeta equilibrate runs/tb
scripts/in_docker.sh --gpus 0,1,2,3,4 --detach --name fmeta-tb fmeta run runs/tb --mode parallel
scripts/in_docker.sh fmeta status runs/tb                  # time reached, ns/day and GPU per replica
scripts/in_docker.sh fmeta analyze runs/tb --plot          # after the run, or during it
```

`docker stop fmeta-tb` stops the run; the same `fmeta run` command continues it from the last
checkpoints. With one GPU use `--gpus 0` and `--mode sequential`.

## What the preparation gives (measured on 2026-11-15 UTC, commit 8d1f10a)

- Removed from the entry: BEN × 1, CA × 1, HOH × 62. 1591 hydrogens added at pH 7. The six
  disulfide bonds of trypsin are in the topology, and the chain has no break.
- Funnel: found by the automatic placement (open path to bulk 2.30 nm long, axis within 0.0° of
  the exit direction); lower wall at s = 0.70 nm, upper wall at s = 4.30 nm, width 0.6 nm.
  Because the ligand is charged, the unbound region is 1.5 nm from the solute instead of 1.0 nm.
- Box: rhombic dodecahedron with an image distance of 10.20 nm, 72434 atoms (OpenFF ligand
  parameters, 0.15 M NaCl).
- Equilibration: mean funnel wall energy during the free stage 0.000 kJ/mol.
- Speed: 660 to 700 ns/day per replica with the bias (RTX PRO 6000 Blackwell, 4 fs, mixed
  precision), five replicas on five GPUs at the same time.

## Result

Not available yet. The run of five replicas × 500 ns with the official defaults was started on
2026-11-15 (UTC) on commit 8d1f10a in the official image; at the measured speed it needs about
18 hours. This section will hold the standard binding free energy with its confidence interval,
the rebinding counts and the convergence verdict, whatever they turn out to be.

The calcium ion is removed by default. Keeping it (`system.keep_hetero`) is a variation that
has not been run.
