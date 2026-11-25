# Trypsin-benzamidine (PDB entry 3PTB)

The classic test case of funnel metadynamics: a small charged ligand (benzamidinium, +1 e) in
the S1 pocket of bovine beta-trypsin. No membrane.

| Quantity | Value | Source |
|---|---|---|
| Experimental binding free energy | −6.4 to −7.3 kcal/mol | the measurements quoted by Limongelli et al. (their references 40 and 41) |
| Funnel metadynamics, published | −8.5 ± 0.7 kcal/mol | Limongelli, Bonomi, Parrinello, PNAS 110, 6358 (2013), doi:10.1073/pnas.1303186110 |
| fmeta, official defaults (5 replicas of 40 ns), two runs | −6.8 kcal/mol, 95% interval −16.0 to 2.5; −10.5 kcal/mol, 95% interval −19.2 to −1.8 | see "Result" below |

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

In the official image the production run is made with NVIDIA's cuPotential plugin of OpenMM:
`run.backend` is `auto` by default (FM-D18, `docs/design/engine.md`). `fmeta status` names the
back end of every replica; `run: {backend: openmm}` in `fmeta.yaml` gives OpenMM's own forces
and integrator. Every number in this file was measured with OpenMM's own forces and integrator,
before `run.backend` existed.

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

With the defaults the five replicas give a value with a wide interval. A replica of 40 ns does
not converge on its own: the five together are the measurement, and the interval is what they
say (FM-D9, second amendment, `docs/design/engine.md`). The value of a replica is the mean over
the second half of the run of dG(t) from the accumulated bias, with a block error (the published
procedure; FM-D9, third amendment). The numbers are reported as measured; the mean without its
interval is not a binding free energy to quote.

Two runs have the values of the defaults. The first is the scan run `sig2` of the section
below (commit f2ea653): five replicas at the same time on one GPU with NVIDIA MPS, from the
equilibrated system of the first run of this example. The second follows the steps above on the
commit of the defaults (47e84f7) with one replica after another on one GPU
(`--mode sequential`), on 2026-11-16 (UTC). Both were analysed with the code of commit c59ec99.

| Quantity | Scan run `sig2` (5 replicas at the same time on one GPU with MPS) | Run by the steps above (`--mode sequential`, one GPU) |
|---|---|---|
| dG0 | −28.28 kJ/mol (−6.76 kcal/mol) | −44.04 kJ/mol (−10.53 kcal/mol) |
| 95% interval of the five replicas | −66.93 to 10.37 kJ/mol (standard error 13.92) | −80.51 to −7.57 kJ/mol (standard error 13.14) |
| Each replica, with its block error (kJ/mol) | −78.17 ± 16.18, −4.60 ± 1.06, −39.18 ± 2.57, −6.60 ± 1.61, −12.87 ± 2.21 | −60.16 ± 3.60, −39.03 ± 2.87, −10.94 ± 2.93, −85.27 ± 3.57, −24.80 ± 4.81 |
| From the samples after the first 2 ns (cross-check, kJ/mol) | −87.04, −1.73, −44.29, −3.42, −9.13 | −36.48, −43.22, −5.18, −75.35, −20.89 |
| Rebinding events per replica | 0, 3, 5, 5, 5 | 2, 3, 6, 3, 4 |
| Replicas with a note in the diagnostics | 3 of 5 | 5 of 5 |

What the diagnostics of `fmeta analyze` say about the scan run:

- Replica 0 left the site once, at 27.5 ns, and did not come back (`COLVAR`). dG(t) from the
  bias changed by +98.8 kJ/mol over the last half of the run: the bias was filling the site for
  most of it. Its samples after the first 2 ns give a value 8.9 kJ/mol lower, and those of the
  second half alone one 24.4 kJ/mol higher.
- For replica 1, dG(t) from the bias changed by +6.5 kJ/mol over the last half of the run, and
  the samples of the second half give a value 4.0 kJ/mol higher.
- For replica 2, dG(t) from the bias changed by +15.3 kJ/mol over the last half of the run; the
  samples after the first 2 ns give a value 5.1 kJ/mol lower and those of the second half one
  33.1 kJ/mol higher.

And about the sequential run:

- dG(t) from the bias changed over the last half of the run by +21.6, +18.1, +14.8 and
  +28.3 kJ/mol for replicas 0, 1, 2 and 4.
- The samples after the first 2 ns differ from dG by +23.7, −4.2, +5.8 and +9.9 kJ/mol for
  replicas 0, 1, 2 and 3; those of the second half alone by +39.9 and +8.7 kJ/mol for replicas
  1 and 4.
- Replica 0 has 2 rebinding events, fewer than the 3 of the check.

The two runs of the same settings differ by 15.8 kJ/mol, which is within their standard errors.
The ten replicas together give −36.16 kJ/mol with a standard error of 9.40 and a 95% interval
from −57.42 to −14.90 kJ/mol. The experimental range is −26.8 to −30.5 kJ/mol. The replicas of
one run differ by 74 kJ/mol, so a mean near or away from that range says nothing about agreement.
The block error of a replica (1.1 to 16.2 kJ/mol) is far below that difference: it says how far
a value moved within the second half of its own run, not how far the run is from the answer.

The sequential run is also the record of that mode on a real system. The steps above, from the
download to `fmeta equilibrate`, took 6 minutes. The five replicas ran one after another on one
GPU at 703 to 725 ns/day (the first replica at 555 to 593 ns/day until the restart with
`OMP_NUM_THREADS=1`, see `docs/gpu-runbook.md`). `docker stop` was tried twice during the run.
The first stop returned in 0.4 s with exit code 1 and the message "stopped (SIGTERM) before
rep1, rep2, rep3, rep4, rep0 finished; run the same command again to continue". The same command
continued replica 0 from its checkpoint at 2.5 ns: the 2500 hills before it were identical to
the file saved at the stop, with no gap and no duplicate after it. At the end every replica has
40000 hills, and the run ended with exit code 0.

The cross-check from the samples of replica 0 of the scan run and of replica 3 of the
sequential run was missing before commit c59ec99: the weighted histogram lost the bins of the
unbound region, whose weights are more than 16 digits below those of the site. The values from
the bias were not affected.

The calcium ion is removed by default. Keeping it (`system.keep_hetero`) is a variation that
has not been run.

## How the run length and the bias were chosen

The defaults before 2026-11-16 were 500 ns per replica, taken from the literature that was
read, with bias factor 10, hills of 1.5 kJ/mol and widths of 0.025 / 0.05 nm. On 2026-11-16 the
owner decided that 500 ns is too long, and then that 100 to 200 ns is too long as well: 20 ns
first, at most 40 ns. Two rules that this file held for the longer lengths (the shortest of
100, 150 and 200 ns; a scan of `run.pace_ps`, `metad.sigma_rho` and the time step at that
length) were withdrawn with those decisions before either gave a value. Of their runs only the
one with 2 fs and real masses had started; it was stopped at 15 to 18 ns and not analysed.

On the same day the owner asked that the start of a run, where the bias changes fastest, is not
used. The headline of `fmeta analyze` then was the estimate from the samples after the first
2 ns, with the mean of dG(t) from the bias as a cross-check (FM-D9, first amendment,
`docs/design/engine.md`). The bias was chosen with that analysis, and every number below is from
it. Since the third amendment of FM-D9 the two have changed places: the Result section above is
from the analysis with the headline from the bias.

**The former default bias.** The run of five replicas by the steps above (commit 8d1f10a, five
GPUs at the same time) was stopped at 75 to 81 ns. Cut at a length and analysed:

| Length (ns) | dG0 (kJ/mol) | 95% interval (kJ/mol) | Rebinding events per replica | Replicas that pass |
|---|---|---|---|---|
| 10 | −66.68 | −87.97 to −45.39 | 1, 1, 1, 1, 0 | 0 of 5 |
| 20 | −48.60 | −66.09 to −31.12 | 2, 4, 1, 2, 2 | 0 of 5 |
| 30 | −36.47 | −54.50 to −18.45 | 2, 5, 3, 2, 3 | 0 of 5 |
| 40 | −27.95 | −44.43 to −11.47 | 4, 5, 3, 4, 3 | 0 of 5 |
| 60 | −18.42 | −30.49 to −6.36 | 7, 6, 4, 6, 4 | 0 of 5 |
| 75 | −14.30 | −25.06 to −3.54 | 9, 10, 5, 8, 5 | 1 of 5 |

The mean rises with the length over the whole range. With that bias the ligand stays in the
site for 4.5 to 10 ns while the bias fills, so most of what the samples after 2 ns see at short
lengths is the site.

**The scan.** Five faster bias settings, five replicas each, from the equilibrated system of
that run (commit f2ea653, one setting per GPU, the five replicas sharing it with NVIDIA MPS).
All 25 replicas ran to 40 ns and were analysed as they stood at 10, 15, 20, 30 and 40 ns.

| Setting | `metad.bias_factor` | `metad.height` (kJ/mol) | `metad.sigma_s` / `sigma_rho` (nm) |
|---|---|---|---|
| `bf20` | 20 | 1.5 | 0.025 / 0.05 |
| `h3` | 10 | 3.0 | 0.025 / 0.05 |
| `sig2` | 10 | 1.5 | 0.05 / 0.10 |
| `bf20_h3` | 20 | 3.0 | 0.025 / 0.05 |
| `bf20_h3_sig2` | 20 | 3.0 | 0.05 / 0.10 |

The rule, written before a scan run started (FM-D16, second amendment,
`docs/design/engine.md`). It counts replicas that passed the convergence check the analysis had
at that time. The analysis has no such verdict any more (see "How the result is read" below);
the rule and its counts stay here as the record of how the defaults were chosen. The default
bias still comes from it; the length of 40 ns and "Nothing runs longer" were replaced on
2026-11-25 by the owner's 100 ns (FM-D16, fourth amendment):

- The lengths are taken in the order 10, 15, 20, 30, 40 ns. The first length at which a setting
  has all five replicas without a convergence problem becomes `run.length_ns`, and that setting
  the default bias; of several at the same length, the one with the smallest standard error of
  the five.
- If no setting has five at 40 ns, the setting with the most passing replicas at 40 ns and
  40 ns become the defaults, and the count is stated. Nothing runs longer.
- Only the convergence diagnostics on this system decide. The distance to experiment is
  reported and decides nothing; no criterion changes; no benchmark is tuned.

What the scan gave: dG0 in kJ/mol, the mean of the five replicas with its standard error in
brackets. No replica of any setting passes at 10, 15, 20 or 30 ns.

| Setting | 10 ns | 15 ns | 20 ns | 30 ns | 40 ns | Replicas that pass at 40 ns |
|---|---|---|---|---|---|---|
| `bf20` | −23.14 (11.19) | −19.35 (8.34) | −12.98 (5.64) | −10.84 (5.00) | −8.07 (5.14) | 1 of 5 |
| `h3` | −33.08 (16.15) | −26.19 (13.31) | −22.81 (11.86) | −16.17 (8.88) | −12.68 (6.96) | 0 of 5 |
| `sig2` | −58.64 (25.79) | −52.73 (22.63) | −45.57 (21.95) | −36.72 (21.22) | −27.35 (14.88) | 2 of 5 |
| `bf20_h3` | −7.11 (4.47) | −14.80 (10.62) | −16.72 (10.49) | −14.35 (8.78) | −15.67 (11.05) | 0 of 5 |
| `bf20_h3_sig2` | −5.44 (3.05) | −5.91 (1.71) | −5.85 (3.80) | −5.95 (4.14) | −2.89 (1.42) | 0 of 5 |
| former default | −66.68 (7.67) | not cut | −48.60 (6.30) | −36.47 (6.49) | −27.95 (5.94) | 0 of 5 |

By the second point of the rule the defaults are `sig2` (hill widths 0.05 / 0.1 nm; bias factor
10 and hills of 1.5 kJ/mol stay) and 40 ns per replica: 2 of 5 replicas pass at 40 ns. With the
analysis before the first amendment of FM-D9 the choice is the same (`sig2` 2 of 5, every other setting
0 of 5).

What the count does not say:

- `sig2` has the widest scatter between replicas of all settings, and one of its replicas left
  the site only once, at 27.5 ns, and did not come back (corrected on 2026-11-16: this line said
  that it never left). The rule counts replicas that pass and was applied as written.
- A run can be extended exactly (`fmeta run runs/tb --length-ns 100`); how long this system
  needs for a narrow interval was not measured. The limit of 40 ns that held then is removed
  and the default is 100 ns (owner, 2026-11-25; FM-D16, fourth amendment).
- The faster settings (`bf20_h3`, `bf20_h3_sig2`) push the ligand out early. Their samples then
  rarely visit the site and the estimate is shallow: at 40 ns 3 and 4 of their five replicas are
  within 1 kT of the empty funnel (2, 2 and 1 for `bf20`, `h3` and `sig2`).

## How the result is read (since 2026-11-16)

The owner's instruction of 2026-11-16: a replica of this length is not a simulation that
converges. The analysis therefore gives no verdict. The set of replicas is the measurement, a
value with an interval, and the checks of a replica are diagnostics (FM-D9, second amendment).
Read that way, the scan says:

- The value depends on the bias setting. At 40 ns the six settings give −2.9 to −28.0 kJ/mol
  (the last column of values in the table above), a spread of 25.1 kJ/mol, with the estimate
  from the samples, the headline when the scan was read. With the headline from the bias (FM-D9,
  third amendment) they give −14.9 to −29.1 kJ/mol, a spread of 14.1.
- Another estimator from the samples does not remove that. Two estimators that use only local
  equilibrium under the bias of the moment (mean-force integration, per replica and pooled over
  the replicas) were compared with the estimate from the samples by a rule written first. All
  three are within 0.80 kJ/mol of the exact value on the toy system. On this scan their spread
  between the settings is 32.8 and 42.0 kJ/mol, more than the 25.1 of the estimate from the
  samples, which therefore stayed (FM-D16, third amendment, with the table).
- The reason is in the data. The replicas fall in two groups, about 0 to −9 and −30 to
  −78 kJ/mol, with 0 to 17 rebinding events each in 40 ns: too few crossings to average the
  two. That is a limit of the sampling at this length.
- The defaults stay. No criterion that leaves experiment out makes another setting better:
  the smallest interval belongs to `bf20_h3_sig2`, whose value is within 1 kT of the empty
  funnel and would report this system as no binding.
- The roadmap item R2 now asks for the value with its interval over the replicas, compared
  with experiment and the published value. For the defaults that is the table under "Result".

Not scanned: the deposition interval, the time step, the wall force constants, the force fields
and the shape of the funnel. They are listed with their sources in `docs/configuration.md`.
