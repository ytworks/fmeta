# 100 ns by default; the TYK2 and A2A examples at full length with cuPotential

Status: In progress
Refs: roadmap R11, R3, R10; FM-D16 (amendment of 2026-11-25), FM-D18, FM-D9

## Context

Owner request (2026-11-25): cuPotential by default in the engine, and the TYK2 and A2A
examples run for 100 ns, to see both the speed and the accuracy.

- cuPotential is the default on `main` already (PR #3, c163eb2; FM-D18): `run.backend: auto`
  in the code and both presets. The plugin takes the nonbonded force, the integrator and the
  barostat of a soluble system, and the force and the integrator of a membrane system; the
  equilibration stays on OpenMM's own forces by design. `src/` is the same on c163eb2 as on
  80d7dc7, whose GPU gates passed with both back ends.
- What is missing is a run of full length with the plugin: the examples ran 1 ns per replica
  with it, and every 40 ns value and benchmark report so far is from OpenMM's own forces.

Owner decisions of 2026-11-25:

1. The rule that nothing runs longer than 40 ns (FM-D16, owner 2026-11-16) is removed, and
   `run.length_ns` is 100 by default.
2. The accuracy is judged on the cuPotential runs alone (no OpenMM run of 100 ns beside them),
   5 replicas per example, as the defaults give.
3. All development (code, documents, the procedure of the analysis, the PR) is done in a
   session without the GPUs; the 8-GPU machine only runs, in a separate session on the branch
   of the PR (the runbook below).

The rule on speeds stays (owner 2026-11-17, clause 2.12 of NVIDIA's licence): a speed or a
duration measured with the plugin goes to no tracked file, commit message, PR text or PR
comment. It stays in `bench/private/` of the GPU machine and is given to the owner directly.

## Approach

### A. The default (this branch)

- `run.length_ns` 100 in `src/fmeta/config.py`, both presets and `docs/configuration.md`;
  `tests/test_config.py` already holds the three to one value.
- FM-D16 amendment: the 40 ns cap is gone; 100 ns is the default by the owner's decision. The
  scan of FM-D16 stays as the record of how the bias was chosen.
- The sentences that state the cap or the old default are rewritten (example READMEs and
  their `fmeta.yaml` comments, `README.md`, `docs/gpu-runbook.md`); values measured at 40 ns
  stay as measured. A benchmark work directory keeps the length its ligands were prepared
  with (`benchmark.prepare_set`); a new one gets 100 ns.

### B. Runbook of the GPU session (only runs; on the clean commit named in the PR)

All commands from the root of the checkout. `G…` are free host GPUs (`nvidia-smi`).

1. Image and machine: `scripts/in_docker.sh python -c "import fmeta"` (builds `fmeta:dev`
   from the lock ee32ef34… if it is not there); `docker image inspect fmeta:dev --format
   '{{.Id}}'` (the gates of 80d7dc7 ran in 648480d3…; another ID from the same lock is
   noted, not an error); `nproc`.
2. GPU gates, once per back end, on two free GPUs:
   ```bash
   scripts/in_docker.sh --gpus G1,G2 python scripts/gpu_gates.py --gpus 0,1
   scripts/in_docker.sh --gpus G1,G2 python scripts/gpu_gates.py --gpus 0,1 --backend openmm
   ```
   At the same time, on two other GPUs, the two examples by their READMEs with the defaults
   (no `--length-ns`: the default is now 100 ns):
   ```bash
   # TYK2 (examples/tyk2_ejm46/README.md)
   mkdir -p runs/tyk2_100_dl runs/tyk2_100_in
   scripts/in_docker.sh python -I -c "import urllib.request; urllib.request.urlretrieve('https://files.rcsb.org/download/4GIH.pdb', '/work/runs/tyk2_100_dl/4gih.pdb')"
   scripts/in_docker.sh fmeta split runs/tyk2_100_dl/4gih.pdb --resname 0X5 --smiles "Clc1cccc(Cl)c1C(=O)Nc2ccnc(NC(=O)C3CC3)c2" -o runs/tyk2_100_in
   cp runs/tyk2_100_dl/4gih.pdb runs/tyk2_100_in/4gih.pdb
   mv runs/tyk2_100_in/ligand.sdf runs/tyk2_100_in/ejm_46.sdf
   cp examples/tyk2_ejm46/fmeta.yaml runs/tyk2_100_in/fmeta.yaml
   scripts/in_docker.sh --gpus GA fmeta prepare runs/tyk2_100_in/fmeta.yaml -o runs/tyk2_100
   scripts/in_docker.sh fmeta funnel runs/tyk2_100
   scripts/in_docker.sh --gpus GA fmeta equilibrate runs/tyk2_100

   # A2A (examples/a2a_theophylline/README.md)
   mkdir -p runs/a2a_100
   cp examples/a2a_theophylline/fmeta.yaml runs/a2a_100/
   scripts/in_docker.sh python examples/a2a_theophylline/fetch.py runs/a2a_100/download
   scripts/in_docker.sh fmeta split runs/a2a_100/download/5mzj_opm.pdb --resname TEP --smiles "Cn1c(=O)c2[nH]cnc2n(C)c1=O" -o runs/a2a_100/inputs
   scripts/in_docker.sh --gpus GB fmeta prepare runs/a2a_100/fmeta.yaml -o runs/a2a_100/work
   scripts/in_docker.sh fmeta funnel runs/a2a_100/work
   scripts/in_docker.sh --gpus GB fmeta equilibrate runs/a2a_100/work --device 0
   ```
   Check: both gate runs end with all items passed; `runs/tyk2_100/config.yaml` and
   `runs/a2a_100/work/config.yaml` have `length_ns: 100.0` and `backend: auto`; about 88.5k and
   221k atoms; the funnel validation gives no error.
3. Speed, before production and with every GPU idle (gates and builds finished):
   `scripts/mps.sh start`, then copies of the equilibrated work directories, made before any
   replica exists (a run reads only `prepared/` and `equil/state.xml` of its work directory):
   ```bash
   for x in tyk2:runs/tyk2_100 a2a:runs/a2a_100/work; do
     for b in openmm auto; do for n in 1 5; do
       mkdir -p runs/speed && cp -a "${x#*:}" "runs/speed/${x%%:*}_${b}_${n}"
     done; done
   done
   ```
   One copy per GPU, all eight at once, 2 ns each (`--backend openmm` only for the `openmm`
   copies; the `auto` copies take the default):
   ```bash
   scripts/in_docker.sh --gpus Gi --detach --name sp-tyk2-auto-1 fmeta run runs/speed/tyk2_auto_1 --replica 0 --device 0 --length-ns 2
   scripts/in_docker.sh --mps --gpus Gj --detach --name sp-tyk2-auto-5 fmeta run runs/speed/tyk2_auto_5 --gpus 0 --per-gpu 5 --length-ns 2
   # ... the same for tyk2_openmm_{1,5}, a2a_auto_{1,5}, a2a_openmm_{1,5}
   ```
   Read ns/day per replica from the last line of each `rep*/run.log` and `fmeta status`, and
   the total per GPU; check that every `auto` copy's `device.json` names cuPotential. Write
   the table to `bench/private/2026-11-25-100ns-speed.md` (untracked). The OpenMM values may
   be published; the cuPotential values and any ratio may not.
4. Production, 5 × 100 ns with the default back end, in the layout of the 40 ns runs of both
   examples (five replicas on one GPU with MPS):
   ```bash
   scripts/in_docker.sh --mps --gpus GA --detach --name fmeta-tyk2-100 fmeta run runs/tyk2_100 --gpus 0 --per-gpu 5
   scripts/in_docker.sh --mps --gpus GB --detach --name fmeta-a2a-100 fmeta run runs/a2a_100/work --gpus 0 --per-gpu 5
   ```
   After the first checkpoint: every `rep*/device.json` names `cupotential` with the parts
   force, integrator and barostat (TYK2) and force and integrator (A2A), and has no
   `backend_reason`. Follow with `fmeta status` and `docker logs`. A run that stops (an MPS
   server that fails) is continued with the same command; note it and `restored_from`. A
   replica that fails again: stop and report to the owner, change nothing. At the end:
   `docker inspect -f '{{.State.ExitCode}}' fmeta-tyk2-100` (and `fmeta-a2a-100`),
   `finish` and return code 0 of every replica in `schedule.jsonl`, 100000 rows in every
   HILLS. `scripts/mps.sh stop` when nothing runs.
5. Analysis at 100 ns and, from the same runs, at 40, 60 and 80 ns. `fmeta analyze` reads only
   `run.yaml`, `HILLS` and `COLVAR` of a replica, so a copy cut at time T is analysed as a run
   of length T would be (`truncate_data_file`, the function of the restart rollback):
   ```bash
   scripts/in_docker.sh fmeta analyze runs/tyk2_100 --plot
   scripts/in_docker.sh fmeta analyze runs/a2a_100/work --plot
   for x in runs/tyk2_100 runs/a2a_100/work; do for T in 40 60 80; do for r in "$x"/rep*; do
     mkdir -p "${x}_at${T}/${r##*/}" && cp "$r"/run.yaml "$r"/HILLS "$r"/COLVAR "${x}_at${T}/${r##*/}/"
   done; done; done
   scripts/in_docker.sh python -c "
   import glob
   from fmeta.engine.hills import truncate_data_file
   for x in ('runs/tyk2_100', 'runs/a2a_100/work'):
       for t in (40, 60, 80):
           for d in sorted(glob.glob(f'{x}_at{t}/rep*')):
               print(d, [truncate_data_file(f'{d}/{f}', max_time=1000.0 * t) for f in ('HILLS', 'COLVAR')])
   "
   for x in runs/tyk2_100 runs/a2a_100/work; do for T in 40 60 80; do
     scripts/in_docker.sh fmeta analyze "${x}_at${T}"
   done; done
   ```
   Every cut HILLS has 1000 × T rows.
6. Results back to the branch:
   - commit the two gate records and `bench/results/index.json` (nothing else) and push;
   - one PR comment with the accuracy and the completion conditions only: the eight outputs
     of `fmeta analyze` (both examples at 100, 40, 60, 80 ns), the two `summary.json` of
     100 ns, per replica the `backend`, `backend_parts`, `backend_reason` and
     `restored_from` of `device.json`, the HILLS row counts, the return codes of
     `schedule.jsonl`, the two container exit codes, the image ID and the commit. Before
     posting: `grep -n -i -E "ns/day|hour|[0-9] ?h\b|min\b|second"` on the text finds nothing
     of the plugin;
   - the speed table to the owner directly, never to GitHub.

The layout of production repeats the 40 ns runs of both examples (OpenMM, 47e84f7: about 300
and 106–107 ns/day per replica), and step 3 gives OpenMM and cuPotential side by side in the
same layout on the same day. With OpenMM's speeds of the image (FM-D14) this layout would take
about 8 h (TYK2) and 23 h (A2A).

### C. Back in the development session

- Pull the gate records; read the comment (`gh pr view --comments`).
- Accuracy, as numbers without a verdict (FM-D9): (1) the 100 ns value and its 95% interval
  beside experiment (TYK2 with ejm_46 −47.3 kJ/mol; A2A with theophylline −33.0, all
  measurements −33.0 to −27.0); (2) the value at 40 ns of the same runs beside the 40 ns runs
  with OpenMM (TYK2 −14.85, 95% −26.60 to −3.11, standard error 4.23; A2A −18.64, −33.31 to
  −3.97, 5.28): the difference with √(SE₁² + SE₂²) and a Welch interval; the OpenMM runs were
  made in the former image (OpenMM 8.6.1) from another equilibration, so the difference is not
  that of the back end alone; (3) 40 → 100 ns: width of the interval, rebinding events, the
  notes (drift, the cross-checks from the samples, no binding).
- Documents (no plugin speed, no duration): a 100 ns section in both example READMEs,
  `README.md`, `docs/roadmap.md`, FM-D18 (dG0 of a real system at full length, not a gate),
  `PROGRESS.md`. Then mark the PR ready.

## Invariants

- No criterion changes; no code under `src/fmeta/` changes but the default of
  `run.length_ns`. The GPU gates run on the commit with that change, once per back end.
- The analysis of a cut copy uses the code of `fmeta analyze` unchanged.

## Verification

- Development session, in the image: `ruff check .`, `scripts/check_progress_size.py`,
  `scripts/verify_bench_results.py`, `pytest` (`tests/test_config.py` fails on the code
  default alone and passes with the presets and the table, checked once). CI green.
- GPU session: both gate records all passed; both examples, every replica: cuPotential without
  a reason, return code 0, 100000 hills; container exit codes 0; `fmeta analyze` exit 0 at
  100, 40, 60 and 80 ns.
- Before every commit and the PR text: no speed or duration of the plugin in the diff.
