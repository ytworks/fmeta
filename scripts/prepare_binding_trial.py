"""Start a new candidate from a prepared, equilibrated example without inheriting any hills."""

import argparse
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory

from score_binding import ARMS, ENDPOINTS

from fmeta.config import Config


def prepare_trial(source, target, arm, length_ns, seed):
    if seed <= 0:
        raise ValueError("supply an explicit positive base seed for this independent trial")
    if target.exists():
        raise FileExistsError(target)
    replicas, walkers, sigma_s, sigma_rho, pace = ARMS[arm]
    data = Config.load(source / "config.yaml").to_dict()
    data["metad"].update(bias_factor=10.0, height=1.5, sigma_s=sigma_s, sigma_rho=sigma_rho)
    data["run"].update(replicas=replicas, walkers=walkers, seed=seed, length_ns=length_ns,
                       timestep_fs=4.0, pace_ps=pace, walker_sync_hills=1, checkpoint_ps=250.0,
                       trajectory_ps=100.0, trajectory_solute_only=True, backend="cupotential",
                       platform="CUDA", precision="mixed", mode="parallel", gpus="all", per_gpu=4)
    config = Config.from_dict(data)
    if config.metad.temperature != 300 or not (source / "equil/state.xml").is_file():
        raise ValueError("the comparison requires a 300 K example with a saved equilibration state")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as tmp:
        stage = Path(tmp) / "trial"
        stage.mkdir()
        for dirname in ("prepared", "equil"):
            copytree(source / dirname, stage / dirname)
        config.save(stage / "config.yaml")
        stage.rename(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="prepared example work directory")
    parser.add_argument("target", type=Path, help="new candidate work directory")
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--length-ns", required=True, type=int, choices=ENDPOINTS)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    prepare_trial(args.source, args.target, args.arm, args.length_ns, args.seed)


if __name__ == "__main__":
    main()
