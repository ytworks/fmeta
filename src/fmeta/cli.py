"""Command line interface: ``fmeta <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .config import TEMPLATE, Config


def _config_from_args(args) -> Config:
    if args.config:
        config = Config.load(args.config)
    else:
        if args.complex and args.ligand_resname:
            data = {"input": {"complex": args.complex, "ligand_resname": args.ligand_resname,
                              "ligand_smiles": args.ligand_smiles}}
        elif args.protein and args.ligand:
            data = {"input": {"protein": args.protein, "ligand": args.ligand}}
        else:
            sys.exit("give a configuration file, --protein and --ligand, or --complex and "
                     "--ligand-resname")
        config = Config.from_dict(data, base_dir=Path.cwd())
    if args.membrane:
        from .system.builder import MembraneSettings
        config.membrane = config.membrane or MembraneSettings()
        config.membrane.lipid = args.membrane
    if args.ligand_xml:
        config.system.ligand_method = "xml"
        config.system.ligand_xml = [str(Path(p).resolve()) for p in args.ligand_xml]
    if args.length_ns is not None:
        config.run.length_ns = args.length_ns
    if args.replicas is not None:
        config.run.replicas = args.replicas
    if args.gpus:
        config.run.gpus = [int(g) for g in args.gpus.split(",")]
    if args.platform:
        config.run.platform = args.platform
    return config


def _add_setup_options(parser) -> None:
    parser.add_argument("config", nargs="?", help="YAML configuration (see 'fmeta init')")
    parser.add_argument("--protein", help="protein structure (PDB or mmCIF)")
    parser.add_argument("--ligand", help="posed ligand with hydrogens (SDF, MOL, MOL2)")
    parser.add_argument("--complex", help="structure holding protein and ligand (PDB or mmCIF)")
    parser.add_argument("--ligand-resname", help="residue name of the ligand in --complex")
    parser.add_argument("--ligand-smiles", help="ligand SMILES, to add its hydrogens (needs RDKit)")
    parser.add_argument("--membrane", nargs="?", const="POPC", metavar="LIPID",
                        help="embed in a lipid bilayer (default POPC); the protein must be "
                             "oriented with the membrane normal along z")
    parser.add_argument("--ligand-xml", nargs="+", help="OpenMM ffxml file(s) with the ligand template")
    parser.add_argument("-o", "--out", default="fmeta_run", help="work directory")
    parser.add_argument("--length-ns", type=float, help="production length per replica")
    parser.add_argument("--replicas", type=int)
    parser.add_argument("--gpus", help="comma-separated device indices, e.g. 0,1")
    parser.add_argument("--platform", help="OpenMM platform (default: auto)")


def cmd_init(args) -> None:
    path = Path(args.path)
    if path.exists():
        sys.exit(f"{path} already exists")
    path.write_text(TEMPLATE)
    print(f"wrote {path}")


def cmd_split(args) -> None:
    from .system.inputs import split_complex
    protein, ligand = split_complex(args.complex, args.resname, args.out, args.smiles,
                                    args.chain, args.resid)
    print(f"wrote {protein} and {ligand}")


def cmd_prepare(args) -> None:
    from .workflow import prepare_stage
    prepared = prepare_stage(_config_from_args(args), args.out)
    print(prepared.report)
    print(f"\n{prepared.topology.getNumAtoms()} atoms; files in {Path(args.out) / 'prepared'}")


def cmd_funnel(args) -> None:
    report = json.loads((Path(args.workdir) / "prepared" / "report.json").read_text())
    for check in report["checks"]:
        print(f"[{check['status']:7s}] {check['name']}: {check['message']}")
    print(f"\nfunnel definition: {Path(args.workdir) / 'prepared' / 'funnel.yaml'}")
    print(f"view with the structure: {Path(args.workdir) / 'prepared' / 'funnel.pdb'}")


def cmd_equilibrate(args) -> None:
    from .workflow import equilibrate_stage
    summary = equilibrate_stage(Config.load(Path(args.workdir) / "config.yaml"), args.workdir,
                                args.device)
    print(f"funnel wall energy during the free stage: mean {summary['wall_energy_mean']:.3f} kJ/mol"
          f" ({'ok' if summary['wall_ok'] else 'TOO HIGH: widen the funnel'})")


def cmd_run(args) -> None:
    from .workflow import run_replicas, run_stage
    config = Config.load(Path(args.workdir) / "config.yaml")
    if args.replica is not None:
        run_stage(config, args.workdir, args.replica, args.device, args.length_ns, args.walker)
    else:
        gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else None
        run_replicas(config, args.workdir, gpus, args.length_ns)


def cmd_analyze(args) -> None:
    from .workflow import analyze_stage, format_summary
    print(format_summary(analyze_stage(args.workdir, plot=args.plot)))
    if args.plot:
        print(f"figures: {Path(args.workdir) / 'summary.png'} and rep*/analysis.png")


def cmd_auto(args) -> None:
    from .workflow import format_summary, run_all
    print(format_summary(run_all(_config_from_args(args), args.out)))


def cmd_check(args) -> None:
    from .check import Checker
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else None
    checker = Checker(args.out, args.platform or "auto", args.precision, args.device, gpus,
                      args.quick)
    if not checker.run_all():
        sys.exit(1)


def cmd_doctor(args) -> None:
    import numpy as np
    import openmm as mm

    from .engine import BiasSettings, FunnelBias
    from .engine.platform import available_platforms, select_platform
    from .testing.toy import make_toy

    print(f"OpenMM {mm.__version__}; platforms: {', '.join(available_platforms())}")
    platform, properties = select_platform(args.platform or "auto")
    print(f"selected: {platform.getName()} {properties or ''}")
    try:
        import openmmforcefields  # noqa: F401
        print("openmmforcefields: available (automatic ligand parameters)")
    except ImportError:
        print("openmmforcefields: not installed (use ligand ffxml or pre-parameterised input)")

    # the funnel forces and the bias table on the selected platform, against Reference
    values = {}
    for name in dict.fromkeys(["Reference", platform.getName()]):
        toy = make_toy()
        bias = FunnelBias(toy.system, toy.funnel, BiasSettings(), periodic=True)
        bias.grid.add_hill(1.4, 0.1, 5.0)
        selected = mm.Platform.getPlatformByName(name)
        context = mm.Context(toy.system, mm.VerletIntegrator(0.001), selected,
                             properties if name == platform.getName() else {})
        positions = toy.positions.copy()
        positions[0] += np.array([0.12, -0.05, 0.4])
        context.setPositions(positions)
        bias.push(context)
        state = context.getState(getEnergy=True, getForces=True)
        values[name] = (state.getPotentialEnergy()._value, np.array(state.getForces(asNumpy=True)._value))
        if name == platform.getName():
            start = time.time()
            for _ in range(20):
                bias.push(context)
            cost = (time.time() - start) / 20
            print(f"bias table update: {1000 * cost:.1f} ms for {bias.grid.values.size} grid points")
    reference, selected = values["Reference"], values[platform.getName()]
    print(f"funnel + bias energy {selected[0]:.4f} kJ/mol; difference from Reference: "
          f"{abs(selected[0] - reference[0]):.2e} kJ/mol, "
          f"max force difference {np.abs(selected[1] - reference[1]).max():.2e} kJ/mol/nm")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="fmeta", description="Funnel metadynamics with OpenMM")
    parser.add_argument("-v", "--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    sub = commands.add_parser("init", help="write a template configuration")
    sub.add_argument("path", nargs="?", default="fmeta.yaml")
    sub.set_defaults(func=cmd_init)

    sub = commands.add_parser("split", help="cut a complex into protein.pdb and ligand.sdf")
    sub.add_argument("complex")
    sub.add_argument("--resname", required=True, help="ligand residue name")
    sub.add_argument("--smiles", help="ligand SMILES in the protonation state to simulate")
    sub.add_argument("--chain")
    sub.add_argument("--resid")
    sub.add_argument("-o", "--out", default="inputs")
    sub.set_defaults(func=cmd_split)

    sub = commands.add_parser("prepare", help="build the system and place the funnel")
    _add_setup_options(sub)
    sub.set_defaults(func=cmd_prepare)

    sub = commands.add_parser("funnel", help="show the funnel report of a prepared system")
    sub.add_argument("workdir")
    sub.set_defaults(func=cmd_funnel)

    sub = commands.add_parser("equilibrate", help="minimise, heat and equilibrate")
    sub.add_argument("workdir")
    sub.add_argument("--device", type=int)
    sub.set_defaults(func=cmd_equilibrate)

    sub = commands.add_parser("run", help="run or continue production")
    sub.add_argument("workdir")
    sub.add_argument("--replica", type=int, help="run only this replica")
    sub.add_argument("--walker", type=int, help="with --replica: run only this walker of it")
    sub.add_argument("--device", type=int, help="GPU device index for --replica")
    sub.add_argument("--gpus", help="comma-separated device indices for all replicas")
    sub.add_argument("--length-ns", type=float, help="total length per replica")
    sub.set_defaults(func=cmd_run)

    sub = commands.add_parser("analyze", help="binding free energy and convergence")
    sub.add_argument("workdir")
    sub.add_argument("--plot", action="store_true", help="also write figures (needs matplotlib)")
    sub.set_defaults(func=cmd_analyze)

    sub = commands.add_parser("auto", help="prepare, equilibrate, run and analyse")
    _add_setup_options(sub)
    sub.set_defaults(func=cmd_auto)

    sub = commands.add_parser(
        "check", help="acceptance test on this machine: forces, toy free energy, full pipeline, "
                      "membrane, ligand parameters, throughput")
    sub.add_argument("-o", "--out", default="fmeta_check", help="directory for the test runs")
    sub.add_argument("--platform", help="OpenMM platform (default: the GPU platform found)")
    sub.add_argument("--precision", default="mixed")
    sub.add_argument("--device", type=int, help="GPU device index")
    sub.add_argument("--gpus", help="comma-separated device indices, to test the multi-GPU launcher")
    sub.add_argument("--quick", action="store_true",
                     help="minimal run lengths (a dry run of the steps, not a validation)")
    sub.set_defaults(func=cmd_check)

    sub = commands.add_parser("doctor", help="quick look at platforms and optional packages")
    sub.add_argument("--platform")
    sub.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
