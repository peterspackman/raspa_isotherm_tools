"""
Command line interface for raspa_isotherm_tools.

This module provides a unified CLI that combines the generator and parallel runner.
"""

import argparse
import sys

from .generator import main as generator_main
from .parallel_runner import main as runner_main


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="RASPA3 Isotherm Tools - Generate and run isotherm calculations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Generate command
    generate_parser = subparsers.add_parser(
        'generate',
        help='Generate RASPA3 input files from CIF structures',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    generate_parser.add_argument("cif", help="Crystal structure file (CIF format)")
    generate_parser.add_argument("-p1", "--min-pressure", type=float, default=0.0,
                               help="Pressure lower bound (Pa)")
    generate_parser.add_argument("-p2", "--max-pressure", type=float, default=1000000.0,
                               help="Pressure upper bound (Pa)")
    generate_parser.add_argument("-n", "--count", type=int, default=100,
                               help="Number of pressure steps")
    generate_parser.add_argument("-j", "--job-name", type=str, default="CO2_iso",
                               help="Job name for SLURM script")
    generate_parser.add_argument("-m", "--memory", type=str, default="1G",
                               help="Memory for SLURM job script")
    generate_parser.add_argument("-t", "--time", type=str, default="00:10:00",
                               help="Time for SLURM job script")
    generate_parser.add_argument("-T", "--temperature", type=float, default=323.0,
                               help="Temperature (K)")
    generate_parser.add_argument("--force-field", type=str, default="uff",
                               choices=["uff", "uff4mof"],
                               help="Force field to use")
    generate_parser.add_argument("--charge-scale-factor", type=float, default=1.0,
                               help="Factor to scale all framework charges (default: 1.0)")
    generate_parser.add_argument("-o", "--output-dir", type=str, default="jobs",
                               help="Output directory name")

    # Simulation settings for generate
    generate_parser.add_argument("--cycles", type=int, default=10000,
                               help="Number of Monte Carlo cycles")
    generate_parser.add_argument("--init-cycles", type=int, default=200,
                               help="Number of initialization cycles")
    generate_parser.add_argument("--print-every", type=int, default=500,
                               help="Print frequency")
    generate_parser.add_argument("--simulation-type", type=str, default="MonteCarlo",
                               choices=["MonteCarlo", "MolecularDynamics"],
                               help="Type of simulation")
    generate_parser.add_argument("--unit-cells", type=int, nargs=3, default=[1, 1, 1],
                               help="Number of unit cells in each direction")
    generate_parser.add_argument("--charge-method", type=str, default="Ewald",
                               choices=["Ewald", "Wolf", "None"],
                               help="Charge calculation method")
    generate_parser.add_argument("--density-grid", action="store_true",
                               help="Enable density grid computation")
    generate_parser.add_argument("--pdb-movie", action="store_true",
                               help="Enable PDB movie output")

    # Component settings for generate
    generate_parser.add_argument("--fugacity-coeff", type=float, default=1.0,
                               help="Fugacity coefficient for CO2")
    generate_parser.add_argument("--translation-prob", type=float, default=0.5,
                               help="Translation move probability")
    generate_parser.add_argument("--rotation-prob", type=float, default=0.5,
                               help="Rotation move probability")
    generate_parser.add_argument("--reinsertion-prob", type=float, default=0.5,
                               help="Reinsertion move probability")
    generate_parser.add_argument("--swap-prob", type=float, default=1.0,
                               help="Swap move probability")
    generate_parser.add_argument("--widom-prob", type=float, default=1.0,
                               help="Widom insertion probability")
    generate_parser.add_argument("--initial-molecules", type=int, default=0,
                               help="Initial number of CO2 molecules")

    # Template file option for generate
    generate_parser.add_argument("--template", type=str,
                               help="JSON template file for simulation settings")

    # Single-point command
    single_point_parser = subparsers.add_parser(
        'single-point',
        help='Generate single-point calculations for multiple crystals at fixed pressure/temperature',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    single_point_parser.add_argument("cifs", nargs="+", help="Crystal structure files (CIF format)")
    single_point_parser.add_argument("-p", "--pressure", type=float, required=True,
                                   help="Pressure (Pa)")
    single_point_parser.add_argument("-T", "--temperature", type=float, default=323.0,
                                   help="Temperature (K)")
    single_point_parser.add_argument("-j", "--job-name", type=str, default="CO2_single",
                                   help="Job name for SLURM script")
    single_point_parser.add_argument("-m", "--memory", type=str, default="1G",
                                   help="Memory for SLURM job script")
    single_point_parser.add_argument("-t", "--time", type=str, default="00:10:00",
                                   help="Time for SLURM job script")
    single_point_parser.add_argument("--force-field", type=str, default="uff",
                                   choices=["uff", "uff4mof"],
                                   help="Force field to use")
    single_point_parser.add_argument("--charge-scale-factor", type=float, default=1.0,
                                   help="Factor to scale all framework charges (default: 1.0)")
    single_point_parser.add_argument("-o", "--output-dir", type=str, default="single_point_jobs",
                                   help="Output directory name")

    # Simulation settings for single-point
    single_point_parser.add_argument("--cycles", type=int, default=10000,
                                   help="Number of Monte Carlo cycles")
    single_point_parser.add_argument("--init-cycles", type=int, default=200,
                                   help="Number of initialization cycles")
    single_point_parser.add_argument("--print-every", type=int, default=500,
                                   help="Print frequency")
    single_point_parser.add_argument("--simulation-type", type=str, default="MonteCarlo",
                                   choices=["MonteCarlo", "MolecularDynamics"],
                                   help="Type of simulation")
    single_point_parser.add_argument("--unit-cells", type=int, nargs=3, default=[1, 1, 1],
                                   help="Number of unit cells in each direction")
    single_point_parser.add_argument("--charge-method", type=str, default="Ewald",
                                   choices=["Ewald", "Wolf", "None"],
                                   help="Charge calculation method")
    single_point_parser.add_argument("--density-grid", action="store_true",
                                   help="Enable density grid computation")
    single_point_parser.add_argument("--pdb-movie", action="store_true",
                                   help="Enable PDB movie output")

    # Component settings for single-point
    single_point_parser.add_argument("--fugacity-coeff", type=float, default=1.0,
                                   help="Fugacity coefficient for CO2")
    single_point_parser.add_argument("--translation-prob", type=float, default=0.5,
                                   help="Translation move probability")
    single_point_parser.add_argument("--rotation-prob", type=float, default=0.5,
                                   help="Rotation move probability")
    single_point_parser.add_argument("--reinsertion-prob", type=float, default=0.5,
                                   help="Reinsertion move probability")
    single_point_parser.add_argument("--swap-prob", type=float, default=1.0,
                                   help="Swap move probability")
    single_point_parser.add_argument("--widom-prob", type=float, default=1.0,
                                   help="Widom insertion probability")
    single_point_parser.add_argument("--initial-molecules", type=int, default=0,
                                   help="Initial number of CO2 molecules")

    # Template file option for single-point
    single_point_parser.add_argument("--template", type=str,
                                   help="JSON template file for simulation settings")

    # Run command
    run_parser = subparsers.add_parser(
        'run',
        help='Run RASPA3 simulations in parallel',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    run_parser.add_argument("directory", help="Base directory containing job subdirectories")
    run_parser.add_argument("-w", "--workers", type=int, default=None,
                           help="Number of parallel workers (default: auto-detect)")
    run_parser.add_argument("--timeout", type=float, default=3600.0,
                           help="Timeout per job in seconds")
    run_parser.add_argument("--no-chmpy", action="store_true",
                           help="Use subprocess instead of chmpy Raspa class")
    run_parser.add_argument("-v", "--verbose", action="store_true",
                           help="Enable verbose logging")
    run_parser.add_argument("--summary-file", type=str,
                           help="Save results summary to JSON file")

    return parser


def main(args=None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()

    if args is None:
        args = sys.argv[1:]

    # If no arguments provided, show help
    if not args:
        parser.print_help()
        return 1

    parsed_args = parser.parse_args(args)

    # If no command specified, show help
    if not parsed_args.command:
        parser.print_help()
        return 1

    try:
        if parsed_args.command == 'generate':
            # Convert parsed args to sys.argv format for generator_main
            generator_args = [
                parsed_args.cif,
                '-p1', str(parsed_args.min_pressure),
                '-p2', str(parsed_args.max_pressure),
                '-n', str(parsed_args.count),
                '-j', parsed_args.job_name,
                '-m', parsed_args.memory,
                '-t', parsed_args.time,
                '-T', str(parsed_args.temperature),
                '--force-field', parsed_args.force_field,
                '--charge-scale-factor', str(parsed_args.charge_scale_factor),
                '-o', parsed_args.output_dir,
                '--cycles', str(parsed_args.cycles),
                '--init-cycles', str(parsed_args.init_cycles),
                '--print-every', str(parsed_args.print_every),
                '--simulation-type', parsed_args.simulation_type,
                '--unit-cells', *map(str, parsed_args.unit_cells),
                '--charge-method', parsed_args.charge_method,
                '--fugacity-coeff', str(parsed_args.fugacity_coeff),
                '--translation-prob', str(parsed_args.translation_prob),
                '--rotation-prob', str(parsed_args.rotation_prob),
                '--reinsertion-prob', str(parsed_args.reinsertion_prob),
                '--swap-prob', str(parsed_args.swap_prob),
                '--widom-prob', str(parsed_args.widom_prob),
                '--initial-molecules', str(parsed_args.initial_molecules),
            ]

            if parsed_args.density_grid:
                generator_args.append('--density-grid')
            if parsed_args.pdb_movie:
                generator_args.append('--pdb-movie')
            if parsed_args.template:
                generator_args.extend(['--template', parsed_args.template])

            # Temporarily replace sys.argv for the generator
            original_argv = sys.argv
            sys.argv = ['raspa-isotherm'] + generator_args
            try:
                return generator_main()
            finally:
                sys.argv = original_argv

        elif parsed_args.command == 'single-point':
            # Import here to avoid circular imports
            from .generator import single_point_main
            
            # Convert parsed args to sys.argv format for single_point_main
            single_point_args = parsed_args.cifs.copy()
            single_point_args.extend([
                '-p', str(parsed_args.pressure),
                '-T', str(parsed_args.temperature),
                '-j', parsed_args.job_name,
                '-m', parsed_args.memory,
                '-t', parsed_args.time,
                '--force-field', parsed_args.force_field,
                '--charge-scale-factor', str(parsed_args.charge_scale_factor),
                '-o', parsed_args.output_dir,
                '--cycles', str(parsed_args.cycles),
                '--init-cycles', str(parsed_args.init_cycles),
                '--print-every', str(parsed_args.print_every),
                '--simulation-type', parsed_args.simulation_type,
                '--unit-cells', *map(str, parsed_args.unit_cells),
                '--charge-method', parsed_args.charge_method,
                '--fugacity-coeff', str(parsed_args.fugacity_coeff),
                '--translation-prob', str(parsed_args.translation_prob),
                '--rotation-prob', str(parsed_args.rotation_prob),
                '--reinsertion-prob', str(parsed_args.reinsertion_prob),
                '--swap-prob', str(parsed_args.swap_prob),
                '--widom-prob', str(parsed_args.widom_prob),
                '--initial-molecules', str(parsed_args.initial_molecules),
            ])

            if parsed_args.density_grid:
                single_point_args.append('--density-grid')
            if parsed_args.pdb_movie:
                single_point_args.append('--pdb-movie')
            if parsed_args.template:
                single_point_args.extend(['--template', parsed_args.template])

            # Temporarily replace sys.argv for the single_point function
            original_argv = sys.argv
            sys.argv = ['raspa-single-point'] + single_point_args
            try:
                return single_point_main()
            finally:
                sys.argv = original_argv

        elif parsed_args.command == 'run':
            # Convert parsed args to sys.argv format for runner_main
            runner_args = [parsed_args.directory]

            if parsed_args.workers:
                runner_args.extend(['-w', str(parsed_args.workers)])
            if parsed_args.timeout != 3600.0:
                runner_args.extend(['-t', str(parsed_args.timeout)])
            if parsed_args.no_chmpy:
                runner_args.append('--no-chmpy')
            if parsed_args.verbose:
                runner_args.append('-v')
            if parsed_args.summary_file:
                runner_args.extend(['--summary-file', parsed_args.summary_file])

            # Temporarily replace sys.argv for the runner
            original_argv = sys.argv
            sys.argv = ['raspa-runner'] + runner_args
            try:
                return runner_main()
            finally:
                sys.argv = original_argv

        else:
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
