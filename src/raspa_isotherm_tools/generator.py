"""
Main RASPA input generator module.

This module provides the main functionality for generating RASPA3 input files
from CIF structures using chmpy for force field parameters and charges.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from chmpy.crystal import Crystal

from .constants import CO2_MOLECULE
from .force_field import create_force_field_json, generate_default_labels, update_crystal_labels
from .simulation import create_simulation_json, create_slurm_script, generate_pressure_range
from .space_group_utils import (
    ensure_space_group_in_cif,
    save_crystal_with_space_group,
    update_crystal_cif_with_space_group,
)


class RASPAInputGenerator:
    """
    Main class for generating RASPA input files from CIF structures.
    """

    def __init__(self, cif_path: str, force_field: str = "uff4mof", charge_scale_factor: float = 1.0,
                 charge_file: str = None):
        """
        Initialize the generator with a CIF file.

        Args:
            cif_path: Path to the CIF file
            force_field: Force field to use ("uff", "uff4mof", or "fit_lj")
            charge_scale_factor: Factor to scale all framework charges (default: 1.0)
            charge_file: Path to JSON file containing atom charges (optional)
        """
        self.cif_path = Path(cif_path)
        self.force_field = force_field
        self.charge_scale_factor = charge_scale_factor
        self.charge_file = Path(charge_file) if charge_file else None

        # Load crystal structure
        try:
            self.crystal = Crystal.load(str(self.cif_path))
            print(f"Loaded crystal: {self.crystal.titl}")
            print(f"Asymmetric unit: {len(self.crystal.asymmetric_unit.atomic_numbers)} atoms")
            print(f"Unit cell: {len(self.crystal.unit_cell_atoms()['element'])} atoms")
        except Exception as e:
            raise RuntimeError(f"Error loading crystal structure: {e}") from e

        # Ensure space group information is available for RASPA3
        ensure_space_group_in_cif(self.crystal)
        update_crystal_cif_with_space_group(self.crystal)

        # Generate and apply default labels
        self.default_labels = generate_default_labels(self.crystal)
        update_crystal_labels(self.crystal, self.default_labels)

        print(f"Using default labels: {set(self.default_labels.values())}")

    def generate_jobs(self, output_dir: str = "jobs",
                     min_pressure: float = 0.0, max_pressure: float = 1000000.0,
                     pressure_count: int = 100, temperature: float = 323.0,
                     simulation_settings: dict[str, Any] | None = None,
                     job_name: str = "CO2_iso", memory: str = "1G",
                     time: str = "00:10:00") -> None:
        """
        Generate all RASPA input files and job directories.

        Args:
            output_dir: Directory to create jobs in
            min_pressure: Minimum pressure in Pa
            max_pressure: Maximum pressure in Pa
            pressure_count: Number of pressure points
            temperature: Temperature in K
            simulation_settings: Custom simulation settings
            job_name: Name for SLURM jobs
            memory: Memory allocation for SLURM jobs
            time: Time limit for SLURM jobs
        """
        base_dir = Path(output_dir)
        base_dir.mkdir(exist_ok=True)
        print(f"Creating simulation directories in {base_dir}/")

        # Generate force field using chmpy
        charge_method = "from file" if self.charge_file else "EEQ"
        print(f"Generating force field parameters using chmpy {self.force_field} and {charge_method} charges...")
        try:
            force_field_data = create_force_field_json(
                self.crystal, self.force_field, self.default_labels, self.charge_scale_factor,
                self.charge_file
            )
            print(f"Generated force field with {len(force_field_data['PseudoAtoms'])} pseudo atoms")
        except Exception as e:
            raise RuntimeError(f"Error generating force field: {e}") from e

        # Generate pressure range
        pressures = generate_pressure_range(min_pressure, max_pressure, pressure_count)

        # Create simulation directories
        for i, pressure in enumerate(pressures):
            # Create numbered directory for SLURM array compatibility
            sim_dir = base_dir / f"{i:03d}"
            sim_dir.mkdir(exist_ok=True)

            # Save CIF file with updated space group info and default labels
            dest_cif = sim_dir / self.cif_path.name
            self._save_crystal_with_space_group(dest_cif)

            # Create simulation.json
            sim_data = create_simulation_json(
                pressure,
                framework_name=self.cif_path.stem,
                temperature=temperature,
                simulation_settings=simulation_settings
            )
            sim_file = sim_dir / "simulation.json"
            with sim_file.open("w") as f:
                json.dump(sim_data, f, indent=2)

            # Create force_field.json
            ff_file = sim_dir / "force_field.json"
            with ff_file.open("w") as f:
                json.dump(force_field_data, f, indent=2)

            # Create CO2.json
            co2_file = sim_dir / "CO2.json"
            with co2_file.open("w") as f:
                json.dump(CO2_MOLECULE, f, indent=2)

        print(f"Created {pressure_count} simulation directories")
        print(f"Pressure range: {min_pressure/1000:.1f} - {max_pressure/1000:.1f} kPa")
        print(f"Temperature: {temperature} K")
        print(f"Force field: {self.force_field.upper()}")

        # Generate directory index file
        self._create_directory_index(base_dir, pressures)

        # Generate SLURM submission script
        self._create_slurm_script(base_dir, pressure_count, job_name, memory, time)

    def _save_crystal_with_space_group(self, output_path: Path) -> None:
        """
        Save crystal with space group information included.

        Args:
            output_path: Path to save the CIF file
        """
        save_crystal_with_space_group(self.crystal, output_path)

    def _create_directory_index(self, base_dir: Path, pressures: list) -> None:
        """
        Create an index file mapping directories to pressures.

        Args:
            base_dir: Base directory containing simulation folders
            pressures: List of pressures corresponding to each directory
        """
        index_file = base_dir / "directory_index.txt"
        with index_file.open("w") as f:
            f.write(f"# Directory index for {self.cif_path.name}\n")
            f.write(f"# Force field: {self.force_field.upper()}\n")
            f.write("# Directory: Pressure (Pa)\n")
            for i, pressure in enumerate(pressures):
                f.write(f"{i:03d}: {pressure:.2e}\n")

        print(f"Created directory index: {index_file}")

    def _create_slurm_script(self, base_dir: Path, job_count: int,
                            job_name: str, memory: str, time: str) -> None:
        """
        Create SLURM submission script.

        Args:
            base_dir: Base directory for jobs
            job_count: Number of jobs
            job_name: Job name
            memory: Memory allocation
            time: Time limit
        """
        slurm_script = create_slurm_script(job_name, job_count, time, memory)

        submit_file = base_dir / "submit.sh"
        with submit_file.open("w") as f:
            f.write(slurm_script)

        # Make submit script executable
        submit_file.chmod(0o755)

        print(f"Created SLURM submission script: {submit_file}")
        print(f"To submit jobs: cd {base_dir} && sbatch submit.sh")

    def generate_single_point_jobs(self, output_dir: str = "single_point_jobs",
                                  pressure: float = 100000.0, temperature: float = 323.0,
                                  simulation_settings: dict[str, Any] | None = None,
                                  job_name: str = "CO2_single", memory: str = "1G",
                                  time: str = "00:10:00", crystal_index: int = 0) -> None:
        """
        Generate a single RASPA input file for one crystal at one pressure/temperature.

        Args:
            output_dir: Directory to create jobs in
            pressure: Pressure in Pa
            temperature: Temperature in K
            simulation_settings: Custom simulation settings
            job_name: Name for SLURM jobs
            memory: Memory allocation for SLURM jobs
            time: Time limit for SLURM jobs
            crystal_index: Index for this crystal in the batch
        """
        base_dir = Path(output_dir)

        # Create numbered directory for SLURM array compatibility
        sim_dir = base_dir / f"{crystal_index:03d}_{self.cif_path.stem}"
        sim_dir.mkdir(parents=True, exist_ok=True)

        print(f"Creating simulation directory: {sim_dir}")

        # Generate force field using chmpy
        charge_method = "from file" if self.charge_file else "EEQ"
        print(f"Generating force field parameters using chmpy {self.force_field} and {charge_method} charges...")
        try:
            force_field_data = create_force_field_json(
                self.crystal, self.force_field, self.default_labels, self.charge_scale_factor,
                self.charge_file
            )
        except Exception as e:
            raise RuntimeError(f"Error generating force field for {self.cif_path.name}: {e}") from e

        # Save CIF file with updated space group info and default labels
        dest_cif = sim_dir / self.cif_path.name
        self._save_crystal_with_space_group(dest_cif)

        # Create simulation.json
        sim_data = create_simulation_json(
            pressure,
            framework_name=self.cif_path.stem,
            temperature=temperature,
            simulation_settings=simulation_settings
        )
        sim_file = sim_dir / "simulation.json"
        with sim_file.open("w") as f:
            json.dump(sim_data, f, indent=2)

        # Create force_field.json
        ff_file = sim_dir / "force_field.json"
        with ff_file.open("w") as f:
            json.dump(force_field_data, f, indent=2)

        # Create CO2.json
        co2_file = sim_dir / "CO2.json"
        with co2_file.open("w") as f:
            json.dump(CO2_MOLECULE, f, indent=2)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate RASPA input files using chmpy UFF/EEQ parameters"
    )
    parser.add_argument("cif", help="Crystal structure file (CIF format)")
    parser.add_argument("-p1", "--min-pressure", type=float, default=0.0,
                       help="Pressure lower bound (Pa)")
    parser.add_argument("-p2", "--max-pressure", type=float, default=1000000.0,
                       help="Pressure upper bound (Pa)")
    parser.add_argument("-n", "--count", type=int, default=100,
                       help="Number of pressure steps")
    parser.add_argument("-j", "--job-name", type=str, default="CO2_iso",
                       help="Job name for SLURM script")
    parser.add_argument("-m", "--memory", type=str, default="1G",
                       help="Memory for SLURM job script")
    parser.add_argument("-t", "--time", type=str, default="00:10:00",
                       help="Time for SLURM job script")
    parser.add_argument("-T", "--temperature", type=float, default=323.0,
                       help="Temperature (K)")
    parser.add_argument("--force-field", type=str, default="uff",
                       choices=["uff", "uff4mof", "fit_lj"],
                       help="Force field to use")
    parser.add_argument("--charge-scale-factor", type=float, default=1.0,
                       help="Factor to scale all framework charges (default: 1.0)")
    parser.add_argument("--charge-file", type=str, default=None,
                       help="JSON file containing atom charges (optional, otherwise uses EEQ)")
    parser.add_argument("-o", "--output-dir", type=str, default="jobs",
                       help="Output directory name")

    # Simulation settings
    parser.add_argument("--cycles", type=int, default=10000,
                       help="Number of Monte Carlo cycles")
    parser.add_argument("--init-cycles", type=int, default=2000,
                       help="Number of initialization cycles")
    parser.add_argument("--print-every", type=int, default=500,
                       help="Print frequency")
    parser.add_argument("--simulation-type", type=str, default="MonteCarlo",
                       choices=["MonteCarlo", "MolecularDynamics"],
                       help="Type of simulation")
    parser.add_argument("--unit-cells", type=int, nargs=3, default=[1, 1, 1],
                       help="Number of unit cells in each direction")
    parser.add_argument("--charge-method", type=str, default="Ewald",
                       choices=["Ewald", "Wolf", "None"],
                       help="Charge calculation method")
    parser.add_argument("--density-grid", action="store_true",
                       help="Enable density grid computation")
    parser.add_argument("--pdb-movie", action="store_true",
                       help="Enable PDB movie output")

    # Component settings
    parser.add_argument("--fugacity-coeff", type=float, default=1.0,
                       help="Fugacity coefficient for CO2")
    parser.add_argument("--translation-prob", type=float, default=0.5,
                       help="Translation move probability")
    parser.add_argument("--rotation-prob", type=float, default=0.5,
                       help="Rotation move probability")
    parser.add_argument("--reinsertion-prob", type=float, default=0.5,
                       help="Reinsertion move probability")
    parser.add_argument("--swap-prob", type=float, default=1.0,
                       help="Swap move probability")
    parser.add_argument("--widom-prob", type=float, default=1.0,
                       help="Widom insertion probability")
    parser.add_argument("--initial-molecules", type=int, default=0,
                       help="Initial number of CO2 molecules")

    # Template file option
    parser.add_argument("--template", type=str,
                       help="JSON template file for simulation settings")

    return parser


def create_single_point_parser() -> argparse.ArgumentParser:
    """Create command line argument parser for single-point calculations."""
    parser = argparse.ArgumentParser(
        description="Generate single-point RASPA calculations for multiple crystals"
    )
    parser.add_argument("cifs", nargs="+", help="Crystal structure files (CIF format)")
    parser.add_argument("-p", "--pressure", type=float, required=True,
                       help="Pressure (Pa)")
    parser.add_argument("-T", "--temperature", type=float, default=323.0,
                       help="Temperature (K)")
    parser.add_argument("-j", "--job-name", type=str, default="CO2_single",
                       help="Job name for SLURM script")
    parser.add_argument("-m", "--memory", type=str, default="1G",
                       help="Memory for SLURM job script")
    parser.add_argument("-t", "--time", type=str, default="00:10:00",
                       help="Time for SLURM job script")
    parser.add_argument("--force-field", type=str, default="uff",
                       choices=["uff", "uff4mof", "fit_lj"],
                       help="Force field to use")
    parser.add_argument("--charge-scale-factor", type=float, default=1.0,
                       help="Factor to scale all framework charges (default: 1.0)")
    parser.add_argument("--charge-file", type=str, default=None,
                       help="JSON file containing atom charges (optional, otherwise uses EEQ)")
    parser.add_argument("-o", "--output-dir", type=str, default="single_point_jobs",
                       help="Output directory name")

    # Simulation settings
    parser.add_argument("--cycles", type=int, default=10000,
                       help="Number of Monte Carlo cycles")
    parser.add_argument("--init-cycles", type=int, default=2000,
                       help="Number of initialization cycles")
    parser.add_argument("--print-every", type=int, default=500,
                       help="Print frequency")
    parser.add_argument("--simulation-type", type=str, default="MonteCarlo",
                       choices=["MonteCarlo", "MolecularDynamics"],
                       help="Type of simulation")
    parser.add_argument("--unit-cells", type=int, nargs=3, default=[1, 1, 1],
                       help="Number of unit cells in each direction")
    parser.add_argument("--charge-method", type=str, default="Ewald",
                       choices=["Ewald", "Wolf", "None"],
                       help="Charge calculation method")
    parser.add_argument("--density-grid", action="store_true",
                       help="Enable density grid computation")
    parser.add_argument("--pdb-movie", action="store_true",
                       help="Enable PDB movie output")

    # Component settings
    parser.add_argument("--fugacity-coeff", type=float, default=1.0,
                       help="Fugacity coefficient for CO2")
    parser.add_argument("--translation-prob", type=float, default=0.5,
                       help="Translation move probability")
    parser.add_argument("--rotation-prob", type=float, default=0.5,
                       help="Rotation move probability")
    parser.add_argument("--reinsertion-prob", type=float, default=0.5,
                       help="Reinsertion move probability")
    parser.add_argument("--swap-prob", type=float, default=1.0,
                       help="Swap move probability")
    parser.add_argument("--widom-prob", type=float, default=1.0,
                       help="Widom insertion probability")
    parser.add_argument("--initial-molecules", type=int, default=0,
                       help="Initial number of CO2 molecules")

    # Template file option
    parser.add_argument("--template", type=str,
                       help="JSON template file for simulation settings")

    return parser


def single_point_main() -> int:
    """Main entry point for single-point calculations."""
    parser = create_single_point_parser()
    args = parser.parse_args()

    print(f"Setting up single-point calculations for {len(args.cifs)} crystals")
    print(f"Pressure: {args.pressure/1000:.1f} kPa, Temperature: {args.temperature} K")

    base_dir = Path(args.output_dir)
    base_dir.mkdir(exist_ok=True)

    successful_crystals = []
    failed_crystals = []

    try:
        # Process each CIF file
        for i, cif_path in enumerate(args.cifs):
            print(f"\nProcessing {cif_path} ({i+1}/{len(args.cifs)})")

            try:
                # Create generator for this crystal
                generator = RASPAInputGenerator(cif_path, args.force_field,
                                               args.charge_scale_factor, args.charge_file)

                # Load template settings if provided
                simulation_settings = None
                if args.template:
                    try:
                        with open(args.template) as f:
                            simulation_settings = json.load(f)
                        print(f"Loaded simulation template: {args.template}")
                    except Exception as e:
                        print(f"Error loading template file: {e}")
                        return 1
                else:
                    # Build settings from command line arguments
                    simulation_settings = {
                        "SimulationType": args.simulation_type,
                        "NumberOfCycles": args.cycles,
                        "NumberOfInitializationCycles": args.init_cycles,
                        "PrintEvery": args.print_every,
                        "Systems": {
                            "NumberOfUnitCells": args.unit_cells,
                            "ChargeMethod": args.charge_method,
                            "ComputeDensityGrid": args.density_grid,
                            "OutputPDBMovie": args.pdb_movie
                        },
                        "Components": {
                            "FugacityCoefficient": args.fugacity_coeff,
                            "TranslationProbability": args.translation_prob,
                            "RotationProbability": args.rotation_prob,
                            "ReinsertionProbability": args.reinsertion_prob,
                            "SwapProbability": args.swap_prob,
                            "WidomProbability": args.widom_prob,
                            "CreateNumberOfMolecules": args.initial_molecules
                        }
                    }

                # Generate single-point job for this crystal
                generator.generate_single_point_jobs(
                    output_dir=args.output_dir,
                    pressure=args.pressure,
                    temperature=args.temperature,
                    simulation_settings=simulation_settings,
                    job_name=args.job_name,
                    memory=args.memory,
                    time=args.time,
                    crystal_index=i
                )

                successful_crystals.append(cif_path)

            except Exception as e:
                print(f"Error processing {cif_path}: {e}")
                failed_crystals.append(cif_path)
                continue

        # Create summary files
        create_single_point_summary(base_dir, successful_crystals, failed_crystals,
                                  args.pressure, args.temperature, args.force_field)

        # Create SLURM submission script for all crystals
        if successful_crystals:
            create_single_point_slurm_script(base_dir, len(successful_crystals),
                                           args.job_name, args.memory, args.time)

        print("\nSummary:")
        print(f"Successfully processed: {len(successful_crystals)} crystals")
        print(f"Failed: {len(failed_crystals)} crystals")
        if failed_crystals:
            print(f"Failed crystals: {', '.join(failed_crystals)}")

        return 0 if not failed_crystals else 1

    except Exception as e:
        print(f"Error: {e}")
        return 1


def create_single_point_summary(base_dir: Path, successful_crystals: list, failed_crystals: list,
                               pressure: float, temperature: float, force_field: str) -> None:
    """Create summary files for single-point calculations."""
    # Create crystal index file (no comments for script compatibility)
    index_file = base_dir / "crystal_index.txt"
    with index_file.open("w") as f:
        for i, crystal in enumerate(successful_crystals):
            crystal_name = Path(crystal).stem
            f.write(f"{i:03d}_{crystal_name}: {crystal}\n")

    print(f"Created crystal index: {index_file}")

    # Create separate info file with metadata
    info_file = base_dir / "calculation_info.txt"
    with info_file.open("w") as f:
        f.write("Single-point calculation summary\n")
        f.write(f"Pressure: {pressure:.2e} Pa ({pressure/1000:.1f} kPa)\n")
        f.write(f"Temperature: {temperature} K\n")
        f.write(f"Force field: {force_field.upper()}\n")
        f.write(f"Total crystals processed: {len(successful_crystals)}\n")
        if failed_crystals:
            f.write(f"Failed crystals: {len(failed_crystals)}\n")

    print(f"Created calculation info: {info_file}")

    # Create failed crystals list if any failed (no comments)
    if failed_crystals:
        failed_file = base_dir / "failed_crystals.txt"
        with failed_file.open("w") as f:
            for crystal in failed_crystals:
                f.write(f"{crystal}\n")
        print(f"Created failed crystals list: {failed_file}")


def create_single_point_slurm_script(base_dir: Path, crystal_count: int, job_name: str,
                                    memory: str, time: str) -> None:
    """Create SLURM submission script for single-point calculations."""
    slurm_script = create_slurm_script(job_name, crystal_count, time, memory)

    submit_file = base_dir / "submit.sh"
    with submit_file.open("w") as f:
        f.write(slurm_script)

    # Make submit script executable
    submit_file.chmod(0o755)

    print(f"Created SLURM submission script: {submit_file}")
    print(f"To submit jobs: cd {base_dir} && sbatch submit.sh")


def main() -> int:
    """Main entry point for the command line interface."""
    parser = create_parser()
    args = parser.parse_args()

    print(f"Loading crystal structure from {args.cif}")

    try:
        # Create generator
        generator = RASPAInputGenerator(args.cif, args.force_field, args.charge_scale_factor, args.charge_file)

        # Load template settings if provided
        simulation_settings = None
        if args.template:
            try:
                with open(args.template) as f:
                    simulation_settings = json.load(f)
                print(f"Loaded simulation template: {args.template}")
            except Exception as e:
                print(f"Error loading template file: {e}")
                return 1
        else:
            # Build settings from command line arguments
            simulation_settings = {
                "SimulationType": args.simulation_type,
                "NumberOfCycles": args.cycles,
                "NumberOfInitializationCycles": args.init_cycles,
                "PrintEvery": args.print_every,
                "Systems": {
                    "NumberOfUnitCells": args.unit_cells,
                    "ChargeMethod": args.charge_method,
                    "ComputeDensityGrid": args.density_grid,
                    "OutputPDBMovie": args.pdb_movie
                },
                "Components": {
                    "FugacityCoefficient": args.fugacity_coeff,
                    "TranslationProbability": args.translation_prob,
                    "RotationProbability": args.rotation_prob,
                    "ReinsertionProbability": args.reinsertion_prob,
                    "SwapProbability": args.swap_prob,
                    "WidomProbability": args.widom_prob,
                    "CreateNumberOfMolecules": args.initial_molecules
                }
            }

        # Generate jobs
        generator.generate_jobs(
            output_dir=args.output_dir,
            min_pressure=args.min_pressure,
            max_pressure=args.max_pressure,
            pressure_count=args.count,
            temperature=args.temperature,
            simulation_settings=simulation_settings,
            job_name=args.job_name,
            memory=args.memory,
            time=args.time
        )

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
