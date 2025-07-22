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

    def __init__(self, cif_path: str, force_field: str = "uff"):
        """
        Initialize the generator with a CIF file.

        Args:
            cif_path: Path to the CIF file
            force_field: Force field to use ("uff" or "uff4mof")
        """
        self.cif_path = Path(cif_path)
        self.force_field = force_field

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
        print("Generating force field parameters using chmpy UFF and EEQ...")
        try:
            force_field_data = create_force_field_json(
                self.crystal, self.force_field, self.default_labels
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
                       choices=["uff", "uff4mof"],
                       help="Force field to use")
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


def main() -> int:
    """Main entry point for the command line interface."""
    parser = create_parser()
    args = parser.parse_args()

    print(f"Loading crystal structure from {args.cif}")

    try:
        # Create generator
        generator = RASPAInputGenerator(args.cif, args.force_field)

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
