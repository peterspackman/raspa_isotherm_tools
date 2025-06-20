"""
Physical constants and default parameters for RASPA simulations.
"""

# Physical constants
BOLTZMANN_K_KCAL_MOL = 0.001987204  # kcal/(mol·K)

# CO2 force field parameters (Garcia-Sanchez et al., J. Phys. Chem. C 2009)
CO2_FORCE_FIELD_PARAMS = {
    "C_co2": {
        "type": "lennard-jones",
        "parameters": [29.933, 2.745],  # [epsilon (K), sigma (Å)]
        "source": "A. Garcia-Sanchez et al., J. Phys. Chem. C 2009, 113, 8814-8820",
        "charge": 0.6512
    },
    "O_co2": {
        "type": "lennard-jones",
        "parameters": [85.671, 3.017],  # [epsilon (K), sigma (Å)]
        "source": "A. Garcia-Sanchez et al., J. Phys. Chem. C 2009, 113, 8814-8820",
        "charge": -0.3256
    }
}

# CO2 molecule definition for RASPA
CO2_MOLECULE = {
    "CriticalTemperature": 304.1282,
    "CriticalPressure": 7377300.0,
    "AcentricFactor": 0.22394,
    "Type": "rigid",
    "pseudoAtoms": [
        ["O_co2", [0.0, 0.0, 1.149]],
        ["C_co2", [0.0, 0.0, 0.0]],
        ["O_co2", [0.0, 0.0, -1.149]]
    ],
    "Bonds": [[0, 1], [1, 2]]
}

# Default simulation settings
DEFAULT_SIMULATION_SETTINGS = {
    "SimulationType": "MonteCarlo",
    "NumberOfCycles": 10000,
    "NumberOfInitializationCycles": 2000,
    "PrintEvery": 500,
    "Systems": [
        {
            "Type": "Framework",
            "NumberOfUnitCells": [1, 1, 1],
            "ChargeMethod": "Ewald",
            "ComputeDensityGrid": False,
            "OutputPDBMovie": False
        }
    ],
    "Components": [
        {
            "Name": "CO2",
            "MoleculeDefinition": "ExampleDefinitions",
            "FugacityCoefficient": 1.0,
            "TranslationProbability": 0.5,
            "RotationProbability": 0.5,
            "ReinsertionProbability": 0.5,
            "SwapProbability": 1.0,
            "WidomProbability": 1.0,
            "CreateNumberOfMolecules": 0
        }
    ]
}

# SLURM job script template
SLURM_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err
#SBATCH --array=0-{job_count}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time={time}
#SBATCH --mem={memory}

# Go to the appropriate simulation directory
SIM_DIR="$(printf "%03d" $SLURM_ARRAY_TASK_ID)"
cd $SIM_DIR || exit 1

echo "Running simulation in directory: $(pwd)"
echo "Pressure: $(grep ExternalPressure simulation.json | awk '{{print $2}}' | tr -d ',')"
echo "Start time: $(date)"

# Run RASPA3
raspa3

echo "End time: $(date)"

# Optional: Quick analysis of results
echo "Adsorption results:"
if [ -d outputs ]; then
    grep "Average loading" outputs/*.txt 2>/dev/null || echo "No loading results found"
fi

# Return to the original directory
cd - > /dev/null
"""
