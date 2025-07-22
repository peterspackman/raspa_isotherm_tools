# RASPA3 Isotherm Tools

[![CI](https://github.com/peterspackman/raspa_isotherm_tools/workflows/CI/badge.svg)](https://github.com/peterspackman/raspa_isotherm_tools/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A Python package for setting up, running, and analyzing adsorption isotherm simulations with RASPA3 using chmpy for force field generation.

## Overview

This package provides tools for:

1. **RASPA3 Input Generation**: Automatically generate RASPA3 simulation inputs from CIF files
2. **Force Field Generation**: Create UFF force fields with EEQ charges using chmpy
3. **Space Group Handling**: Ensure RASPA3 compatibility with proper space group information
4. **Parallel Execution**: Run multiple RASPA3 simulations in parallel with progress tracking
5. **Results Analysis**: Parse output files, create plots, and export data to CSV/Excel
6. **Unified CLI**: Single command-line interface for the entire workflow

## Features

- **Automatic force field generation** using Universal Force Field (UFF) parameters via chmpy
- **EEQ charge calculation** for framework atoms
- **Space group information** automatically added to CIF files for RASPA3 compatibility
- **Parallel job execution** with ThreadPoolExecutor and progress bars
- **Comprehensive testing** with 29+ unit tests
- **Modern Python packaging** with pyproject.toml and type hints

## Installation

### Prerequisites

- Python 3.10+
- RASPA3 (must be installed and available in PATH)
- chmpy >= 1.1.7

### From Source

```bash
git clone https://github.com/peterspackman/raspa_isotherm_tools.git
cd raspa_isotherm_tools
pip install -e .
```

For development with all dependencies:

```bash
pip install -e ".[dev,parallel]"
```

## Dependencies

### Core Dependencies
- **numpy**: Numerical computations
- **chmpy**: Crystallographic calculations and force field generation

### Optional Dependencies
- **tqdm**: Progress bars for parallel execution (install with `[parallel]`)

### Development Dependencies
- **pytest**: Testing framework
- **pytest-cov**: Test coverage
- **black**: Code formatting
- **ruff**: Fast linting
- **mypy**: Type checking

## Usage

### Command-line Interface

The package provides three CLI entry points:

#### 1. Generate RASPA3 Inputs

```bash
# Generate simulation inputs from a CIF file
raspa-generate framework.cif --output-dir jobs --pressure-count 50 --max-pressure 1000000

# With custom settings
raspa-generate framework.cif \
  --output-dir isotherms \
  --min-pressure 1000 \
  --max-pressure 5000000 \
  --pressure-count 100 \
  --temperature 298.15 \
  --cycles 10000 \
  --init-cycles 2000
```

#### 2. Run Simulations in Parallel

```bash
# Run all jobs in the directory
raspa-run jobs --max-workers 4

# With timeout and custom settings
raspa-run isotherms --max-workers 8 --timeout 3600 --use-chmpy
```

#### 3. Analyze Results

```bash
# Parse RASPA3 outputs and create database
raspa-analyze jobs --database results.db

# Plot isotherms with error bars  
raspa-analyze jobs --database results.db --plot Cu-BTC CO2

# Export data to CSV/Excel
raspa-analyze jobs --database results.db --export csv Cu-BTC CO2
```

#### 4. Unified Interface

```bash
# Combined generate and run workflow
raspa-isotherm generate framework.cif --output-dir jobs
raspa-isotherm run jobs --max-workers 4
```

### Python API

```python
from raspa_isotherm_tools.generator import RASPAInputGenerator
from raspa_isotherm_tools.parallel_runner import RASPAParallelRunner
from raspa_isotherm_tools.database import RASPADatabase

# Generate simulation inputs
generator = RASPAInputGenerator("framework.cif")
generator.generate_jobs(
    output_dir="jobs",
    min_pressure=1000,
    max_pressure=1000000,
    pressure_count=50,
    temperature=323.0
)

# Run simulations in parallel
runner = RASPAParallelRunner("jobs", max_workers=4)
results = runner.run_all_jobs()

# Parse results into database
from raspa_isotherm_tools.output_parser import RASPAOutputParser
from raspa_isotherm_tools.plotting import plot_isotherm_from_database

parser = RASPAOutputParser("jobs")
db = RASPADatabase("results.db")
parser.parse_all_outputs(db)

# Create plots and export data
plot_isotherm_from_database("results.db", "Cu-BTC", "CO2")
db.export_isotherm_to_csv("isotherm.csv", "Cu-BTC", "CO2")
```

### Force Field Generation

The package automatically generates force fields using:

- **UFF parameters** for Lennard-Jones interactions
- **EEQ charges** for electrostatic interactions
- **Automatic atom labeling** based on asymmetric unit
- **CO2 parameters** from literature values

```python
from raspa_isotherm_tools.force_field import create_force_field_json
from chmpy import Crystal

crystal = Crystal.load("framework.cif")
force_field = create_force_field_json(crystal)
```

### Space Group Handling

RASPA3 requires space group information in CIF files. This package automatically:

- Extracts space group number and symbol from chmpy
- Updates CIF files with proper space group data
- Ensures RASPA3 compatibility

```python
from raspa_isotherm_tools.space_group_utils import get_space_group_info

crystal = Crystal.load("framework.cif")
sg_number, sg_symbol = get_space_group_info(crystal)
print(f"Space group: {sg_symbol} (#{sg_number})")
```

## Configuration

### Simulation Settings

Default simulation parameters can be customized:

```python
custom_settings = {
    "NumberOfCycles": 50000,
    "NumberOfInitializationCycles": 5000,
    "PrintEvery": 1000,
    "RestartFile": False
}

generator.generate_jobs(simulation_settings=custom_settings)
```

### Force Field Parameters

CO2 parameters are built-in but can be customized by modifying `constants.py`.

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/raspa_isotherm_tools

# Run specific test file
pytest tests/test_generator.py -v
```

## Development

### Code Quality

The project uses modern Python development tools:

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/

# Run all checks
ruff check src/ tests/ --fix
black --check src/ tests/
mypy src/
pytest
```

### Continuous Integration

GitHub Actions CI runs on:
- Multiple Python versions (3.10-3.13)
- Multiple platforms (Ubuntu, macOS, Windows)
- Code quality checks (ruff, black, mypy)
- Full test suite

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure all tests and quality checks pass
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/peterspackman/raspa_isotherm_tools.git
cd raspa_isotherm_tools
pip install -e ".[dev,parallel]"
pre-commit install  # Optional: set up pre-commit hooks
```

## License

This project is licensed under the GNU General Public License v3.0 or later (GPLv3+) - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **chmpy**: Crystallographic calculations and force field generation
- **RASPA3**: Molecular simulation software
- **UFF**: Universal Force Field implementation
- **EEQ**: Electronegativity equalization method for charge calculation
