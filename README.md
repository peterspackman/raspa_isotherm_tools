# RASPA3 Isotherm Tools

A Python package for setting up, running, and analyzing adsorption isotherm simulations with RASPA3.

## Overview

This package contains tools for:

1. **Setting up simulations**: Create simulation directories for different pressures
2. **Processing results**: Extract and store simulation results
3. **Visualization**: Plot isotherms and analyze data
4. **Workflow management**: Unified interface for the entire workflow

## Installation

### From PyPI

```bash
pip install raspa_isotherm_tools
```

### From Source

```bash
git clone https://github.com/username/raspa_isotherm_tools.git
cd raspa_isotherm_tools
pip install -e .
```

For development:

```bash
pip install -e ".[dev,docs]"
```

## Dependencies

- **Core**: Python standard library and SQLite (built into Python)
- **Visualization**: Matplotlib

## Usage

### Command-line Interface

After installation, you can use the `raspa-isotherm` command:

```bash
# Setup simulations for multiple pressures
raspa-isotherm setup \
  --framework MFI_SI.cif \
  --framework-id MFI_SI \
  --component CO2.json \
  --component-id CO2 \
  --forcefield force_field.json \
  --pressure-range 1000:1000000:log:10 \
  --temperature 298 \
  --cycles 50000 \
  --init-cycles 10000 \
  --output-dir isotherms/MFI_CO2_298K \
  --job-name MFI_CO2_298K \
  --memory 2G \
  --time 05:00:00

# Submit to Slurm
raspa-isotherm submit --output-dir isotherms/MFI_CO2_298K

# Process results after jobs complete
raspa-isotherm process --output-dir isotherms/MFI_CO2_298K --db MFI_isotherms.db

# Plot the isotherm
raspa-isotherm plot --db MFI_isotherms.db --plot-output MFI_CO2_298K.png

# Export the data
raspa-isotherm export --db MFI_isotherms.db --export-output MFI_CO2_298K.csv
```

### Python API

You can also use the package in your Python scripts:

```python
from raspa_isotherm_tools.isotherm_setup import setup_isotherm
from raspa_isotherm_tools.isotherm_process import process_results
from raspa_isotherm_tools.isotherm_workflow import plot_isotherm

# Setup simulations
setup_params = {
    "framework": "IRMOF-1.cif",
    "component": "CO2.json",
    "forcefield": "force_field.json",
    "pressure_range": "1000:100000:log:10",
    "temperature": 298,
    "cycles": 50000,
    "output_dir": "isotherms/IRMOF-1_CO2_298K"
}
setup_isotherm(setup_params)
```

See the [documentation](https://raspa-isotherm-tools.readthedocs.io) for more details.

## Help

Each command includes detailed help information:

```bash
raspa-isotherm --help
raspa-isotherm setup --help
raspa-isotherm process --help
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.