"""
Simulation setup utilities for RASPA calculations.

This module handles the creation of simulation.json files and related
simulation configuration.
"""

from copy import deepcopy
from typing import Any

from .constants import DEFAULT_SIMULATION_SETTINGS


def create_simulation_json(pressure: float, framework_name: str = "framework",
                          temperature: float = 323.0,
                          simulation_settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Create simulation.json with specified pressure and settings.

    Args:
        pressure: External pressure in Pa
        framework_name: Name of the framework
        temperature: Temperature in K
        simulation_settings: Custom simulation settings to override defaults

    Returns:
        Dict containing simulation configuration
    """
    # Start with default settings
    result = deepcopy(DEFAULT_SIMULATION_SETTINGS)

    # Set basic system parameters
    result["Systems"][0]["Name"] = framework_name
    result["Systems"][0]["ExternalTemperature"] = temperature
    result["Systems"][0]["ExternalPressure"] = pressure

    # Override with user settings if provided
    if simulation_settings:
        # Deep merge the settings
        for key, value in simulation_settings.items():
            if key == "Systems" and isinstance(value, dict):
                # Merge system settings
                for sys_key, sys_value in value.items():
                    result["Systems"][0][sys_key] = sys_value
            elif key == "Components" and isinstance(value, dict):
                # Merge component settings
                for comp_key, comp_value in value.items():
                    result["Components"][0][comp_key] = comp_value
            else:
                result[key] = value

        # Always set pressure and temperature from function args
        result["Systems"][0]["ExternalPressure"] = pressure
        result["Systems"][0]["ExternalTemperature"] = temperature
        result["Systems"][0]["Name"] = framework_name

    return result


def generate_pressure_range(min_pressure: float, max_pressure: float,
                           count: int) -> list[float]:
    """
    Generate a range of pressures for isotherm calculations.

    Args:
        min_pressure: Minimum pressure in Pa
        max_pressure: Maximum pressure in Pa
        count: Number of pressure points

    Returns:
        List of pressures
    """
    if count <= 1:
        return [min_pressure]

    pressures = []
    for i in range(count):
        pressure = min_pressure + i * (max_pressure - min_pressure) / (count - 1)
        pressures.append(pressure)

    return pressures


def create_slurm_script(job_name: str, job_count: int, time: str = "00:10:00",
                       memory: str = "1G") -> str:
    """
    Create a SLURM submission script for the simulation jobs.

    Args:
        job_name: Name for the SLURM job
        job_count: Number of jobs (for array indexing)
        time: Time limit for each job
        memory: Memory allocation per job

    Returns:
        SLURM script content as string
    """
    from .constants import SLURM_SCRIPT_TEMPLATE

    return SLURM_SCRIPT_TEMPLATE.format(
        job_name=job_name,
        job_count=job_count - 1,  # SLURM array is 0-indexed
        time=time,
        memory=memory
    )
