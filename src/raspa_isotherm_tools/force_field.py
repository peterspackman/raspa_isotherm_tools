"""
Force field generation utilities for RASPA simulations.

This module handles the generation of force field parameters using chmpy's
UFF parameters and EEQ charges.
"""

from typing import Any

from chmpy.core.element import Element
from chmpy.crystal import Crystal
from chmpy.crystal.eeq_pbc import calculate_eeq_charges_crystal
from chmpy.ff.params import assign_uff_type_from_coordination, load_lj_params, get_lj_parameters

from .constants import BOLTZMANN_K_KCAL_MOL, CO2_FORCE_FIELD_PARAMS

import json
from pathlib import Path


def get_asymmetric_unit_uff_parameters(crystal: Crystal, force_field: str = "uff") -> tuple[dict[int, str], dict[int, dict[str, float]]]:
    """
    Get UFF parameters for asymmetric unit atoms only.

    Args:
        crystal: chmpy Crystal object
        force_field: "uff" or "uff4mof"

    Returns:
        Tuple of (atom_types, parameters) for asymmetric unit atoms
    """
    # Get asymmetric unit info
    asym_unit = crystal.asymmetric_unit
    atomic_nums = asym_unit.atomic_numbers

    # Calculate coordination numbers for asymmetric unit
    coord_nums = crystal.asymmetric_unit_coordination_numbers()

    # Load parameter database
    lj_params = load_lj_params()[force_field.lower()]

    atom_types = {}
    parameters = {}

    for i, (atomic_num, coord_num) in enumerate(zip(atomic_nums, coord_nums, strict=False)):
        # Assign UFF type based on coordination
        uff_type = assign_uff_type_from_coordination(atomic_num, coord_num)
        atom_types[i] = uff_type

        # Get parameters
        if uff_type in lj_params:
            sigma, epsilon = lj_params[uff_type]
            parameters[i] = {"sigma": sigma, "epsilon": epsilon}
        else:
            # Fallback parameters
            parameters[i] = {"sigma": 3.0, "epsilon": 0.1}
            print(f"Warning: No parameters found for {uff_type}, using defaults")

    return atom_types, parameters


def generate_default_labels(crystal: Crystal) -> dict[int, str]:
    """
    Generate default labels for asymmetric unit atoms (H1, H2, C1, etc.).

    Args:
        crystal: chmpy Crystal object

    Returns:
        Dict mapping from atom index to default label
    """
    # Get asymmetric unit atoms info (not unit cell!)
    asym_unit = crystal.asymmetric_unit
    atomic_nums = asym_unit.atomic_numbers

    # Count atoms by element symbol
    element_counts = {}
    labels = {}

    for i, atomic_num in enumerate(atomic_nums):
        element = Element.from_atomic_number(atomic_num)
        symbol = element.symbol

        if symbol not in element_counts:
            element_counts[symbol] = 0
        element_counts[symbol] += 1

        # Create label like H1, H2, C1, etc.
        labels[i] = f"{symbol}{element_counts[symbol]}"

    return labels


def update_crystal_labels(crystal: Crystal, new_labels: dict[int, str]) -> None:
    """
    Update crystal asymmetric unit labels and clear cached data.

    Args:
        crystal: chmpy Crystal object
        new_labels: Dict mapping atom indices to new labels
    """
    import numpy as np

    # Convert current labels to a list for easier manipulation
    current_labels = list(crystal.asymmetric_unit.labels)

    # Update with new labels
    for i, label in new_labels.items():
        current_labels[i] = label

    # Replace the entire labels array with a new one that can accommodate longer strings
    # Using dtype=object to handle variable-length strings without truncation
    crystal.asymmetric_unit.labels = np.array(current_labels, dtype=object)

    # Clear cached CIF data so it gets regenerated with new labels
    if hasattr(crystal, 'properties') and 'cif_data' in crystal.properties:
        del crystal.properties['cif_data']


def create_force_field_json(crystal: Crystal, force_field: str = "uff",
                           asym_labels: dict[int, str] = None, 
                           charge_scale_factor: float = 1.0,
                           charge_file: Path = None) -> dict[str, Any]:
    """
    Create force_field.json using chmpy UFF parameters and EEQ or file-based charges.

    Args:
        crystal: chmpy Crystal object
        force_field: "uff", "uff4mof", or "fit_lj"
        asym_labels: Dict mapping atom indices to asymmetric unit labels
        charge_scale_factor: Factor to scale all framework charges (default: 1.0)
        charge_file: Path to JSON file containing atom charges (optional)

    Returns:
        Dict containing force field JSON structure
    """
    # Get asymmetric unit info
    asym_unit = crystal.asymmetric_unit
    atomic_nums = asym_unit.atomic_numbers

    # Get force field parameters
    if force_field == "fit_lj":
        # Use get_lj_parameters directly for fitted parameters
        atom_types, uff_params = get_lj_parameters(crystal, force_field="fit_lj")
    else:
        # Get UFF parameters for asymmetric unit only
        atom_types, uff_params = get_asymmetric_unit_uff_parameters(crystal, force_field)

    # Get charges either from file or calculate with EEQ
    if charge_file and charge_file.exists():
        # Load charges from JSON file
        with open(charge_file, 'r') as f:
            charge_data = json.load(f)
        
        # Extract charges for asymmetric unit atoms
        # Assume the file contains a mapping of atom labels to charges
        charges = []
        for i in range(len(atomic_nums)):
            label = asym_labels[i] if asym_labels else f"atom{i+1}"
            if label in charge_data:
                charges.append(charge_data[label])
            elif str(i) in charge_data:
                charges.append(charge_data[str(i)])
            else:
                # Try to find by element and number
                element = Element.from_atomic_number(atomic_nums[i])
                default_label = f"{element.symbol}{i+1}"
                if default_label in charge_data:
                    charges.append(charge_data[default_label])
                else:
                    print(f"Warning: No charge found for atom {label}, using 0.0")
                    charges.append(0.0)
    else:
        # Get EEQ charges for unit cell, then extract asymmetric unit portion
        all_charges = calculate_eeq_charges_crystal(crystal)
        
        # The issue is that some CIFs have more atoms than the asymmetric unit
        # We need to use only the asymmetric unit charges
        num_asym_atoms = len(atomic_nums)
        charges = all_charges[:num_asym_atoms]

    # Create pseudo atoms list - one per asymmetric unit atom
    pseudo_atoms = []

    # Framework atoms - one entry per asymmetric unit atom (each can have different charge)
    for i, atomic_num in enumerate(atomic_nums):
        element = Element.from_atomic_number(atomic_num)

        # Use asymmetric unit label
        atom_label = asym_labels[i] if asym_labels else f"atom{i+1}"

        pseudo_atom = {
            "name": atom_label,
            "framework": True,
            "print_to_output": True,
            "element": element.symbol,
            "print_as": element.symbol,
            "mass": element.mass,
            "charge": float(charges[i] * charge_scale_factor)  # Each atom has its own EEQ charge, scaled
        }
        pseudo_atoms.append(pseudo_atom)

    # Add CO2 atoms
    pseudo_atoms.extend([
        {
            "name": "C_co2",
            "framework": False,
            "print_to_output": True,
            "element": "C",
            "print_as": "C",
            "mass": 12.0,
            "charge": CO2_FORCE_FIELD_PARAMS["C_co2"]["charge"]
        },
        {
            "name": "O_co2",
            "framework": False,
            "print_to_output": True,
            "element": "O",
            "print_as": "O",
            "mass": 15.9994,
            "charge": CO2_FORCE_FIELD_PARAMS["O_co2"]["charge"]
        }
    ])

    # Create self interactions - one per asymmetric unit atom (each needs its own entry)
    self_interactions = []

    # Framework interactions - one per asymmetric unit atom
    for i in range(len(atomic_nums)):
        # Use asymmetric unit label to match pseudo atom name
        atom_label = asym_labels[i] if asym_labels else f"atom{i+1}"

        # Get parameters for this atom
        params = uff_params[i]

        # Convert epsilon from kcal/mol to K for RASPA
        epsilon_K = params["epsilon"] / BOLTZMANN_K_KCAL_MOL

        # Set source based on force field type
        if force_field == "fit_lj":
            source = "Fitted LJ via chmpy"
        elif force_field == "uff4mof":
            source = "UFF4MOF via chmpy"
        else:
            source = "UFF via chmpy"
        
        interaction = {
            "name": atom_label,  # Must match pseudo atom name
            "type": "lennard-jones",
            "parameters": [epsilon_K, params["sigma"]],
            "source": source
        }
        self_interactions.append(interaction)

    # Add CO2 interactions
    for atom_name, params in CO2_FORCE_FIELD_PARAMS.items():
        interaction = {
            "name": atom_name,
            "type": params["type"],
            "parameters": params["parameters"],
            "source": params["source"]
        }
        self_interactions.append(interaction)

    return {
        "PseudoAtoms": pseudo_atoms,
        "SelfInteractions": self_interactions,
        "MixingRule": "Lorentz-Berthelot",
        "TruncationMethod": "shifted",
        "TailCorrections": False
    }
