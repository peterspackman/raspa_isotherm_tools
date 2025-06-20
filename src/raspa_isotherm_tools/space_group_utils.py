"""
Space group utilities for ensuring RASPA3 compatibility.

This module provides functions to handle space group information in CIF files,
ensuring they have the necessary international tables number and hall symbol
required by RASPA3.
"""

from typing import Optional, Tuple

from chmpy.crystal import Crystal


def get_space_group_info(crystal: Crystal) -> Tuple[Optional[int], Optional[str]]:
    """
    Get space group information from a chmpy Crystal object.

    Args:
        crystal: chmpy Crystal object

    Returns:
        Tuple of (international_tables_number, hermann_mauguin_symbol)
    """
    try:
        # Access the space group from the crystal
        space_group = crystal.space_group

        # Get international tables number
        int_tables_number = getattr(space_group, 'international_tables_number', None)

        # Get Hermann-Mauguin symbol
        hm_symbol = getattr(space_group, 'symbol', None)

        return int_tables_number, hm_symbol

    except Exception as e:
        print(f"Warning: Could not extract space group information: {e}")
        return None, None


def ensure_space_group_in_cif(crystal: Crystal) -> None:
    """
    Ensure that the crystal's CIF data contains the necessary space group information
    for RASPA3 compatibility.

    Args:
        crystal: chmpy Crystal object to update
    """
    int_tables_number, hm_symbol = get_space_group_info(crystal)

    if int_tables_number is None:
        print("Warning: Could not determine space group information from crystal.")
        print("RASPA3 may require _symmetry_Int_Tables_number and _symmetry_space_group_name_H-M")
        return

    # Update the crystal's properties to include space group info
    # This ensures it gets written to the CIF when saved
    if not hasattr(crystal, 'properties'):
        crystal.properties = {}

    # Store space group information for CIF output
    crystal.properties['space_group_info'] = {
        'int_tables_number': int_tables_number,
        'hm_symbol': hm_symbol
    }


def add_space_group_to_cif_content(cif_content: str, int_tables_number: int,
                                   hall_symbol: str, hm_symbol: str) -> str:
    """
    Add space group information to CIF content if missing.

    Args:
        cif_content: Original CIF file content
        int_tables_number: International Tables number
        hall_symbol: Hall symbol
        hm_symbol: Hermann-Mauguin symbol

    Returns:
        Updated CIF content with space group information
    """
    lines = cif_content.split('\n')

    # Check if space group info already exists
    has_int_tables = any('_symmetry_Int_Tables_number' in line for line in lines)
    any('_symmetry_space_group_name_Hall' in line for line in lines)
    any('_symmetry_space_group_name_H-M' in line for line in lines)

    # Find insertion point (after cell parameters, before symmetry operations)
    insert_idx = -1
    for i, line in enumerate(lines):
        if '_cell_angle_gamma' in line or '_cell_volume' in line:
            insert_idx = i + 1
            break

    if insert_idx == -1:
        # If no cell parameters found, insert after data_ line
        for i, line in enumerate(lines):
            if line.startswith('data_'):
                insert_idx = i + 1
                break

    # Insert space group information
    space_group_lines = []

    if not has_int_tables and int_tables_number:
        space_group_lines.append('')
        space_group_lines.append('_symmetry_cell_setting          ?')
        space_group_lines.append(f'_symmetry_space_group_name_Hall \'{hall_symbol}\'')
        space_group_lines.append(f'_symmetry_space_group_name_H-M  \'{hm_symbol}\'')
        space_group_lines.append(f'_symmetry_Int_Tables_number     {int_tables_number}')

    if space_group_lines and insert_idx >= 0:
        lines[insert_idx:insert_idx] = space_group_lines

    return '\n'.join(lines)


def update_crystal_cif_with_space_group(crystal: Crystal) -> None:
    """
    Update a crystal's CIF data to include space group information.

    Args:
        crystal: chmpy Crystal object to update
    """
    int_tables_number, hm_symbol = get_space_group_info(crystal)

    if int_tables_number is None:
        print("Warning: Cannot update CIF - space group information not available")
        return

    # Store space group information in crystal properties for CIF generation
    if not hasattr(crystal, 'properties'):
        crystal.properties = {}

    # Store space group info that will be used during CIF generation
    crystal.properties['space_group_override'] = {
        'int_tables_number': int_tables_number,
        'hm_symbol': hm_symbol
    }


def save_crystal_with_space_group(crystal: Crystal, output_path) -> None:
    """
    Save crystal to CIF file with space group information included.

    Args:
        crystal: chmpy Crystal object
        output_path: Path to save the CIF file
    """
    try:
        # Get space group info
        int_tables_number, hm_symbol = get_space_group_info(crystal)

        if int_tables_number is not None:
            # Get CIF data
            cif_data = crystal.to_cif_data()

            # CIF data is a nested dict with structure {block_name: {data}}
            # Get the first (and usually only) data block
            block_name = list(cif_data.keys())[0]
            block_data = cif_data[block_name]

            # Add space group information to the CIF data block
            block_data['symmetry_cell_setting'] = '?'
            block_data['symmetry_Int_Tables_number'] = str(int_tables_number)
            if hm_symbol:
                block_data['symmetry_space_group_name_H-M'] = f"'{hm_symbol}'"

            # Convert CIF data to string and write to file
            from chmpy.fmt.cif import Cif
            cif = Cif(cif_data)
            cif.to_file(output_path)
        else:
            # Fall back to regular save
            crystal.save(output_path)

    except Exception as e:
        print(f"Warning: Error saving CIF with space group info: {e}")
        # Fallback to regular save
        crystal.save(output_path)
