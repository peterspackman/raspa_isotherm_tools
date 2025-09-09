#!/usr/bin/env python3
"""
Script to extract atomic positions and charges from DMA (Distributed Multipole Analysis) files.

DMA files contain multipole expansions for atoms, but we only extract:
- Atom labels
- Positions (x, y, z) in Angstroms  
- Charges (monopole moments)
"""

import argparse
import json
from pathlib import Path


class DMAParser:
    def __init__(self, filename):
        self.filename = filename
        self.atoms = []
        self.positions = []
        self.charges = []
        self.units = None

    def parse(self):
        """Parse DMA file to extract positions and charges."""
        with open(self.filename) as f:
            lines = f.readlines()

        # Default to Angstroms
        self.units = 'angstrom'

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines, comments, and molecule markers
            if not line or line.startswith('!') or line.startswith('#'):
                i += 1
                continue

            # Check for units specification
            if 'Units' in line or 'UNITS' in line:
                if 'angstrom' in line.lower() or 'angs' in line.lower():
                    self.units = 'angstrom'
                elif 'bohr' in line.lower():
                    self.units = 'bohr'
                i += 1
                continue

            # Parse atom line: label x y z Rank n
            parts = line.split()

            # Check if this looks like an atom line
            # Format: LABEL x y z Rank n
            if len(parts) >= 5 and 'Rank' in line:
                try:
                    atom_label = parts[0]
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])

                    # Store atom info
                    self.atoms.append(atom_label)

                    # Note: assuming positions are already in Angstrom
                    # Add conversion here if needed:
                    # if self.units == 'bohr':
                    #     bohr_to_ang = 0.529177210903
                    #     x *= bohr_to_ang
                    #     y *= bohr_to_ang
                    #     z *= bohr_to_ang

                    self.positions.append([x, y, z])

                    # Get charge from next line
                    i += 1
                    if i < len(lines):
                        charge_line = lines[i].strip()
                        if charge_line:
                            try:
                                # The charge is on its own line
                                charge = float(charge_line.split()[0])
                                self.charges.append(charge)
                            except (ValueError, IndexError):
                                self.charges.append(0.0)
                        else:
                            self.charges.append(0.0)
                    else:
                        self.charges.append(0.0)

                except (ValueError, IndexError):
                    pass

            i += 1

        # Ensure all arrays have same length
        n_atoms = len(self.atoms)
        if len(self.positions) != n_atoms:
            print(f"Warning: positions count ({len(self.positions)}) != atoms count ({n_atoms})")
        if len(self.charges) != n_atoms:
            print(f"Warning: charges count ({len(self.charges)}) != atoms count ({n_atoms})")
            # Pad with zeros if needed
            while len(self.charges) < n_atoms:
                self.charges.append(0.0)

    def get_data(self):
        """Return parsed data as a dictionary."""
        return {
            'atoms': self.atoms,
            'positions': self.positions,
            'charges': self.charges,
            'n_atoms': len(self.atoms),
            'total_charge': sum(self.charges),
            'units': 'angstrom'  # Always convert to Angstrom
        }

    def save_json(self, output_file):
        """Save parsed data to JSON file."""
        data = self.get_data()
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved charge data to {output_file}")

    def save_xyz(self, output_file):
        """Save as XYZ file with proper element symbols."""
        # Extract element symbols from DMA labels
        def clean_atom_label(label):
            """Extract element symbol from DMA label like 'C_F1_1____' -> 'C'"""
            parts = label.split('_')
            if parts:
                element = parts[0]
                # Remove any trailing numbers from element
                element = ''.join(c for c in element if c.isalpha())
                return element
            return label

        with open(output_file, 'w') as f:
            f.write(f"{len(self.atoms)}\n")
            f.write("DMA molecule\n")
            for atom, pos in zip(self.atoms, self.positions, strict=False):
                element = clean_atom_label(atom)
                f.write(f"{element:<3} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
        print(f"Saved XYZ format to {output_file}")

    def print_summary(self):
        """Print summary of parsed data."""
        print(f"\nDMA File: {self.filename}")
        print(f"Number of atoms: {len(self.atoms)}")
        print(f"Total charge: {sum(self.charges):.6f}")
        print("\nAtom  Label     x (Å)      y (Å)      z (Å)     Charge")
        print("-" * 60)
        for i, (atom, pos, charge) in enumerate(zip(self.atoms, self.positions, self.charges, strict=False)):
            print(f"{i+1:4d}  {atom:<6} {pos[0]:10.4f} {pos[1]:10.4f} {pos[2]:10.4f} {charge:10.6f}")


def match_with_cif(dma_data, cif_file):
    """
    Attempt to match DMA atoms with CIF structure.
    This is a placeholder for more sophisticated matching algorithms.
    """
    print(f"\nMatching with CIF file: {cif_file}")
    print("Note: Automatic matching requires handling rotations/translations")
    print("      and is not yet implemented. Manual matching may be needed.")

    # TODO: Implement structure matching algorithm
    # - Read CIF file
    # - Extract positions and atom types
    # - Find best rotation/translation to match DMA positions
    # - Map DMA charges to CIF atoms


def main():
    parser = argparse.ArgumentParser(description='Extract charges and positions from DMA files')
    parser.add_argument('dma_file', help='Input DMA file')
    parser.add_argument('-o', '--output', help='Output JSON file',
                       default=None)
    parser.add_argument('--xyz', help='Also save as XYZ file',
                       action='store_true')
    parser.add_argument('--cif', help='CIF file to match against',
                       default=None)
    parser.add_argument('-v', '--verbose', help='Verbose output',
                       action='store_true')

    args = parser.parse_args()

    # Parse DMA file
    dma_parser = DMAParser(args.dma_file)
    dma_parser.parse()

    # Print summary
    if args.verbose:
        dma_parser.print_summary()
    else:
        data = dma_parser.get_data()
        print(f"Parsed {data['n_atoms']} atoms from {args.dma_file}")
        print(f"Total charge: {data['total_charge']:.6f}")

    # Save outputs
    if args.output:
        dma_parser.save_json(args.output)
    else:
        # Default output name
        output_file = Path(args.dma_file).stem + '_charges.json'
        dma_parser.save_json(output_file)

    if args.xyz:
        xyz_file = Path(args.dma_file).stem + '.xyz'
        dma_parser.save_xyz(xyz_file)

    # Match with CIF if provided
    if args.cif:
        dma_data = dma_parser.get_data()
        match_with_cif(dma_data, args.cif)


if __name__ == "__main__":
    main()
