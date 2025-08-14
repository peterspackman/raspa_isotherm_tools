#!/usr/bin/env python3
"""
Match DMA charges to crystal structure using chmpy and Kabsch algorithm.

This script:
1. Reads charges and positions from DMA file
2. Loads crystal structure from CIF
3. Extracts symmetry-unique molecules
4. Uses Kabsch algorithm to find optimal alignment
5. Maps charges to crystal atoms
"""

import numpy as np
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from itertools import permutations
from dataclasses import dataclass

# Import chmpy components
try:
    from chmpy import Crystal
    from chmpy.util.num import kabsch_rotation_matrix
except ImportError:
    print("Error: chmpy is required. Install with: pip install chmpy")
    exit(1)


@dataclass
class AlignmentResult:
    """Result of molecular alignment transformation."""
    positions: np.ndarray
    moments: np.ndarray
    axes: np.ndarray
    

@dataclass 
class MatchingResult:
    """Result of atom matching between two molecules."""
    mapping: List[int]
    rmsd: float
    rotation: np.ndarray
    translation: np.ndarray
    success: bool
    

class MatchingError(Exception):
    """Raised when molecular matching fails."""
    pass


def load_dma_data(dma_json_file: str) -> Dict:
    """Load DMA charge data from JSON file."""
    with open(dma_json_file, 'r') as f:
        return json.load(f)


def extract_molecule_from_dma(dma_data: Dict) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Extract a single molecule from DMA data.
    
    Returns:
        labels: List of atom labels
        positions: numpy array of positions (N x 3)
        charges: numpy array of charges (N,)
    """
    n_atoms = dma_data['n_atoms']
    
    # Check if there are multiple molecules (repeated patterns)
    # For ACETAC01_rank0.dma, we have 2 identical molecules
    if n_atoms == 16:  # Two 8-atom molecules
        # Take first molecule only
        labels = dma_data['atoms'][:8]
        positions = np.array(dma_data['positions'][:8])
        charges = np.array(dma_data['charges'][:8])
    else:
        # Single molecule
        labels = dma_data['atoms']
        positions = np.array(dma_data['positions'])
        charges = np.array(dma_data['charges'])
    
    return labels, positions, charges


def clean_atom_labels(labels: List[str]) -> List[str]:
    """
    Clean atom labels to extract element symbols.
    E.g., 'C_F1_1____' -> 'C'
    """
    cleaned = []
    for label in labels:
        # Extract element symbol (first part before underscore)
        parts = label.split('_')
        if parts:
            element = parts[0]
            # Remove any trailing numbers from element
            element = ''.join(c for c in element if c.isalpha())
            cleaned.append(element)
        else:
            cleaned.append(label)
    return cleaned


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_rmsd(pos1: np.ndarray, pos2: np.ndarray, mapping: List[int] = None) -> float:
    """
    Calculate RMSD between two sets of positions.
    
    Args:
        pos1: First set of positions (N x 3)
        pos2: Second set of positions (N x 3) 
        mapping: Optional mapping from pos1 indices to pos2 indices
    """
    if mapping is not None:
        pos2_mapped = pos2[mapping]
    else:
        pos2_mapped = pos2
    
    return np.sqrt(np.mean(np.sum((pos1 - pos2_mapped)**2, axis=1)))


def get_distance_matrix(pos: np.ndarray) -> np.ndarray:
    """Calculate pairwise distance matrix for positions."""
    n = len(pos)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(pos[i] - pos[j])
            distances[i, j] = distances[j, i] = dist
    return distances


# =============================================================================
# MOLECULAR ALIGNMENT CLASS
# =============================================================================

class MolecularAlignment:
    """Handles molecular axis frame transformations using moments of inertia."""
    
    @staticmethod
    def to_principal_axis_frame(positions: np.ndarray, masses: np.ndarray = None) -> AlignmentResult:
        """
        Transform positions to principal axis frame using moment of inertia tensor.
        Returns positions aligned to principal axes of inertia.
        """
        # Center positions
        centered = positions - np.mean(positions, axis=0)
        
        # Use unit masses if not provided
        if masses is None:
            masses = np.ones(len(positions))
        
        # Calculate moment of inertia tensor
        inertia = MolecularAlignment._calculate_inertia_tensor(centered, masses)
        
        # Get principal moments and axes
        moments, axes = np.linalg.eigh(inertia)
        
        # Sort by moment of inertia (ascending - smallest moment first)
        idx = np.argsort(moments)
        moments = moments[idx]
        axes = axes[:, idx]
        
        # Ensure right-handed coordinate system
        if np.linalg.det(axes) < 0:
            axes[:, 2] = -axes[:, 2]
        
        # Transform positions to principal axis frame
        principal_positions = centered @ axes
        
        return AlignmentResult(
            positions=principal_positions,
            moments=moments,
            axes=axes
        )
    
    @staticmethod
    def _calculate_inertia_tensor(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """Calculate the moment of inertia tensor."""
        inertia = np.zeros((3, 3))
        
        for k in range(len(positions)):
            r = positions[k]
            
            # Diagonal terms: I_xx = sum(m * (y^2 + z^2))
            inertia[0, 0] += masses[k] * (r[1]**2 + r[2]**2)
            inertia[1, 1] += masses[k] * (r[0]**2 + r[2]**2)  
            inertia[2, 2] += masses[k] * (r[0]**2 + r[1]**2)
            
            # Off-diagonal terms: I_xy = -sum(m * x * y)
            inertia[0, 1] -= masses[k] * r[0] * r[1]
            inertia[0, 2] -= masses[k] * r[0] * r[2]
            inertia[1, 2] -= masses[k] * r[1] * r[2]
        
        # Symmetrize
        inertia[1, 0] = inertia[0, 1]
        inertia[2, 0] = inertia[0, 2]
        inertia[2, 1] = inertia[1, 2]
        
        return inertia


# =============================================================================
# ATOM MATCHER CLASS
# =============================================================================

class AtomMatcher:
    """Handles atom correspondence matching between molecules using grouped permutations."""
    
    def __init__(self, rmsd_threshold: float = 1.0):
        self.rmsd_threshold = rmsd_threshold
        self.logger = logging.getLogger(__name__)
    
    def match_molecules(self, pos1: np.ndarray, pos2: np.ndarray, 
                       elements1: List[str], elements2: List[str]) -> MatchingResult:
        """
        Match two molecules using grouped permutations.
        
        Args:
            pos1: Positions of first molecule (N x 3)
            pos2: Positions of second molecule (N x 3)
            elements1: Element symbols for first molecule
            elements2: Element symbols for second molecule
            
        Returns:
            MatchingResult with mapping and RMSD
        """
        if len(pos1) != len(pos2):
            raise MatchingError(f"Molecules have different numbers of atoms: {len(pos1)} vs {len(pos2)}")
        
        # Try different matching strategies in order
        for strategy_name, strategy_func in [
            ("identity", self._try_identity_mapping),
            ("kabsch", self._try_kabsch_alignment), 
            ("grouped_permutations", self._try_grouped_permutations)
        ]:
            self.logger.info(f"Trying {strategy_name} matching strategy")
            
            try:
                result = strategy_func(pos1, pos2, elements1, elements2)
                if result.success:
                    self.logger.info(f"{strategy_name} matching succeeded with RMSD: {result.rmsd:.4f}")
                    return result
                else:
                    self.logger.info(f"{strategy_name} matching failed with RMSD: {result.rmsd:.4f}")
            except Exception as e:
                self.logger.warning(f"{strategy_name} matching failed: {e}")
        
        raise MatchingError(f"Could not match molecules within RMSD threshold {self.rmsd_threshold}")
    
    def _try_identity_mapping(self, pos1: np.ndarray, pos2: np.ndarray, 
                             elements1: List[str], elements2: List[str]) -> MatchingResult:
        """Try identity mapping (as-is)."""
        identity_mapping = list(range(len(pos1)))
        rmsd = calculate_rmsd(pos1, pos2, identity_mapping)
        
        return MatchingResult(
            mapping=identity_mapping,
            rmsd=rmsd,
            rotation=np.eye(3),
            translation=np.zeros(3),
            success=rmsd <= self.rmsd_threshold
        )
    
    def _try_kabsch_alignment(self, pos1: np.ndarray, pos2: np.ndarray,
                             elements1: List[str], elements2: List[str]) -> MatchingResult:
        """Try Kabsch rotation alignment."""
        center1 = np.mean(pos1, axis=0)
        center2 = np.mean(pos2, axis=0)
        
        pos1_centered = pos1 - center1
        pos2_centered = pos2 - center2
        
        try:
            R = kabsch_rotation_matrix(pos1_centered, pos2_centered)
            t = center2 - R @ center1
            
            pos1_transformed = (R @ pos1_centered.T).T + center2
            rmsd = calculate_rmsd(pos1_transformed, pos2)
            
            return MatchingResult(
                mapping=list(range(len(pos1))),
                rmsd=rmsd,
                rotation=R,
                translation=t,
                success=rmsd <= self.rmsd_threshold
            )
        except Exception as e:
            return MatchingResult(
                mapping=list(range(len(pos1))),
                rmsd=float('inf'),
                rotation=np.eye(3),
                translation=np.zeros(3),
                success=False
            )
    
    def _try_grouped_permutations(self, pos1: np.ndarray, pos2: np.ndarray,
                                 elements1: List[str], elements2: List[str]) -> MatchingResult:
        """Try grouped permutations matching by element type."""
        # Group atoms by element type
        groups1 = self._group_atoms_by_element(elements1)
        groups2 = self._group_atoms_by_element(elements2)
        
        # Check that groups match
        if set(groups1.keys()) != set(groups2.keys()):
            return MatchingResult(
                mapping=list(range(len(pos1))),
                rmsd=float('inf'),
                rotation=np.eye(3),
                translation=np.zeros(3),
                success=False
            )
        
        # Find best permutation for each element group independently
        final_mapping = [-1] * len(pos1)
        total_rmsd_sq = 0.0
        
        for element in groups1.keys():
            indices1 = groups1[element]
            indices2 = groups2[element]
            
            if len(indices1) != len(indices2):
                return MatchingResult(
                    mapping=list(range(len(pos1))),
                    rmsd=float('inf'),
                    rotation=np.eye(3),
                    translation=np.zeros(3),
                    success=False
                )
            
            # Find best permutation for this element group
            group_pos1 = pos1[indices1]
            group_pos2 = pos2[indices2]
            
            best_rmsd = float('inf')
            best_perm = None
            
            for perm in permutations(range(len(indices2))):
                perm_indices = [indices2[i] for i in perm]
                group_pos2_perm = pos2[perm_indices]
                rmsd = calculate_rmsd(group_pos1, group_pos2_perm)
                
                if rmsd < best_rmsd:
                    best_rmsd = rmsd
                    best_perm = perm_indices
            
            # Add this group's contribution to total RMSD
            total_rmsd_sq += (best_rmsd ** 2) * len(indices1)
            
            # Set mapping for this group
            for i, idx1 in enumerate(indices1):
                final_mapping[idx1] = best_perm[i]
        
        # Calculate final RMSD
        final_rmsd = np.sqrt(total_rmsd_sq / len(pos1))
        
        return MatchingResult(
            mapping=final_mapping,
            rmsd=final_rmsd,
            rotation=np.eye(3),
            translation=np.zeros(3),
            success=final_rmsd <= self.rmsd_threshold
        )
    
    def _group_atoms_by_element(self, elements: List[str]) -> Dict[str, List[int]]:
        """Group atom indices by element type."""
        groups = {}
        for i, element in enumerate(elements):
            if element not in groups:
                groups[element] = []
            groups[element].append(i)
        return groups


# =============================================================================
# MOLECULE MATCHING PIPELINE
# =============================================================================

class MoleculeMatchingPipeline:
    """Main pipeline for matching DMA charges to crystal molecules."""
    
    def __init__(self, rmsd_threshold: float = 1.0, verbose: bool = False):
        self.rmsd_threshold = rmsd_threshold
        self.verbose = verbose
        self.alignment = MolecularAlignment()
        self.matcher = AtomMatcher(rmsd_threshold)
        
        # Set up logging
        if verbose:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        else:
            logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
            
        self.logger = logging.getLogger(__name__)
    
    def match_dma_to_molecule(self, dma_positions: np.ndarray, dma_elements: List[str],
                            crystal_positions: np.ndarray, crystal_elements: List[str]) -> MatchingResult:
        """
        Match DMA molecule to crystal molecule using the complete pipeline.
        
        Steps:
        1. Transform both molecules to principal axis frames
        2. Try matching strategies in order of complexity
        3. Return best result
        """
        self.logger.info("=== Starting molecular matching pipeline ===")
        
        # Step 1: Transform to principal axis frames
        self.logger.info("Step 1: Transforming molecules to principal axis frames")
        
        dma_centered = dma_positions - np.mean(dma_positions, axis=0)
        crystal_centered = crystal_positions - np.mean(crystal_positions, axis=0)
        
        dma_alignment = self.alignment.to_principal_axis_frame(dma_centered)
        crystal_alignment = self.alignment.to_principal_axis_frame(crystal_centered)
        
        if self.verbose:
            self._log_alignment_info("DMA", dma_alignment)
            self._log_alignment_info("Crystal", crystal_alignment)
            self._log_positions("DMA", dma_alignment.positions, dma_elements)
            self._log_positions("Crystal", crystal_alignment.positions, crystal_elements)
        
        # Step 2: Try matching strategies
        self.logger.info("Step 2: Attempting molecular matching")
        
        result = self.matcher.match_molecules(
            dma_alignment.positions, 
            crystal_alignment.positions,
            dma_elements, 
            crystal_elements
        )
        
        if result.success:
            self.logger.info(f"Successfully matched molecules with RMSD: {result.rmsd:.4f} Å")
        else:
            self.logger.error(f"Failed to match molecules. Best RMSD: {result.rmsd:.4f} Å")
        
        return result
    
    def _log_alignment_info(self, name: str, alignment: AlignmentResult):
        """Log alignment information."""
        self.logger.info(f"{name} principal moments of inertia:")
        for i, moment in enumerate(alignment.moments):
            self.logger.info(f"  I{i+1}: {moment:12.5f}")
        
        self.logger.info(f"{name} principal axes (original frame):")
        for i in range(3):
            self.logger.info(f"  Axis{i+1}: {alignment.axes[0,i]:8.5f} {alignment.axes[1,i]:8.5f} {alignment.axes[2,i]:8.5f}")
    
    def _log_positions(self, name: str, positions: np.ndarray, elements: List[str]):
        """Log molecular positions."""
        self.logger.info(f"{name} axis positions (centered at origin):")
        for i, pos in enumerate(positions):
            self.logger.info(f"  {i:2d} {elements[i]:>2s}: {pos[0]:12.5f} {pos[1]:12.5f} {pos[2]:12.5f}")


# =============================================================================
# MAIN MATCHING FUNCTION (updated to use new pipeline)
# =============================================================================

def match_dma_to_crystal(dma_file: str, cif_file: str, output_file: Optional[str] = None,
                         tolerance: float = 0.5, verbose: bool = False):
    """
    Match DMA charges to crystal structure using the new modular pipeline.
    
    Args:
        dma_file: Path to DMA JSON file (from parse_dma_charges.py)
        cif_file: Path to CIF crystal structure file
        output_file: Optional output JSON file for mapped charges
        tolerance: Distance tolerance for matching (Angstroms)
        verbose: Print detailed output
    """
    # Create pipeline
    pipeline = MoleculeMatchingPipeline(rmsd_threshold=tolerance, verbose=verbose)
    
    # Load DMA data
    dma_data = load_dma_data(dma_file)
    dma_labels, dma_positions, dma_charges = extract_molecule_from_dma(dma_data)
    dma_elements = clean_atom_labels(dma_labels)
    
    if verbose:
        print(f"DMA molecule: {len(dma_labels)} atoms")
        print(f"Elements: {', '.join(dma_elements)}")
        print(f"Total charge: {np.sum(dma_charges):.6f}")
    
    # Load crystal structure
    crystal = Crystal.load(cif_file)
    
    if verbose:
        print(f"\nCrystal loaded from: {cif_file}")
        print(f"Space group: {crystal.space_group.symbol if crystal.space_group else 'Unknown'}")
        print(f"Asymmetric unit: {crystal.asymmetric_unit.formula}")
        print(f"Unit cell molecules: {len(crystal.unit_cell_molecules())}")
    
    # Get symmetry-unique molecules
    molecules = crystal.symmetry_unique_molecules()
    
    if verbose:
        print(f"Found {len(molecules)} symmetry-unique molecule(s)")
    
    # Try to match with each molecule
    best_match = None
    best_rmsd = float('inf')
    
    for mol_idx, molecule in enumerate(molecules):
        mol_positions = molecule.positions
        mol_elements = [str(elem) for elem in molecule.elements]
        
        if verbose:
            print(f"\nMolecule {mol_idx + 1}: {len(mol_elements)} atoms")
            print(f"Elements: {', '.join(mol_elements)}")
        
        # Check basic compatibility
        if len(mol_elements) != len(dma_elements):
            if verbose:
                print(f"  Skipping - different number of atoms")
            continue
        
        if sorted(mol_elements) != sorted(dma_elements):
            if verbose:
                print(f"  Skipping - different element composition")
            continue
        
        try:
            # Use the new pipeline to match molecules
            result = pipeline.match_dma_to_molecule(
                dma_positions, dma_elements,
                mol_positions, mol_elements
            )
            
            if result.success and result.rmsd < best_rmsd:
                best_rmsd = result.rmsd
                best_match = {
                    'molecule_idx': mol_idx,
                    'molecule': molecule,
                    'result': result
                }
                
                if verbose:
                    print(f"  New best match! RMSD: {result.rmsd:.4f} Å")
            elif verbose:
                print(f"  Match failed or worse RMSD: {result.rmsd:.4f} Å")
                
        except MatchingError as e:
            if verbose:
                print(f"  Matching failed: {e}")
            continue
    
    if best_match is None:
        print("ERROR: Could not match DMA molecule to any crystal molecule")
        return None
    
    # Extract results
    molecule = best_match['molecule']
    result = best_match['result']
    mapping = result.mapping
    
    print(f"\nBest match: Molecule {best_match['molecule_idx'] + 1}")
    print(f"RMSD: {result.rmsd:.4f} Å")
    
    # Create charge mapping
    charge_mapping = {}
    mol_elements = [str(elem) for elem in molecule.elements]
    mol_positions = molecule.positions
    
    for dma_idx, crystal_idx in enumerate(mapping):
        element = mol_elements[crystal_idx]
        position = mol_positions[crystal_idx]
        charge = dma_charges[dma_idx]
        
        charge_mapping[f"{element}_{crystal_idx}"] = {
            'dma_label': dma_labels[dma_idx],
            'element': element,
            'charge': charge,
            'position': position.tolist(),
            'crystal_idx': crystal_idx,
            'dma_idx': dma_idx
        }
    
    if verbose:
        print("\nCharge mapping:")
        print(f"{'DMA Label':<12} {'Element':<8} {'Charge':>10} {'Crystal Idx'}")
        print("-" * 45)
        for dma_idx, crystal_idx in enumerate(mapping):
            element = mol_elements[crystal_idx]
            print(f"{dma_labels[dma_idx]:<12} {element:<8} {dma_charges[dma_idx]:>10.6f} {crystal_idx:>5}")
    
    # Create output data
    output_data = {
        'cif_file': str(cif_file),
        'dma_file': str(dma_file),
        'molecule_idx': best_match['molecule_idx'],
        'rmsd': result.rmsd,
        'rotation_matrix': result.rotation.tolist(),
        'translation_vector': result.translation.tolist(),
        'charge_mapping': charge_mapping,
        'total_charge': float(np.sum(dma_charges)),
        'charges_by_element': {}
    }
    
    # Summarize charges by element
    for element in set(dma_elements):
        element_charges = [dma_charges[i] for i, e in enumerate(dma_elements) if e == element]
        output_data['charges_by_element'][element] = {
            'charges': element_charges,
            'mean': float(np.mean(element_charges)),
            'std': float(np.std(element_charges))
        }
    
    # Save output
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved charge mapping to {output_file}")
    
    return output_data


def save_raspa_charges(output_data: dict, crystal: Crystal, charge_file: str):
    """
    Save charges in RASPA-compatible format for asymmetric unit atoms.
    Uses the same labeling scheme as the RASPA generator.
    
    Args:
        output_data: Output from match_dma_to_crystal
        crystal: Crystal object
        charge_file: Path to save RASPA charges JSON
    """
    # Import the exact same function used by RASPA generator
    from raspa_isotherm_tools.force_field import generate_default_labels
    
    # Generate labels using the exact same function as RASPA generator
    default_labels = generate_default_labels(crystal)
    n_asym = len(crystal.asymmetric_unit.atomic_numbers)
    
    raspa_charges = {}
    
    for i in range(n_asym):
        label = default_labels[i]
        
        # Find corresponding charge from DMA mapping
        # Look for the crystal index that matches this asymmetric unit atom
        charge = 0.0  # Default if not found
        
        for key, data in output_data['charge_mapping'].items():
            if data['crystal_idx'] == i:
                charge = data['charge']
                break
        
        raspa_charges[label] = charge
    
    # Save to file
    with open(charge_file, 'w') as f:
        json.dump(raspa_charges, f, indent=2)
    
    print(f"\nSaved RASPA-compatible charges to {charge_file}")
    print("Charges for asymmetric unit atoms (using RASPA generator labeling):")
    for label, charge in raspa_charges.items():
        print(f"  {label}: {charge:>10.6f}")
    
    return raspa_charges


def main():
    parser = argparse.ArgumentParser(description='Match DMA charges to crystal structure')
    parser.add_argument('dma_json', help='DMA charges JSON file (from parse_dma_charges.py)')
    parser.add_argument('cif_file', help='Crystal structure CIF file')
    parser.add_argument('-o', '--output', help='Output JSON file for charge mapping')
    parser.add_argument('--raspa-charges', help='Output JSON file for RASPA-compatible charges')
    parser.add_argument('-t', '--tolerance', type=float, default=0.5,
                       help='Distance tolerance for matching (Angstroms, default: 0.5)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Run matching
    result = match_dma_to_crystal(
        args.dma_json,
        args.cif_file,
        args.output,
        args.tolerance,
        args.verbose
    )
    
    if result:
        print(f"\nSuccessfully matched DMA charges to crystal structure")
        print(f"Total charge: {result['total_charge']:.6f}")
        print("\nCharges by element:")
        for element, data in result['charges_by_element'].items():
            print(f"  {element}: mean={data['mean']:.6f}, std={data['std']:.6f}")
        
        # Save RASPA-compatible charges if requested
        if args.raspa_charges:
            crystal = Crystal.load(args.cif_file)
            from chmpy.core.element import Element
            save_raspa_charges(result, crystal, args.raspa_charges)


if __name__ == "__main__":
    main()
