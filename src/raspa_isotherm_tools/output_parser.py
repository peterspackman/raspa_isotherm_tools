"""
RASPA3 output file parser.

This module provides functionality to parse RASPA3 output files (JSON and text formats).
"""

import json
import re
from pathlib import Path
from typing import Any


class RASPAOutputParser:
    """
    Parser for RASPA3 output files.
    
    Supports both JSON and text output formats, with JSON being preferred
    for metadata and text files required for loading data.
    """
    
    def __init__(self):
        self.results = []
    
    def parse_json_output(self, json_path: Path) -> dict[str, Any]:
        """
        Parse RASPA3 JSON output file for metadata and energy data.
        
        Args:
            json_path: Path to JSON output file
            
        Returns:
            Dict containing extracted simulation metadata
        """
        with json_path.open() as f:
            data = json.load(f)
        
        # Extract simulation metadata
        init_conditions = data.get("initialization", {}).get("initialConditions", {})
        
        result = {
            "file_path": str(json_path),
            "pressure_pa": init_conditions.get("pressure", 0.0),
            "temperature_k": init_conditions.get("temperature", 0.0),
            "framework_name": self._extract_framework_name(data),
            "component_name": self._extract_component_name(data),
        }
        
        # Extract properties
        properties = data.get("properties", {})
        
        # Extract energy data
        self._extract_energy_data(properties, result)
        
        # Extract enthalpy data
        component_name = result["component_name"]
        self._extract_enthalpy_data(properties, result, component_name)
        
        # Initialize loading data (will be filled from text file)
        result.update({
            "loading_abs_molecules_per_cell": 0.0,
            "loading_abs_mol_per_kg": 0.0,
            "loading_abs_mg_per_g": 0.0,
            "loading_excess_molecules_per_cell": 0.0,
            "loading_excess_mol_per_kg": 0.0,
            "loading_excess_mg_per_g": 0.0,
            "loading_abs_error": 0.0,
            "loading_excess_error": 0.0,
            "loading_abs_mol_per_kg_error": 0.0,
            "loading_excess_mol_per_kg_error": 0.0,
        })
        
        return result
    
    def parse_combined_output(self, json_path: Path, txt_path: Path = None) -> dict[str, Any]:
        """
        Parse both JSON and text output files to get complete data.
        
        Args:
            json_path: Path to JSON output file
            txt_path: Path to text output file (optional, will be inferred if not provided)
            
        Returns:
            Dict containing complete simulation results
        """
        # Start with JSON data for metadata and energies
        result = self.parse_json_output(json_path)
        
        # Find corresponding text file if not provided
        if txt_path is None:
            txt_path = json_path.with_suffix('.txt')
        
        # Extract loading data from text file
        if txt_path.exists():
            self._extract_loading_from_text_file(txt_path, result)
        
        return result
    
    def parse_text_output(self, txt_path: Path) -> dict[str, Any]:
        """
        Parse RASPA3 text output file.
        
        Args:
            txt_path: Path to text output file
            
        Returns:
            Dict containing extracted simulation results
        """
        with txt_path.open() as f:
            content = f.read()
        
        result = {
            "file_path": str(txt_path),
            "pressure_pa": self._extract_pressure_from_text(content),
            "temperature_k": self._extract_temperature_from_text(content),
            "framework_name": self._extract_framework_name_from_text(content),
            "component_name": "CO2",  # Default assumption
        }
        
        # Extract loading data from text
        self._extract_loading_from_text(content, result)
        
        return result
    
    def _extract_framework_name(self, data: dict) -> str:
        """Extract framework name from JSON data."""
        components = data.get("initialization", {}).get("components", {})
        for name, info in components.items():
            if name != "CO2":  # Skip adsorbate components
                return name
        return "unknown"
    
    def _extract_component_name(self, data: dict) -> str:
        """Extract adsorbate component name from JSON data."""
        components = data.get("initialization", {}).get("components", {})
        for name, info in components.items():
            if info.get("swappable", False):  # Swappable components are adsorbates
                return name
        return "CO2"  # Default assumption
    
    def _extract_loading_from_text_file(self, txt_path: Path, result: dict) -> None:
        """Extract loading data from text output file."""
        with txt_path.open() as f:
            content = f.read()
        
        # Extract absolute loading with proper error handling
        self._extract_loading_from_text(content, result)
    
    def _extract_energy_data(self, properties: dict, result: dict) -> None:
        """Extract energy data from properties section."""
        avg_energies = properties.get("averageEnergies", {})
        
        # Framework-molecule energies
        fw_mol = avg_energies.get("Framework-Molecule", {})
        total_fw_mol = 0.0
        
        for interaction, energy_data in fw_mol.items():
            if "total" in energy_data:
                total_fw_mol += energy_data["total"].get("mean", 0.0)
        
        result.update({
            "energy_framework_molecule_k": total_fw_mol,
            "energy_total_k": avg_energies.get("totalEnergy", {}).get("mean", 0.0),
        })
    
    def _extract_enthalpy_data(self, properties: dict, result: dict, component_name: str) -> None:
        """Extract enthalpy data from properties section."""
        avg_enthalpy = properties.get("averageEnthalpy", {})
        
        if component_name in avg_enthalpy:
            enthalpy_data = avg_enthalpy[component_name]
            result.update({
                "enthalpy_adsorption_k": enthalpy_data.get("mean", {}).get("[K]", 0.0),
                "enthalpy_adsorption_kj_mol": enthalpy_data.get("mean", {}).get("[kJ/mol]", 0.0),
            })
        else:
            result.update({
                "enthalpy_adsorption_k": 0.0,
                "enthalpy_adsorption_kj_mol": 0.0,
            })
    
    def _extract_pressure_from_text(self, content: str) -> float:
        """Extract pressure from text output."""
        # Look for pressure in the initial conditions
        match = re.search(r"pressure:\s*([0-9.e+-]+)", content, re.IGNORECASE)
        return float(match.group(1)) if match else 0.0
    
    def _extract_temperature_from_text(self, content: str) -> float:
        """Extract temperature from text output."""
        match = re.search(r"temperature:\s*([0-9.e+-]+)", content, re.IGNORECASE)
        return float(match.group(1)) if match else 0.0
    
    def _extract_framework_name_from_text(self, content: str) -> str:
        """Extract framework name from text output."""
        # This would need to be implemented based on text format
        return "unknown"
    
    def _extract_loading_from_text(self, content: str, result: dict) -> None:
        """Extract loading data from text output using regex."""
        # Initialize all values to 0
        result.update({
            "loading_abs_molecules_per_cell": 0.0,
            "loading_abs_mol_per_kg": 0.0,
            "loading_abs_mg_per_g": 0.0,
            "loading_excess_molecules_per_cell": 0.0,
            "loading_excess_mol_per_kg": 0.0,
            "loading_excess_mg_per_g": 0.0,
            "loading_abs_error": 0.0,
            "loading_excess_error": 0.0,
            "loading_abs_mol_per_kg_error": 0.0,
            "loading_excess_mol_per_kg_error": 0.0,
        })
        
        # Extract absolute loading in molecules/cell
        abs_loading_match = re.search(
            r"Abs\. loading average\s+([0-9.e+-]+)\s+\+/-\s+([0-9.e+-]+)\s+\[molecules/cell\]",
            content
        )
        if abs_loading_match:
            result["loading_abs_molecules_per_cell"] = float(abs_loading_match.group(1))
            result["loading_abs_error"] = float(abs_loading_match.group(2))  # molecules/cell error
        
        # Extract absolute loading in mol/kg
        abs_mol_kg_match = re.search(
            r"Abs\. loading average\s+([0-9.e+-]+)\s+\+/-\s+([0-9.e+-]+)\s+\[mol/kg-framework\]",
            content
        )
        if abs_mol_kg_match:
            result["loading_abs_mol_per_kg"] = float(abs_mol_kg_match.group(1))
            result["loading_abs_mol_per_kg_error"] = float(abs_mol_kg_match.group(2))  # mol/kg error
        
        # Extract absolute loading in mg/g
        abs_mg_g_match = re.search(
            r"Abs\. loading average\s+([0-9.e+-]+)\s+\+/-\s+([0-9.e+-]+)\s+\[mg/g-framework\]",
            content
        )
        if abs_mg_g_match:
            result["loading_abs_mg_per_g"] = float(abs_mg_g_match.group(1))
        
        # Extract excess loading in molecules/cell
        excess_loading_match = re.search(
            r"Excess loading average\s+([0-9.e+-]+)\s+\+/-\s+([0-9.e+-]+)\s+\[molecules/cell\]",
            content
        )
        if excess_loading_match:
            result["loading_excess_molecules_per_cell"] = float(excess_loading_match.group(1))
            result["loading_excess_error"] = float(excess_loading_match.group(2))  # molecules/cell error
        
        # Extract excess loading in mol/kg
        excess_mol_kg_match = re.search(
            r"Excess loading average\s+([0-9.e+-]+)\s+\+/-\s+([0-9.e+-]+)\s+\[mol/kg-framework\]",
            content
        )
        if excess_mol_kg_match:
            result["loading_excess_mol_per_kg"] = float(excess_mol_kg_match.group(1))
            result["loading_excess_mol_per_kg_error"] = float(excess_mol_kg_match.group(2))  # mol/kg error
        
        # Extract excess loading in mg/g  
        excess_mg_g_match = re.search(
            r"Excess loading average\s+([0-9.e+-]+)\s+\+/-\s+([0-9.e+-]+)\s+\[mg/g-framework\]",
            content
        )
        if excess_mg_g_match:
            result["loading_excess_mg_per_g"] = float(excess_mg_g_match.group(1))