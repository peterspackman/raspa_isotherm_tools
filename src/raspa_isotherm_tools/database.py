"""
Database management for RASPA simulation results.

This module provides functionality to store and retrieve RASPA simulation results
using DuckDB as the backend database.
"""

from typing import Any

import duckdb
import pandas as pd


class RASPADatabase:
    """
    DuckDB database manager for RASPA simulation results.
    """
    
    def __init__(self, db_path: str = "raspa_results.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables for storing RASPA results."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS simulation_results (
                job_directory VARCHAR PRIMARY KEY,
                file_path VARCHAR,
                framework_name VARCHAR,
                component_name VARCHAR,
                pressure_pa DOUBLE,
                temperature_k DOUBLE,
                loading_abs_molecules_per_cell DOUBLE,
                loading_abs_mol_per_kg DOUBLE,
                loading_abs_mg_per_g DOUBLE,
                loading_excess_molecules_per_cell DOUBLE,
                loading_excess_mol_per_kg DOUBLE,
                loading_excess_mg_per_g DOUBLE,
                loading_abs_error DOUBLE,
                loading_excess_error DOUBLE,
                loading_abs_mol_per_kg_error DOUBLE,
                loading_excess_mol_per_kg_error DOUBLE,
                energy_framework_molecule_k DOUBLE,
                energy_total_k DOUBLE,
                enthalpy_adsorption_k DOUBLE,
                enthalpy_adsorption_kj_mol DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def insert_result(self, result: dict[str, Any]) -> None:
        """
        Insert a simulation result into the database.
        
        Args:
            result: Dict containing simulation results
        """
        columns = [
            "job_directory", "file_path", "framework_name", "component_name", "pressure_pa", "temperature_k",
            "loading_abs_molecules_per_cell", "loading_abs_mol_per_kg", "loading_abs_mg_per_g",
            "loading_excess_molecules_per_cell", "loading_excess_mol_per_kg", "loading_excess_mg_per_g",
            "loading_abs_error", "loading_excess_error", "loading_abs_mol_per_kg_error", "loading_excess_mol_per_kg_error",
            "energy_framework_molecule_k", "energy_total_k",
            "enthalpy_adsorption_k", "enthalpy_adsorption_kj_mol"
        ]
        
        values = [result.get(col, 0.0) for col in columns]
        placeholders = ", ".join(["?" for _ in columns])
        
        self.conn.execute(
            f"INSERT OR REPLACE INTO simulation_results ({', '.join(columns)}) VALUES ({placeholders})",
            values
        )
    
    def get_results_dataframe(self, framework_name: str = None, 
                            component_name: str = None) -> pd.DataFrame:
        """
        Get simulation results as a pandas DataFrame.
        
        Args:
            framework_name: Filter by framework name
            component_name: Filter by component name
            
        Returns:
            DataFrame containing filtered results
        """
        query = "SELECT * FROM simulation_results WHERE 1=1"
        params = []
        
        if framework_name:
            query += " AND framework_name = ?"
            params.append(framework_name)
        
        if component_name:
            query += " AND component_name = ?"
            params.append(component_name)
        
        query += " ORDER BY pressure_pa"
        
        return self.conn.execute(query, params).df()
    
    def get_isotherm_data(self, framework_name: str, component_name: str, 
                         temperature_k: float = None) -> pd.DataFrame:
        """
        Get isotherm data (pressure vs loading) for a specific framework-component pair.
        
        Args:
            framework_name: Framework name
            component_name: Component name  
            temperature_k: Temperature filter (optional)
            
        Returns:
            DataFrame with pressure and loading columns
        """
        query = """
            SELECT 
                pressure_pa,
                pressure_pa / 1000 as pressure_kpa,
                pressure_pa / 100000 as pressure_bar,
                loading_abs_molecules_per_cell,
                loading_abs_mol_per_kg,
                loading_abs_mg_per_g,
                loading_excess_molecules_per_cell,
                loading_excess_mol_per_kg,
                loading_excess_mg_per_g,
                loading_abs_error,
                loading_excess_error,
                loading_abs_mol_per_kg_error,
                loading_excess_mol_per_kg_error
            FROM simulation_results
            WHERE framework_name = ? AND component_name = ?
        """
        params = [framework_name, component_name]
        
        if temperature_k:
            query += " AND ABS(temperature_k - ?) < 0.1"
            params.append(temperature_k)
        
        query += " ORDER BY pressure_pa"
        
        return self.conn.execute(query, params).df()
    
    def get_framework_names(self) -> list[str]:
        """Get list of all framework names in the database."""
        result = self.conn.execute("SELECT DISTINCT framework_name FROM simulation_results ORDER BY framework_name").fetchall()
        return [row[0] for row in result]
    
    def get_component_names(self) -> list[str]:
        """Get list of all component names in the database."""
        result = self.conn.execute("SELECT DISTINCT component_name FROM simulation_results ORDER BY component_name").fetchall()
        return [row[0] for row in result]
    
    def export_to_csv(self, output_path: str, framework_name: str = None, 
                     component_name: str = None) -> None:
        """
        Export simulation results to CSV file.
        
        Args:
            output_path: Path to save the CSV file
            framework_name: Filter by framework name (optional)
            component_name: Filter by component name (optional)
        """
        df = self.get_results_dataframe(framework_name, component_name)
        df.to_csv(output_path, index=False)
    
    def export_to_excel(self, output_path: str, framework_name: str = None,
                       component_name: str = None) -> None:
        """
        Export simulation results to Excel file.
        
        Args:
            output_path: Path to save the Excel file  
            framework_name: Filter by framework name (optional)
            component_name: Filter by component name (optional)
        """
        df = self.get_results_dataframe(framework_name, component_name)
        df.to_excel(output_path, index=False)
    
    def export_isotherm_to_csv(self, output_path: str, framework_name: str, 
                              component_name: str, temperature_k: float = None) -> None:
        """
        Export isotherm data to CSV file.
        
        Args:
            output_path: Path to save the CSV file
            framework_name: Framework name
            component_name: Component name
            temperature_k: Temperature filter (optional)
        """
        df = self.get_isotherm_data(framework_name, component_name, temperature_k)
        df.to_csv(output_path, index=False)
    
    def export_isotherm_to_excel(self, output_path: str, framework_name: str,
                                component_name: str, temperature_k: float = None) -> None:
        """
        Export isotherm data to Excel file.
        
        Args:
            output_path: Path to save the Excel file
            framework_name: Framework name
            component_name: Component name  
            temperature_k: Temperature filter (optional)
        """
        df = self.get_isotherm_data(framework_name, component_name, temperature_k)
        df.to_excel(output_path, index=False)

    def close(self):
        """Close database connection."""
        self.conn.close()