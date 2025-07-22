"""
Plotting utilities for RASPA simulation results.

This module provides functions to create publication-quality plots of adsorption isotherms
and other simulation data.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from .database import RASPADatabase


def plot_isotherm(df: pd.DataFrame, 
                 framework_name: str,
                 component_name: str,
                 output_path: Optional[Path] = None,
                 loading_type: str = "abs",
                 loading_units: str = "mol_per_kg",
                 pressure_units: str = "bar",
                 show_excess: bool = False,
                 show_grid: bool = False,
                 figsize: tuple = (10, 6),
                 dpi: int = 300) -> Path:
    """
    Plot an adsorption isotherm with error bars.
    
    Args:
        df: DataFrame containing isotherm data (from RASPADatabase.get_isotherm_data())
        framework_name: Name of the framework
        component_name: Name of the adsorbate component
        output_path: Path to save the plot (optional, will generate if not provided)
        loading_type: "abs" for absolute or "excess" for excess loading
        loading_units: "mol_per_kg", "mg_per_g", or "molecules_per_cell"
        pressure_units: "bar", "kpa", or "pa"
        show_excess: Whether to show both absolute and excess on same plot
        show_grid: Whether to show grid lines (default: False for cleaner look)
        figsize: Figure size tuple
        dpi: Resolution for saved figure
        
    Returns:
        Path to saved plot file
    """
    # Generate output path if not provided
    if output_path is None:
        output_path = Path(f"{framework_name}_{component_name}_isotherm.png")
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine column names based on units
    pressure_col = f"pressure_{pressure_units}"
    loading_col = f"loading_{loading_type}_{loading_units}"
    error_col = f"loading_{loading_type}_{loading_units}_error"
    
    # Handle special case for molecules/cell (uses different error column)
    if loading_units == "molecules_per_cell":
        error_col = f"loading_{loading_type}_error"
    
    # Plot the main isotherm
    if error_col in df.columns and df[error_col].sum() > 0:
        # Plot with error bars if error data is available
        ax.errorbar(df[pressure_col], df[loading_col], yerr=df[error_col],
                   fmt='o-', linewidth=2, markersize=6, capsize=3,
                   label=f"{loading_type.capitalize()} loading")
    else:
        # Plot without error bars
        ax.plot(df[pressure_col], df[loading_col], 'o-', 
               linewidth=2, markersize=6, 
               label=f"{loading_type.capitalize()} loading")
    
    # Plot excess loading if requested and different from main plot
    if show_excess and loading_type != "excess":
        excess_col = f"loading_excess_{loading_units}"
        excess_error_col = f"loading_excess_{loading_units}_error"
        
        if loading_units == "molecules_per_cell":
            excess_error_col = "loading_excess_error"
        
        if excess_col in df.columns:
            if excess_error_col in df.columns and df[excess_error_col].sum() > 0:
                ax.errorbar(df[pressure_col], df[excess_col], yerr=df[excess_error_col],
                           fmt='s--', linewidth=2, markersize=6, capsize=3,
                           label="Excess loading", alpha=0.7)
            else:
                ax.plot(df[pressure_col], df[excess_col], 's--',
                       linewidth=2, markersize=6, 
                       label="Excess loading", alpha=0.7)
    
    # Customize the plot appearance
    ax.set_xlabel(_get_pressure_label(pressure_units))
    ax.set_ylabel(_get_loading_label(loading_units))
    ax.set_title(f"{framework_name} + {component_name} Isotherm")
    ax.legend(frameon=False)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make left and bottom spines thicker
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Optional grid
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Set reasonable axis limits with small margin
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    # Improve tick appearance
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_isotherm_from_database(db_path: str,
                               framework_name: str, 
                               component_name: str,
                               output_path: Optional[Path] = None,
                               **kwargs) -> Path:
    """
    Plot an isotherm directly from a database file.
    
    Args:
        db_path: Path to the DuckDB database file
        framework_name: Name of the framework
        component_name: Name of the adsorbate component
        output_path: Path to save the plot (optional)
        **kwargs: Additional arguments passed to plot_isotherm()
        
    Returns:
        Path to saved plot file
    """
    db = RASPADatabase(db_path)
    try:
        df = db.get_isotherm_data(framework_name, component_name)
        if len(df) == 0:
            raise ValueError(f"No data found for {framework_name}-{component_name}")
        
        return plot_isotherm(df, framework_name, component_name, output_path, **kwargs)
    finally:
        db.close()


def _get_pressure_label(pressure_units: str) -> str:
    """Get formatted pressure axis label."""
    labels = {
        "pa": "Pressure (Pa)",
        "kpa": "Pressure (kPa)", 
        "bar": "Pressure (bar)"
    }
    return labels.get(pressure_units, "Pressure")


def _get_loading_label(loading_units: str) -> str:
    """Get formatted loading axis label."""
    labels = {
        "mol_per_kg": "Loading (mol/kg)",
        "mg_per_g": "Loading (mg/g)",
        "molecules_per_cell": "Loading (molecules/unit cell)"
    }
    return labels.get(loading_units, "Loading")


def create_comparison_plot(db_path: str,
                          frameworks: list[str],
                          component_name: str,
                          output_path: Optional[Path] = None,
                          **kwargs) -> Path:
    """
    Create a comparison plot of isotherms for multiple frameworks.
    
    Args:
        db_path: Path to the DuckDB database file
        frameworks: List of framework names to compare
        component_name: Name of the adsorbate component
        output_path: Path to save the plot (optional)
        **kwargs: Additional arguments for plot customization
        
    Returns:
        Path to saved plot file
    """
    if output_path is None:
        framework_names = "_vs_".join(frameworks)
        output_path = Path(f"{framework_names}_{component_name}_comparison.png")
    
    db = RASPADatabase(db_path)
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    
    try:
        for framework in frameworks:
            df = db.get_isotherm_data(framework, component_name)
            if len(df) == 0:
                print(f"Warning: No data found for {framework}-{component_name}")
                continue
            
            # Use same column logic as single isotherm plot
            loading_type = kwargs.get('loading_type', 'abs')
            loading_units = kwargs.get('loading_units', 'mol_per_kg')
            pressure_units = kwargs.get('pressure_units', 'bar')
            
            pressure_col = f"pressure_{pressure_units}"
            loading_col = f"loading_{loading_type}_{loading_units}"
            error_col = f"loading_{loading_type}_{loading_units}_error"
            
            if loading_units == "molecules_per_cell":
                error_col = f"loading_{loading_type}_error"
            
            # Plot each framework
            if error_col in df.columns and df[error_col].sum() > 0:
                ax.errorbar(df[pressure_col], df[loading_col], yerr=df[error_col],
                           fmt='o-', linewidth=2, markersize=6, capsize=3,
                           label=framework)
            else:
                ax.plot(df[pressure_col], df[loading_col], 'o-',
                       linewidth=2, markersize=6, label=framework)
        
        # Customize plot appearance
        ax.set_xlabel(_get_pressure_label(pressure_units))
        ax.set_ylabel(_get_loading_label(loading_units))
        ax.set_title(f"Framework Comparison: {component_name} Isotherms")
        ax.legend(frameon=False)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Make left and bottom spines thicker
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        
        # No grid by default
        # ax.grid(True, alpha=0.3)  # Users can uncomment if they want grid
        
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        
        # Improve tick appearance
        ax.tick_params(axis='both', which='major', labelsize=11, width=1.2)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
        plt.close()
        
        return output_path
        
    finally:
        db.close()


def plot_and_export_isotherm(db_path: str,
                            framework_name: str,
                            component_name: str,
                            output_dir: Optional[Path] = None,
                            **kwargs) -> dict[str, Path]:
    """
    Create both a plot and data export for an isotherm.
    
    Args:
        db_path: Path to the DuckDB database file
        framework_name: Name of the framework
        component_name: Name of the adsorbate component
        output_dir: Directory to save files (optional, uses current dir if not provided)
        **kwargs: Additional arguments passed to plot_isotherm()
        
    Returns:
        Dict with paths to created files: {'plot': path, 'csv': path, 'excel': path}
    """
    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    base_name = f"{framework_name}_{component_name}_isotherm"
    
    # Create plot
    plot_path = output_dir / f"{base_name}.png"
    plot_isotherm_from_database(db_path, framework_name, component_name, plot_path, **kwargs)
    
    # Export data
    db = RASPADatabase(db_path)
    try:
        csv_path = output_dir / f"{base_name}.csv"
        excel_path = output_dir / f"{base_name}.xlsx"
        
        db.export_isotherm_to_csv(str(csv_path), framework_name, component_name)
        db.export_isotherm_to_excel(str(excel_path), framework_name, component_name)
        
        return {
            'plot': plot_path,
            'csv': csv_path, 
            'excel': excel_path
        }
    finally:
        db.close()