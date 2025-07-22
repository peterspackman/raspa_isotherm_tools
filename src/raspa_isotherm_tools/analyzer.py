"""
CLI interface for RASPA3 output analysis.

This module provides a command-line interface for analyzing RASPA3 output files
and storing results in a database.
"""

import argparse
from pathlib import Path

from .database import RASPADatabase
from .output_parser import RASPAOutputParser
from .plotting import plot_isotherm_from_database


def analyze_raspa_outputs(jobs_dir: Path, db_path: str = "raspa_results.db", 
                         overwrite: bool = False) -> None:
    """
    Analyze all RASPA output files in a jobs directory and store results in database.
    
    Args:
        jobs_dir: Path to jobs directory containing simulation subdirectories
        db_path: Path to DuckDB database file
        overwrite: Whether to overwrite existing database
    """
    if overwrite and Path(db_path).exists():
        Path(db_path).unlink()
    
    parser = RASPAOutputParser()
    db = RASPADatabase(db_path)
    
    processed_count = 0
    error_count = 0
    
    # Find all output directories
    for job_dir in jobs_dir.iterdir():
        if not job_dir.is_dir():
            continue
            
        output_dir = job_dir / "output"
        if not output_dir.exists():
            continue
        
        # Look for JSON output first (preferred), then text
        json_files = list(output_dir.glob("*.json"))
        txt_files = list(output_dir.glob("*.txt"))
        
        try:
            if json_files and txt_files:
                # Use combined parsing (JSON for metadata, text for loading data)
                result = parser.parse_combined_output(json_files[0], txt_files[0])
            elif json_files:
                # JSON only (metadata only, no loading data)
                result = parser.parse_json_output(json_files[0])
            elif txt_files:
                # Fall back to text output only
                result = parser.parse_text_output(txt_files[0])
            else:
                print(f"No output files found in {output_dir}")
                continue
            
            # Add job directory info
            result["job_directory"] = job_dir.name
            
            # Insert into database
            db.insert_result(result)
            processed_count += 1
            
            print(f"Processed {job_dir.name}: P={result['pressure_pa']:.0f} Pa, "
                  f"Loading={result.get('loading_abs_mol_per_kg', 0):.3f} ± "
                  f"{result.get('loading_abs_mol_per_kg_error', 0):.3f} mol/kg")
            
        except Exception as e:
            print(f"Error processing {job_dir.name}: {e}")
            error_count += 1
    
    db.close()
    print(f"Analysis complete: {processed_count} jobs processed, {error_count} errors")
    print(f"Results stored in: {db_path}")


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Analyze RASPA3 output files and store results in DuckDB database"
    )
    parser.add_argument("jobs_dir", help="Directory containing RASPA job subdirectories")
    parser.add_argument("-o", "--output", default="raspa_results.db",
                       help="Output database file (default: raspa_results.db)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing database")
    parser.add_argument("--plot", nargs='*', metavar=("FRAMEWORK", "COMPONENT"),
                       help="Plot isotherm and save as PNG. Auto-detects framework/component if not specified.")
    parser.add_argument("--plot-output", metavar="FILE",
                       help="Output file for plot (default: auto-generated)")
    parser.add_argument("--export-csv", metavar="FILE",
                       help="Export all data to CSV file")
    parser.add_argument("--export-excel", metavar="FILE",
                       help="Export all data to Excel file")
    parser.add_argument("--export-isotherm-csv", metavar="FILE",
                       help="Export isotherm data to CSV file (auto-detects framework/component)")
    parser.add_argument("--export-isotherm-excel", metavar="FILE",
                       help="Export isotherm data to Excel file (auto-detects framework/component)")
    return parser


def main() -> int:
    """Main entry point for the command line interface."""
    parser = create_parser()
    args = parser.parse_args()
    
    jobs_dir = Path(args.jobs_dir)
    if not jobs_dir.exists() or not jobs_dir.is_dir():
        print(f"Error: Jobs directory {jobs_dir} does not exist")
        return 1
    
    try:
        # Analyze outputs and store in database
        analyze_raspa_outputs(jobs_dir, args.output, args.overwrite)
        
        # Create plot if requested
        if args.plot is not None:
            # Auto-detect framework and component if not specified
            if len(args.plot) == 0:
                # Auto-detect from database
                db = RASPADatabase(args.output)
                try:
                    frameworks = db.get_framework_names()
                    components = db.get_component_names()
                    
                    if len(frameworks) == 1 and len(components) == 1:
                        framework, component = frameworks[0], components[0]
                        print(f"Auto-detected: {framework} + {component}")
                    else:
                        print(f"Multiple frameworks/components found. Please specify: --plot FRAMEWORK COMPONENT")
                        print(f"Available frameworks: {frameworks}")
                        print(f"Available components: {components}")
                        db.close()
                        return 1
                finally:
                    db.close()
            elif len(args.plot) == 2:
                framework, component = args.plot
            else:
                print("Error: --plot requires either 0 arguments (auto-detect) or 2 arguments (FRAMEWORK COMPONENT)")
                return 1
            
            output_path = Path(args.plot_output) if args.plot_output else None
            
            try:
                plot_path = plot_isotherm_from_database(
                    args.output, framework, component, output_path
                )
                print(f"Isotherm plot saved: {plot_path}")
            except Exception as e:
                print(f"Error creating plot: {e}")
                return 1
        
        # Handle data exports
        db = None
        try:
            # Export all data to CSV
            if args.export_csv:
                if db is None:
                    db = RASPADatabase(args.output)
                db.export_to_csv(args.export_csv)
                print(f"Data exported to CSV: {args.export_csv}")
            
            # Export all data to Excel
            if args.export_excel:
                if db is None:
                    db = RASPADatabase(args.output)
                db.export_to_excel(args.export_excel)
                print(f"Data exported to Excel: {args.export_excel}")
            
            # Export isotherm data to CSV (auto-detect framework/component)
            if args.export_isotherm_csv:
                if db is None:
                    db = RASPADatabase(args.output)
                
                frameworks = db.get_framework_names()
                components = db.get_component_names()
                
                if len(frameworks) == 1 and len(components) == 1:
                    framework, component = frameworks[0], components[0]
                    db.export_isotherm_to_csv(args.export_isotherm_csv, framework, component)
                    print(f"Isotherm data exported to CSV: {args.export_isotherm_csv}")
                else:
                    print(f"Multiple frameworks/components found. Cannot auto-detect for export.")
                    print(f"Available frameworks: {frameworks}")
                    print(f"Available components: {components}")
                    return 1
            
            # Export isotherm data to Excel (auto-detect framework/component)
            if args.export_isotherm_excel:
                if db is None:
                    db = RASPADatabase(args.output)
                
                frameworks = db.get_framework_names()
                components = db.get_component_names()
                
                if len(frameworks) == 1 and len(components) == 1:
                    framework, component = frameworks[0], components[0]
                    db.export_isotherm_to_excel(args.export_isotherm_excel, framework, component)
                    print(f"Isotherm data exported to Excel: {args.export_isotherm_excel}")
                else:
                    print(f"Multiple frameworks/components found. Cannot auto-detect for export.")
                    print(f"Available frameworks: {frameworks}")
                    print(f"Available components: {components}")
                    return 1
                    
        except Exception as e:
            print(f"Error exporting data: {e}")
            return 1
        finally:
            if db is not None:
                db.close()
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())