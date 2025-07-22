"""
Parallel RASPA3 runner for batch job execution.

This module provides functionality to run RASPA3 simulations in parallel
across multiple directories with progress tracking.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from chmpy.exe.raspa import Raspa
    HAS_CHMPY_RASPA = True
except ImportError:
    HAS_CHMPY_RASPA = False

LOG = logging.getLogger(__name__)


class SimpleProgressBar:
    """Simple progress bar implementation when tqdm is not available."""

    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.current = 0
        self.desc = desc
        self.start_time = time.time()

    def update(self, n: int = 1):
        self.current += n
        self._display()

    def _display(self):
        elapsed = time.time() - self.start_time
        percent = (self.current / self.total) * 100 if self.total > 0 else 0
        rate = self.current / elapsed if elapsed > 0 else 0

        bar_length = 40
        filled_length = int(bar_length * self.current // self.total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        print(f'\r{self.desc}: |{bar}| {self.current}/{self.total} '
              f'({percent:.1f}%) [{rate:.2f}it/s]', end='', flush=True)

    def close(self):
        print()  # New line when done


def create_progress_bar(total: int, desc: str = "Processing") -> Any:
    """Create a progress bar, using tqdm if available, otherwise simple fallback."""
    if HAS_TQDM:
        return tqdm(total=total, desc=desc, unit='job')
    else:
        return SimpleProgressBar(total=total, desc=desc)


class JobResult:
    """Container for job execution results."""

    def __init__(self, job_dir: Path, success: bool,
                 output: str | None = None, error: str | None = None,
                 execution_time: float | None = None):
        self.job_dir = job_dir
        self.success = success
        self.output = output
        self.error = error
        self.execution_time = execution_time


class RASPAParallelRunner:
    """Parallel runner for RASPA3 simulations."""

    def __init__(self, base_directory: str, max_workers: int | None = None,
                 timeout: float = 3600.0, use_chmpy: bool = True):
        """
        Initialize the parallel runner.

        Args:
            base_directory: Directory containing numbered subdirectories (000, 001, etc.)
            max_workers: Maximum number of parallel workers (None = auto-detect)
            timeout: Timeout per job in seconds
            use_chmpy: Whether to use chmpy's Raspa class (if available)
        """
        self.base_directory = Path(base_directory)
        self.max_workers = max_workers
        self.timeout = timeout
        self.use_chmpy = use_chmpy and HAS_CHMPY_RASPA

        if not self.base_directory.exists():
            raise FileNotFoundError(f"Base directory {base_directory} does not exist")

        # Find all numbered job directories
        self.job_directories = self._find_job_directories()

        LOG.info(f"Found {len(self.job_directories)} job directories to process")

    def _find_job_directories(self) -> list[Path]:
        """Find all numbered job directories in the base directory."""
        job_dirs = []

        # Look for directories named with 3-digit numbers (000, 001, etc.)
        for item in self.base_directory.iterdir():
            if item.is_dir() and item.name.isdigit() and len(item.name) == 3:
                # Verify it has the required RASPA input files
                if self._is_valid_job_directory(item):
                    job_dirs.append(item)
                else:
                    LOG.warning(f"Directory {item} missing required RASPA files")

        return sorted(job_dirs)

    def _is_valid_job_directory(self, job_dir: Path) -> bool:
        """Check if a directory contains the required RASPA input files."""
        required_files = ['simulation.json']

        for required_file in required_files:
            if not (job_dir / required_file).exists():
                return False

        return True

    def _run_job_chmpy(self, job_dir: Path) -> JobResult:
        """Run a single job using chmpy's Raspa class."""
        start_time = time.time()

        try:
            # Read input files
            sim_file = job_dir / 'simulation.json'
            with sim_file.open() as f:
                simulation_json = json.load(f)

            # Check for optional files
            force_field_json = None
            ff_file = job_dir / 'force_field.json'
            if ff_file.exists():
                with ff_file.open() as f:
                    force_field_json = json.load(f)

            # Find framework file (CIF) - but don't copy if it's already in the right place
            framework_file = None
            cif_files = list(job_dir.glob('*.cif'))
            if cif_files:
                # If CIF is already in the job directory, pass None to avoid copying
                cif_file = cif_files[0]
                if cif_file.parent == job_dir:
                    # Framework is already in place, no need to copy
                    framework_file = None
                else:
                    framework_file = str(cif_file)

            # Find component files (JSON files that aren't simulation or force field)
            # But don't include them if they're already in the job directory
            component_files = {}
            for json_file in job_dir.glob('*.json'):
                if json_file.name not in ['simulation.json', 'force_field.json']:
                    component_name = json_file.stem
                    # Since files are already in job directory, load content instead of path
                    with json_file.open() as f:
                        component_files[component_name] = json.load(f)

            # Create and run RASPA job
            raspa_job = Raspa(
                simulation_json=simulation_json,
                force_field_json=force_field_json,
                framework_file=framework_file,
                component_files=component_files,
                working_directory=str(job_dir),
                timeout=self.timeout
            )

            # Resolve dependencies and run
            raspa_job.resolve_dependencies()
            raspa_job.run()
            raspa_job.post_process()

            execution_time = time.time() - start_time

            return JobResult(
                job_dir=job_dir,
                success=True,
                output=raspa_job.output_contents,
                error=raspa_job.error_contents,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            LOG.error(f"Job {job_dir.name} failed: {e}")

            return JobResult(
                job_dir=job_dir,
                success=False,
                error=str(e),
                execution_time=execution_time
            )

    def _run_job_subprocess(self, job_dir: Path) -> JobResult:
        """Run a single job using subprocess to call raspa3 directly."""
        import subprocess

        start_time = time.time()

        try:
            # Run raspa3 in the job directory
            result = subprocess.run(
                ['raspa3'],
                cwd=job_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            execution_time = time.time() - start_time

            # Check for output files
            output_content = result.stdout
            output_dir = job_dir / 'output'
            if output_dir.exists():
                output_files = list(output_dir.glob('*.txt'))
                if output_files:
                    latest_output = max(output_files, key=lambda p: p.stat().st_mtime)
                    output_content = latest_output.read_text()

            return JobResult(
                job_dir=job_dir,
                success=result.returncode == 0,
                output=output_content,
                error=result.stderr if result.returncode != 0 else None,
                execution_time=execution_time
            )

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            LOG.error(f"Job {job_dir.name} timed out after {self.timeout} seconds")

            return JobResult(
                job_dir=job_dir,
                success=False,
                error=f"Timeout after {self.timeout} seconds",
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            LOG.error(f"Job {job_dir.name} failed: {e}")

            return JobResult(
                job_dir=job_dir,
                success=False,
                error=str(e),
                execution_time=execution_time
            )

    def run_single_job(self, job_dir: Path) -> JobResult:
        """Run a single RASPA job."""
        if self.use_chmpy:
            return self._run_job_chmpy(job_dir)
        else:
            return self._run_job_subprocess(job_dir)

    def run_all_jobs(self) -> dict[str, JobResult]:
        """
        Run all jobs in parallel.

        Returns:
            Dict mapping job directory names to JobResult objects
        """
        if not self.job_directories:
            LOG.warning("No valid job directories found")
            return {}

        results = {}
        failed_jobs = []

        # Create progress bar
        progress_bar = create_progress_bar(
            total=len(self.job_directories),
            desc="Running RASPA3 jobs"
        )

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all jobs
                future_to_job = {
                    executor.submit(self.run_single_job, job_dir): job_dir
                    for job_dir in self.job_directories
                }

                # Process completed jobs
                for future in as_completed(future_to_job):
                    job_dir = future_to_job[future]
                    try:
                        result = future.result()
                        results[job_dir.name] = result

                        if not result.success:
                            failed_jobs.append(job_dir.name)

                        progress_bar.update(1)

                    except Exception as e:
                        LOG.error(f"Job {job_dir.name} raised exception: {e}")
                        results[job_dir.name] = JobResult(
                            job_dir=job_dir,
                            success=False,
                            error=f"Exception: {e}"
                        )
                        failed_jobs.append(job_dir.name)
                        progress_bar.update(1)

        finally:
            progress_bar.close()

        # Summary
        successful_jobs = len(results) - len(failed_jobs)
        total_time = sum(r.execution_time for r in results.values() if r.execution_time)
        avg_time = total_time / len(results) if results else 0

        print("\n=== RASPA3 Batch Execution Summary ===")
        print(f"Total jobs: {len(self.job_directories)}")
        print(f"Successful: {successful_jobs}")
        print(f"Failed: {len(failed_jobs)}")
        print(f"Average execution time: {avg_time:.2f} seconds")
        print(f"Total execution time: {total_time:.2f} seconds")

        if failed_jobs:
            print(f"\nFailed jobs: {', '.join(failed_jobs)}")

        return results

    def get_results_summary(self, results: dict[str, JobResult]) -> dict[str, Any]:
        """
        Extract a summary of results for analysis.

        Args:
            results: Results dictionary from run_all_jobs()

        Returns:
            Summary dictionary with extracted data
        """
        summary = {
            'job_directories': [],
            'pressures': [],
            'loadings': [],
            'success_flags': [],
            'execution_times': []
        }

        for job_name, result in sorted(results.items()):
            summary['job_directories'].append(job_name)
            summary['success_flags'].append(result.success)
            summary['execution_times'].append(result.execution_time or 0)

            # Try to extract pressure from simulation.json
            try:
                sim_file = result.job_dir / 'simulation.json'
                if sim_file.exists():
                    with sim_file.open() as f:
                        sim_data = json.load(f)
                    pressure = sim_data['Systems'][0]['ExternalPressure']
                    summary['pressures'].append(pressure)
                else:
                    summary['pressures'].append(None)
            except Exception:
                summary['pressures'].append(None)

            # Try to extract loading from output
            loading = None
            if result.success and result.output:
                # Simple regex to find loading values - this may need adjustment
                # based on actual RASPA3 output format
                import re
                loading_match = re.search(r'Average loading.*?(\d+\.?\d*)', result.output)
                if loading_match:
                    loading = float(loading_match.group(1))

            summary['loadings'].append(loading)

        return summary


def main():
    """Command line interface for the parallel runner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run RASPA3 simulations in parallel across job directories"
    )
    parser.add_argument("directory", help="Base directory containing job subdirectories")
    parser.add_argument("-w", "--workers", type=int, default=None,
                       help="Number of parallel workers (default: auto-detect)")
    parser.add_argument("-t", "--timeout", type=float, default=3600.0,
                       help="Timeout per job in seconds (default: 3600)")
    parser.add_argument("--no-chmpy", action="store_true",
                       help="Use subprocess instead of chmpy Raspa class")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--summary-file", type=str,
                       help="Save results summary to JSON file")

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Check if chmpy is available
    if not HAS_CHMPY_RASPA and not args.no_chmpy:
        print("Warning: chmpy.exe.raspa not available, falling back to subprocess")
        args.no_chmpy = True

    # Create runner and execute
    runner = RASPAParallelRunner(
        base_directory=args.directory,
        max_workers=args.workers,
        timeout=args.timeout,
        use_chmpy=not args.no_chmpy
    )

    # Run all jobs
    results = runner.run_all_jobs()

    # Save summary if requested
    if args.summary_file:
        summary = runner.get_results_summary(results)
        with open(args.summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Results summary saved to {args.summary_file}")

    # Exit with error code if any jobs failed
    failed_count = sum(1 for r in results.values() if not r.success)
    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    exit(main())
