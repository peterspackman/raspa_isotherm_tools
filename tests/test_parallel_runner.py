"""
Tests for the parallel runner functionality.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from raspa_isotherm_tools.parallel_runner import JobResult, RASPAParallelRunner, create_progress_bar


class TestParallelRunner(unittest.TestCase):
    """Tests for the parallel RASPA runner."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create mock job directories with RASPA input files
        for i in range(3):
            job_dir = self.temp_dir / f"{i:03d}"
            job_dir.mkdir()

            # Create simulation.json
            sim_data = {
                "SimulationType": "MonteCarlo",
                "Systems": [{"Name": "test_framework", "ExternalPressure": 1000 * (i + 1)}],
                "Components": [{"Name": "CO2"}]
            }
            (job_dir / "simulation.json").write_text(json.dumps(sim_data, indent=2))

            # Create force_field.json
            ff_data = {"PseudoAtoms": [], "SelfInteractions": []}
            (job_dir / "force_field.json").write_text(json.dumps(ff_data, indent=2))

            # Create component file
            co2_data = {"Type": "rigid"}
            (job_dir / "CO2.json").write_text(json.dumps(co2_data, indent=2))

            # Create CIF file
            (job_dir / "test_framework.cif").write_text("data_test\n_cell_length_a 10.0")

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_find_job_directories(self):
        """Test finding valid job directories."""
        runner = RASPAParallelRunner(str(self.temp_dir), use_chmpy=False)

        self.assertEqual(len(runner.job_directories), 3)
        self.assertEqual([d.name for d in runner.job_directories], ['000', '001', '002'])

    def test_is_valid_job_directory(self):
        """Test validation of job directories."""
        runner = RASPAParallelRunner(str(self.temp_dir), use_chmpy=False)

        # Valid directory
        self.assertTrue(runner._is_valid_job_directory(self.temp_dir / "000"))

        # Invalid directory (missing simulation.json)
        invalid_dir = self.temp_dir / "invalid"
        invalid_dir.mkdir()
        self.assertFalse(runner._is_valid_job_directory(invalid_dir))

    @patch('subprocess.run')
    def test_run_job_subprocess_success(self, mock_run):
        """Test running a job using subprocess (successful case)."""
        # Mock successful subprocess result
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "RASPA simulation completed successfully"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        runner = RASPAParallelRunner(str(self.temp_dir), use_chmpy=False)
        job_dir = self.temp_dir / "000"

        result = runner._run_job_subprocess(job_dir)

        self.assertTrue(result.success)
        self.assertEqual(result.job_dir, job_dir)
        self.assertIsNotNone(result.execution_time)
        self.assertIn("completed successfully", result.output)

    @patch('subprocess.run')
    def test_run_job_subprocess_failure(self, mock_run):
        """Test running a job using subprocess (failure case)."""
        # Mock failed subprocess result
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "RASPA simulation failed"
        mock_run.return_value = mock_result

        runner = RASPAParallelRunner(str(self.temp_dir), use_chmpy=False)
        job_dir = self.temp_dir / "000"

        result = runner._run_job_subprocess(job_dir)

        self.assertFalse(result.success)
        self.assertEqual(result.job_dir, job_dir)
        self.assertIsNotNone(result.execution_time)
        self.assertIn("failed", result.error)

    @patch('raspa_isotherm_tools.parallel_runner.HAS_CHMPY_RASPA', False)
    def test_run_all_jobs_no_chmpy(self):
        """Test running all jobs without chmpy."""
        with patch('subprocess.run') as mock_run:
            # Mock successful results for all jobs
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Success"
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            runner = RASPAParallelRunner(str(self.temp_dir), use_chmpy=False, max_workers=1)
            results = runner.run_all_jobs()

            self.assertEqual(len(results), 3)
            self.assertTrue(all(r.success for r in results.values()))

            # Check that subprocess was called for each job
            self.assertEqual(mock_run.call_count, 3)

    def test_get_results_summary(self):
        """Test extracting results summary."""
        runner = RASPAParallelRunner(str(self.temp_dir), use_chmpy=False)

        # Create mock results
        results = {}
        for i in range(3):
            job_dir = self.temp_dir / f"{i:03d}"
            results[f"{i:03d}"] = JobResult(
                job_dir=job_dir,
                success=True,
                output="Average loading: 5.2 mol/kg",
                execution_time=10.0 + i
            )

        summary = runner.get_results_summary(results)

        self.assertEqual(len(summary['job_directories']), 3)
        self.assertEqual(summary['pressures'], [1000, 2000, 3000])
        self.assertTrue(all(summary['success_flags']))
        self.assertEqual(summary['execution_times'], [10.0, 11.0, 12.0])

    def test_create_progress_bar(self):
        """Test progress bar creation."""
        # This test mainly ensures the function doesn't crash
        pbar = create_progress_bar(10, "Test")
        self.assertIsNotNone(pbar)

        # Clean up if it's a real tqdm object
        if hasattr(pbar, 'close'):
            pbar.close()

    def test_job_result(self):
        """Test JobResult class."""
        job_dir = Path("/test")
        result = JobResult(
            job_dir=job_dir,
            success=True,
            output="test output",
            error=None,
            execution_time=5.0
        )

        self.assertEqual(result.job_dir, job_dir)
        self.assertTrue(result.success)
        self.assertEqual(result.output, "test output")
        self.assertIsNone(result.error)
        self.assertEqual(result.execution_time, 5.0)


if __name__ == '__main__':
    unittest.main()
