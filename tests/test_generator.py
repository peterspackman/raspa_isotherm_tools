"""
Tests for the main generator functionality.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from raspa_isotherm_tools.generator import RASPAInputGenerator


class TestRASPAInputGenerator(unittest.TestCase):
    """Tests for the main RASPA input generator."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_cif = self.temp_dir / "test.cif"

        # Create a simple test CIF file
        cif_content = """data_test
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.0 0.0 0.0
H1 H 0.1 0.1 0.1
"""
        self.test_cif.write_text(cif_content)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch('raspa_isotherm_tools.generator.Crystal')
    @patch('raspa_isotherm_tools.generator.ensure_space_group_in_cif')
    @patch('raspa_isotherm_tools.generator.update_crystal_cif_with_space_group')
    @patch('raspa_isotherm_tools.generator.generate_default_labels')
    @patch('raspa_isotherm_tools.generator.update_crystal_labels')
    def test_init(self, mock_update_labels, mock_gen_labels, mock_update_cif,
                  mock_ensure_sg, mock_crystal_class):
        """Test generator initialization."""
        # Mock the Crystal.load method
        mock_crystal = Mock()
        mock_crystal.titl = "Test Crystal"
        mock_crystal.asymmetric_unit.atomic_numbers = [6, 1]
        mock_crystal.unit_cell_atoms.return_value = {'element': ['C', 'H', 'C', 'H']}
        mock_crystal_class.load.return_value = mock_crystal

        # Mock label generation
        mock_gen_labels.return_value = {0: 'C1', 1: 'H1'}

        # Create generator
        generator = RASPAInputGenerator(str(self.test_cif), "uff")

        # Check that methods were called
        mock_crystal_class.load.assert_called_once_with(str(self.test_cif))
        mock_ensure_sg.assert_called_once_with(mock_crystal)
        mock_update_cif.assert_called_once_with(mock_crystal)
        mock_gen_labels.assert_called_once_with(mock_crystal)
        mock_update_labels.assert_called_once()

        # Check attributes
        self.assertEqual(generator.cif_path, self.test_cif)
        self.assertEqual(generator.force_field, "uff")
        self.assertEqual(generator.crystal, mock_crystal)

    @patch('raspa_isotherm_tools.generator.Crystal')
    def test_init_crystal_load_error(self, mock_crystal_class):
        """Test handling of crystal loading errors."""
        mock_crystal_class.load.side_effect = Exception("Load error")

        with self.assertRaises(RuntimeError) as context:
            RASPAInputGenerator(str(self.test_cif), "uff")

        self.assertIn("Error loading crystal structure", str(context.exception))

    @patch('raspa_isotherm_tools.generator.save_crystal_with_space_group')
    @patch('raspa_isotherm_tools.generator.Crystal')
    @patch('raspa_isotherm_tools.generator.ensure_space_group_in_cif')
    @patch('raspa_isotherm_tools.generator.update_crystal_cif_with_space_group')
    @patch('raspa_isotherm_tools.generator.generate_default_labels')
    @patch('raspa_isotherm_tools.generator.update_crystal_labels')
    @patch('raspa_isotherm_tools.generator.create_force_field_json')
    @patch('raspa_isotherm_tools.generator.generate_pressure_range')
    @patch('raspa_isotherm_tools.generator.create_simulation_json')
    def test_generate_jobs(self, mock_sim_json, mock_pressure_range,
                          mock_ff_json, mock_update_labels, mock_gen_labels,
                          mock_update_cif, mock_ensure_sg, mock_crystal_class, mock_save_cif):
        """Test job generation."""
        # Setup mocks
        mock_crystal = Mock()
        mock_crystal.titl = "Test Crystal"
        mock_crystal.asymmetric_unit.atomic_numbers = [6, 1]
        mock_crystal.unit_cell_atoms.return_value = {'element': ['C', 'H']}
        mock_crystal_class.load.return_value = mock_crystal

        mock_gen_labels.return_value = {0: 'C1', 1: 'H1'}
        mock_pressure_range.return_value = [1000.0, 2000.0, 3000.0]
        mock_ff_json.return_value = {"PseudoAtoms": [], "SelfInteractions": []}
        mock_sim_json.return_value = {"SimulationType": "MonteCarlo"}

        # Mock the save function to actually create the CIF file
        def create_cif_file(crystal, output_path):
            output_path.write_text("data_test\n_cell_length_a 10.0")
        mock_save_cif.side_effect = create_cif_file

        # Create generator
        generator = RASPAInputGenerator(str(self.test_cif), "uff")

        # Generate jobs
        output_dir = self.temp_dir / "test_jobs"
        generator.generate_jobs(
            output_dir=str(output_dir),
            min_pressure=1000.0,
            max_pressure=3000.0,
            pressure_count=3
        )

        # Check that directories were created
        self.assertTrue(output_dir.exists())
        self.assertTrue((output_dir / "000").exists())
        self.assertTrue((output_dir / "001").exists())
        self.assertTrue((output_dir / "002").exists())

        # Check that files were created in each directory
        for i in range(3):
            job_dir = output_dir / f"{i:03d}"
            self.assertTrue((job_dir / "simulation.json").exists())
            self.assertTrue((job_dir / "force_field.json").exists())
            self.assertTrue((job_dir / "CO2.json").exists())
            self.assertTrue((job_dir / self.test_cif.name).exists())

        # Check auxiliary files
        self.assertTrue((output_dir / "directory_index.txt").exists())
        self.assertTrue((output_dir / "submit.sh").exists())


if __name__ == '__main__':
    unittest.main()
