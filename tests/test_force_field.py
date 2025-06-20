"""
Tests for force field generation functionality.
"""

import unittest
from unittest.mock import Mock, patch

from raspa_isotherm_tools.force_field import (
    create_force_field_json,
    generate_default_labels,
    update_crystal_labels,
)


class TestForceField(unittest.TestCase):
    """Tests for force field generation functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock crystal object
        self.mock_crystal = Mock()

        # Mock asymmetric unit
        mock_asym_unit = Mock()
        mock_asym_unit.atomic_numbers = [6, 1, 1, 8]  # C, H, H, O
        mock_asym_unit.labels = ['C1', 'H1', 'H2', 'O1']
        self.mock_crystal.asymmetric_unit = mock_asym_unit

        # Mock coordination numbers
        self.mock_crystal.asymmetric_unit_coordination_numbers.return_value = [4, 1, 1, 2]

        # Mock element data
        self.mock_elements = {
            6: Mock(symbol='C', mass=12.01),
            1: Mock(symbol='H', mass=1.008),
            8: Mock(symbol='O', mass=15.999)
        }

    @patch('raspa_isotherm_tools.force_field.Element')
    def test_generate_default_labels(self, mock_element_class):
        """Test generation of default atom labels."""
        # Mock Element.from_atomic_number calls
        mock_element_class.from_atomic_number.side_effect = lambda x: self.mock_elements[x]

        labels = generate_default_labels(self.mock_crystal)

        expected_labels = {0: 'C1', 1: 'H1', 2: 'H2', 3: 'O1'}
        self.assertEqual(labels, expected_labels)

    def test_update_crystal_labels(self):
        """Test updating crystal labels."""
        new_labels = {0: 'C_new', 1: 'H_new'}

        # Mock the properties attribute
        self.mock_crystal.properties = {}

        update_crystal_labels(self.mock_crystal, new_labels)

        # Check that labels were updated
        self.mock_crystal.asymmetric_unit.labels[0] = 'C_new'
        self.mock_crystal.asymmetric_unit.labels[1] = 'H_new'

    @patch('raspa_isotherm_tools.force_field.calculate_eeq_charges_crystal')
    @patch('raspa_isotherm_tools.force_field.get_asymmetric_unit_uff_parameters')
    @patch('raspa_isotherm_tools.force_field.Element')
    def test_create_force_field_json(self, mock_element_class, mock_uff_params, mock_eeq_charges):
        """Test creation of force field JSON structure."""
        # Mock return values
        mock_element_class.from_atomic_number.side_effect = lambda x: self.mock_elements[x]
        mock_uff_params.return_value = (
            {0: 'C_3', 1: 'H_', 2: 'H_', 3: 'O_2'},  # atom types
            {0: {'sigma': 3.4, 'epsilon': 0.1}, 1: {'sigma': 2.5, 'epsilon': 0.05},
             2: {'sigma': 2.5, 'epsilon': 0.05}, 3: {'sigma': 3.0, 'epsilon': 0.08}}  # parameters
        )
        mock_eeq_charges.return_value = [0.1, 0.05, 0.05, -0.2]

        labels = {0: 'C1', 1: 'H1', 2: 'H2', 3: 'O1'}

        ff_data = create_force_field_json(self.mock_crystal, "uff", labels)

        # Check structure
        self.assertIn('PseudoAtoms', ff_data)
        self.assertIn('SelfInteractions', ff_data)
        self.assertIn('MixingRule', ff_data)

        # Check that we have framework atoms + CO2 atoms
        pseudo_atoms = ff_data['PseudoAtoms']
        self.assertTrue(len(pseudo_atoms) >= 6)  # 4 framework + 2 CO2

        # Check that framework atoms have correct charges
        framework_atoms = [atom for atom in pseudo_atoms if atom['framework']]
        self.assertEqual(len(framework_atoms), 4)

        # Check CO2 atoms are present
        co2_atoms = [atom for atom in pseudo_atoms if not atom['framework']]
        self.assertEqual(len(co2_atoms), 2)
        co2_names = {atom['name'] for atom in co2_atoms}
        self.assertEqual(co2_names, {'C_co2', 'O_co2'})


if __name__ == '__main__':
    unittest.main()
