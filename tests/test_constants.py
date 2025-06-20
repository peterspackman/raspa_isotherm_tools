"""
Tests for constants and default values.
"""

import unittest

from raspa_isotherm_tools.constants import (
    BOLTZMANN_K_KCAL_MOL,
    CO2_FORCE_FIELD_PARAMS,
    CO2_MOLECULE,
    DEFAULT_SIMULATION_SETTINGS,
)


class TestConstants(unittest.TestCase):
    """Tests for package constants."""

    def test_boltzmann_constant(self):
        """Test Boltzmann constant value."""
        # Check that the value is reasonable
        self.assertAlmostEqual(BOLTZMANN_K_KCAL_MOL, 0.001987204, places=6)

    def test_co2_molecule_structure(self):
        """Test CO2 molecule definition structure."""
        # Check required keys
        required_keys = ['CriticalTemperature', 'CriticalPressure', 'AcentricFactor',
                        'Type', 'pseudoAtoms', 'Bonds']
        for key in required_keys:
            self.assertIn(key, CO2_MOLECULE)

        # Check that it's a rigid molecule
        self.assertEqual(CO2_MOLECULE['Type'], 'rigid')

        # Check pseudo atoms structure
        pseudo_atoms = CO2_MOLECULE['pseudoAtoms']
        self.assertEqual(len(pseudo_atoms), 3)  # O-C-O

        # Check bonds
        bonds = CO2_MOLECULE['Bonds']
        self.assertEqual(len(bonds), 2)  # Two bonds: O-C and C-O
        self.assertEqual(bonds, [[0, 1], [1, 2]])

    def test_co2_force_field_params(self):
        """Test CO2 force field parameters."""
        # Check that both C and O parameters are present
        self.assertIn('C_co2', CO2_FORCE_FIELD_PARAMS)
        self.assertIn('O_co2', CO2_FORCE_FIELD_PARAMS)

        # Check structure of parameters
        for _atom_type, params in CO2_FORCE_FIELD_PARAMS.items():
            self.assertIn('type', params)
            self.assertIn('parameters', params)
            self.assertIn('source', params)
            self.assertIn('charge', params)

            # Check that parameters are lists with 2 elements [epsilon, sigma]
            self.assertEqual(len(params['parameters']), 2)
            self.assertIsInstance(params['parameters'][0], (int, float))
            self.assertIsInstance(params['parameters'][1], (int, float))

    def test_default_simulation_settings(self):
        """Test default simulation settings structure."""
        # Check required top-level keys
        required_keys = ['SimulationType', 'NumberOfCycles', 'NumberOfInitializationCycles',
                        'PrintEvery', 'Systems', 'Components']
        for key in required_keys:
            self.assertIn(key, DEFAULT_SIMULATION_SETTINGS)

        # Check simulation type
        self.assertEqual(DEFAULT_SIMULATION_SETTINGS['SimulationType'], 'MonteCarlo')

        # Check that Systems is a list with one system
        systems = DEFAULT_SIMULATION_SETTINGS['Systems']
        self.assertIsInstance(systems, list)
        self.assertEqual(len(systems), 1)

        # Check system structure
        system = systems[0]
        system_keys = ['Type', 'NumberOfUnitCells', 'ChargeMethod']
        for key in system_keys:
            self.assertIn(key, system)

        # Check that Components is a list with one component
        components = DEFAULT_SIMULATION_SETTINGS['Components']
        self.assertIsInstance(components, list)
        self.assertEqual(len(components), 1)

        # Check component structure
        component = components[0]
        self.assertEqual(component['Name'], 'CO2')
        self.assertIn('FugacityCoefficient', component)
        self.assertIn('SwapProbability', component)

    def test_numeric_values_reasonable(self):
        """Test that numeric values are reasonable."""
        # Check cycle counts are positive
        self.assertGreater(DEFAULT_SIMULATION_SETTINGS['NumberOfCycles'], 0)
        self.assertGreater(DEFAULT_SIMULATION_SETTINGS['NumberOfInitializationCycles'], 0)

        # Check probabilities are between 0 and 1
        component = DEFAULT_SIMULATION_SETTINGS['Components'][0]
        probabilities = [
            'TranslationProbability', 'RotationProbability', 'ReinsertionProbability',
            'SwapProbability', 'WidomProbability', 'FugacityCoefficient'
        ]
        for prob_key in probabilities:
            if prob_key in component:
                prob_value = component[prob_key]
                self.assertGreaterEqual(prob_value, 0.0)
                self.assertLessEqual(prob_value, 1.0)


if __name__ == '__main__':
    unittest.main()
