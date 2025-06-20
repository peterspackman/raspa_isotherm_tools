"""
Tests for simulation setup functionality.
"""

import unittest

from raspa_isotherm_tools.simulation import (
    create_simulation_json,
    create_slurm_script,
    generate_pressure_range,
)


class TestSimulation(unittest.TestCase):
    """Tests for simulation setup functions."""

    def test_create_simulation_json_basic(self):
        """Test basic simulation JSON creation."""
        sim_json = create_simulation_json(
            pressure=100000.0,
            framework_name="test_framework",
            temperature=298.15
        )

        # Check basic structure
        self.assertIn('SimulationType', sim_json)
        self.assertIn('Systems', sim_json)
        self.assertIn('Components', sim_json)

        # Check system parameters
        system = sim_json['Systems'][0]
        self.assertEqual(system['Name'], 'test_framework')
        self.assertEqual(system['ExternalPressure'], 100000.0)
        self.assertEqual(system['ExternalTemperature'], 298.15)

        # Check component
        component = sim_json['Components'][0]
        self.assertEqual(component['Name'], 'CO2')

    def test_create_simulation_json_with_custom_settings(self):
        """Test simulation JSON creation with custom settings."""
        custom_settings = {
            "NumberOfCycles": 50000,
            "Systems": {
                "NumberOfUnitCells": [2, 2, 2],
                "ChargeMethod": "Wolf"
            },
            "Components": {
                "TranslationProbability": 0.8
            }
        }

        sim_json = create_simulation_json(
            pressure=200000.0,
            framework_name="custom_framework",
            temperature=323.0,
            simulation_settings=custom_settings
        )

        # Check that custom settings were applied
        self.assertEqual(sim_json['NumberOfCycles'], 50000)
        self.assertEqual(sim_json['Systems'][0]['NumberOfUnitCells'], [2, 2, 2])
        self.assertEqual(sim_json['Systems'][0]['ChargeMethod'], 'Wolf')
        self.assertEqual(sim_json['Components'][0]['TranslationProbability'], 0.8)

        # Check that pressure and temperature are still set from function args
        self.assertEqual(sim_json['Systems'][0]['ExternalPressure'], 200000.0)
        self.assertEqual(sim_json['Systems'][0]['ExternalTemperature'], 323.0)

    def test_generate_pressure_range(self):
        """Test pressure range generation."""
        # Test normal case
        pressures = generate_pressure_range(1000.0, 10000.0, 5)
        expected = [1000.0, 3250.0, 5500.0, 7750.0, 10000.0]
        self.assertEqual(len(pressures), 5)
        for i, expected_p in enumerate(expected):
            self.assertAlmostEqual(pressures[i], expected_p, places=1)

        # Test single point
        pressures = generate_pressure_range(5000.0, 10000.0, 1)
        self.assertEqual(pressures, [5000.0])

        # Test zero count
        pressures = generate_pressure_range(1000.0, 10000.0, 0)
        self.assertEqual(pressures, [1000.0])

    def test_create_slurm_script(self):
        """Test SLURM script creation."""
        script = create_slurm_script(
            job_name="test_job",
            job_count=10,
            time="02:00:00",
            memory="2G"
        )

        # Check that parameters are in the script
        self.assertIn("#SBATCH --job-name=test_job", script)
        self.assertIn("#SBATCH --array=0-9", script)  # job_count-1
        self.assertIn("#SBATCH --time=02:00:00", script)
        self.assertIn("#SBATCH --mem=2G", script)

        # Check that it contains the essential commands
        self.assertIn("raspa3", script)
        self.assertIn("SIM_DIR=", script)


if __name__ == '__main__':
    unittest.main()
