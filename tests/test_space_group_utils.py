"""
Tests for space group utility functions.
"""

import unittest
from unittest.mock import Mock

from raspa_isotherm_tools.space_group_utils import (
    add_space_group_to_cif_content,
    ensure_space_group_in_cif,
    get_space_group_info,
)


class TestSpaceGroupUtils(unittest.TestCase):
    """Tests for space group utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock crystal with space group
        self.mock_crystal = Mock()
        self.mock_space_group = Mock()
        self.mock_space_group.international_tables_number = 225
        self.mock_space_group.symbol = 'F m -3 m'
        self.mock_crystal.space_group = self.mock_space_group

    def test_get_space_group_info_success(self):
        """Test successful extraction of space group information."""
        int_tables, hm_symbol = get_space_group_info(self.mock_crystal)

        self.assertEqual(int_tables, 225)
        self.assertEqual(hm_symbol, 'F m -3 m')

    def test_get_space_group_info_missing_attributes(self):
        """Test handling of missing space group attributes."""
        # Create a space group with missing attributes
        incomplete_space_group = Mock()
        incomplete_space_group.international_tables_number = 225
        # Missing symbol attribute
        del incomplete_space_group.symbol

        self.mock_crystal.space_group = incomplete_space_group

        int_tables, hm_symbol = get_space_group_info(self.mock_crystal)

        self.assertEqual(int_tables, 225)
        self.assertIsNone(hm_symbol)

    def test_get_space_group_info_exception(self):
        """Test handling of exceptions during space group access."""
        # Create a crystal that raises an exception when accessing space_group
        error_crystal = Mock()

        # Make space_group property raise an exception when accessed
        def raise_error():
            raise Exception("Test error")

        type(error_crystal).space_group = property(lambda self: raise_error())

        int_tables, hm_symbol = get_space_group_info(error_crystal)

        self.assertIsNone(int_tables)
        self.assertIsNone(hm_symbol)

    def test_add_space_group_to_cif_content(self):
        """Test adding space group information to CIF content."""
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
"""

        updated_cif = add_space_group_to_cif_content(
            cif_content, 225, '-F 4 2 3', 'F m -3 m'
        )

        # Check that space group information was added
        self.assertIn('_symmetry_Int_Tables_number     225', updated_cif)
        self.assertIn("_symmetry_space_group_name_Hall '-F 4 2 3'", updated_cif)
        self.assertIn("_symmetry_space_group_name_H-M  'F m -3 m'", updated_cif)

    def test_add_space_group_to_cif_content_already_present(self):
        """Test that existing space group info is not duplicated."""
        cif_content = """data_test
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_symmetry_Int_Tables_number 225

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.0 0.0 0.0
"""

        updated_cif = add_space_group_to_cif_content(
            cif_content, 225, '-F 4 2 3', 'F m -3 m'
        )

        # Should not add duplicate space group information
        count_int_tables = updated_cif.count('_symmetry_Int_Tables_number')
        self.assertEqual(count_int_tables, 1)

    def test_ensure_space_group_in_cif(self):
        """Test ensuring space group information is stored in crystal."""
        # Mock the properties attribute
        self.mock_crystal.properties = {}

        ensure_space_group_in_cif(self.mock_crystal)

        # Check that properties were set
        self.assertTrue(hasattr(self.mock_crystal, 'properties'))
        self.assertIn('space_group_info', self.mock_crystal.properties)

        space_group_info = self.mock_crystal.properties['space_group_info']
        self.assertEqual(space_group_info['int_tables_number'], 225)
        self.assertEqual(space_group_info['hm_symbol'], 'F m -3 m')


if __name__ == '__main__':
    unittest.main()
