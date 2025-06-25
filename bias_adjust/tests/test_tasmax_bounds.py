"""Test maximum temperature bounds in a dataset."""


def test_maximum_temperature_bounds(sanity_checker):
    """Test maximum temperature bounds."""
    sanity_checker.test_tasmax_bounds()
