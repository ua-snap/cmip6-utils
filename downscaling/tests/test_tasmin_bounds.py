"""Test minimum temperature bounds in a dataset.

Example usage:
    pytest -s test_tasmin_bounds.py --data-folder /center1/CMIP6/kmredilla/cmip6_4km_downscaling/adjusted
"""


def test_minimum_temperature_bounds(sanity_checker):
    """Test tasmin bounds."""
    sanity_checker.test_tasmin_bounds()
