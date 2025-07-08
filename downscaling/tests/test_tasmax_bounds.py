"""Test maximum temperature bounds in a dataset.

Example usage:
    pytest -s test_pr_bounds.py --data-folder /center1/CMIP6/kmredilla/cmip6_4km_downscaling/adjusted
"""


def test_maximum_temperature_bounds(sanity_checker):
    """Test maximum temperature bounds."""
    sanity_checker.test_tasmax_bounds()
