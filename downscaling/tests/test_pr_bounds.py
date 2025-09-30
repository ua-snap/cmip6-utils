"""Test precipitation bounds in a dataset.

Example usage:
    pytest -s test_pr_bounds.py --data-folder /center1/CMIP6/kmredilla/cmip6_4km_downscaling/adjusted
"""


def test_precipitation_bounds(sanity_checker):
    """Test precipitation data bounds."""
    sanity_checker.test_pr_bounds()
