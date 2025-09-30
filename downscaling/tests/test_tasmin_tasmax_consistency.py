"""Test that tasmin < tasmax.

Example usage:
    pytest -s test_tasmin_tasmax_consistency.py --data-folder /center1/CMIP6/kmredilla/cmip6_4km_downscaling/adjusted
"""


def test_tasmin_tasmax_consistency(sanity_checker):
    """Test that tasmin < tasmax."""
    sanity_checker.test_tasmin_tasmax_consistency()
