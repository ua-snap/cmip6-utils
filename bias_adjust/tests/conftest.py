# Pytest fixtures and test functions

import pytest
from sanity_checks import SanityChecker


@pytest.fixture(scope="session")
def sanity_checker(request):
    """Create ClimateDataTester instance from command line argument or default."""
    data_folder = request.config.getoption("--data-folder")
    return SanityChecker(data_folder)


def pytest_addoption(parser):
    """Add command line option for data folder."""
    parser.addoption(
        "--data-folder",
        action="store",
        default="./data",
        help="Path to folder containing zarr files",
    )
