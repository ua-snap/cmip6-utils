
def pytest_addoption(parser):
    parser.addoption(
        "--tasmax_dir", action="store", help="Directory containing daily maiximum temperature data"
    )
    parser.addoption(
        "--tasmin_dir",
        action="store",
        help="Directory containing daily minimum temperature data.",
    )
    parser.addoption(
        "--output_dir",
        action="store",
        help="Directory where DTR files were written",
    )