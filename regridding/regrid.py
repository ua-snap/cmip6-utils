"""Script for regridding a batch of files listed in a batch file"""

import argparse
import dask


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-b",
        dest="regrid_batch_fp",
        type=str,
        help="Batch file containing filepaths to be regridded"
        required=True
    )
    parser.add_argument(
        "-o", dest="out_dir", type=str, help="Path to directory where regridded data should be written", required=True
    )
    
    return args.regrid_batch_fp, args.out_dir


if __name__ == "__main__":
    # parse args
    regrid_batch_fp, out_dir = parse_args()
    
    # get the paths of files to regrid from the batch file
    with open(regrid_batch_fp) as f:
        lines = f.readlines()
    src_fps = [Path(line.replace("\n", "")) for line in lines]

