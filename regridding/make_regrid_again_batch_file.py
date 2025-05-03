"""Script to make the batch file for a "second regridding" (aka regrid_again), 
where we are regridding a set of files that have already been regridded to a common grid.

Example usage:
    python make_regrid_again_batch_file.py \
        --regridded_dir /center1/CMIP6/kmredilla/cmip6_4km_downscaling/regrid \
        --regrid_again_batch_file /center1/CMIP6/kmredilla/cmip6_4km_downscaling/slurm/regrid_again_batch.txt \
        --output_dir /center1/CMIP6/kmredilla/cmip6_4km_downscaling/final_regrid

"""

import argparse
from pathlib import Path


def parse_args():
    """Parse some command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regridded_dir",
        type=str,
        help="Path to directory where CMIP6 files are stored",
        required=True,
    )
    parser.add_argument(
        "--regrid_again_batch_file",
        type=str,
        help="Path to batch file containing the files to regrid again",
        required=True,
    )
    args = parser.parse_args()
    return (
        Path(args.regridded_dir),
        args.regrid_again_batch_file,
    )


if __name__ == "__main__":
    regridded_dir, regrid_again_batch_file = parse_args()

    src_fps = list(regridded_dir.glob("**/*.nc"))
    with open(regrid_again_batch_file, "w") as f:
        print("Writing regrid again batch file to", regrid_again_batch_file)
        for src_fp in src_fps:
            f.write(f"{src_fp}\n")
