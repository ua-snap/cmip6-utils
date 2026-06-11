"""Squeeze hurs and hursmin zarr files to the valid [0, 100] range.

Finds all hurs and hursmin *_adjusted.zarr files in the input directory, clips
values below 0 up to 0 and values above 100 down to 100 (NaN-preserving), writes
squeezed zarr files to hurs_squeezed/ and hursmin_squeezed/ subdirectories inside
the input directory, and saves a combined summary CSV there as well.

Example usage:
    python squeeze_hurs.py /beegfs/CMIP6/.../sfcWind_hurs_hursmin/adjusted/
    python squeeze_hurs.py --skip-existing /beegfs/CMIP6/.../sfcWind_hurs_hursmin/adjusted/
"""

import argparse
import gc
import logging
import shutil
import sys
from pathlib import Path

import dask
import dask.array as da
import numcodecs
import pandas as pd
import xarray as xr

SQUEEZE_MIN = 0.0
SQUEEZE_MAX = 100.0

# Output zarr chunk layout — matches what bias_adjust.py produces
OUTPUT_CHUNKS = (365, 100, 100)  # (time, y, x)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


dask.config.set({"array.slicing.split_large_chunks": True})


def parse_zarr_name(zarr_name):
    """Parse variable, model, and scenario from an adjusted zarr filename.

    Expected format: {var}_{model}_{scenario}_adjusted.zarr
    Models may contain hyphens; variable is always the first token,
    scenario is always the last (historical / ssp126 / ssp245 / ssp370 / ssp585).
    """
    stem = zarr_name.replace("_adjusted.zarr", "")
    parts = stem.split("_")
    var_id = parts[0]
    scenario = parts[-1]
    model = "_".join(parts[1:-1])
    return var_id, model, scenario


def squeeze_zarr(input_path, output_path, var_id):
    """Open a zarr, count out-of-range values, squeeze, and write.

    Returns a dict with count/percentage stats.
    """
    logging.info(f"  Opening {input_path.name}")
    ds = xr.open_zarr(input_path, chunks={}, consolidated=True)
    try:
        arr = ds[var_id]

        # Count out-of-range values in a single pass (NaN excluded from denominator)
        total_valid_da = da.count_nonzero(da.notnull(arr.data))
        count_below_da = (arr.data < SQUEEZE_MIN).sum()
        count_above_da = (arr.data > SQUEEZE_MAX).sum()
        total_valid, count_below, count_above = dask.compute(
            total_valid_da, count_below_da, count_above_da
        )
        total_valid = int(total_valid)
        count_below = int(count_below)
        count_above = int(count_above)

        pct_below = 100.0 * count_below / total_valid if total_valid > 0 else 0.0
        pct_above = 100.0 * count_above / total_valid if total_valid > 0 else 0.0

        logging.info(f"  Valid values : {total_valid:,}")
        logging.info(f"  Below {SQUEEZE_MIN}   : {count_below:,} ({pct_below:.6f}%)")
        logging.info(f"  Above {SQUEEZE_MAX} : {count_above:,} ({pct_above:.6f}%)")

        # Apply squeeze — NaN cells are left as NaN
        squeezed = arr.where((arr >= SQUEEZE_MIN) | arr.isnull(), other=SQUEEZE_MIN)
        squeezed = squeezed.where((squeezed <= SQUEEZE_MAX) | squeezed.isnull(), other=SQUEEZE_MAX)
        ds[var_id] = squeezed

        # No rechunk needed: open_zarr(chunks={}) preserves stored chunk layout and
        # element-wise where keeps it intact — dask chunks already match output chunks.

        # Remove existing output if present
        if output_path.exists():
            logging.info(f"  Removing existing output: {output_path}")
            shutil.rmtree(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        encoding = {
            var_id: {
                "compressor": numcodecs.Blosc(
                    cname="zstd",
                    clevel=3,
                    shuffle=numcodecs.Blosc.BITSHUFFLE,
                ),
                "chunks": OUTPUT_CHUNKS,
            }
        }

        # safe_chunks=False: the overlap check is a false positive for element-wise ops
        # where dask and zarr chunks are already aligned. Avoids a retry+rmtree race.
        logging.info(f"  Writing to {output_path}")
        ds.to_zarr(
            output_path,
            encoding=encoding,
            consolidated=True,
            compute=True,
            safe_chunks=False,
        )

        logging.info(f"  Done: {output_path.name}")

        return {
            "count_below_0": count_below,
            "count_above_100": count_above,
            "pct_below_0": round(pct_below, 6),
            "pct_above_100": round(pct_above, 6),
        }
    finally:
        ds.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing hurs/hursmin *_adjusted.zarr files",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files whose squeezed output already exists",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    zarr_paths = sorted(input_dir.glob("hurs*_adjusted.zarr"))
    if not zarr_paths:
        logging.error(f"No hurs/hursmin *_adjusted.zarr files found in {input_dir}")
        sys.exit(1)

    logging.info(f"Found {len(zarr_paths)} zarr files to process")

    out_dirs = {
        "hurs": input_dir / "hurs_squeezed",
        "hursmin": input_dir / "hursmin_squeezed",
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    failed = []

    for zarr_path in zarr_paths:
        var_id, model, scenario = parse_zarr_name(zarr_path.name)
        out_name = zarr_path.name.replace("_adjusted.zarr", "_squeezed.zarr")
        out_path = out_dirs[var_id] / out_name

        logging.info(f"--- {var_id} | {model} | {scenario} ---")

        if args.skip_existing and out_path.exists():
            logging.info(f"  Skipping (output exists): {out_path.name}")
            continue

        try:
            stats = squeeze_zarr(zarr_path, out_path, var_id)
            rows.append(
                {
                    "original_filepath": str(zarr_path.resolve()),
                    "squeezed_filepath": str(out_path.resolve()),
                    "variable": var_id,
                    "model": model,
                    "scenario": scenario,
                    **stats,
                }
            )
        except Exception as e:
            logging.error(f"FAILED: {zarr_path.name} — {e}")
            failed.append(zarr_path.name)

        gc.collect()

    if rows:
        csv_path = input_dir / "hurs_hursmin_squeeze_summary.csv"
        df = pd.DataFrame(rows, columns=[
            "original_filepath",
            "squeezed_filepath",
            "variable",
            "model",
            "scenario",
            "count_below_0",
            "count_above_100",
            "pct_below_0",
            "pct_above_100",
        ])
        df.to_csv(csv_path, index=False)
        logging.info(f"Summary CSV written to {csv_path}")
        logging.info(f"Processed {len(rows)} files successfully")

    if failed:
        logging.error(f"{len(failed)} files FAILED: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
