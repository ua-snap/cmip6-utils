"""Delete intermediate/fragments/ once a pipeline run AND qc.py have both
been run and reviewed.

This is a separate, manually-invoked step -- not part of run_pipeline.sh,
and must run *after* qc.py, not before: qc.py's coverage-gap cross-check
reads fragment filenames from intermediate/fragments/, so deleting them
first would make that check vacuous (every indicator/model/scenario would
look like "no fragment", and the check could no longer tell a missing
fragment from a deleted one).

intermediate/job_list.json is intentionally left alone -- it's tiny and
useful as a record of exactly what was run.

Usage:
    python cleanup_intermediate.py --config config_12km.yaml   # dry run, default
    python cleanup_intermediate.py --config config_12km.yaml --yes   # actually delete
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_CONFIG_PATH, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--yes", action="store_true", help="actually delete (default is a dry run)")
    args = parser.parse_args()
    config = load_config(args.config)

    frag_dir = config.fragments_dir
    if not frag_dir.exists():
        print(f"{frag_dir} does not exist -- nothing to clean up")
        return

    fragments = sorted(frag_dir.glob("*.zarr"))
    total_bytes = sum(f.stat().st_size for store in fragments for f in store.rglob("*") if f.is_file())
    print(f"{len(fragments)} fragment(s) under {frag_dir} ({total_bytes / 1e9:.2f} GB)")

    if not args.yes:
        print()
        print("Dry run -- nothing deleted. Re-run with --yes to actually delete.")
        print("Make sure qc.py has been run AND its results reviewed first --")
        print("it reads this directory for the coverage-gap cross-check.")
        return

    shutil.rmtree(frag_dir)
    print(f"Deleted {frag_dir}")


if __name__ == "__main__":
    main()
