"""Test the mirroring of files discovered from auditing an ESGF node

Usage: from transfers folder, rung python -m pytest tests/test_mirror.py
"""

from pathlib import Path
import pandas as pd
from transfers.luts import prod_variant_lu, model_inst_lu


def get_activity(scenario):
        if scenario == "historical":
            activity = "CMIP"
        else:
            activity = "ScenarioMIP"

        return activity
    
    
def test_mirror():
    """Iterate over all filenames in the ESGF LLNL audit table (hardcoded for now) and assert that they are present in the ACDN"""
    
    manifest = pd.read_csv("llnl_manifest.csv")
    
    tmp_fp = "/beegfs/CMIP6/arctic-cmip6/CMIP6/{activity}/{institution}/{model}/{scenario}/{variant}/{frequency}/{variable}/{grid_type}/{version}/{filename}"
    
    # test that all files to be mirrored are found on the filesystem
    # (individual assertions rather than aggregate)
    for i, row in manifest.iterrows():
        scenario = row["scenario"]
        model = row["model"]
        fp_kw = {
            "activity": get_activity(scenario),
            "institution": model_inst_lu[model],
            "model": model,
            "scenario": scenario,
            "variant": row["variant"],
            "frequency": row["frequency"],
            "variable": row["variable"],
            "grid_type": row["grid_type"],
            "version": row["version"],
        }
        mirror_fp = tmp_fp.format(**fp_kw, filename=row["filename"])
        assert Path(mirror_fp).exists()
