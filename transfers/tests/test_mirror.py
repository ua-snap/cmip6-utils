"""Test the mirroring of files discovered from auditing an ESGF node

Usage: from transfers folder, rung python -m pytest tests/test_mirror.py
"""

from pathlib import Path
import pandas as pd
from transfers.luts import model_inst_lu


def test_mirror():
    """Iterate over all filenames in the ESGF LLNL audit table (hardcoded for now) and assert that they are present in the ACDN"""
    
    holdings = pd.read_csv("llnl_esgf_holdings.csv", converters={"filenames": lambda x: x.strip("[]").replace("'","").split(", ")})
    
    models = [model for model in model_inst_lu.keys() if "mirror_variant" in list(model_inst_lu[model].keys())]
    
    mirror_holdings = []

    for model in models:
        variant = model_inst_lu[model]["mirror_variant"]
        mirror_holdings.append(holdings.query(f"model == '{model}' & variant == '{variant}'"))

    mirror_holdings = pd.concat(mirror_holdings).dropna(axis=0).reset_index(drop=True)
    
    tmp_fp = "/beegfs/CMIP6/arctic-cmip6/CMIP6/{activity}/{institution}/{model}/{scenario}/{variant}/{frequency}/{variable}/{grid_type}/{version}/{filename}"

    def get_activity(scenario):
        if scenario == "historical":
            activity = "CMIP"
        else:
            activity = "ScenarioMIP"

        return activity

    for i, row in mirror_holdings.iterrows():
        scenario = row["scenario"]
        model = row["model"]
        fp_kw = {
            "activity": get_activity(scenario),
            "institution": model_inst_lu[model]["institution"],
            "model": model,
            "scenario": scenario,
            "variant": row["variant"],
            "frequency": row["frequency"],
            "variable": row["variable"],
            "grid_type": row["grid_type"],
            "version": row["version"],
        }
        mirror_filepaths = [tmp_fp.format(**fp_kw, filename=fn) for fn in row["filenames"]]
        for fp in mirror_filepaths:
            assert Path(fp).exists()
