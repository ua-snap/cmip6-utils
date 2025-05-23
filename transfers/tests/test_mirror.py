"""Test the mirroring of files discovered from auditing an ESGF node

Usage: from transfers folder, run python -m pytest tests/test_mirror.py
"""

import pandas as pd
import xarray as xr
from config import model_inst_lu
from multiprocessing import Pool


def get_activity(scenario):
    if scenario == "historical":
        activity = "CMIP"
    else:
        activity = "ScenarioMIP"
    return activity


# Check if files are readable with xarray.
def read_file(file):
    try:
        xr.open_dataset(file)
    except:
        assert False
    assert True


def test_mirror():
    """Iterate over all filenames in the ESGF LLNL audit table (hardcoded for now) and assert that they are present and readable in the ACDN"""
    manifest = pd.read_csv("llnl_manifest.csv")
    tmp_fp = "/beegfs/CMIP6/arctic-cmip6/CMIP6/{activity}/{institution}/{model}/{scenario}/{variant}/{frequency}/{variable}/{grid_type}/{version}/{filename}"

    mirror_fps = []
    for i, row in manifest.iterrows():
        scenario = row["scenario"]
        model = row["model"]
        if isinstance(model_inst_lu[model], list):
        # if more than one inst for the model, choose the first if historical and second if scenario
            institution = model_inst_lu[model][0] if get_activity(scenario) == "CMIP" else model_inst_lu[model][1]
        else: 
            institution = model_inst_lu[model]
        fp_kw = {
            "activity": get_activity(scenario),
            "institution": institution,
            "model": model,
            "scenario": scenario,
            "variant": row["variant"],
            "frequency": row["frequency"],
            "variable": row["variable"],
            "grid_type": row["grid_type"],
            "version": row["version"],
        }
        mirror_fps.append(tmp_fp.format(**fp_kw, filename=row["filename"]))

    with Pool(5) as p:
        p.imap_unordered(read_file, mirror_fps)
