from config import PROJECT_DIR

# load the luts file from the transfers pipeline
import imp

transfers_luts = imp.load_source(
    "luts", str(PROJECT_DIR.joinpath("transfers", "luts.py"))
)

model_inst_lu = transfers_luts.model_inst_lu
