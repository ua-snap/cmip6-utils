import pandas as pd
from config import SCRATCH_DIR



qc_file = SCRATCH_DIR.joinpath("slurm", "indicators", "qc", "qc.csv")
error_file = SCRATCH_DIR.joinpath("slurm", "indicators", "qc", "qc_error.csv")


def qc_by_row(row):

    #set up list to collect errors
    error_strings = []

    #QC 1: do the indicator .nc files exist?
    check_if_file_exists(indicator_fp)

    #QC 2: do the files contain reasonable values? might need to create a min/max lookup table for this?
    check_for_reasonable_values(indicator, indicator_fp)

    #QC 3: are there any suspicious NA values, or all NA values?
    check_na_values(indicator_fp)

    #QC 4: does slurm job output show success?
    check_sbatch_output(sbatch_out_fp)

    #Log the errors: write any errors into the error file
    write_errors(error_strings)



df = pd.read_csv(qc_file)
for row in df.iterrows():
    qc_by_row(row)