from pathlib import Path
import os
import pandas as pd
import argparse

# Example usage:
# python generate_ardn_cmip6_holdings_csv.py /beegfs/CMIP6/arctic-cmip6/CMIP6/ $HOME/scratch/


def make_df(directory):
    files = list(directory.rglob("*.nc"))
    dirs = [f.parts[7:] for f in files]
    df = pd.DataFrame(
        dirs,
        columns=[
            "model",
            "scenario",
            "variant",
            "frequency",
            "variable_id",
            "grid_label",
            "version",
            "filename",
        ],
    )
    return df


def generate_holdings_table(df):
    df = pd.get_dummies(df, columns=["variable_id"])
    df.columns = df.columns.str.replace("variable_id_", "")
    df = df.groupby(["model", "scenario"]).sum()
    df = df.applymap(lambda x: "x" if x >= 1 else "")
    df = df.reset_index()
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate CMIP6 holdings table")
    parser.add_argument("directory", type=Path, help="Directory containing CMIP6 data")
    parser.add_argument("output", type=Path, help="Output directory")
    args = parser.parse_args()

    df = make_df(args.directory)
    holdings = generate_holdings_table(df)
    # add timestamp to file name and create filepath using output directory
    timestamp = pd.Timestamp.now().strftime("%Y%m%d")
    holdings.to_csv(args.output / f"cmip6_holdings_{timestamp}.csv", index=False)


if __name__ == "__main__":
    main()
