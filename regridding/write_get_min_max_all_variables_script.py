from pathlib import Path
import numpy as np
from config import regrid_batch_dir


if __name__ == "__main__":
    regrid_vars = []
    for fp in regrid_batch_dir.glob("*.txt"):
        with open(fp) as f:
            # Return a list of all the files inside f with a basename that starts with args.variable.
            regrid_vars.extend([Path(line.strip()).name.split("_")[0] for line in f])

    regrid_vars = np.unique(regrid_vars)

    # create a shell script to run each process
    shell_script_str = (
        "#!/bin/sh\n\n"
        "echo Gathering min and max values for all regrid variables\n"
        f"for var in {' '.join(regrid_vars)}\n"
        "do\n"
        "  echo working on $var\n"
        "  timeout 30m $PROJECT_DIR/regridding/get_min_max.py -v $var\n"
        "  while [ $? == 124 ]; do\n"
        "    timeout 30m $PROJECT_DIR/regridding/get_min_max.py -v $var\n"
        '    echo ""\n'
        "  done\n"
        "done\n"
    )

    with open("get_min_max.sh", "w") as f:
        f.write(shell_script_str)

    print("shell script written", flush=True)
