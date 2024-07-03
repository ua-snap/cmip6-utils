#!/bin/sh
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -p {partition}
#SBATCH --output {sbatch_out_file}
echo Start slurm && date
source {conda_init_script}
conda activate {env_name}

time python {profile_script} --model_dir {model_dir} --ref_dir {ref_dir} --results_file {results_file} --ju_thresh "{ju_thresh}" --af_thresh "{af_thresh}" 