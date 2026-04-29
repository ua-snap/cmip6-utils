#!/bin/sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=FAIL
#SBATCH -p t2small
#SBATCH --output=%x_%j.out

date && echo Job Started
./find_duplicates.py
date && echo Job Completed
