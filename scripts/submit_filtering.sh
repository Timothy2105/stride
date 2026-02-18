#!/bin/sh
#SBATCH --account=juno
#SBATCH --partition=juno-lo
#SBATCH --exclude=juno1
#SBATCH --cpus-per-task=12
#SBATCH --mem=90G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/juno/u/tyu2105/projects/stride/sbatch_logs/job-%j.out
#SBATCH --mail-user=tyu2105@stanford.edu
#SBATCH --mail-type=FAIL,END

eval "$1"
