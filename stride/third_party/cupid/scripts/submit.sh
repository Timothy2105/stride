#!/bin/sh
#SBATCH --account=juno
#SBATCH --partition=juno-lo # partition can be juno or juno-lo
#SBATCH --exclude=juno1 # exclude juno1, which has slower gpus
#SBATCH --cpus-per-task=12
#SBATCH --mem=90G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=sbatch_logs/job-%j.out

cd /juno/u/tyu2105/projects/stride/stride/third_party/cupid
eval "$1"
