#!/bin/bash
#SBATCH -J train_NN                  		# job name
#SBATCH -N 1 -n 6 --mem-per-cpu 20gb
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -t 12:00:00
#SBATCH -A gts-skalidindi7
#SBATCH --mail-type NONE                          
#SBATCH -o slurm_outputs/%j.out   


source $SLURM_SUBMIT_DIR/sandbox/bin/activate

cd $SLURM_SUBMIT_DIR

echo $SLURM_JOB_ID

pwd

# Add any cli overrides
python3 main.py "$@"