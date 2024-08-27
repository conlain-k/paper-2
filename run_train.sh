#!/bin/bash
#SBATCH -J train_NN                  		# job name
#SBATCH -N 1 --mem-per-gpu 80gb
##SBATCH --gres=gpu:A100:1
#SBATCH --gres=gpu:RTX_6000:1
##SBATCH --gres=gpu:V100:1
#SBATCH -t 12:00:00
#SBATCH -q inferno 
#SBATCH -A gts-skalidindi7-coda20
#SBATCH --mail-type NONE                          
#SBATCH -o slurm_outputs/%j.out   


source $SLURM_SUBMIT_DIR/sandbox/bin/activate

cd $SLURM_SUBMIT_DIR

echo $SLURM_JOB_ID

nvidia-smi

#export PATH=/storage/scratch1/3/ckelly84/miniforge/bin:$PATH
#conda activate ml

pwd

# Add any cli overrides
python3 main.py "$@"
