#!/bin/bash
#SBATCH --ur account
#SBATCH --mem-per-gpu=12GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-1:35
#SBATCH --array=0

module load cuda cudnn

source /envMAP/bin/activate

python ./model_save_load_generate.py --run_num=$SLURM_ARRAY_TASK_ID
