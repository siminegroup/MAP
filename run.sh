#!/bin/bash
#SBATCH --account=ur account
#SBATCH --mem-per-gpu=40G
#SBATCH --gres=gpu:1
#SBATCH --time=0-120:00
#SBATCH --array=0
source /envMAP/bin/activate

# WORLD_SIZE as gpus/node * num_nodes
export WORLD_SIZE=1

### get the first node name as master address - customized for vgg slurm
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR


python ./main.py --run_num=$SLURM_ARRAY_TASK_ID
