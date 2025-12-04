#!/bin/bash
#SBATCH --job-name=MONTE_CARLO
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8  
#SBATCH --gres=gpu
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --output=slurm/%x-%A-%a-%j.out
#SBATCH --error=slurm/%x-%A-%a-%j.err

source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh;
conda activate /scratch/ml9715/CLASSIFIER/myenv
cd /scratch/ml9715/monte-carlo-collective

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python mat_mul.py & 

torchrun --nproc_per_node=1 --master_port=$((12000 + $RANDOM % 1000)) /scratch/ml9715/monte-carlo-collective/experiments.py  --local_rank 0
