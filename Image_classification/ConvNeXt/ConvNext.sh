#!/bin/bash
#SBATCH -J deepmd
#SBATCH -p gpu-high
#SBATCH -N 1
#SBATCH -n 6
##SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH -o out

export PATH=/gs/home/caixy/software/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/gs/home/caixy/software/cuda-11.3/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/gs/home/caixy/software/cuda-11.3/lib64:$LIBRARY_PATH
export INCLUDE=/gs/home/caixy/software/cuda-11.3/include:$INCLUDE
. "/gs/home/caixy/software/anaconda3/etc/profile.d/conda.sh"
conda activate convnext
wandb offline
nvidia-smi

# dp train input.json --init-frz-model graph.pb

python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model convnext_tiny --drop_path 0.1 --batch_size 128 --lr 4e-3 --update_freq 4 --model_ema true --model_ema_eval true --data_path ./image_folder --output_dir ./output

