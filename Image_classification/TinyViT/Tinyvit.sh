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
conda activate tinyvit
wandb offline
nvidia-smi

# dp train input.json --init-frz-model graph.pb
python -m torch.distributed.launch --nproc_per_node 2 --use_env main.py --cfg configs/1k/tiny_vit_5m.yaml --data-path ./datasets/ImageNet --batch-size 128 --output ./output

