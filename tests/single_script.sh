#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:4
#SBATCH --account=vector
#SBATCH -p gpu
#SBATCH --qos=high
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=tfi_qaoa

# I use this at Vector to tell Python where all the CUDA stuff is GPU
export PATH=/pkgs/anaconda3/bin:$PATH
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate /h/roeland/condaenvs/tfqc/
export PATH=/pkgs/cuda-10.0/bin:$PATH
export LD_LIBRARY_PATH=/pkgs/cudnn-10.0-v7.6.3.30/lib64:/pkgs/cuda-10.0/lib64
echo $LD_LIBRARY_PATH
nvidia-smi
echo "Starting"

python multi_gpu_setup.py &
wait