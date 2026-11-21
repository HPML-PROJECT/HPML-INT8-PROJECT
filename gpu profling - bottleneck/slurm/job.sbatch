#!/usr/bin/env bash
#SBATCH --job-name=ws2-profile
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/ws2_%j.log
# Edit partition/account/constraint so you land on the SAME GPU type as WS1. Submit once per GPU type.
# Submit from ws2/:  sbatch slurm/job.sbatch
# module load cuda                      # whatever provides ncu/nsys on Insomnia
# source ~/venvs/vllm/bin/activate
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
./run_all.sh
