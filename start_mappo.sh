#!/bin/bash
#SBATCH --job-name=Gfootball_impala_mamba
#SBATCH --partition=mobile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem-per-cpu=6GB          # evtl. anpassen
#SBATCH --gres=gpu:1               # entfernen, falls keine GPU
#SBATCH --time=30-00:00:00         # realistisch anpassen
#SBATCH --output=/home/john/rlib_gfootball/ray_%j.out
#SBATCH --error=/home/john/rlib_gfootball/ray_%j.err

set -x

source /home/john/miniforge/etc/profile.d/conda.sh
conda activate football_cuda

echo "=== Version Check ==="
echo "Python: $(python --version)"
echo "Ray: $(ray --version)"
echo "===================="

export RAY_TEMP_DIR="/home/john/rlib_gfootball/ray_temp_${SLURM_JOB_ID}"
mkdir -p "$RAY_TEMP_DIR"
echo "Ray temp dir set to: $RAY_TEMP_DIR"

echo "Starte Python-App (main.py)..."
python -u /home/john/rlib_gfootball/main.py

echo "Python-Skript beendet."