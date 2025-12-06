#!/bin/bash
#SBATCH --job-name=gfootball_ippo_hard_stochastic
#SBATCH --partition=mobile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem-per-cpu=6GB
#SBATCH --gres=gpu:1
#SBATCH --time=30-00:00:00
#SBATCH --output=/home/john/rlib_gfootball/main/grf_ippo/ray_%j.out
#SBATCH --error=/home/john/rlib_gfootball/main/grf_ippo/ray_%j.err

set -x

# --- Conda aktivieren ---
source /home/john/miniforge/etc/profile.d/conda.sh
conda activate football_cuda

# --- Pfade aufräumen/setzen ---
export PATH=$(echo $PATH | tr ':' '\n' | grep -v "^/home/john/.local/bin$" | tr '\n' ':' | sed 's/:$//')
export PATH="/home/john/miniforge/envs/football_cuda/bin:$PATH"
export LD_LIBRARY_PATH="/home/john/miniforge/envs/football_cuda/lib:${LD_LIBRARY_PATH}"

echo "=== Version Check ==="
echo "Python: $(python --version)"
echo "Ray: $(ray --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "===================="

# --- Ray Temp-Pfad (optional, verhindert /tmp overflow) ---
export RAY_TMPDIR=/home/john/grf_ippo/ray_temp_${SLURM_JOB_ID}
mkdir -p "$RAY_TMPDIR"

echo "=== Training Start ==="
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Ray temp: $RAY_TMPDIR"
echo "======================"

# === Hauptskript ausführen (ray.init() startet automatisch) ===
python -u /home/john/rlib_gfootball/main/mamba_hammer.py

# === Aufräumen ===
echo "Räume auf..."
rm -rf "$RAY_TMPDIR"