#!/bin/bash
#SBATCH --job-name=Gfootball_impala_mamba
#SBATCH --partition=mobile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem-per-cpu=6GB                # evtl. anpassen
#SBATCH --gres=gpu:1                     # entfernen, falls keine GPU
#SBATCH --time=30-00:00:00               # realistisch anpassen
#SBATCH --output=/home/john/rlib_gfootball/ray_%j.out
#SBATCH --error=/home/john/rlib_gfootball/ray_%j.err

set -x

# === CPU/GPU-Defaults für Single-Node ===
: "${SLURM_GPUS_PER_TASK:=1}"            # auf 0 setzen, wenn keine GPU

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
echo "===================="

# --- Kurzer Temp-Pfad für Ray ---
BASE_STORAGE=/home/john/rlib_gfootball/ray_temp_${SLURM_JOB_ID}
NODE=$(hostname -s)
mkdir -p "$BASE_STORAGE/$NODE"
SHORT_LINK=/tmp/r_${SLURM_JOB_ID}
ln -sfn "$BASE_STORAGE/$NODE" "$SHORT_LINK"
export RAY_TEMP_DIR="$SHORT_LINK"

echo "=== Storage Setup ==="
echo "Real storage: $BASE_STORAGE/$NODE"
echo "Short link:   $SHORT_LINK"
echo "Ray temp dir: $RAY_TEMP_DIR"
echo "====================="

# === Single-Node Ray starten (nur HEAD, keine Worker) ===
# IP lokal binden – bei Single-Node reicht 127.0.0.1
port=6379
ip_head="127.0.0.1:${port}"
echo "IP Head: $ip_head"

# Vorsichtshalber alte Instanz stoppen
ray stop || true

/home/john/miniforge/envs/football_cuda/bin/ray start --head \
  --node-ip-address=127.0.0.1 \
  --port=$port \
  --dashboard-host=0.0.0.0 \
  --num-cpus=${SLURM_CPUS_PER_TASK} \
  --num-gpus=${SLURM_GPUS_PER_TASK} \
  --temp-dir="$RAY_TEMP_DIR"

echo "=== Cluster-Status (Single-Node) ==="
ray status || true
echo "===================================="

# === Hauptskript ausführen ===
echo "Starte Python-App..."
python -u /home/john/rlib_gfootball/main.py "$SLURM_CPUS_PER_TASK"

# === Aufräumen ===
echo "Beende Ray..."
ray stop
