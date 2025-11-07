#!/bin/bash

#SBATCH --job-name=Gfootball_impala_gnn-mamba-KAN
#SBATCH --partition=compute
#SBATCH --nodes=3
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=9GB
#SBATCH --gpus-per-node=8
#SBATCH --time=90-00:00:00
#SBATCH --output=/home/john/rlib_gfootball/ray_%j.out
#SBATCH --error=/home/john/rlib_gfootball/ray_%j.err

set -x

: "${SLURM_GPUS_PER_TASK:=0}"

# Initialize conda
source /home/john/miniforge/etc/profile.d/conda.sh
conda activate base

# Remove ~/.local/bin from PATH and ensure conda env is first
export PATH=$(echo $PATH | tr ':' '\n' | grep -v "^/home/john/.local/bin$" | tr '\n' ':' | sed 's/:$//')
export PATH="/home/john/miniforge/envs/football_cuda/bin:$PATH"
export LD_LIBRARY_PATH="/home/john/miniforge/envs/football_cuda/lib:${LD_LIBRARY_PATH}"

# Verify we're using the correct versions
echo "=== Version Check ==="
echo "Python: $(python --version)"
echo "Ray: $(ray --version)"
echo "===================="

#==============================================================================
# WICHTIG: BASE auf gemounteten Speicher, aber kurzer Symlink in /tmp
#==============================================================================

# Echte Daten auf gemounteten Speicher (viel Platz)
BASE_STORAGE=/home/john/rlib_gfootball/ray_temp_${SLURM_JOB_ID}
mkdir -p "$BASE_STORAGE"

NODE=$(hostname -s)
mkdir -p "$BASE_STORAGE/$NODE"

# Kurzer Symlink in /tmp (für Socket-Pfade < 107 Zeichen)
SHORT_LINK=/tmp/r_${SLURM_JOB_ID}
ln -sfn "$BASE_STORAGE/$NODE" "$SHORT_LINK"

# Ray nutzt den kurzen Pfad
export RAY_TEMP_DIR="$SHORT_LINK"

echo "=== Storage Setup ==="
echo "Real storage: $BASE_STORAGE/$NODE"
echo "Short link: $SHORT_LINK"
echo "Ray temp dir: $RAY_TEMP_DIR"
echo "====================="

#==============================================================================
# Head-Node-Konfiguration
#==============================================================================

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | tr ' ' '\n' | grep -E '^192\.168\.2\.[0-9]{1,3}$' | head -n 1)

port=6379
ip_head="$head_node_ip:$port"

export ip_head
echo "IP Head: $ip_head"

#==============================================================================
# Starten des Ray-Clusters
#==============================================================================

echo "Starte HEAD auf $head_node"

srun --nodes=1 --ntasks=1 -w "$head_node" bash <<EOFHEAD &
export PATH="/home/john/miniforge/envs/football_cuda/bin:\$PATH"
export LD_LIBRARY_PATH="/home/john/miniforge/envs/football_cuda/lib:\$LD_LIBRARY_PATH"
/home/john/miniforge/envs/football_cuda/bin/ray start --head \
    --node-ip-address='$head_node_ip' \
    --port=$port \
    --dashboard-host=0.0.0.0 \
    --num-cpus=${SLURM_CPUS_PER_TASK} \
    --num-gpus=${SLURM_GPUS_PER_TASK} \
    --temp-dir='$RAY_TEMP_DIR' \
    --block
EOFHEAD

sleep 15

worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    node_ip=$(srun --nodes=1 --ntasks=1 -w "$node_i" hostname -I | tr ' ' '\n' | grep -E '^192\.168\.2\.[0-9]{1,3}$' | head -n 1)
    
    echo "Starte WORKER $i auf $node_i mit IP $node_ip"
    
    srun --nodes=1 --ntasks=1 -w "$node_i" bash <<EOFWORKER &
export PATH="/home/john/miniforge/envs/football_cuda/bin:\$PATH"
export LD_LIBRARY_PATH="/home/john/miniforge/envs/football_cuda/lib:\$LD_LIBRARY_PATH"

# Real storage auf gemounteten Speicher
BASE_STORAGE=$BASE_STORAGE
mkdir -p \$BASE_STORAGE/$node_i

# Kurzer Symlink
SHORT_LINK=/tmp/r_${SLURM_JOB_ID}
ln -sfn \$BASE_STORAGE/$node_i \$SHORT_LINK

export RAY_TEMP_DIR=\$SHORT_LINK

echo "Worker $node_i: Storage=\$BASE_STORAGE/$node_i, Link=\$SHORT_LINK"

/home/john/miniforge/envs/football_cuda/bin/ray start --address='$ip_head' \
    --node-ip-address='$node_ip' \
    --num-cpus=${SLURM_CPUS_PER_TASK} \
    --num-gpus=${SLURM_GPUS_PER_TASK} \
    --temp-dir='\$SHORT_LINK' \
    --block
EOFWORKER
    
    sleep 5
done

#==============================================================================
# Starten einer zusätzlichen GPU Worker-Node
#==============================================================================

echo "Starte GPU Worker-Node aus der GPU-Partition..."

GPU_WORKER_JOB=$(sbatch --parsable <<EOFGPU
#!/bin/bash
#SBATCH --job-name=ray_gpu_worker_${SLURM_JOB_ID}
#SBATCH --partition=mobile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=6GB
#SBATCH --time=90-00:00:00
#SBATCH --output=/home/john/rlib_gfootball/ray_gpu_worker_${SLURM_JOB_ID}_%j.out
#SBATCH --error=/home/john/rlib_gfootball/ray_gpu_worker_${SLURM_JOB_ID}_%j.err

source /home/john/miniforge/etc/profile.d/conda.sh
conda activate football_cuda

export PATH=\$(echo \$PATH | tr ':' '\n' | grep -v "^/home/john/.local/bin\$" | tr '\n' ':' | sed 's/:\$//')
export PATH="/home/john/miniforge/envs/football_cuda/bin:\$PATH"
export LD_LIBRARY_PATH="/home/john/miniforge/envs/football_cuda/lib:\${LD_LIBRARY_PATH}"

# Real storage auf gemounteten Speicher
BASE_STORAGE=/home/john/rlib_gfootball/ray_temp_${SLURM_JOB_ID}
GPU_NODE=\$(hostname -s)
mkdir -p \$BASE_STORAGE/\$GPU_NODE

# Kurzer Symlink
SHORT_LINK=/tmp/r_${SLURM_JOB_ID}
ln -sfn \$BASE_STORAGE/\$GPU_NODE \$SHORT_LINK

export RAY_TEMP_DIR=\$SHORT_LINK

GPU_NODE_IP=\$(hostname -I | tr ' ' '\n' | grep -E '^192\.168\.2\.[0-9]{1,3}\$' | head -n 1)

echo "=== GPU Worker Info ==="
echo "Node: \$GPU_NODE"
echo "IP: \$GPU_NODE_IP"
echo "Real storage: \$BASE_STORAGE/\$GPU_NODE"
echo "Short link: \$SHORT_LINK -> \$(readlink \$SHORT_LINK)"
echo "Ray temp dir: \$RAY_TEMP_DIR"
ls -lah \$SHORT_LINK || echo "Link not found!"
df -h \$BASE_STORAGE || echo "Storage not mounted!"
echo "GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "Connecting to: $ip_head"
echo "======================"

ping -c 3 $head_node_ip && echo "✓ Head reachable" || echo "✗ Cannot reach head!"

echo "Starting Ray worker..."
/home/john/miniforge/envs/football_cuda/bin/ray start --address='$ip_head' \
    --node-ip-address="\$GPU_NODE_IP" \
    --num-cpus=256 \
    --num-gpus=1 \
    --temp-dir='\$SHORT_LINK' \
    --block
EOFGPU
)

if [ -z "$GPU_WORKER_JOB" ]; then
    echo "FEHLER: GPU Worker Job konnte nicht gestartet werden!"
else
    echo "GPU Worker Job ID: $GPU_WORKER_JOB"
    echo "Warte auf GPU Worker Verbindung..."
    
    echo "Prüfe GPU-Verfügbarkeit..."
    for i in {1..120}; do
        sleep 2
        GPU_STATUS=$(ray status 2>/dev/null | grep -i "gpu" || echo "")
        if echo "$GPU_STATUS" | grep -q "[1-9]"; then
            echo "✓ GPUs im Cluster gefunden!"
            ray status
            break
        fi
        if [ $((i % 10)) -eq 0 ]; then
            echo "Warte noch auf GPUs... (${i}/120 Sekunden)"
        fi
    done
    
    echo "=== Finale Cluster-Ressourcen ==="
    ray status
    echo "=================================="
fi

#==============================================================================
# Ausführen des Hauptskripts
#==============================================================================

echo "Führe die Python-Anwendung aus..."
python -u /home/john/rlib_gfootball/main.py "$SLURM_CPUS_PER_TASK"

#==============================================================================
# Aufräumen
#==============================================================================

echo "Beende Ray Cluster..."
ray stop