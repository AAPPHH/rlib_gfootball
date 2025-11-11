#!/bin/bash
#SBATCH --job-name=xgboost_dgx
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --gres=gpu:8
#SBATCH --mem=1500G
#SBATCH --time=72:00:00
#SBATCH --output=xgboost_%j.out
#SBATCH --error=xgboost_%j.err
#SBATCH --exclusive

set -x

source /home/john/miniforge/etc/profile.d/conda.sh
conda activate base

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK

export CUDA_LAUNCH_BLOCKING=0

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_LEVEL=NVL

export XGBOOST_USE_CUDA=1
export XGBOOST_USE_NCCL=1

echo "=========================================="
echo "Job Information:"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: 8"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "=========================================="

echo "Initial GPU status:"
nvidia-smi

echo "CPU Information:"
lscpu | grep -E '^CPU\(s\):|^Thread\(s\) per core:|^Core\(s\) per socket:|^Socket\(s\):'

echo "=== Environment Check ==="
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Conda Env: $CONDA_PREFIX"
echo "========================="

echo "Testing XGBoost GPU support..."
python -c "import xgboost as xgb; print(f'XGBoost version: {xgb.__version__}')"

cd /home/john/rlib_gfootball/cold_start

echo "SLURM & BASH see CUDA_VISIBLE_DEVICES as: [$CUDA_VISIBLE_DEVICES]"

echo "Starting GPU monitoring..."
(
    while true; do
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "=== GPU Status at $timestamp ===" >> gpu_monitor_${SLURM_JOB_ID}.log
        nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv >> gpu_monitor_${SLURM_JOB_ID}.log
        echo "" >> gpu_monitor_${SLURM_JOB_ID}.log
        sleep 30
    done
) &
MONITOR_PID=$!

echo "=========================================="
echo "Starting XGBoost Multi-GPU Training..."
echo "=========================================="

python /home/john/rlib_gfootball/cold_start/xgboost_train_dgx.py

EXIT_CODE=$?

kill $MONITOR_PID 2>/dev/null
echo "GPU monitoring stopped"

echo "=========================================="
echo "Final GPU status:"
nvidia-smi

if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "✅ Training completed successfully!"
    echo "=========================================="
    
    echo "Output files:"
    ls -lh /home/john/rlib_gfootball/cold_start/xgboost_dgx_output/ 2>/dev/null || echo "No output files found yet"
    
    echo ""
    echo "GPU Usage Summary:"
    if [ -f gpu_monitor_${SLURM_JOB_ID}.log ]; then
        echo "Average GPU utilization:"
        grep -v "===" gpu_monitor_${SLURM_JOB_ID}.log | grep -v "^$" | grep -v "index" | \
            awk -F',' '{sum+=$3; count++} END {if(count>0) printf "%.1f%%\n", sum/count}'
        echo "Peak GPU memory usage:"
        grep -v "===" gpu_monitor_${SLURM_JOB_ID}.log | grep -v "^$" | grep -v "index" | \
            awk -F',' '{if($5>max) max=$5} END {printf "%.1f GB\n", max/1024}'
    fi
else
    echo "=========================================="
    echo "❌ Training failed with exit code $EXIT_CODE"
    echo "=========================================="
fi

echo "Job completed at $(date)"

if [ -f gpu_monitor_${SLURM_JOB_ID}.log ]; then
    mkdir -p /home/john/rlib_gfootball/cold_start/xgboost_dgx_output/
    cp gpu_monitor_${SLURM_JOB_ID}.log /home/john/rlib_gfootball/cold_start/xgboost_dgx_output/
    echo "GPU monitoring log saved to output directory"
fi

echo "=========================================="
echo "Job ${SLURM_JOB_ID} finished"
echo "=========================================="