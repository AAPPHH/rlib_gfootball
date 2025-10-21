#!/bin/bash



#==============================================================================

# SLURM-Direktiven

#==============================================================================

# shellcheck disable=SC2206


#SBATCH --job-name=LIPR

#SBATCH --partition=compute

#SBATCH --nodes=10

#SBATCH --tasks-per-node=1

#SBATCH --cpus-per-task=96

#SBATCH --mem-per-cpu=9GB

#SBATCH --time=90-00:00:00

#SBATCH --output=ray_%j.out

#SBATCH --error=ray_%j.err


set -x


: "${SLURM_GPUS_PER_TASK:=0}"

source /home/john/Ray_LIPR/football-venv/bin/activate


BASE=/home/john/Ray_LIPR/ray_temp_${SLURM_JOB_ID}

mkdir -p "$BASE"


NODE=$(hostname -s)

mkdir -p "$BASE/$NODE"

ln -sfn "$BASE/$NODE" "/tmp/ray_${SLURM_JOB_ID}"


export RAY_TEMP_DIR="/tmp/ray_${SLURM_JOB_ID}"


#==============================================================================

# Head-Node-Konfiguration

#==============================================================================


nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")

nodes_array=($nodes)

head_node=${nodes_array[0]}

head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | tr ' ' '\n' | grep -E '^([0-9]{1,3}\.){2}2\.[0-9]{1,3}$' | head -n 1)


#==============================================================================

# Starten des Ray-Clusters

#==============================================================================


port=6379

ip_head="$head_node_ip:$port"

export ip_head

echo "IP Head: $ip_head"



echo "Starte HEAD auf $head_node"

srun --nodes=1 --ntasks=1 -w "$head_node" \

    ray start --head \

    --node-ip-address="$head_node_ip" \

    --port=$port \

    --dashboard-host 0.0.0.0 \

    --num-cpus "${SLURM_CPUS_PER_TASK}" \

    --num-gpus "${SLURM_GPUS_PER_TASK}" \

    --temp-dir "$RAY_TEMP_DIR" \

    --block &



sleep 15



worker_num=$((SLURM_JOB_NUM_NODES - 1))



for ((i = 1; i <= worker_num; i++)); do

    node_i=${nodes_array[$i]}



    node_ip=$(srun --nodes=1 --ntasks=1 -w "$node_i" hostname -I | tr ' ' '\n' | grep -E '^([0-9]{1,3}\.){2}2\.[0-9]{1,3}$' | head -n 1)



    echo "Starte WORKER $i auf $node_i mit IP $node_ip"

    srun --nodes=1 --ntasks=1 -w "$node_i" \

        bash -c "mkdir -p $BASE/$node_i && \

                 ln -sfn $BASE/$node_i /tmp/ray_${SLURM_JOB_ID} && \

                 RAY_TEMP_DIR=/tmp/ray_${SLURM_JOB_ID} \

                 ray start --address $ip_head \

                           --node-ip-address=$node_ip \

                           --num-cpus ${SLURM_CPUS_PER_TASK} \

                           --num-gpus ${SLURM_GPUS_PER_TASK} \

                           --temp-dir \$RAY_TEMP_DIR \

                           --block" &

    sleep 5

done


#==============================================================================

# Ausführen des Hauptskripts

#==============================================================================

echo "Führe die Python-Anwendung aus..."

python -u /home/john/Ray_LIPR/search_class_file.py "$SLURM_CPUS_PER_TASK"


#==============================================================================

# Aufräumen und Warten

#==============================================================================

ray stop

wait