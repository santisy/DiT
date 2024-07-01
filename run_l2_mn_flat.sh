#!/bin/bash
#SBATCH --time=71:00:0
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="l2_0630_flat_DiT"
#SBATCH --output=./sbatch_logs/%j.log

# List out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURM_TMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# Source the environment, load everything here
source ~/.bashrc
unset LD_LIBRARY_PATH
source ~/DiT/bin/activate

# Set master address and port
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$( echo 2$(($RANDOM % 9000 + 1000)) )
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

# Increase NCCL timeout to prevent socket timeout errors
export NCCL_SOCKET_TIMEOUT=3600
export NCCL_IB_TIMEOUT=22  # Set Infiniband timeout
export NCCL_DEBUG=INFO  # Enable NCCL debug logging

# Copy data to local
WORK_DIR=$(pwd)
DATA_ZIP_PATH=./datasets/shapenet_airplane_discreteL1.zip
DATA_ZIP_FILE=$(basename ${DATA_ZIP_PATH})

# Use srun to copy data to each node's SLURM_TMPDIR
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "
cp $DATA_ZIP_PATH $SLURM_TMPDIR
cd $SLURM_TMPDIR && unzip -q $DATA_ZIP_FILE && rm $DATA_ZIP_FILE
cd $WORK_DIR
"

# Wait for all nodes to finish copying data
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "echo Data copied."

# Ensure all environment variables are set correctly
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "
echo NODE: \$SLURM_NODEID
echo MASTER_ADDR=\$MASTER_ADDR
echo MASTER_PORT=\$MASTER_PORT
echo WORLD_SIZE=\$WORLD_SIZE
echo NCCL_SOCKET_TIMEOUT=\$NCCL_SOCKET_TIMEOUT
echo NCCL_IB_TIMEOUT=\$NCCL_IB_TIMEOUT
echo NCCL_DEBUG=\$NCCL_DEBUG
"

# Run the PyTorch distributed job
srun --ntasks=$WORLD_SIZE --ntasks-per-node=$SLURM_NTASKS_PER_NODE torchrun \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_id="$SLURM_JOBID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    train.py --exp-id l2_0630_flat_DiT \
    --epoch 4000 \
    --global-batch-size 96 \
    --config-file configs/OFALG_config_v9_predV_cos_ra_flat.yaml \
    --data-root ${SLURM_TMPDIR}/shapenet_airplane_discreteL1 \
    --num-workers 40 \
    --ckpt-every 8000 \
    --work-on-tmp-dir \
    --gradient-clipping \
    --level-num 2
