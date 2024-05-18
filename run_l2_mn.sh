#!/bin/bash
#SBATCH --time=71:00:0
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name="diff_raw_0517_l2"
#SBATCH --output=./sbatch_logs/%j.log

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURM_TMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# Source the environment, load everything here
unset LD_LIBRARY_PATH
source ~/.bashrc
source ~/DiT/bin/activate

# Set master address and port
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$( echo 2$(($RANDOM % 9000 + 1000)) )
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

# Copy data to local
WORK_DIR=$(pwd)
DATA_ZIP_PATH=/home/dya62/scratch/datasets/shapenet_airplane_l_corrected.zip
DATA_ZIP_FILE=$(basename ${DATA_ZIP_PATH})

# Use srun to copy data to each node's SLURM_TMPDIR
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=1 bash -c "
cp $DATA_ZIP_PATH $SLURM_TMPDIR
cd $SLURM_TMPDIR && unzip $DATA_ZIP_FILE && rm $DATA_ZIP_FILE
cd $WORK_DIR
"

# Wait for all nodes to finish copying data
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=1 bash -c "echo Data copied to $DATA_DEST"


# Run the PyTorch distributed job
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
train.py --exp-id diff_raw_0517_l2 \
    --epoch 4000 \
    --global-batch-size 56 \
    --config-file configs/OFALG_config_v7_nl_small.yaml \
    --data-root ${SLURM_TMPDIR}/shapenet_airplane_l_corrected \
    --num-workers 32 \
    --ckpt-every 8000 \
    --work-on-tmp-dir \
    --level-num 2
