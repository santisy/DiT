#!/bin/bash
#SBATCH --time=4:00:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --gres=gpu:v100l:1
#SBATCH --job-name="level0"
#SBATCH --output=./sbatch_logs/%j.log

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURM_TMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
# sample process (list hostnames of the nodes you've requested)
NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' | wc -l`
echo NPROCS=$NPROCS

# Source the environment, load everything here
unset LD_LIBRARY_PATH
source ~/.bashrc
source ~/th/bin/activate

# Set master address and port
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=23512

WORK_DIR=$(pwd)
DATA_ZIP_PATH=/home/dya62/scratch/datasets/shapenet_airplane.zip
DATA_ZIP_FILE=$(basename ${DATA_ZIP_PATH})
cp $DATA_ZIP_PATH $SLURM_TMPDIR
cd $SLURM_TMPDIR && unzip $DATA_ZIP_FILE && rm $DATA_ZIP_FILE
cd $WORK_DIR

# Run the PyTorch distributed job
torchrun \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
train.py --exp-id test_exp \
    --epoch 100 \
    --global-batch-size 64 \
    --config-file configs/OFALG_config.yaml \
    --data-root ${SLURM_TMPDIR}/shapenet_airplane \
    --work-on-tmp-dir \
    --level_num 0
