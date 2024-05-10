#!/bin/bash
#SBATCH --time=71:00:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=25
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100l:2
#SBATCH --job-name="diff_0509"
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
source ~/th/bin/activate

# Set master address and port
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$( echo 2$(($RANDOM % 9000 + 1000)) )
NPROCS=$( echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l )
echo "NPROCS=$NPROCS"

# Copy data to local
WORK_DIR=$(pwd)
DATA_ZIP_PATH=/home/dya62/scratch/datasets/shapenet_airplane_l.zip
DATA_ZIP_FILE=$(basename ${DATA_ZIP_PATH})
cp $DATA_ZIP_PATH $SLURM_TMPDIR
cd $SLURM_TMPDIR && unzip $DATA_ZIP_FILE && rm $DATA_ZIP_FILE
cd $WORK_DIR

# Run the PyTorch distributed job
torchrun \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --nproc_per_node=$NPROCS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
train.py --exp-id diff_0509 \
    --epoch 4000 \
    --global-batch-size 48 \
    --config-file configs/OFALG_config_v7_nl_small.yaml \
    --data-root ${SLURM_TMPDIR}/shapenet_airplane_l_corrected \
    --num-workers 24 \
    --ckpt-every 8000 \
    --work-on-tmp-dir \
    --vae-ckpt training_runs/vae_0509_VQ/vae_0020000.pt \
    --level-num 2
