unset LD_LIBRARY_PATH
source ~/.bashrc
source ~/th/bin/activate
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=23510

WORK_DIR=$(pwd)
DATA_ZIP_PATH=/home/dya62/scratch/datasets/shapenet_airplane.zip
DATA_ZIP_FILE=$(basename ${DATA_ZIP_PATH})
cp $DATA_ZIP_PATH $SLURM_TMPDIR
cd $SLURM_TMPDIR && unzip $DATA_ZIP_FILE && rm $DATA_ZIP_FILE
cd $WORK_DIR
