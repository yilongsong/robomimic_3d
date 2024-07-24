#!/bin/bash
#SBATCH -o job-%j.out
#SBATCH -e job-%j.err
#SBATCH --time=0:05:00

#SBATCH --nodes=1
#SBATCH -c 2
#SBATCH --mem-per-cpu 3G


#SBATCH --export=ROBOT_DATASETS_DIR=/users/nharlalk/data/shared/mimicgen/core,ENV_XML_PATH=/users/nharlalk/data/nharlalk/robosuite/robosuite/models/assets/arenas/table_arena.xml


module load miniconda3
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate cam-env

python multi_camera_dataset_to_obs.py
#python test.py
