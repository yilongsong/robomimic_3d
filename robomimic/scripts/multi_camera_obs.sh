#!/bin/bash
#SBATCH -o ../ccv/job-%j.out
#SBATCH -e ../ccv/job-%j.err
#SBATCH --time=1:00:00

#SBATCH --nodes=1
#SBATCH -c 2
#SBATCH --mem-per-cpu 3G

module load miniconda3
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate cam-env

python multi_camera_dataset_to_obs.py --dataset /users/nharlalk/data/shared/mimicgen/core/pick_place_d0.hdf5 --env_xml /users/nharlalk/data/nharlalk/robosuite/robosuite/models/assets/arenas/table_arena.xml --num_cameras 60 --output_file multicamera_pick_place_d0.hdf5
