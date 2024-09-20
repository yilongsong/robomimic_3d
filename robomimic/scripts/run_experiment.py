import os
import subprocess

import submitit


def get_output_file_name(dataset_path):
    filename = f"multicamera_{dataset_path.split('/')[-1]}"
    split_path = dataset_path.split("/")
    split_path[-1] = filename
    return os.path.join(*split_path)


OUTPUT_DIR = "~/data/shared/multicamera_datasets/"
DATASET_DIR = "~/data/nharlalk/datasets/"
DATASET_FILE = "can/ph/demo_v141.hdf5"

ENV_XML_LIST = [
    "models/assets/arenas/bins_arena.xml",
    "models/assets/arenas/empty_arena.xml",
    "models/assets/arenas/table_arena.xml",
    "models/assets/arenas/pegs_arena.xml",
    "models/assets/arenas/multi_table_arena.xml",
]

ROBOSUITE_PATH = "~/data/nharlalk/robosuite/robosuite/"

dataset_path = os.path.join(DATASET_DIR, DATASET_FILE)
output_path = os.path.join(OUTPUT_DIR, get_output_file_name(DATASET_FILE))
env_xml_path = os.path.join(ROBOSUITE_PATH, ENV_XML_LIST[0])
num_workers = int(subprocess.check_output(["nproc"]).decode().strip())
num_demos = 100
num_cameras = 60

command = f"python multicamera_dataset_to_obs.py --dataset {dataset_path} --output_file {output_path} --env_xml {env_xml_path} --num_workers {num_workers} --num_demos {num_demos} --num_cameras {num_cameras}"

print(f"Running command: {command}")

subprocess.run(command, shell=True)
