import os
import subprocess

ROBOSUITE_XML_FOLDER_PATH = "~/data/nharlalk/robosuite/robosuite/"
OUTPUT_DIR = "~/data/shared/multicamera_datasets/"
DATASET_DIR = "~/data/nharlalk/datasets/"
DATASET_FILE = "can/ph/demo_v141.hdf5"
OUTPUT_FILE = "can/ph/multicamera_demo_v141.hdf5"

ENV_XML_LIST = [
    "bins_arena.xml",
    "empty_arena.xml",
    "table_arena.xml",
    "pegs_arena.xml",
    "multi_table_arena.xml",
]

NUM_CAMERAS = 60


def generate_dataset():
    def get_output_file_name(dataset_path):
        filename = f"multicamera_{dataset_path.split('/')[-1]}"
        split_path = dataset_path.split("/")
        split_path[-1] = filename
        return os.path.join(*split_path)

    dataset_path = os.path.join(DATASET_DIR, DATASET_FILE)
    output_path = os.path.join(OUTPUT_DIR, get_output_file_name(DATASET_FILE))
    env_xml_path = os.path.join(ROBOSUITE_XML_FOLDER_PATH, ENV_XML_LIST[0])
    num_workers = int(subprocess.check_output(["nproc"]).decode().strip())
    num_demos = 100

    command = f"python multicamera_dataset_to_obs.py --dataset {dataset_path} --output_file {output_path} --env_xml {env_xml_path} --num_workers {num_workers} --num_demos {num_demos} --num_cameras {NUM_CAMERAS}"

    print(f"Running command: {command}")

    subprocess.run(command, shell=True)


def play_dataset():
    dataset_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    for i in range(NUM_CAMERAS):
        command = f"python playback_dataset.py --n 1 --render_image_names camera{i} \
            --dataset {dataset_path} --video_path ~/data/nharlalk/robomimic/robomimic/playbacks/{OUTPUT_FILE.split('/')[-1].split('.')[0]}_camera{i}.mp4 \
                --multicamera"

        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    play_dataset()
