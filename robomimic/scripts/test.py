import h5py
import os

def main():
    # "../../../camera-invariant-robot-learning/datasets/core/coffee_d0.hdf5"
    dataset_folder = os.environ.get("ROBOT_DATASETS_DIR")
    f = h5py.File(os.path.join(dataset_folder, "core/coffee_d0.hdf5"), "r")

    demos = f["data"].keys()
    print(demos)
    print("Number of demos in dataset: ", len(demos))

if __name__ == "__main__":
    main()