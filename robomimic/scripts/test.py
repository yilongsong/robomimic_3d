import h5py

def main():
    f = h5py.File("coffee_data_84.hdf5", "r")

    # Check the type of the data
    print(f["data"]["demo_0"]["obs"]["camera0_rgbd"][0][0][0])
    print("The type of the data is:", type(f["data"]["demo_0"]["obs"]["robot0_joint_pos"][0][0]))
    

if __name__ == "__main__":
    main()
