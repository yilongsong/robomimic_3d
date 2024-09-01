import h5py

def main():
    f = h5py.File("coffee_data.hdf5", "w")

    # Check the type of the data
    print("The type of the data is:", type(f["data"]["demo_0"]["obs"]["camera0_rgbd"][0][0]))

if __name__ == "__main__":
    main()
