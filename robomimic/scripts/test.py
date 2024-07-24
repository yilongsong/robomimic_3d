import h5py
import os

def main():
    f_out = h5py.File("processed_data.hdf5", "w")
    f_out.close()
    f = h5py.File("processed_data.hdf5", "w")

if __name__ == "__main__":
    main()
