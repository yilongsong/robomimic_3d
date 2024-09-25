import h5py


def add_xml_to_dataset(xml):
    f = h5py.File(
        "/users/nharlalk/data/shared/multicamera_datasets/can/ph/multicamera_demo_v141.hdf5",
        "r+",
    )

    data_group = f["data"]

    data_group.attrs["xml_filename"] = "bins_arena.xml"
    data_group.attrs["xml_multicamera"] = xml


def test_if_xml_attr_added():
    f = h5py.File(
        "/users/nharlalk/data/shared/multicamera_datasets/can/ph/multicamera_demo_v141.hdf5",
        "r",
    )

    data_group = f["data"]

    print(data_group.attrs["xml_filename"])
    print(data_group.attrs["xml_multicamera"])


if __name__ == "__main__":
    test_if_xml_attr_added()
