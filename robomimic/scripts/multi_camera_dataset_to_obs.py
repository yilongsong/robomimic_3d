import numpy as np

def array_to_string(arr):
        """
        Converts a numpy array to a string.

        Args:
            arr (np.array): numpy array to convert to string
        """
        return " ".join([str(x) for x in arr])
    
def find_index_to_add_camera(xml):
    """
    Finds the index to add a camera in the xml.

    Args:
        xml (str): xml string
    """
    last_cam_starting_point = xml.rfind("<camera")

    return xml.find(">", last_cam_starting_point) + 1

class Simulation():
    def __init__(self, dataset_path, env_xml_path):
        self.dataset_path = dataset_path
        self.env_xml_path = env_xml_path

    def generate_camera_pos_and_quat(self, num_cameras=1):
        """
        Generates equally-spaced camera positions and quaternions using a spherical distribution.

        Args:
            num_cameras (int): number of cameras to generate
        """
        pass

    def add_camera(self, pos, quat, name):
        """
        Adds a camera to the simulation.

        Args:
            pos (np.array): camera position
            quat (np.array): camera quaternion
        """
        pos_string = array_to_string(pos)
        quat_string = array_to_string(quat)

        # Load xml
        with open(self.env_xml_path, "r") as f:
            xml = f.read()

        # Add camera to xml
        index = find_index_to_add_camera(xml)
        xml = xml[:index] + \
            f'\n    <camera mode="fixed" name="{name}" pos="{pos_string}" quat="{quat_string}" />' + \
            xml[index:]
        

        
        # Save xml
        with open(self.env_xml_path, "w") as f:
            f.write(xml)

if __name__ == "__main__":
    sim = Simulation("", "")