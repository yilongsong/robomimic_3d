import numpy as np
import os
import h5py
import json
from copy import deepcopy

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase
from robomimic.utils.obs_utils import DEPTH_MINMAX

def array_to_string(arr):
        """
        Converts a numpy array to a string.

        Args:
            arr (np.array): numpy array to convert to string
        """
        return " ".join([str(x) for x in arr])
    
def find_index_to_add_camera(xml, camera_string = "<camera"):
    """
    Finds the index to add a camera in the xml.

    Args:
        xml (str): xml string
    """
    last_cam_starting_point = xml.rfind(camera_string)

    return xml.find(">", last_cam_starting_point) + 1

def check_output_file():
    f = h5py.File("processed_data.hdf5", "r")

    print(f.keys())
    import matplotlib.pyplot as plt
    plt.imshow(f["data"]["demo_0"]["obs"]["camera1_image"][0])
    plt.show()

class Simulation():
    def __init__(self, datasets_path, env_xml_path):
        self.env_xml_path = env_xml_path

        # Load all .hdf5 files in the datasets folder
        self.datasets = []
        for root, dirs, files in os.walk(datasets_path):
            for file in files:
                if file.endswith(".hdf5"):
                    self.datasets.append(os.path.join(root, file))

        self.added_cameras= {}

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

        self.old_xml = xml

        # Add camera to xml
        index = find_index_to_add_camera(xml)
        xml = xml[:index] + \
            f'\n    <camera mode="fixed" name="{name}" pos="{pos_string}" quat="{quat_string}" />' + \
            xml[index:]
        
        # Save xml
        with open(self.env_xml_path, "w") as f:
            f.write(xml)

        # TODO: Add depth to DEPTH_MINMAX. Should be calculated using sphere_radius

        if name not in self.added_cameras:
            self.added_cameras[name] = dict(
                pos=pos,
                quat=quat,
            )

    def restore_xml(self):
        with open(self.env_xml_path, "w") as f:
            f.write(self.old_xml)

    def generate_obs_for_dataset(self):
        dataset = self.datasets[0]

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset)

        dataset_cameras = env_meta["env_kwargs"]["camera_names"]
        camera_names = dataset_cameras + list(self.added_cameras.keys())
        env = EnvUtils.create_env_for_data_processing(
            env_meta=env_meta,
            camera_names=camera_names, 
            camera_height=128, 
            camera_width=128,
            reward_shaping=False,
        )

        is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

        f = h5py.File(dataset, "r")

        demos = list(f["data"].keys())
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]
        
        # TODO: testing line. Remove later
        demos = demos[:1]

        f_out = h5py.File("processed_data.hdf5", "w")
        data_group = f_out.create_group("data")

        total_samples = 0
        for ind in range(len(demos)):
            ep = demos[ind]

            states = f["data/{}/states".format(ep)][()]
            initial_state = dict(states=states[0])
            if is_robosuite_env:
                initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]


            # extract obs, rewards, dones
            actions = f["data/{}/actions".format(ep)][()]
            traj = self.extract_trajectory(
                env=env, 
                initial_state=initial_state, 
                states=states, 
                actions=actions,
                done_mode=1,
            )

            # Args that need to be changed
            compress = False
            exclude_next_obs = True

            # maybe copy reward or done signal from source file
            # if args.copy_rewards:
            #     traj["rewards"] = f["data/{}/rewards".format(ep)][()]
            # if args.copy_dones:
            #     traj["dones"] = f["data/{}/dones".format(ep)][()]

            ep_data_grp = data_group.create_group(ep)
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            for k in traj["obs"]:
                if compress:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]), compression="gzip")
                else:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                if not exclude_next_obs:
                    if compress:
                        ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]), compression="gzip")
                    else:
                        ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if is_robosuite_env:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]
            print("ep {}: wrote {} transitions to group {}".format(ind, ep_data_grp.attrs["num_samples"], ep))

        if "mask" in f:
            f.copy("mask", f_out)

        # global metadata
        data_group.attrs["total"] = total_samples
        data_group.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
        print("Wrote {} trajectories to {}".format(len(demos), "processed_data.hdf5"))

        f.close()
        f_out.close()

    def extract_trajectory(
        self,
        env, 
        initial_state, 
        states, 
        actions,
        done_mode,
    ):
        """
        Helper function to extract observations, rewards, and dones along a trajectory using
        the simulator environment.

        Args:
            env (instance of EnvBase): environment
            initial_state (dict): initial simulation state to load
            states (np.array): array of simulation states to load to extract information
            actions (np.array): array of actions
            done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a 
                success state. If 1, done is 1 at the end of each trajectory. 
                If 2, do both.
        """
        assert isinstance(env, EnvBase)
        assert states.shape[0] == actions.shape[0]

        # load the initial state
        env.reset()

        insert_index = find_index_to_add_camera(initial_state['model'], camera_string="<camera name=\"sideview")

        for camera_name, camera in self.added_cameras.items():
            new_cameras_xml = f'''\n    <camera mode="fixed" name="{camera_name}" pos="{array_to_string(camera['pos'])}" quat="{array_to_string(camera['quat'])}" />'''
            initial_state['model'] = initial_state['model'][:insert_index] + new_cameras_xml + initial_state['model'][insert_index:]

        obs = env.reset_to(initial_state)

        traj = dict(
            obs=[], 
            next_obs=[], 
            rewards=[], 
            dones=[], 
            actions=np.array(actions), 
            states=np.array(states), 
            initial_state_dict=initial_state,
        )
        traj_len = states.shape[0]
        # iteration variable @t is over "next obs" indices
        for t in range(1, traj_len + 1):

            # get next observation
            if t == traj_len:
                # play final action to get next observation for last timestep
                next_obs, _, _, _ = env.step(actions[t - 1])
            else:
                # reset to simulator state to get observation
                next_obs = env.reset_to({"states" : states[t]})

            # infer reward signal
            # note: our tasks use reward r(s'), reward AFTER transition, so this is
            #       the reward for the current timestep
            r = env.get_reward()

            # infer done signal
            done = False
            if (done_mode == 1) or (done_mode == 2):
                # done = 1 at end of trajectory
                done = done or (t == traj_len)
            if (done_mode == 0) or (done_mode == 2):
                # done = 1 when s' is task success state
                done = done or env.is_success()["task"]
            done = int(done)

            # collect transition
            traj["obs"].append(obs)
            traj["next_obs"].append(next_obs)
            traj["rewards"].append(r)
            traj["dones"].append(done)

            # update for next iter
            obs = deepcopy(next_obs)

        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

        # list to numpy array
        for k in traj:
            if k == "initial_state_dict":
                continue
            if isinstance(traj[k], dict):
                for kp in traj[k]:
                    traj[k][kp] = np.array(traj[k][kp])
            else:
                traj[k] = np.array(traj[k])

        return traj

if __name__ == "__main__":
    check_output_file()
    quit()
    env_xml_path = os.environ.get("ENV_XML_PATH")
    dataset_folder = os.environ.get("ROBOT_DATASETS_DIR")
    sim = Simulation(dataset_folder, env_xml_path)
    sim.add_camera(
        pos=np.array([0, -1.5, 1.4879572214102434]),
        quat=np.array([0.7933533, 0.6087614, 0, 0]),
        name="camera1"
    )
    try:
        sim.generate_obs_for_dataset()
    except Exception as e:
        print(e)
    finally:
        sim.restore_xml() 