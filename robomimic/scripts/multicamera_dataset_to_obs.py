import os
import h5py
import json
import time
import math
import argparse
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils

from copy import deepcopy
from robomimic.envs.env_base import EnvBase
from performance_analysis import check_memory_and_time

multiprocessing.set_start_method('spawn', force=True)

def debug_print(*args):
    print("-----DEBUG-----")
    print(*args)
    print("---------------")

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

    if last_cam_starting_point == -1:
        raise ValueError("Could not find camera string in xml to add camera.")

    return xml.find(">", last_cam_starting_point) + 1

def visualize_points_on_sphere(points, radius):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color='b', alpha=0.1)

    plt.savefig('camera_positions_on_sphere.png')

def get_angle_from_sin_cos(sin, cos):
    if sin >= 0 and cos >= 0:
        return np.arcsin(sin)
    if sin >= 0 and cos < 0:
        return np.pi - np.arcsin(sin)
    if sin < 0 and cos < 0:
        return np.pi - np.arcsin(sin)
    if sin < 0 and cos >= 0:
        return 2 * np.pi + np.arcsin(sin)
    
def combine_quaternions(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])

def visualize_camera_views(cameras, data_file):
        f = h5py.File(data_file, "r")

        num_cameras = len(cameras.keys())

        fig, axes = plt.subplots(
            math.ceil(num_cameras / 3), 
            3, 
            figsize=(12, 4)
        )
        
        for i, camera in enumerate(cameras.keys()):
            camera_name = camera
            image = f["data"]["demo_0"]["obs"][f"{camera_name}_image"][0]
            axes[i // 3, i % 3].imshow(image)
            axes[i // 3, i % 3].set_title(camera_name)
            axes[i // 3, i % 3].axis("off")
        plt.savefig('multiview_figure.png')

def calculate_quaternion(normalized_target_orientation, pos, original_orientation=np.array([0, 0, -1])):
        normalized_x_y = np.array([pos[0], pos[1]]) / np.linalg.norm([pos[0], pos[1]])
        horizontal_rotation_angle = (get_angle_from_sin_cos(normalized_x_y[1], normalized_x_y[0]) + np.pi / 2) % (2 * np.pi)

        horizontal_rotation_quat = np.array([
            np.cos(horizontal_rotation_angle / 2),
            0,
            0,
            np.sin(horizontal_rotation_angle / 2),
        ])

        angle = np.arccos(np.dot(original_orientation, normalized_target_orientation))
        cross = np.cross(original_orientation, normalized_target_orientation)

        rotation_quat = np.array([
            np.cos(angle / 2),
            np.sin(angle / 2) * cross[0],
            np.sin(angle / 2) * cross[1],
            np.sin(angle / 2) * cross[2],
        ])

        quat = combine_quaternions(rotation_quat, horizontal_rotation_quat)

        return quat

def generate_equally_spaced_points_on_sphere(num_points, camera_sphere_radius):
        """
        Samples equally spaced points on a sphere.

        Args:
            num_points (int): number of points to sample
        """
        points = np.zeros((num_points, 3))

        # Adjustable parameters
        theta_max_angle = np.pi / 3
        theta_adjustment = 0.0
        phi_max_angle = 3 * np.pi / 2
        phi_adjustment = 5 * np.pi / 6

        # Derived parameters
        m_theta = int(np.round(np.sqrt(num_points)))
        n_count = 0

        for m in range(1, m_theta + 1):
            if n_count >= num_points:
                break
            
            theta = theta_max_angle * (m / m_theta) - theta_adjustment

            # Adjust the number of points in the phi direction based on theta. 
            # Adding 0.7 to the sin(theta) term since not using 0 angle and 
            # want to make up the difference between num_points and m_theta^2
            m_phi = int(np.round(np.sqrt(num_points) * (np.sin(theta) + 0.5) ))

            for n in range(1, m_phi + 1):
                if n_count >= num_points:
                    break
                phi = phi_max_angle * (n / m_phi) - phi_adjustment
                points[n_count] = np.array([
                    camera_sphere_radius * np.sin(theta) * np.cos(phi),
                    camera_sphere_radius * np.sin(theta) * np.sin(phi),
                    camera_sphere_radius * np.cos(theta)
                ])
                n_count += 1

        return points

def generate_camera_pos_and_quat(args):
        """
        Generates camera positions and quaternions.

        Args:
            num_cameras (int): number of cameras to generate
        """

        num_cameras = args.num_cameras
        camera_sphere_radius = args.camera_sphere_radius

        pos = generate_equally_spaced_points_on_sphere(num_cameras, camera_sphere_radius)

        origin = [
            np.array(
                [
                    -pos[i][0] / 2,
                    -pos[i][1] / 2,
                    -pos[i][2] / 2 + 0.85 * camera_sphere_radius
                ]
            ) for i in range(num_cameras)]

        target_vectors = np.array([origin[i] - pos[i] for i in range(num_cameras)])
        normalized_target_vectors = np.array([target_vectors[i] / np.linalg.norm(target_vectors[i]) for i in range(num_cameras)])

        quats = np.array([calculate_quaternion(normalized_target_vectors[i], pos[i]) for i in range(num_cameras)])

        return pos, quats

def setup_additional_cameras(pos, quat, args):
        """
        Adds a camera to the simulation.

        Args:
            pos (np.array): camera position
            quat (np.array): camera quaternion
        """
        pos_string = [array_to_string(p) for p in pos]
        quat_string = [array_to_string(q) for q in quat]

        names = [f"camera{i}" for i in range(len(pos))]

        # Load xml
        with open(args.env_xml, "r") as f:
            xml = f.read()

        old_xml = xml

        # Add camera to xml
        index = find_index_to_add_camera(xml)
        for i in range(len(pos)):
            xml = xml[:index] + \
                f'\n    <camera mode="fixed" name="{names[i]}" pos="{pos_string[i]}" quat="{quat_string[i]}"/>' + \
                xml[index:]
        
        # Save xml
        with open(args.env_xml, "w") as f:
            f.write(xml)

        for i in range(len(pos)):
            args.cameras[names[i]] = dict(
                pos=pos[i],
                quat=quat[i],
            )
            args.custom_camera_names = [
                names[i] for i in range(len(pos))
            ]

        return old_xml

def generate_obs_for_dataset(args):
    env_meta = FileUtils.get_env_metadata_from_dataset(args.dataset)

    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=list(args.cameras.keys()), 
        camera_height=84,   
        camera_width=84,
        reward_shaping=False,
    )

    # Add the depth ranges for the custom cameras
    for custom_camera_name in args.custom_camera_names:
        env.depthminmax.add(
            custom_camera_name + "_depth",
            [0.1, args.camera_sphere_radius + 1.0]
        )

    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    f = h5py.File(args.dataset, "r")

    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # f_out = h5py.File(generated_dataset_path, "w")
    # data_group = f_out.create_group("data")
    if os.path.exists(args.output_file):
        f_out = h5py.File(args.output_file, "r+")
        data_group = f_out["data"]
        start_ind = len(data_group.keys())
    else:
        f_out = h5py.File(args.output_file, "w")
        data_group = f_out.create_group("data")
        start_ind = 0
    
    demos = demos[:start_ind + args.num_demos]

    total_samples = 0
    for ind in range(start_ind, len(demos)):
        ep = demos[ind]

        states = f["data/{}/states".format(ep)][()]

        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        # extract obs, rewards, dones
        actions = f["data/{}/actions".format(ep)][()]
        
        traj = extract_trajectory(
            args,
            env=env, 
            initial_state=initial_state, 
            states=states, 
            actions=actions,
            done_mode=1
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
        f_out.flush()
        print("ep {}: wrote {} transitions to group {}".format(ind, ep_data_grp.attrs["num_samples"], ep))

    # if "mask" in f:
    #     f.copy("mask", f_out)

    # global metadata
    data_group.attrs["total"] = total_samples
    data_group.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
    print("Wrote {} trajectories to {}".format(len(demos), args.output_file))

    f.close()
    f_out.close()

def extract_trajectory(
        args,
        env_meta, 
        initial_state, 
        states, 
        actions,
        done_mode = 1
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
        env = EnvUtils.create_env_for_data_processing(
            env_meta=env_meta,
            camera_names=list(args.cameras.keys()), 
            camera_height=84,   
            camera_width=84,
            reward_shaping=False,
        )

        # Add the depth ranges for the custom cameras
        for custom_camera_name in args.custom_camera_names:
            env.depthminmax.add(
                custom_camera_name + "_depth",
                [0.1, args.camera_sphere_radius + 1.0]
            )
        
        assert isinstance(env, EnvBase)
        assert states.shape[0] == actions.shape[0]

        # load the initial state
        env.reset()
        insert_index = find_index_to_add_camera(initial_state['model'], camera_string="<camera name=\"agentview")

        for camera in args.custom_camera_names:
            new_cameras_xml = f'''\n    <camera mode="fixed" name="{camera}" pos="{array_to_string(args.cameras[camera]['pos'])}" quat="{array_to_string(args.cameras[camera]['quat'])}" />'''
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

def worker(input):
    args, env_meta, initial_state, states, actions = input

    traj = extract_trajectory(
        args,
        env_meta=env_meta, 
        initial_state=initial_state, 
        states=states, 
        actions=actions,
        done_mode=1
    )

    return traj

def generate_obs_for_dataset_parallel(args):
    env_meta = FileUtils.get_env_metadata_from_dataset(args.dataset)

    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=list(args.cameras.keys()), 
        camera_height=84,   
        camera_width=84,
        reward_shaping=False,
    )

    # Add the depth ranges for the custom cameras
    for custom_camera_name in args.custom_camera_names:
        env.depthminmax.add(
            custom_camera_name + "_depth",
            [0.1, args.camera_sphere_radius + 1.0]
        )

    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    f = h5py.File(args.dataset, "r")

    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    if os.path.exists(args.output_file):
        f_out = h5py.File(args.output_file, "r+")
        data_group = f_out["data"]
        start_ind = len(data_group.keys())
    else:
        f_out = h5py.File(args.output_file, "w")
        data_group = f_out.create_group("data")
        start_ind = 0

    demos = demos[:start_ind + args.num_demos]
    num_workers = args.num_workers
    total_samples = 0

    for i in range(start_ind, len(demos), num_workers):
        current_demos = demos[i:min(i + num_workers, len(demos))]
        initial_state_list = []
        states_list = []
        actions_list = []

        for ep in current_demos:
            states_list.append(f["data/{}/states".format(ep)][()])
            actions_list.append(f["data/{}/actions".format(ep)][()])

            initial_state = dict(states=states_list[-1][0])
            if is_robosuite_env:
                initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

            initial_state_list.append(initial_state)

        inputs = [(args, env_meta, initial_state_list[j], states_list[j], actions_list[j]) for j in range(len(current_demos))]

        with multiprocessing.Pool(num_workers) as p:
            trajs = p.map(worker, inputs)

        for j in range(len(current_demos)):
            ep = current_demos[j]
            traj = trajs[j]

            ep_data_grp = data_group.create_group(ep)
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            for k in traj["obs"]:
                ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
            for k in traj["next_obs"]:
                ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if is_robosuite_env:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]

            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]
            total_samples += traj["actions"].shape[0]
            f_out.flush()
            print("{}: wrote {} transitions to group {}".format(
                ep, 
                ep_data_grp.attrs["num_samples"], 
                ep
            ))
        del inputs
        del trajs

    # global metadata
    data_group.attrs["total"] = total_samples
    data_group.attrs["env_args"] = json.dumps(env.serialize(), indent=4)
    print("Wrote {} demos to {}".format(len(demos), args.output_file))

    f.close()
    f_out.close()

def restore_xml(args, old_xml):
    with open(args.env_xml, "w") as f:
        f.write(old_xml)

@check_memory_and_time
def generate_multicamera_obs_dataset(args):
    print("Generating camera positions and quarternions...")
    camera_positions, camera_quarternions = generate_camera_pos_and_quat(
        args
    )

    print("Setting up cameras...")
    old_xml = setup_additional_cameras(
        camera_positions,
        camera_quarternions,
        args
    )

    print("Generating observations...")

    try:
        generate_obs_for_dataset_parallel(args)
        # visualize_camera_views(args.cameras, args.output_file)
    except Exception as e:
        print(e)
    finally:
        restore_xml(args, old_xml)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
        default="/users/nharlalk/data/shared/mimicgen/core/pick_place_d0.hdf5"
    )
    parser.add_argument(
        "--num_cameras",
        type=int,
        default=5,
        help="number of cameras to add to the simulation"
    )
    parser.add_argument(
        "--env_xml",
        type=str,
        help="path to the environment xml file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="multicamera_pick_place_d0.hdf5",
        help="path to the output file"
    )
    parser.add_argument(
        "--num_demos",
        type=int,
        default=1,
        help="number of demos to process"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of workers to use for parallel processing"
    )

    args = parser.parse_args()

    # Additional args
    args.cameras= {
        "frontview": {
            "pos": np.array([1.6, 0, 1.45]),
            "quat": np.array([0.56, 0.43, 0.43, 0.56])
        },
        "birdview": {
            "pos": np.array([-0.2, 0, 3.0]),
            "quat": np.array([0.7071, 0, 0, 0.7071])
        },
        "agentview": {
            "pos": np.array([0.5, 0, 1.35]),
            "quat": np.array([0.653, 0.271, 0.271, 0.653])
        }
    }

    args.camera_sphere_radius = 1.8

    
    generate_multicamera_obs_dataset(args)


