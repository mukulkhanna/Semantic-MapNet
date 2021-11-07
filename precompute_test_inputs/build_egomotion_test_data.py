import json
import os
import sys

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "../"))
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.utils.visualizations import maps
from scipy.spatial.transform import Rotation as R

from projector import _transform3D
from projector.projector import Projector
from utils.habitat_utils import HabitatUtils

"""
Directory structure:
- /PointNav-VO
- /Semantic-MapNet
    - /precompute_test_inputs
        - build_egomotion_test_data.py
"""
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VO_DIR = os.path.join(BASE_DIR, "../../PointNav-VO")
sys.path.append(VO_DIR)

from pointnav_vo.vo.common.base_vo_trainer import BaseVOTrainer as VO

DIRNAME = f"test_data/egomotion_data"
VO_CONFIG = '/nethome/mkhanna38/disk/PointNav-VO/configs/vo_eval/vo_mp3d_100k_barebones_action_conditioned.yaml'

output_dir = f"data/{DIRNAME}/projections/"
traj_viz_output_dir = f"data/{DIRNAME}/trajectory_viz/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(traj_viz_output_dir, exist_ok=True)

device = torch.device("cuda")

# Settings
resolution = 0.02  # topdown resolution
default_ego_dim = (480, 640)  # egocentric resolution
z_clip = 0.50  # detections over z_clip will be ignored
vfov = 67.5
vfov = vfov * np.pi / 180.0

# -- instantiate VO module
vo = VO(VO_CONFIG)

# -- -- Load json
paths = json.load(open("data/paths.json", "r"))

info = json.load(open("data/semmap_GT_info.json", "r"))

envs_splits = json.load(open("data/envs_splits.json", "r"))
test_envs = envs_splits["test_envs"]
test_envs = [x for x in test_envs if x in paths]
test_envs.sort()


for env in tqdm(test_envs):

    # -- instantiate Habitat
    house, level = env.split("_")
    scene = "data/mp3d/{}/{}.glb".format(house, house)
    habitat = HabitatUtils(scene, int(level))

    # -- get house info
    world_dim_discret = info[env]["dim"]
    map_world_shift = info[env]["map_world_shift"]
    map_world_shift = np.array(map_world_shift)
    world_shift_origin = torch.from_numpy(map_world_shift).float().to(device=device)

    # -- instantiate projector
    projector = Projector(
        vfov,
        1,
        default_ego_dim[0],
        default_ego_dim[1],
        world_dim_discret[2],  # height
        world_dim_discret[0],  # width
        resolution,
        world_shift_origin,
        z_clip,
        device=device,
    )

    path = paths[env]

    N = len(path["positions"])

    projections_wtm = np.zeros((N, 480, 640, 2), dtype=np.uint16)
    projections_masks = np.zeros((N, 480, 640), dtype=np.bool)
    projections_heights = np.zeros((N, 480, 640), dtype=np.float32)

    ego_map_coords = []
    cur_obs = {}
    prev_obs = {}
    topdown_map = None

    with torch.no_grad():
        for n in tqdm(range(N)):

            pos = path["positions"][n]
            ori = path["orientations"][n]

            # place agent at gt positions
            habitat.position = list(pos)
            habitat.rotation = list(ori)
            habitat.set_agent_state()

            cur_obs["rgb"] = habitat.render(mode="rgb")
            depth = habitat.render(mode="depth")  # capturing depth obs at gt loc
            cur_obs["depth"] = depth

            if n == 0:
                prev_ego_global_state = (
                    habitat.get_agent_state().rotation,
                    habitat.get_agent_state().position,
                )

                topdown_map = vo.get_topdown_map_with_trajectory(
                    habitat, path["positions"]
                )

            else:
                action = (
                    path["actions"][n - 1] + 1
                )  # FWD = 1; LEFT = 2; RIGHT = 3 in VO code

                # get predicted egomotion deltas
                ego_local_deltas = vo._compute_local_delta_states_from_vo(
                    prev_obs, cur_obs, action
                )[0]
                # compute global state and 2d topdown map coords using egomotion deltas
                ego_map_coord, ego_global_state = vo.get_global_state_and_map_coords(
                    prev_ego_global_state, ego_local_deltas, topdown_map.shape, habitat
                )
                ego_map_coords.append(ego_map_coord)

                # place agent at predicted points
                ego_global_rot, ego_global_pos = ego_global_state
                habitat.position = ego_global_pos
                habitat.rotation = ego_global_rot
                habitat.set_agent_state()

                prev_ego_global_state = ego_global_state

            # getting sensor pose based on egomotion estimates (unless when n==0)
            sensor_pos = habitat.get_sensor_pos()
            sensor_ori = habitat.get_sensor_ori()

            # -- get egomotion-based T transorm
            sensor_ori = np.array(
                [sensor_ori.x, sensor_ori.y, sensor_ori.z, sensor_ori.w]
            )
            r = R.from_quat(sensor_ori)
            elevation, heading, bank = r.as_rotvec()

            xyzhe = np.array(
                [
                    [
                        sensor_pos[0],
                        sensor_pos[1],
                        sensor_pos[2],
                        heading,
                        elevation + np.pi,
                    ]
                ]
            )

            xyzhe = torch.FloatTensor(xyzhe).to(device)
            T = _transform3D(xyzhe, device=device)

            # -- using depth at gt obs for projection
            depth = depth[:, :, 0]
            depth = depth.astype(np.float32)
            depth *= 10.0
            depth_var = torch.FloatTensor(depth).unsqueeze(0).unsqueeze(0).to(device)

            # -- projection
            world_to_map, mask_outliers, heights = projector.forward(
                depth_var, T, return_heights=True
            )

            world_to_map = world_to_map[0].cpu().numpy()
            mask_outliers = mask_outliers[0].cpu().numpy()
            heights = heights[0].cpu().numpy()

            world_to_map = world_to_map.astype(np.uint16)
            mask_outliers = mask_outliers.astype(np.bool)
            heights = heights.astype(np.float32)

            projections_wtm[n, ...] = world_to_map
            projections_masks[n, ...] = mask_outliers
            projections_heights[n, ...] = heights

            prev_obs["rgb"] = cur_obs["rgb"]
            prev_obs["depth"] = cur_obs["depth"]

        topdown_map = vo.draw_egomotion_trajectory(topdown_map, ego_map_coords)

    trajectory_filename = os.path.join(traj_viz_output_dir, f"{env}.png")
    filename = os.path.join(output_dir, env + ".h5")

    cv2.imwrite(trajectory_filename, topdown_map)

    with h5py.File(filename, "w") as f:
        f.create_dataset("proj_world_to_map", data=projections_wtm, dtype=np.uint16)
        f.create_dataset("mask_outliers", data=projections_masks, dtype=np.bool)
        f.create_dataset("heights", data=projections_heights, dtype=np.float32)

    del habitat, projector
