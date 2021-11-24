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

DIRNAME = f"test_data/egomotion"
# VO_CONFIG = '/nethome/mkhanna38/disk/PointNav-VO/configs/vo_eval/vo_pretrained.yaml'
VO_CONFIG = '/nethome/mkhanna38/disk/PointNav-VO/configs/vo_eval/vo_mp3d_100k_barebones_action_conditioned.yaml'
# VO_CONFIG = '/nethome/mkhanna38/disk/PointNav-VO/configs/vo_eval/vo_mp3d_100k_barebones_unified.yaml'
# VO_CONFIG = '/nethome/mkhanna38/disk/PointNav-VO/configs/vo_eval/vo_mp3d_1mil_barebones_unified.yaml'

traj_viz_output_dir = f"data/{DIRNAME}/trajectory_viz/"
os.makedirs(traj_viz_output_dir, exist_ok=True)

device = torch.device("cuda")

# -- instantiate VO module
vo = VO(VO_CONFIG)

# -- -- Load json
paths = json.load(open("data/paths.json", "r"))

info = json.load(open("data/semmap_GT_info.json", "r"))

envs_splits = json.load(open("data/envs_splits.json", "r"))
test_envs = envs_splits["test_envs"]
test_envs = [x for x in test_envs if x in paths]
test_envs.sort()

rmses = []
relative_diffs_arr = []

for env in tqdm(test_envs):

    # -- instantiate Habitat
    house, level = env.split("_")
    scene = "data/mp3d/{}/{}.glb".format(house, house)
    habitat = HabitatUtils(scene, int(level))

    path = paths[env]

    N = len(path["positions"])

    ego_map_coords = []
    cur_obs = {}
    prev_obs = {}
    topdown_map = None

    delta_diffs = []
    gt_deltas_arr = []
    with torch.no_grad():
        for n in tqdm(range(N)):

            pos = path["positions"][n]
            ori = path["orientations"][n]
            # place agent at gt positions
            habitat.position = list(pos)
            habitat.rotation = list(ori)
            habitat.set_agent_state()

            cur_agent_state = habitat.get_agent_state()
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

                gt_deltas = vo.get_gt_deltas(prev_agent_state, cur_agent_state)
                # print('action:', action)
                # print('ego_deltas:', ego_local_deltas)
                
                q = R.from_quat(gt_deltas[0])
                delta_elevation, delta_heading, delta_bank = q.as_rotvec()
                delta_x, delta_y, delta_z = gt_deltas[1]
                gt_deltas = (delta_x, delta_z, delta_heading)
                # print('gt_deltas:', gt_deltas)
                
                gt_deltas = np.array(gt_deltas)
                ego_local_deltas = np.array(ego_local_deltas)

                delta_diff = (gt_deltas - ego_local_deltas) ** 2
                # print('delta_diff', delta_diff)
                # print('-------------')
                delta_diffs.append(delta_diff)
                gt_deltas_arr.append(gt_deltas)
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

            prev_obs["rgb"] = cur_obs["rgb"]
            prev_obs["depth"] = cur_obs["depth"]
            prev_agent_state = cur_agent_state

        topdown_map = vo.draw_egomotion_trajectory(topdown_map, ego_map_coords)
    delta_diffs_mean = np.mean(np.array(delta_diffs), axis=0)
    gt_deltas_mean = np.mean(np.abs(np.array(gt_deltas_arr)), axis=0)

    rmse = np.sqrt(delta_diffs_mean)
    rmses.append(rmse)

    relative_diffs = np.sqrt(np.mean(np.array(delta_diffs) / (np.array(gt_deltas_arr)**2 + 1E-8), axis=0))

    # relative_diffs = rmse / (gt_deltas_mean + 1E-8)
    relative_diffs_arr.append(relative_diffs)
    trajectory_filename = os.path.join(traj_viz_output_dir, f"{env}.png")

    cv2.imwrite(trajectory_filename, topdown_map)

    del habitat
    break

rmses = np.array(rmses)
rmses_mean = np.mean(rmses, axis=0)
relative_diffs_arr = np.array(relative_diffs_arr)
relative_diffs_arr_mean = np.mean(relative_diffs_arr, axis=0)
print('Average RMSE over all episodes', rmses_mean)
print('Rounded off to 4 places', np.round(rmses_mean, 4))
print('Average relative diffs over all episodes', relative_diffs_arr_mean)
print('Average relative diffs over all episodes(rounded off)', np.around(relative_diffs_arr_mean, 3))

