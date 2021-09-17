import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "../"))

import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation as R

from projector import _transform3D
from projector.point_cloud import PointCloud
from semseg.rednet import RedNet
from utils import convert_weights_cuda_cpu
from utils.habitat_utils import HabitatUtils

parser = argparse.ArgumentParser(description="config")
parser.add_argument(
    "--noise",
    "-n",
    action="store_true",
    help="Whether to add noise to localisation using noisy actuations.",
)
parser.add_argument(
    "--noise-multiplier",
    "-m",
    type=float,
    default=0.5,
    help="Noise multiplier for noisy actuations.",
)
args = parser.parse_args()

if args.noise:
    DIR_NAME = f"training_noise_{args.noise_multiplier}"
else:
    DIR_NAME = "training"

output_dir = f"data/{DIR_NAME}/smnet_training_data/"
os.makedirs(output_dir, exist_ok=True)

short_tour_start_indices_json = (
    "/srv/flash1/vcartillier3/SMNet/start_index_in_video_training_data.json"
)

# Settings
resolution = 0.02  # topdown resolution
default_ego_dim = (480, 640)  # egocentric resolution
z_clip = 0.50  # detections over z_clip will be ignored
vfov = 67.5
vfov = vfov * np.pi / 180.0
features_spatial_dimensions = (480, 640)


nb_samples_per_env = 50
nb_frames_per_sample = 20

paths = json.load(open("data/paths.json", "r"))
short_tour_start_indices = json.load(open(short_tour_start_indices_json, "r"))

device = torch.device("cuda")

# -- Create model
# -- instantiate RedNet
cfg_rednet = {
    "arch": "rednet",
    "resnet_pretrained": False,
    "finetune": True,
    "SUNRGBD_pretrained_weights": "",
    "n_classes": 13,
    "upsample_prediction": True,
    "model_path": "rednet_mp3d_best_model.pkl",
}
model = RedNet(cfg_rednet)
model = model.to(device)

print("Loading pre-trained weights: ", cfg_rednet["model_path"])
state = torch.load(cfg_rednet["model_path"])
model_state = state["model_state"]
model_state = convert_weights_cuda_cpu(model_state, "cpu")
model.load_state_dict(model_state)
model = model.eval()

normalize = transforms.Compose(
    [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

depth_normalize = transforms.Normalize(mean=[0.213], std=[0.285])


# -- build projector
map_world_shift = np.zeros(3)
world_shift_origin = torch.from_numpy(map_world_shift).float().to(device=device)
projector = PointCloud(
    vfov,
    1,
    features_spatial_dimensions[0],
    features_spatial_dimensions[1],
    world_shift_origin,
    z_clip,
    device=device,
)


"""
 -->> START
"""
info = {}
for env, path in tqdm(paths.items()):

    house, level = env.split("_")

    if house == "2n8kARJN3HM":  # some issue with env.glb file on disk
        scene = f"data/mp3d_orig/{house}/{house}.glb"
    else:
        scene = "data/mp3d/{}/{}.glb".format(house, house)

    habitat = HabitatUtils(
        scene, int(level), noise=args.noise, noise_mul=args.noise_multiplier
    )

    N = len(path["positions"])

    info[env] = {}

    for m in range(nb_samples_per_env):

        # start = np.random.randint(0, high=N-nb_frames_per_sample)
        # choose fixed starting point for each short tour -- the ones that were first created
        start = short_tour_start_indices[env][str(m)]["start"]

        info[env][m] = {"start": start}

        sub_path = {}
        sub_path["positions"] = path["positions"][
            start : start + nb_frames_per_sample + 1
        ]
        sub_path["orientations"] = path["orientations"][
            start : start + nb_frames_per_sample + 1
        ]
        sub_path["actions"] = path["actions"][start : start + nb_frames_per_sample + 1]

        frames_RGB = []
        frames_depth = []
        sensor_positions = []
        sensor_rotations = []
        projection_indices = []
        masks_outliers = []

        features_encoder = []
        features_lastlayer = []
        features_scores = []

        with torch.no_grad():
            for n in tqdm(range(nb_frames_per_sample)):
                pos = sub_path["positions"][n]
                ori = sub_path["orientations"][n]

                if n == 0 or not args.noise:
                    habitat.position = list(pos)
                    habitat.rotation = list(ori)
                    habitat.set_agent_state()

                    sensor_pos = habitat.get_sensor_pos()
                    sensor_ori = habitat.get_sensor_ori()

                else:
                    action = sub_path["actions"][n - 1]
                    noisy_action = habitat.noisy_action_id_map[action]
                    habitat.step(noisy_action)

                    # get (noisy) loc corresponding to noisy actuation
                    sensor_pos = habitat.get_sensor_pos()
                    sensor_ori = habitat.get_sensor_ori()

                    # repositioning agent to GT for correct observations
                    habitat.position = list(pos)
                    habitat.rotation = list(ori)
                    habitat.set_agent_state()

                sensor_positions.append(sensor_pos)
                sensor_rotations.append(
                    [sensor_ori.x, sensor_ori.y, sensor_ori.z, sensor_ori.w]
                )

                # -- get T transorm
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

                # -- depth for projection
                depth = habitat.render(mode="depth")
                depth = depth[:, :, 0]
                depth = depth[np.newaxis, ...]
                frames_depth.append(depth)
                depth = habitat.render(mode="depth")
                depth = depth[:, :, 0]
                depth = depth.astype(np.float32)
                depth *= 10.0
                depth_var = (
                    torch.FloatTensor(depth).unsqueeze(0).unsqueeze(0).to(device)
                )

                pc, mask = projector.forward(depth_var, T)

                pc = pc.cpu().numpy()
                mask_outliers = mask.cpu().numpy()
                projection_indices.append(pc)
                masks_outliers.append(mask_outliers)

                # -- get semantic labels
                rgb = habitat.render()
                rgb = rgb[np.newaxis, ...]
                frames_RGB.append(rgb)
                rgb = habitat.render()
                rgb = rgb.astype(np.float32)
                rgb = rgb / 255.0
                rgb = torch.FloatTensor(rgb).permute(2, 0, 1)
                rgb = normalize(rgb)
                rgb = rgb.unsqueeze(0).to(device)

                depth_enc = habitat.render(mode="depth")
                depth_enc = depth_enc[:, :, 0]
                depth_enc = depth_enc.astype(np.float32)
                depth_enc = torch.FloatTensor(depth_enc).unsqueeze(0)
                depth_enc = depth_normalize(depth_enc)
                depth_enc = depth_enc.unsqueeze(0).to(device)

                semfeat_lastlayer = model(rgb, depth_enc)
                semfeat_lastlayer = semfeat_lastlayer.cpu().numpy()
                features_lastlayer.append(semfeat_lastlayer)

        frames_RGB = np.concatenate(frames_RGB, axis=0)
        frames_depth = np.concatenate(frames_depth, axis=0)
        sensor_positions = np.array(sensor_positions)
        sensor_rotations = np.array(sensor_rotations)
        masks_outliers = np.concatenate(masks_outliers, axis=0)
        projection_indices = np.concatenate(projection_indices, axis=0)

        features_lastlayer = np.concatenate(features_lastlayer, axis=0)

        filename = os.path.join(output_dir, env + "_{}.h5".format(m))
        with h5py.File(filename, "w") as f:
            f.create_dataset("rgb", data=frames_RGB, dtype=np.uint8)
            f.create_dataset("depth", data=frames_depth, dtype=np.float32)
            f.create_dataset(
                "sensor_positions", data=sensor_positions, dtype=np.float32
            )
            f.create_dataset(
                "sensor_rotations", data=sensor_rotations, dtype=np.float32
            )
            f.create_dataset(
                "projection_indices", data=projection_indices, dtype=np.float32
            )
            f.create_dataset("masks_outliers", data=masks_outliers, dtype=np.bool)
            # f.create_dataset('features_encoder', data=features_encoder, dtype=np.float32)
            f.create_dataset(
                "features_lastlayer", data=features_lastlayer, dtype=np.float32
            )
            # f.create_dataset('features_scores', data=features_scores, dtype=np.float32)

    del habitat

json.dump(info, open(f"data/{DIR_NAME}/info_training_data.json", "w"))
