"""
Script for visualising semmap predictions saved as h5 files after testing.
"""
import os

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from utils.semantic_utils import color_label

DIR = [
    # 'data/test_outputs/noise_0.5_ngs'
    # 'data/test_outputs/noise_0.5_ngs_weighted',
    "data/test_outputs/egomotion_data",
    "data/test_outputs/egomotion_data_unified_1mil_e25",
    "data/test_outputs/egomotion_data_unified_100k",
    # 'data/test_outputs/noise_1.0',
]

FILES = [
    "2t7WUuJeko7_0.h5",
    "5ZKStnWn8Zo_0.h5",
    "5ZKStnWn8Zo_1.h5",
    "ARNzJeq3xxb_0.h5",
    "RPmz2sHmrrY_0.h5",
]

os.makedirs("data/viz_semmaps/output_viz/obs_map", exist_ok=True)
os.makedirs("data/viz_semmaps/output_viz/semmap", exist_ok=True)

for odir in tqdm(DIR):
    for _file in tqdm(FILES):
        file_path = os.path.join(odir, _file)
        file = h5py.File(file_path, "r")
        exp_name = odir.split("/")[-1]

        obs_map = np.array(file["observed_map"])
        obs_map_filename = os.path.join(
            "data/viz_semmaps/output_viz",
            "obs_map",
            f"{exp_name}_{_file[0:4]}{_file[-7:-3]}.png",
        )
        cv2.imwrite(obs_map_filename, obs_map * 255)

        semmap = np.array(file["semmap"])
        semmap[~obs_map] = 0  # mask away the outliers
        semmap_color = color_label(semmap)
        semmap_color = semmap_color.transpose(1, 2, 0)
        semmap_color = semmap_color.astype(np.uint8)

        semmap_filename = os.path.join(
            "data/viz_semmaps/output_viz",
            "semmap",
            f"{exp_name}_{_file[0:4]}{_file[-7:-3]}.png",
        )
        cv2.imwrite(semmap_filename, semmap_color)

        print("\nSemmap saved at", semmap_filename)
        file.close()
