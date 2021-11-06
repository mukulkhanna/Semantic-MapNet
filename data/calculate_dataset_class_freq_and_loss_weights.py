"""
Script that calculates class-frequency across dataset for calculating loss weighting during training.
"""

import json
import os

import cv2
import h5py
import numpy as np
from tqdm import tqdm

root_dir = "data/training"
# root_dir = '../SMNet-new/data/vince_training_data'
# root_dir = 'data/training_noise_0.5'
# root_dir = 'data/training_noise_1.0'

all_data_path = f"{root_dir}/smnet_training_data"
train_semmaps_path = f"{root_dir}/smnet_training_data_semmap.h5"

envs_splits = json.load(open("data/envs_splits.json", "r"))

files = os.listdir(all_data_path)
train_files = [
    x
    for x in files
    if "_".join(x.split("_")[:2]) in envs_splits["{}_envs".format("train")]
]
train_envs = [x.split(".")[0] for x in train_files]

semmap_envs = json.load(open(f"{root_dir}/smnet_training_data_semmap.json", "r"))
semmap_indx = {
    i: semmap_envs.index(train_envs[i] + ".h5") for i in range(len(train_files))
}

h5py_file = h5py.File(train_semmaps_path, "r")
semmaps = np.array(h5py_file["semantic_maps"])
obs_masks = np.array(h5py_file["observed_masks"])
class_freq = np.zeros(13)

for i in tqdm(range(len(semmaps))):
    if i in semmap_indx:
        semmap = semmaps[i, :, :]
        obs_mask = obs_masks[i, :, :]
        unique, counts = np.unique(semmap[obs_mask == True], return_counts=True)
        for uniq, count in zip(unique, counts):
            class_freq[uniq] += count

norm_class_freq = class_freq / class_freq.sum()
print("Percentage frequency distribution: ", np.around(norm_class_freq * 100, 2))
weights = 1 / norm_class_freq
# print(weights)
norm_weights = weights / weights.sum()
print("Weights to be assigned to each class's loss:", np.around(norm_weights, 4))


# from utils.semantic_utils import color_label
# semmap[obs_mask == False] = -1
# semmap_color = color_label(semmap).transpose(1,2,0)
# cv2.imwrite(f'semmap_color_{i}.png', semmap_color)
# break
