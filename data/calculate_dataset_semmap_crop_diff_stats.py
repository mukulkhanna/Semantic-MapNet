"""
Script for calculating stats related to difference between short-tour semantic
map crops generated from GT vs noisy localisation.
"""
import json

import numpy as np

envs_splits = json.load(open("data/envs_splits.json", "r"))

gt_crops_json_file = "data/training/info_training_data_crops.json"
n5_crops_json_file = "data/training_noise_0.5/info_training_data_crops.json"
n1_crops_json_file = "data/training_noise_1.0/info_training_data_crops.json"

gt_crops_json = json.load(open(gt_crops_json_file, "r"))
n5_crops_json = json.load(open(n5_crops_json_file, "r"))
n1_crops_json = json.load(open(n1_crops_json_file, "r"))

n5_diffs_train, n5_diffs_test, n5_diffs_val = [], [], []
n1_diffs_train, n1_diffs_test, n1_diffs_val = [], [], []

for tour in gt_crops_json.keys():
    gt_dim = gt_crops_json[tour]["dim"]
    gt_min_y, gt_max_y, gt_min_x, gt_max_x = gt_dim

    n5_dim = n5_crops_json[tour]["dim"]
    n5_min_y, n5_max_y, n5_min_x, n5_max_x = n5_dim

    n1_dim = n1_crops_json[tour]["dim"]
    n1_min_y, n1_max_y, n1_min_x, n1_max_x = n1_dim

    gt_center_x = (gt_min_x + gt_max_x) / 2
    gt_center_y = (gt_min_y + gt_max_y) / 2

    n5_center_x = (n5_min_x + n5_max_x) / 2
    n5_center_y = (n5_min_y + n5_max_y) / 2

    n1_center_x = (n1_min_x + n1_max_x) / 2
    n1_center_y = (n1_min_y + n1_max_y) / 2

    n5_diff_x = abs(gt_center_x - n5_center_x)
    n5_diff_y = abs(gt_center_y - n5_center_y)

    n1_diff_x = abs(gt_center_x - n1_center_x)
    n1_diff_y = abs(gt_center_y - n1_center_y)

    n5_diff = (n5_diff_x ** 2 + n5_diff_y ** 2) ** 0.5
    n1_diff = (n1_diff_x ** 2 + n1_diff_y ** 2) ** 0.5

    if tour[:13] in envs_splits["train_envs"]:
        n5_diffs_train.append(n5_diff)
        n1_diffs_train.append(n1_diff)
    elif tour[:13] in envs_splits["test_envs"]:
        n5_diffs_test.append(n5_diff)
        n1_diffs_test.append(n1_diff)
    elif tour[:13] in envs_splits["val_envs"]:
        n5_diffs_val.append(n5_diff)
        n1_diffs_val.append(n1_diff)


n5_diffs_train = np.array(n5_diffs_train)
n1_diffs_train = np.array(n1_diffs_train)

n5_diffs_test = np.array(n5_diffs_test)
n1_diffs_test = np.array(n1_diffs_test)

n5_diffs_val = np.array(n5_diffs_val)
n1_diffs_val = np.array(n1_diffs_val)

print("Train")
print("Mean difference (noise 0.5):", np.mean(n5_diffs_train))
print("Mean difference (noise 1.0):", np.mean(n1_diffs_train))
print("Median difference (noise 0.5):", np.median(n5_diffs_train))
print("Median difference (noise 1.0):", np.median(n1_diffs_train))

print("---------------------------")

print("Test")
print("Mean difference (noise 0.5):", np.mean(n5_diffs_test))
print("Mean difference (noise 1.0):", np.mean(n1_diffs_test))
print("Median difference (noise 0.5):", np.median(n5_diffs_test))
print("Median difference (noise 1.0):", np.median(n1_diffs_test))

print("---------------------------")

print("Val")
print("Mean difference (noise 0.5):", np.mean(n5_diffs_val))
print("Mean difference (noise 1.0):", np.mean(n1_diffs_val))
print("Median difference (noise 0.5):", np.median(n5_diffs_val))
print("Median difference (noise 1.0):", np.median(n1_diffs_val))
