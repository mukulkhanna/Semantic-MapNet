import os
import json
import h5py
import torch
import numpy as np

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from utils.crop_memories import crop_memories

from torch_scatter import scatter_add
from tqdm import tqdm

NAME = 'noisy_training_0.5_same_tours'
# NAME = 'noisy_training_1.0_same_tours'

semmap_dir = 'data/semmap/'
data_dir = f'data/{NAME}/smnet_training_data'

gt_data_dir = f'data/training/smnet_training_data'

sample_semmap_output_dir = f'data/{NAME}/smnet_training_data_semmap'
sample_indices_output_dir = f'data/{NAME}/smnet_training_data_indices'

os.makedirs(sample_indices_output_dir, exist_ok=True)
os.makedirs(sample_semmap_output_dir, exist_ok=True)

semmap_info = json.load(open('data/semmap_GT_info.json', 'r'))

#Settings
resolution = 0.02 # topdown resolution
default_ego_dim = (480, 640) #egocentric resolution
z_clip = 0.50 # detections over z_clip will be ignored
vfov = 67.5
vfov = vfov * np.pi / 180.0


files = os.listdir(data_dir)
gt_files = os.listdir(gt_data_dir)

info = {}
semantic_maps = np.zeros((len(files), 250, 250), dtype=np.int32)
instance_maps = np.zeros((len(files), 250, 250), dtype=np.int32)
observed_masks = np.zeros((len(files), 250, 250), dtype=np.bool)
semantic_maps_env_names = []
for n, file in tqdm(enumerate(files), total=len(files)):

    house, level, _ = file.split('_')
    env = '_'.join((house, level))


    map_world_shift = np.array(semmap_info[env]['map_world_shift'])
    world_discret_dim = np.array(semmap_info[env]['dim'])
    map_width = world_discret_dim[0]
    map_height = world_discret_dim[2]

    h5file = h5py.File(os.path.join(semmap_dir, env + '.h5'), 'r')
    semmap = np.array(h5file['map_semantic'], dtype=np.int)
    insmap = np.array(h5file['map_instance'], dtype=np.int)
    h5file.close()

    h5file = h5py.File(os.path.join(data_dir, file), 'r')
    rgb = np.array(h5file['rgb'], dtype=np.uint8)
    projection_indices = np.array(h5file['projection_indices'], dtype=np.float32)
    masks_outliers = np.array(h5file['masks_outliers'], dtype=np.bool)
    sensor_positions = np.array(h5file['sensor_positions'], dtype=np.float32)
    h5file.close()

    h5file = h5py.File(os.path.join(gt_data_dir, file), 'r')
    gt_projection_indices = np.array(h5file['projection_indices'], dtype=np.float32)
    gt_masks_outliers = np.array(h5file['masks_outliers'], dtype=np.bool)
    gt_sensor_positions = np.array(h5file['sensor_positions'], dtype=np.float32)
    h5file.close()

    # transfer to Pytorch
    projection_indices = torch.FloatTensor(projection_indices)
    masks_outliers = torch.BoolTensor(masks_outliers)
    sensor_positions = torch.FloatTensor(sensor_positions)

    gt_projection_indices = torch.FloatTensor(gt_projection_indices)
    gt_masks_outliers = torch.BoolTensor(gt_masks_outliers)
    gt_sensor_positions = torch.FloatTensor(gt_sensor_positions)

    map_world_shift = torch.FloatTensor(map_world_shift)

    projection_indices -= map_world_shift
    gt_projection_indices -= map_world_shift

    pixels_in_map = (projection_indices[:,:,:, [0,2]] / resolution).round().long()
    gt_pixels_in_map = (gt_projection_indices[:,:,:, [0,2]] / resolution).round().long()

    outside_map_indices = (pixels_in_map[:, :, :, 0] >= map_width) +\
                          (pixels_in_map[:, :, :, 1] >= map_height) +\
                          (pixels_in_map[:, :, :, 0] < 0) +\
                          (pixels_in_map[:, :, :, 1] < 0)

    gt_outside_map_indices = (gt_pixels_in_map[:, :, :, 0] >= map_width) +\
                          (gt_pixels_in_map[:, :, :, 1] >= map_height) +\
                          (gt_pixels_in_map[:, :, :, 0] < 0) +\
                          (gt_pixels_in_map[:, :, :, 1] < 0)

    # shape: camera_z (batch_size, features_height, features_width)
    camera_y = sensor_positions[:,1]
    camera_y = camera_y.unsqueeze(-1).unsqueeze(-1).repeat(1, pixels_in_map.shape[1], pixels_in_map.shape[2])
    above_threshold_z_indices = projection_indices[:,:,:,1] > camera_y + z_clip
    
    gt_camera_y = gt_sensor_positions[:,1]
    gt_camera_y = gt_camera_y.unsqueeze(-1).unsqueeze(-1).repeat(1, gt_pixels_in_map.shape[1], gt_pixels_in_map.shape[2])
    gt_above_threshold_z_indices = gt_projection_indices[:,:,:,1] > gt_camera_y + z_clip

    masks_outliers = masks_outliers + outside_map_indices + above_threshold_z_indices
    gt_masks_outliers = gt_masks_outliers + gt_outside_map_indices + gt_above_threshold_z_indices

    masks_inliers = ~masks_outliers
    gt_masks_inliers = ~gt_masks_outliers

    flat_pixels_in_map = pixels_in_map[masks_inliers]
    gt_flat_pixels_in_map = gt_pixels_in_map[gt_masks_inliers]

    flat_indices = map_width * flat_pixels_in_map[:,1] + flat_pixels_in_map[:,0]
    flat_indices = flat_indices.long()
    
    gt_flat_indices = map_width * gt_flat_pixels_in_map[:,1] + gt_flat_pixels_in_map[:,0]
    gt_flat_indices = gt_flat_indices.long()
    
    ones = torch.ones(flat_indices.shape)
    gt_ones = torch.ones(gt_flat_indices.shape)

    flat_map = torch.zeros((map_width * map_height))
    gt_flat_map = torch.zeros((map_width * map_height))

    flat_map = scatter_add(
        ones,
        flat_indices,
        dim=0,
        out=flat_map,
    )
    gt_flat_map = scatter_add(
        gt_ones,
        gt_flat_indices,
        dim=0,
        out=gt_flat_map,
    )

    map = flat_map.reshape(map_height, map_width)
    mask = map > 0
    
    gt_map = gt_flat_map.reshape(map_height, map_width)
    gt_mask = gt_map > 0

    mask = mask.cpu().numpy()
    gt_mask = gt_mask.cpu().numpy()
    
    if mask.any():
        mask_observe, dim = crop_memories(mask, (250,250))
    else:
        print(file)
        continue

    if gt_mask.any():
        gt_mask_observe, gt_dim = crop_memories(gt_mask, (250,250))
    else:
        print(file)
        continue

    min_y, max_y, min_x, max_x = dim
    gt_min_y, gt_max_y, gt_min_x, gt_max_x = gt_dim

    print(dim)
    print(gt_dim)

    outside_sample_map_indices = (pixels_in_map[:, :, :, 0] >= max_x+1) +\
                                 (pixels_in_map[:, :, :, 1] >= max_y+1) +\
                                 (pixels_in_map[:, :, :, 0] < min_x) +\
                                 (pixels_in_map[:, :, :, 1] < min_y)

    mask_outliers_sample = masks_outliers + outside_sample_map_indices

    pixels_in_map[:,:,:,0] -= min_x
    pixels_in_map[:,:,:,1] -= min_y


    sample_semmap = semmap[min_y:max_y+1, min_x:max_x+1]
    sample_insmap = insmap[min_y:max_y+1, min_x:max_x+1]

    # sample_semmap = semmap[gt_min_y:gt_max_y+1, gt_min_x:gt_max_x+1]
    # sample_insmap = insmap[gt_min_y:gt_max_y+1, gt_min_x:gt_max_x+1]

    filename = os.path.join(sample_semmap_output_dir, file)
    with h5py.File(filename, 'w') as f:
        f.create_dataset('mask', data=mask_observe, dtype=np.bool)
        # f.create_dataset('mask', data=gt_mask_observe, dtype=np.bool)
        f.create_dataset('semmap', data=sample_semmap, dtype=np.int32)
        f.create_dataset('insmap', data=sample_insmap, dtype=np.int32)

    filename = os.path.join(sample_indices_output_dir, file)
    with h5py.File(filename, 'w') as f:
        f.create_dataset('masks_outliers', data=mask_outliers_sample, dtype=np.bool)
        f.create_dataset('indices', data=pixels_in_map, dtype=np.int32)


    info[file]={'dim': [min_y, max_y, min_x, max_x]}

    semantic_maps_env_names.append(file)
    semantic_maps[n,:,:] = sample_semmap
    instance_maps[n,:,:] = sample_insmap
    observed_masks[n,:,:] = mask_observe


json.dump(info, open(f'data/{NAME}/info_training_data_crops.json', 'w'))

json.dump(semantic_maps_env_names,
          open(f'data/{NAME}/smnet_training_data_semmap.json', 'w'))

with h5py.File(f'data/{NAME}/smnet_training_data_semmap.h5', 'w') as f:
    f.create_dataset('semantic_maps', data=semantic_maps, dtype=np.int32)
    f.create_dataset('instance_maps', data=instance_maps, dtype=np.int32)
    f.create_dataset('observed_masks', data=observed_masks, dtype=np.bool)



