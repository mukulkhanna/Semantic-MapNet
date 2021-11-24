import cv2
import h5py
import numpy as np
import os

from utils.semantic_utils import color_label

DIR = 'data/training/smnet_training_data_semmap/'

FILES = ['1LXtFkjw3qL_0_20.h5', '1LXtFkjw3qL_0_15.h5']
# FILES = ['2t7WUuJeko7_0.h5', '5ZKStnWn8Zo_0.h5']
# FILES = ['5ZKStnWn8Zo_0.h5']

for _file in FILES:
    file_path = os.path.join(DIR, _file)
    file = h5py.File(file_path, 'r')

    obs_map = np.array(file['mask'])

    # cv2.imwrite(f'short_obs_map_gt_loc_{_file[0:4] + _file[-7:-3]}.png', obs_map * 255)
    cv2.imwrite(f'obs_map_gt_loc_{_file[0:4] + _file[-7:-3]}.png', obs_map * 255)
    # cv2.imwrite(f'obs_map_n_0.5_{_file[0:4]}.png', obs_map * 255)

    semmap = np.array(file['semmap'])
    # semmap[~obs_map] = 0 
    semmap_color = color_label(semmap)
    semmap_color = semmap_color.transpose(1,2,0)
    semmap_color = semmap_color.astype(np.uint8)

    # cv2.imwrite(f'semmap_n_0.5_{_file[0:4]}.png', semmap_color)
    # cv2.imwrite(f'semmap_gt_loc_{_file[0:4]}.png', semmap_color)
    cv2.imwrite(f'semmap_gt_loc_{_file[0:4]  + _file[-7:-3]}.png', semmap_color)
    file.close()

