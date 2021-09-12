import cv2
import h5py
import numpy as np
import os

from utils.semantic_utils import color_label

# DIR = 'data/be_no_mask_b4_loss/semmap'
# DIR = 'data/vince_best_outputs/semmap'
# DIR = 'data/my_gt_outputs_x/semmap'
DIR = 'data/semmap'
# DIR = 'data/noisy_0.5_outputs/semmap'
# DIR = 'data/noisy_1.0_outputs/semmap'
# DIR = 'data/noisy_on_gt_outputs/semmap'
# DIR = 'data/gt_on_noisy_outputs/semmap'
FILES = ['2t7WUuJeko7_0.h5', '5ZKStnWn8Zo_0.h5']
# FILES = ['5ZKStnWn8Zo_0.h5']

for _file in FILES:
    file_path = os.path.join(DIR, _file)
    file = h5py.File(file_path, 'r')
    print(file.keys())
    '''
    obs_map = np.array(file['observed_map'])
    cv2.imwrite(f'obs_map_gt_{_file[0:4]}.png', obs_map* 255)
    '''
    # cv2.imwrite(f'obs_map_best_n_0.5_{_file[0:4]}.png', obs_map* 255)
    # cv2.imwrite('obs_map_gt_on_noisy.png', obs_map* 255)
    # cv2.imwrite(f'obs_map_vince_best_gt_{_file[0:4]}.png', obs_map* 255)
    # cv2.imwrite(f'obs_map_be_no_mask_b4_loss_{_file[0:4]}.png', obs_map* 255)

    semmap = np.array(file['map_semantic'])
    # semmap[obs_map == 0] = 0 
    semmap_color = color_label(semmap)
    semmap_color = semmap_color.transpose(1,2,0)
    semmap_color = semmap_color.astype(np.uint8)

    cv2.imwrite(f'semmap_gt_{_file[0:4]}.png', semmap_color)
    # cv2.imwrite(f'semmap_best_n_0.5_{_file[0:4]}.png', semmap_color)
    # cv2.imwrite('semmap_gt_on_noisy.png', semmap_color)
    # cv2.imwrite(f'semmap_vince_best_gt_{_file[0:4]}.png', semmap_color)
    # cv2.imwrite(f'semmap_best_be_no_mask_b4_loss_{_file[0:4]}.png', semmap_color)
    file.close()

