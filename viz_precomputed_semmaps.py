import cv2
import h5py
import numpy as np
import os

from utils.semantic_utils import color_label

semmaps_path = 'data/vince_training_data/smnet_training_data_semmap'
# semmaps_path = 'data/noisy_training_0.5_same_tours/smnet_training_data_semmap'
# semmaps_path = 'data/noisy_training_1.0_same_tours/smnet_training_data_semmap'

for file in sorted(os.listdir(semmaps_path))[124:]:
    file_path = os.path.join(semmaps_path, file)
    h5py_file = h5py.File(file_path, 'r')
    semmap = np.array(h5py_file['semmap'])
    semmap_color = color_label(semmap).transpose(1,2,0)
    cv2.imwrite(f'precomputed_semmap_gt_{file[0:4]}.png', semmap_color)
    break