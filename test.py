import os
import json
import h5py
import torch
import numpy as np

from tqdm import tqdm

from SMNet.model_test import SMNet

from utils import convert_weights_cuda_cpu
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
split = 'test'

data_dir = 'data/test_data/' # gt test data
# data_dir = 'data/test_data_n_0.5/'
# data_dir = 'data/test_vince_data/'
# data_dir = 'data/test_data_n_1.0/'

# output_dir = 'data/be_yes_mask_b4_loss/semmap/'
# output_dir = 'data/be_no_mask_b4_loss/semmap/'
# output_dir = 'data/my_gt_outputs/semmap/'
output_dir = 'data/vince_best_outputs_check/semmap/'
# output_dir = 'data/noisy_on_gt_outputs/semmap/'
# output_dir = 'data/gt_on_noisy_outputs/semmap/'
# output_dir = 'data/noisy_0.5_outputs/semmap/'
# output_dir = 'data/vince_data_outputs/semmap/'
# output_dir = 'data/noisy_1.0_outputs/semmap/'
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -- create model
cfg_model = {
    'arch': 'smnet',
    'finetune': False,
    'n_obj_classes': 13,
    'ego_feature_dim': 64,
    'mem_feature_dim': 256,
    'mem_update': 'gru',
    'ego_downsample': False,
}
model_path = 'smnet_mp3d_best_model.pkl'
# model_path = 'runs/gru_fullrez_lastlayer_m256/93288/smnet_mp3d_best_model.pkl' # my gt loc model
# model_path = 'runs/gru_fullrez_lastlayer_m256/yes-mask-b4-loss/smnet_mp3d_best_model.pkl' # my gt loc model
# model_path = 'runs/gru_fullrez_lastlayer_m256/smnet-gt/smnet_mp3d_best_model.pkl' # noisy loc model
# model_path = 'runs/gru_fullrez_lastlayer_m256/smnet-noisy-0.5/smnet_mp3d_best_model.pkl' # noisy loc model
# model_path = 'runs/gru_fullrez_lastlayer_m256/smnet-noisy-1.0/smnet_mp3d_best_model.pkl'
model = SMNet(cfg_model, device)
model = model.to(device)

print('Loading pre-trained weights: ', model_path)
state = torch.load(model_path)
model_state = state['model_state']
model_state = convert_weights_cuda_cpu(model_state, 'cpu')
model.load_state_dict(model_state)
model.eval()


# -- load JSONS and select envs
info = json.load(open('data/semmap_GT_info.json','r'))
paths = json.load(open('data/paths.json', 'r'))
envs_splits = json.load(open('data/envs_splits.json', 'r'))
envs = envs_splits['{}_envs'.format(split)]
envs = [x for x in envs if x in paths]
envs.sort()



with torch.no_grad():
    for env in tqdm(envs):

        if os.path.isfile(os.path.join(output_dir, env+'.h5')): continue

        # get env dim
        world_dim_discret = info[env]['dim']
        map_height = world_dim_discret[2]
        map_width  = world_dim_discret[0]

        # load DATA
        h5file = h5py.File(os.path.join(data_dir, 'projections', env+'.h5'), 'r')
        projections_wtm = np.array(h5file['proj_world_to_map'], dtype=np.uint16)
        mask_outliers = np.array(h5file['mask_outliers'], dtype=np.bool)
        heights = np.array(h5file['heights'], dtype=np.float32)
        h5file.close()

        h5file = h5py.File(os.path.join(data_dir, 'features', env+'.h5'), 'r')
        features = np.array(h5file['features_lastlayer'], dtype=np.float32)
        h5file.close()

        features = torch.from_numpy(features)

        projections_wtm = projections_wtm.astype(np.int32)
        projections_wtm = torch.from_numpy(projections_wtm)
        mask_outliers = torch.from_numpy(mask_outliers)
        heights = torch.from_numpy(heights)

        scores, observed_map, height_map = model(features,
                                                 projections_wtm,
                                                 mask_outliers,
                                                 heights,
                                                 map_height,
                                                 map_width)

        semmap = scores.data.max(0)[1]
        semmap = semmap.cpu().numpy()
        
        # values, counts = np.unique(semmap, return_counts=True)

        # ind = np.argmax(counts)
        # print('--------------')
        # print(values[ind])
        # print('--------------')

        semmap = semmap.astype(np.uint8)
        scores = scores.cpu().numpy()
        observed_map = observed_map.cpu().numpy()

        # semmap [~observed_map] = 0
        height_map = height_map.cpu().numpy()


        filename = os.path.join(output_dir, env+'.h5')
        with h5py.File(filename, 'w') as f:
            f.create_dataset('semmap', data=semmap, dtype=np.uint8)
            f.create_dataset('scores', data=scores, dtype=np.float32)
            f.create_dataset('observed_map', data=observed_map, dtype=np.bool)
            f.create_dataset('height_map', data=height_map, dtype=np.float32)

