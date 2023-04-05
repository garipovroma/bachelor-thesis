# create list of files in the directory ../processed_data/processed_16_64_2/

import os
import numpy as np
import stabilization.NumpyRawToData as nrt
import torch

train_files_path = os.listdir('../processed_data/processed_16_64_2/')
train_files = [file for file in train_files_path if file.endswith('.npy')]

test_files_path = os.listdir('../processed_data/test/')
test_files = [file for file in test_files_path if file.endswith('.npy')]

def stabilize(data):
    dataset_tensor = data.reshape(128, 16)
    target_column = np.array([i < 64 for i in range(128)]).reshape(128, 1)
    dataset_tensor = np.concatenate([dataset_tensor, target_column], axis=1)
    x_0, x_1 = nrt.correlation_method(dataset_tensor)
    res = torch.stack([torch.Tensor(x_0), torch.tensor(x_1)], dim=0).numpy()

    return res

# stabilize all train files with tqdm progress bar with desc = 'Stabilizing train files'

from tqdm import tqdm

for file in tqdm(train_files, desc='Stabilizing train files'):
    data = np.load(f'../processed_data/processed_16_64_2/{file}')
    stabilized_data = stabilize(data)
    np.save(f'../processed_data/corr_stabilized_train/{file}', stabilized_data)

# stabilize all test files with tqdm progress bar with desc = 'Stabilizing test files'

for file in tqdm(test_files, desc='Stabilizing test files'):
    data = np.load(f'../processed_data/test/{file}')
    stabilized_data = stabilize(data)
    np.save(f'../processed_data/corr_stabilized_test/{file}', stabilized_data)