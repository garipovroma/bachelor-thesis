import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import torch
from joblib import dump, load
import os

print(f'{os.getcwd()}')

DIR_PATH = 'basic_metaclassifiers'

files_in_folder = ['lm_gan_discriminator_model_0.joblib', 'lm_gan_discriminator_model_1.joblib', 'lm_gan_discriminator_model_2.joblib']

files = [os.path.join(DIR_PATH, file) for file in files_in_folder]

print(files)

models = [load(file) for file in files]

for model, file in zip(models, files):
    print(f'{file}: {model}')
    torch_model_file = file.replace('.joblib', '.pkl')
    torch.save(model, torch_model_file)