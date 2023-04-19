import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

import torch
import os

from custom_metrics import torch_acc, torch_acc_numerator

from train_basic_metaclassifiers import EXP_NUM, model_names, models

from joblib import dump, load

from DatasetLoader import get_deepset_loader

from Models import DeepSetModelV5, CONFIG
import copy

# calculate accuracy for all models in basic_metaclassifiers

# read data
def read_data(folder_path, meta_path, lambda_path):
    full_meta_path = os.path.join(folder_path, meta_path)
    full_lambda_path = os.path.join(folder_path, lambda_path)
    data = np.load(full_meta_path, allow_pickle=True).reshape(-1, 27)
    target = np.load(full_lambda_path, allow_pickle=True)
    return data, target

def load_basic_models_weights(load_from_torch=None):
    models_dict = {model_name: [] for model_name in model_names}
    for model_id in range(len(model_names)):
        loaded_models_count = 0
        model_name = model_names[model_id]
        for exp_id in range(EXP_NUM):
            model_path = os.path.join('basic_metaclassifiers', f'{model_name}_model_{exp_id}.joblib')
            if load_from_torch is None or (load_from_torch is not None and load_from_torch[model_id] is not None):
                try:
                    model = copy.deepcopy(load_from_torch[model_id])
                    model_path = os.path.join('basic_metaclassifiers', f'{model_name}_model_{exp_id}.pkl')
                    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                    model.eval()
                except Exception as e:
                    print(f'Failed to load model {model_name} with exp_id {exp_id}, error: {e}')
                    break
            else:
                try:
                    model = load(model_path)
                except Exception as e:
                    print(f'Failed to load model {model_name} with exp_id {exp_id}, error: {e}')
                    break
            models_dict[model_name].append(model)
            loaded_models_count += 1
        print(f'Loaded {loaded_models_count} models for {model_name}')
    return models_dict


# calculate accuracy

def calculate_accuracy(model, data, target):
    data = torch.from_numpy(data).float()
    target = torch.from_numpy(target).float()
    y_pred = torch.from_numpy(model.predict(data)).float()
    acc = torch_acc(y_pred, target)
    return acc


def calculate_accuracy_with_dataloader(model, dataloader, transform_pred=None):
    acc = 0
    for X_p, X_n, meta, target in dataloader:
        meta = meta.float()
        target = target.float()
        y_pred = model(X_p, X_n, meta)
        if transform_pred:
            y_pred = transform_pred(y_pred)
        acc += torch_acc_numerator(y_pred, target)
    acc = acc / len(dataloader.dataset)
    return acc


def calculate_accuracy_for_all_models(models_dict, data, target, dataloader_for_model):
    acc_dict = {model_name: [] for model_name in model_names}
    for model_name, dataloader in zip(model_names, dataloader_for_model):
        for model in models_dict[model_name]:
            if dataloader is None:
                acc = calculate_accuracy(model, data, target)
            else:
                transform_pred = None
                if model_name == 'deepset':
                    transform_pred = lambda x: torch.sigmoid(x)
                acc = calculate_accuracy_with_dataloader(model, dataloader, transform_pred)
            acc_dict[model_name].append(acc)
    return acc_dict


train_data, train_target = read_data(folder_path='precalculated_data',
                                     meta_path='train_meta_tensors.npy',
                                     lambda_path='train_lambda_lambda_tensors.npy')

test_data, test_target = read_data(folder_path='precalculated_data',
                                   meta_path='test_meta_tensors.npy',
                                   lambda_path='test_lambda_lambda_tensors.npy')

model_names.append('deepset')
# model_names.append('lm_gan_discriminator')

load_from_torch = [None, None, None, None, None,
                   DeepSetModelV5(hidden_size_0=CONFIG.hidden_size_0,
                                  hidden_size_1=CONFIG.hidden_size_1,
                                  predlast_hidden_size=CONFIG.predlast_hidden_size,
                                  meta_size=CONFIG.meta_size,
                                  out_classes=CONFIG.out_classes),
                   False]

models_dict = load_basic_models_weights(load_from_torch)

print(models_dict)

# model_names = ['DecisionTreeClassifier', 'KNeighborsClassifier', 'SVC', 'LogisticRegression', 'SimpleMetaNeuralNetwork']
dataloader_for_model = [None, None, None, None, None, get_deepset_loader(), None]

accuracy = calculate_accuracy_for_all_models(models_dict, test_data, test_target, dataloader_for_model)

accuracy_params = {model_name: (np.mean(accuracy[model_name]), np.std(accuracy[model_name])) for model_name in
                   model_names}
accuracy_with_borders = {
    model_name: f'{accuracy_params[model_name][0]} ± {2 * accuracy_params[model_name][1] / np.sqrt(EXP_NUM)}' for
    model_name in model_names}

print(f'accuracy: {accuracy}')
print(f'accuracy_params: {accuracy_params}')
print(f'accuracy_with_borders: {accuracy_with_borders}')
