import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import torch

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from SimpleMetaNeuralNetwork import SimpleMetaNeuralNetwork
from sklearn.multioutput import MultiOutputClassifier

from sklearn.preprocessing import StandardScaler

from joblib import dump, load
import os
from tqdm import tqdm

# models: sklearn decision tree, knn, svm, logistic regression, torch simple meta neural network

RANDOM_STATE = 42

models = [DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5),
          KNeighborsClassifier(),
          MultiOutputClassifier(SVC(random_state=RANDOM_STATE)),
          MultiOutputClassifier(LogisticRegression(random_state=RANDOM_STATE)),
          SimpleMetaNeuralNetwork()]

model_names = ['DecisionTreeClassifier', 'KNeighborsClassifier', 'SVC', 'LogisticRegression', 'SimpleMetaNeuralNetwork']

model_preprocessing = ['standardscale', 'standardscale', 'standardscale', 'standardscale', 'standardscale']

# read data
def read_data(folder_path, meta_path, lambda_path):
    full_meta_path = os.path.join(folder_path, meta_path)
    full_lambda_path = os.path.join(folder_path, lambda_path)
    data = np.load(full_meta_path, allow_pickle=True).reshape(-1, 27)
    target = np.load(full_lambda_path, allow_pickle=True)
    return data, target


train_data, train_target = read_data(folder_path='precalculated_data',
                                     meta_path='train_meta_tensors.npy',
                                     lambda_path='train_lambda_lambda_tensors.npy')

test_data, test_target = read_data(folder_path='precalculated_data',
                                   meta_path='test_meta_tensors.npy',
                                   lambda_path='test_lambda_lambda_tensors.npy')

train_data_standard_scaler = StandardScaler()
train_data_standard_scaler.fit(train_data)

dump(train_data_standard_scaler, 'precalculated_data/train_data_standard_scaler.joblib')

EXP_NUM = 5

with tqdm(total=EXP_NUM * len(models)) as pbar:
    for exp_id in range(EXP_NUM):
        for model_id in range(len(models)):
            model = models[model_id]
            preprocessing = model_preprocessing[model_id]
            processed_train_data = train_data.copy()
            if preprocessing == 'standardscale':
                processed_train_data = train_data_standard_scaler.transform(train_data)
            model.fit(processed_train_data, train_target)
            dump(model, f'basic_metaclassifiers/{model_names[model_id]}_model_{exp_id}.joblib')
            pbar.update(1)

        RANDOM_STATE = RANDOM_STATE * 3 // 2
        models = [DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5),
                  KNeighborsClassifier(),
                  MultiOutputClassifier(SVC(random_state=RANDOM_STATE)),
                  MultiOutputClassifier(LogisticRegression(random_state=RANDOM_STATE)),
                  SimpleMetaNeuralNetwork()]