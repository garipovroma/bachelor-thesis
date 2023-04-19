import os
from datetime import datetime
from pathlib import Path

import argparse

import torch
from scipy.spatial.distance import mahalanobis
from torch import optim
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, MSELoss
import numpy as np

from DatasetLoader import get_loader
from Models import Generator, Discriminator
from feature_extraction.LambdaFeaturesCollector import LambdaFeaturesCollector
from feature_extraction.MetaFeaturesCollector import MetaFeaturesCollector
import logging

import joblib

from feature_extraction.MetaZerosCollector import MetaZerosCollector

logging.basicConfig(format='%(asctime)s %(message)s', filename='2201_LM.log', level=logging.DEBUG,
                    datefmt='%d-%m %H:%M:%S')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Trainer:
    def __init__(self, num_epochs: int = 21, device: str = "mps", continue_from: int = 0, data_prefix='', exp_prefix=''):
        self.features = 16
        self.instances = 64
        self.classes = 2
        self.z_size = 100
        self.batch_size = 100
        self.workers = 5
        self.num_epochs = num_epochs
        self.device = device
        self.log_step = 10
        self.log_step_print = 50
        self.save_period = 5
        self.continue_from = continue_from
        self.data_prefix = data_prefix
        self.exp_prefix = exp_prefix

        self.models_path = "./models2201_d"

        self.lambdas = LambdaFeaturesCollector(self.features, self.instances)
        self.metas = MetaFeaturesCollector(self.features, self.instances)
        self.data_loader, train_dataset = get_loader(
            "../processed_data/corr_stabilized_train/",
            # f"../processed_data/{data_prefix}processed_{self.features}_{self.instances}_{self.classes}/",
            self.features, self.instances, self.classes, self.metas,
            self.lambdas, self.batch_size,
            self.workers,
            True,
            'precalculated_data/stabilized_corr_train_meta_tensors.npy',
            'precalculated_data/stabilized_corr_train_lambda_tensors.npy', None, None, stabilize=None)

        self.test_loader, _ = get_loader(
                    # f"../processed_data/{data_prefix}test/",
            "../processed_data/corr_stabilized_test/",
             self.features, self.instances, self.classes, self.metas, self.lambdas, 228,
              5,
              False,
              'precalculated_data/stabilized_corr_test_meta_tensors.npy',
              'precalculated_data/stabilized_corr_test_lambda_tensors.npy', None,
              None,
              stabilize=None)

        if continue_from == 0:
            self.generator = Generator(self.features, self.instances, self.classes, self.metas.getLength(), self.z_size)
            self.discriminator = Discriminator(self.features, self.instances, self.classes, self.metas.getLength(),
                                               self.lambdas.getLength())
        else:
            self.generator = Generator(self.features, self.instances, self.classes, self.metas.getLength(), self.z_size)
            self.generator.load_state_dict(
                torch.load(
                    f'{self.models_path}/generator-{self.exp_prefix}-{self.features}_{self.instances}_{self.classes}-{continue_from}.pkl'))
            self.generator.eval()

            self.discriminator = Discriminator(self.features, self.instances, self.classes, self.metas.getLength(),
                                               self.lambdas.getLength())
            self.discriminator.load_state_dict(
                torch.load(
                    f'{self.models_path}/discriminator-{self.exp_prefix}-{self.features}_{self.instances}_{self.classes}-{continue_from}.pkl'))
            self.discriminator.eval()

        self.generator = self.generator.to(self.device)

        self.discriminator = self.discriminator.to(self.device)


        print(f'Generator parameters: {count_parameters(self.generator)}')
        print(f'Discriminator parameters: {count_parameters(self.discriminator)}')
        print(f'D / G: {count_parameters(self.discriminator) / count_parameters(self.generator)}')
        # exit(0)
        self.lr = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999

        self.g_optimizer = optim.Adam(self.generator.parameters(),
                                      self.lr, [self.beta1, self.beta2])
        self.d_optimizer = optim.Adam(self.discriminator.parameters(),
                                      self.lr, [self.beta1, self.beta2])

        self.mse = MSELoss()
        self.mse = self.mse.to(self.device)

        self.loaded = self.discriminator
        exp_id = 2
        path = f'models2201_d/discriminator-stabilization-{exp_id}-16_64_2-20.pkl'
        self.loaded.load_state_dict(torch.load(path))

        joblib.dump(self.loaded, f'basic_metaclassifiers/lm_gan_discriminator_model_{exp_id}.joblib')

if __name__ == '__main__':
    # parse arguments which are fields of Trainer
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--num_epochs', type=int, default=10)
    # parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--learning_rate', type=float, default=0.0002)
    # parser.add_argument('--device', type=str, default='cuda:0')
    # parser.add_argument('--data_prefix', type=str, default='data')
    # parser.add_argument('--models_path', type=str, default='models')
    # parser.add_argument('--log_step', type=int, default=10)
    # parser.add_argument('--log_step_print', type=int, default=10)
    # parser.add_argument('--save_period', type=int, default=1)
    # args = parser.parse_args()

    trainer = Trainer(num_epochs=20,
                      # device=args.device,
                      continue_from=0,
                      exp_prefix='stabilization-2'
                      )
    # trainer = Trainer(data_prefix='')
    # trainer.train()
