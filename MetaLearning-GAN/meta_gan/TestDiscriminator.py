import torch
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from torch.autograd import Variable
from torch.nn import MSELoss
import numpy as np
import os
from tqdm import tqdm

from matplotlib import pyplot as plt

from DatasetLoader import get_loader
from Models import Generator, Discriminator
from feature_extraction.LambdaFeaturesCollector import LambdaFeaturesCollector
from feature_extraction.MetaFeaturesCollector import MetaFeaturesCollector

if __name__ == '__main__':
    datasize = 64
    z_size = 100
    batch_size = 1911
    workers = 16

    lambdas_ = LambdaFeaturesCollector(16, 64)
    metas_ = MetaFeaturesCollector(16, 64)
    dataloader = get_loader(f"../processed_data/processed_16_64_2/", 16, 64, 2, metas_, lambdas_, batch_size, workers)
    # datatest = get_loader(f"../processed_data/test/", 16, 64, 2, metas_, lambdas_, batch_size, workers,
    #                       train_meta=False, precalced_meta_path="test_meta_tensors.npy")

    data_loader, train_dataset = get_loader(
        "../processed_data/processed_16_64_2/",
        16, 64, 2, metas_,
        lambdas_, batch_size,
        workers,
        True,
        'train_meta_tensors.npy', 'train_lambda_lambda_tensors.npy', None, None)

    test_loader, _ = get_loader(
        "../processed_data/test/",
        16, 64, 2, metas_,
        lambdas_, batch_size,
        workers,
        False,
        'test_meta_tensors.npy', 'test_lambda_lambda_tensors.npy',
        train_dataset.data_scaler,
        train_dataset.meta_data_scaler)

    print(len(test_loader))
    # exit(0)


    dir_path = 'models2201_d/'

    # EXP_NAME = 'realmeta_to_discirminator'
    # EXP_NAME = 'smooting-coef'
    EXP_NAME = 'scalers'

    paths = os.listdir(dir_path)
    avg_losses = []
    avg_mses = []
    checkpoints_epochs = []
    paths = []
    with tqdm(total = 10) as pbar:
        for i in range(0, 22):
            path = os.path.join(dir_path, f'discriminator-{EXP_NAME}-16_64_2-{i}.pkl')
            # check if file exists
            if os.path.exists(path):
                paths.append(path)
                checkpoints_epochs.append(i)
            else:
                continue

            discriminator = Discriminator(16, 64, 2, metas_.getLength(),
                                          lambdas_.getLength())
            full_path = path
            discriminator.load_state_dict(
                torch.load(full_path))

            device = torch.device('mps')

            discriminator.eval()
            discriminator = discriminator.to(device)
            mse = MSELoss()


            def to_variable(x):
                x = x.to(device)
                return Variable(x)


            loss = []
            squared_error_sum = 0
            total_elems = 0
            for j, data in enumerate(test_loader):
                print(j)
                dataset = to_variable(data[0])
                metas = to_variable(data[1])
                lambdas = to_variable(data[2])
                real_outputs = discriminator(dataset, metas)
                d_real_labels_loss = mse(real_outputs[:, 1:], lambdas)
                loss.append(d_real_labels_loss.cpu().detach().numpy())
                squared_error_sum += np.sum((lambdas.cpu().detach().numpy() - real_outputs[:, 1:].cpu().detach().numpy()) ** 2)
                total_elems += lambdas.cpu().detach().numpy().shape[0]

            pbar.update(1)
            # print(loss)
            # print(np.mean(loss))
            avg_losses.append(np.mean(loss))
            avg_mses.append(squared_error_sum / total_elems)
            print(f'mse: {squared_error_sum / total_elems}, avg loss: {np.mean(loss)}')
            print(loss)
    print(avg_losses)
    plt.plot(checkpoints_epochs, avg_losses)
    # plt.plot([2 * i + 2 for i in range(len(avg_losses))], avg_losses)
    plt.xlabel('epoch')
    plt.ylabel('mse')
    plt.savefig(f'exp_results/mse-discriminator-{EXP_NAME}.png')
    plt.show()

    # write avg_losses arr to file
    with open(f'exp_results/mse-{EXP_NAME}.txt', 'w') as f:
        f.write(str(avg_losses))
