from Models import DeepSetModelV5
from Models import CONFIG

from DatasetLoader import DatasetWithTargets
from feature_extraction.MetaFeaturesCollector import MetaFeaturesCollector
from feature_extraction.LambdaFeaturesCollector import LambdaFeaturesCollector

def acc(y_pred, y_true):
    y_pred_ = (y_pred == y_pred.max(dim=1, keepdim=True)[0]).to(bool)
    y_pred__ = ((y_true == 1) & (y_pred_ == 1)).to(int)
    return y_pred__.sum() / y_pred__.shape[0]

if __name__ == '__main__':

    model = DeepSetModelV5(hidden_size_0=CONFIG.hidden_size_0,
                           hidden_size_1=CONFIG.hidden_size_1,
                           predlast_hidden_size=CONFIG.predlast_hidden_size,
                           meta_size=CONFIG.meta_size,
                           out_classes=CONFIG.out_classes)

    model.to(CONFIG.device)

    from copy import deepcopy
    from Models import CONFIG

    lambdaFeaturesCollector = LambdaFeaturesCollector(CONFIG.features, CONFIG.instances)
    metaFeaturesCollector = MetaFeaturesCollector(CONFIG.features, CONFIG.instances)
    train_dataset = DatasetWithTargets('../processed_data/processed_16_64_2/',
                                       CONFIG.features,
                                       CONFIG.instances,
                                       CONFIG.classes,
                                       metaFeaturesCollector,
                                       lambdaFeaturesCollector,
                                       True,
                                       meta_precalc_path='precalculated_data/train_meta_tensors.npy',
                                       lambda_precalc_path='precalculated_data/train_lambda_lambda_tensors.npy',
                                       standardscale=CONFIG.standardscaler,
                                       meta_standardscale=CONFIG.meta_standardscaler,
                                       transpose=CONFIG.transpose)

    test_lambdaFeaturesCollector = LambdaFeaturesCollector(CONFIG.features, CONFIG.instances)

    test_dataset = DatasetWithTargets('../processed_data/test/',
                                      CONFIG.features,
                                      CONFIG.instances,
                                      CONFIG.classes,
                                      metaFeaturesCollector,
                                      lambdaFeaturesCollector,
                                      False,
                                      meta_precalc_path='precalculated_data/test_meta_tensors.npy',
                                      lambda_precalc_path='precalculated_data/test_lambda_lambda_tensors.npy',
                                      standardscale=CONFIG.standardscaler,
                                      meta_standardscale=CONFIG.meta_standardscaler,
                                      transpose=CONFIG.transpose)

    test_dataset.data_scaler = deepcopy(train_dataset.data_scaler)
    test_dataset.meta_data_scaler = deepcopy(test_dataset.meta_data_scaler)

    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=CONFIG.train_batch_size,
                                  num_workers=CONFIG.num_workers,
                                  shuffle=True)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 num_workers=CONFIG.num_workers,
                                 shuffle=False)

    import numpy as np
    import torch
    from tqdm import tqdm
    import math

    methods = ['deepset']
    mse = torch.nn.MSELoss()

    exp_num = 1

    methods_results = []

    for w in range(len(methods)):
        print("Method " + methods[w])
        global_reals = []
        global_fakes = []
        global_luckies = []
        for j in range(5, 21, 5):
            g_reals = []
            g_fakes = []
            g_luckies = []
            d_int_reals = []
            d_int_fakes = []
            d_int_luckies = []
            index = 0
            for i in range(exp_num):
                reals = []
                fakes = []
                luckies = []
                index = 0
                print("Epoch " + str(j))
                model.load_state_dict(
                    torch.load(
                        # f'./{methods[w]}{i}/discriminator-16_64_2-{j}.pkl'
                        f'models2201_d/discriminator-{methods[w]}-16_64_2-{j}.pkl',
                        map_location=CONFIG.device
                    ))
                model.eval()

                # test loop
                model.eval()
                test_epoch_loss = 0
                test_squared_error = 0
                test_total_elems = 0
                test_correct_acc = 0
                luckies.append(0)
                cnt = 0
                for (ind, (X_pos, X_neg, meta, y)) in tqdm(enumerate(test_dataloader), total=len(test_dataloader),
                                                           desc=f"Test Epoch {i}"):
                    X_pos = X_pos.to(CONFIG.device)
                    X_neg = X_neg.to(CONFIG.device)
                    meta = meta.to(CONFIG.device)
                    y = y.to(CONFIG.device)

                    y_pred = model(X_pos, X_neg, meta)

                    y_pred_ = torch.sigmoid(y_pred)

                    q = y_pred_.cpu().detach().numpy().flatten()

                    winners = np.argwhere(q == np.amax(q)).flatten().tolist()
                    if (len(winners) >= 2):
                        cnt += 1
                    lambdas_ = y.cpu().detach().numpy()
                    for winner in winners:
                        if y[0][winner] == 1.0:
                            luckies[index] += 1
                luckies[index] /= len(test_dataloader)
                print(cnt)
                # print(reals)
                # print(fakes)
                print(luckies)

                g_luckies.append(luckies[index])
                index += 1
            std_r = np.std(g_reals)
            std_f = np.std(g_fakes)
            std_l = np.std(g_luckies)
            d_int_l = 2.0 * std_l / math.sqrt(exp_num)
            d_int_luckies.append(d_int_l)
            global_luckies.append((np.mean(g_luckies), d_int_luckies))
        methods_results.append((global_reals, global_fakes, global_luckies))
    print(methods_results)
