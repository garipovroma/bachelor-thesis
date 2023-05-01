import os

import torch
from torch.utils import data
import numpy as np
from torch.utils.data import DataLoader

from feature_extraction.MetaFeaturesCollector import MetaFeaturesCollector
from feature_extraction.LambdaFeaturesCollector import LambdaFeaturesCollector
from Models import CONFIG

class DatasetFolder(data.Dataset):

    def __init__(self, path: str, features_size: int, instances_size: int, classes_size: int,
                 meta: MetaFeaturesCollector, lambdas: LambdaFeaturesCollector, train_meta: bool):
        self.root = path
        self.features = features_size
        self.instances = instances_size
        self.classes = classes_size
        paths = []
        for fname in os.listdir(self.root):
            path = os.path.join(self.root, fname)
            if not os.path.isdir(path):
                paths.append(path)
        # print(f'paths = {paths}')
        from collections import Counter
        shapes = []
        for i in paths:
            loaded_np_data = np.load(i, allow_pickle=True)
            shapes.append(loaded_np_data.shape)
            if loaded_np_data.shape == (8000, 27):
                print(i)
                exit(0)
            # print(f'loaded_np_data.shape = {loaded_np_data.}')
        print(f'shapes = {Counter(shapes)}')
        self.data_paths = paths
        self.meta_features = meta
        if train_meta:
            self.meta_features.train(self.root, load_from_fs=True)
        self.lambda_features = lambdas

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        data_path = self.data_paths[index]
        data_np = np.load(data_path, allow_pickle=True)
        dataset_tensor = torch.from_numpy(data_np).float().view((self.classes, self.instances, self.features))

        meta_tensor = self.meta_features.get(data_np)
        meta_tensor = meta_tensor.view(self.meta_features.getLength(), 1, 1)
        lambda_tensor = self.lambda_features.get(data_np)
        return dataset_tensor, meta_tensor, lambda_tensor

    def __len__(self):
        return len(self.data_paths)


from sklearn.preprocessing import StandardScaler


# class DatasetWithMetaFolder(data.Dataset):
#
#     def __init__(self, path: str, features_size: int, instances_size: int, classes_size: int,
#                  meta: MetaFeaturesCollector, lambdas: LambdaFeaturesCollector, train_meta: bool,
#                  meta_precalc_path: str, standardscale=False):
#         self.root = path
#         self.features = features_size
#         self.instances = instances_size
#         self.classes = classes_size
#         self.meta_precalc_path = meta_precalc_path
#         self.meta_precalculated = np.load(self.meta_precalc_path, allow_pickle=True)
#         paths = []
#         for fname in os.listdir(self.root):
#             path = os.path.join(self.root, fname)
#             if not os.path.isdir(path):
#                 paths.append(path)
#         # print(f'paths = {paths}')
#         from collections import Counter
#         shapes = []
#         for i in paths:
#             loaded_np_data = np.load(i, allow_pickle=True)
#             shapes.append(loaded_np_data.shape)
#             if loaded_np_data.shape == (8000, 27):
#                 print(i)
#                 # exit(0)
#             # print(f'loaded_np_data.shape = {loaded_np_data.}')
#         print(f'shapes = {Counter(shapes)}')
#         self.data_paths = paths
#         self.meta_features = meta
#         if train_meta:
#             self.meta_features.train(self.root, load_from_fs=True)
#
#         self.standardscale = standardscale
#         self.lambda_features = lambdas
#         self.data_scaler = StandardScaler()
#         self.all_data = np.concatenate([np.load(i, allow_pickle=True) for i in self.data_paths])
#         self.all_data = self.all_data.reshape((-1, self.all_data.shape[-1]))
#         print(f"shape = {self.all_data.shape}")
#         self.data_scaler.fit(self.all_data)
#         del self.all_data
#
#     def __getitem__(self, index) -> (torch.Tensor, torch.Tensor, torch.Tensor):
#         data_path = self.data_paths[index]
#         data_np = np.load(data_path, allow_pickle=True)
#         dataset_tensor = torch.from_numpy(data_np).float().view((self.classes, self.instances, self.features))
#         # dataset_tensor = dataset_tensor.flatten(0, 1)
#
#         # meta_tensor = self.meta_features.get(data_np)
#         meta_tensor = torch.Tensor(self.meta_precalculated[index])
#         meta_tensor = meta_tensor.view(self.meta_features.getLength(), 1, 1)
#         lambda_tensor = self.lambda_features.get(data_np)
#
#         if self.standardscale:
#             dataset_tensor = torch.Tensor(self.data_scaler.transform(dataset_tensor))
#
#         return dataset_tensor, meta_tensor, lambda_tensor
#
#     def __len__(self):
#         return len(self.data_paths)


class DatasetWithMetaFolder(data.Dataset):

    def __init__(self, path: str, features_size: int, instances_size: int, classes_size: int,
                 meta: MetaFeaturesCollector, lambdas: LambdaFeaturesCollector, train_meta: bool,
                 meta_precalc_path: str = None, lambda_precalc_path: str = None,
                 standardscale=False, meta_standardscale=False, transpose=False,
                 stabilize=False):
        self.root = path
        self.features = features_size
        self.instances = instances_size
        self.classes = classes_size
        self.meta_precalc_path = meta_precalc_path
        self.lambda_precalc_path = lambda_precalc_path
        if self.meta_precalc_path is not None:
            self.meta_precalculated = np.load(self.meta_precalc_path, allow_pickle=True)
        if self.lambda_precalc_path is not None:
            self.lambda_precalculated = np.load(self.lambda_precalc_path, allow_pickle=True)
        self.standardscale = standardscale
        self.meta_standardscale = meta_standardscale
        self.transpose = transpose
        self.stabilize = stabilize
        paths = []
        for fname in os.listdir(self.root):
            path = os.path.join(self.root, fname)
            if not os.path.isdir(path):
                paths.append(path)
        # print(f'paths = {paths}')
        from collections import Counter
        shapes = []
        for i in paths:
            loaded_np_data = np.load(i, allow_pickle=True)
            shapes.append(loaded_np_data.shape)
            if loaded_np_data.shape == (8000, 27):
                print(i)
                # exit(0)
            # print(f'loaded_np_data.shape = {loaded_np_data.}')
        print(f'shapes = {Counter(shapes)}')
        self.data_paths = paths
        self.meta_features = meta
        if train_meta:
            self.meta_features.train(self.root, load_from_fs=True)

        self.lambda_features = lambdas
        # standardscale
        if self.standardscale:
            self.data_scaler = StandardScaler()
            self.all_data = np.concatenate([np.load(i, allow_pickle=True) for i in self.data_paths])
            self.all_data = self.all_data.reshape((-1, self.all_data.shape[-1]))
            print(f"shape = {self.all_data.shape}")
            self.data_scaler.fit(self.all_data)
            del self.all_data

        # meta_standard_scale

        if self.meta_standardscale:
            self.meta_data_scaler = StandardScaler()
            all_meta_data = None
            if self.meta_precalc_path is not None:
                all_meta_data = self.meta_precalculated
            else:
                raise NotImplementedError(":)")
            self.meta_data_scaler.fit(all_meta_data.reshape(-1, self.meta_features.getLength()))

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        data_path = self.data_paths[index]
        data_np = np.load(data_path, allow_pickle=True)
        dataset_tensor = torch.from_numpy(data_np).float().view((self.classes, self.instances, self.features))
        dataset_tensor = dataset_tensor.flatten(0, 1)

        # meta_tensor = self.meta_features.get(data_np)
        meta_tensor = torch.Tensor(self.meta_precalculated[index])
        meta_tensor = meta_tensor.view(self.meta_features.getLength(), 1, 1)
        lambda_tensor = torch.Tensor(self.lambda_precalculated[index])

        # if self.standardscale:
        #     dataset_tensor = torch.Tensor(self.data_scaler.transform(dataset_tensor))
        #
        # if self.meta_standardscale:
        #     meta_tensor = torch.Tensor(
        #         self.meta_data_scaler.transform(meta_tensor.reshape(-1, self.meta_features.getLength()))).reshape(-1)

        # total_elems = self.classes * self.instances
        # dataset_tensor = torch.cat([dataset_tensor,
        #                             (torch.arange(0, total_elems) < total_elems / 2).reshape((total_elems, 1)),
        #                             (torch.arange(0, total_elems) >= total_elems / 2).reshape((total_elems, 1))],
        #                            dim=1)

        # if self.transpose:
        #     dataset_tensor = dataset_tensor.T
        dataset_tensor = dataset_tensor.view((self.classes, self.instances, self.features))
        meta_tensor = meta_tensor.view(self.meta_features.getLength(), 1, 1)
        # if self.stabilize is not None:
        #     dataset_tensor = dataset_tensor.reshape(128, 16)
        #     target_column = np.array([i < 64 for i in range(128)]).reshape(128, 1)
        #     dataset_tensor = np.concatenate([dataset_tensor, target_column], axis=1)
        #     x_0, x_1 = self.stabilize(dataset_tensor)
        #     torch.stack([torch.Tensor(x_0), torch.tensor(x_1)], dim=0)
        return dataset_tensor, meta_tensor, lambda_tensor

    def __len__(self):
        return len(self.data_paths)

    def precalc(self, pref_name):
        from tqdm import tqdm
        meta_tensors = []
        for i in tqdm(range(self.__len__()), total=self.__len__()):
            data_path = self.data_paths[i]
            data_np = np.load(data_path, allow_pickle=True)

            meta_tensor = self.meta_features.get(data_np)
            meta_tensor = meta_tensor.view(self.meta_features.getLength(), 1, 1)
            meta_tensors.append(meta_tensor)

        # write meta_tensors to numpy_file
        meta_tensors = torch.stack(meta_tensors).numpy()
        np.save(f'{pref_name}_meta_tensors.npy', meta_tensors)

    def precalc_lambda(self, pref_name):
        from tqdm import tqdm
        lambda_tensors = []
        for i in tqdm(range(self.__len__()), total=self.__len__()):
            data_path = self.data_paths[i]
            data_np = np.load(data_path, allow_pickle=True)

            lambda_tensor = self.lambda_features.get(data_np)
            lambda_tensors.append(lambda_tensor)

        # write meta_tensors to numpy_file
        lambda_tensors = torch.stack(lambda_tensors).numpy()
        print(lambda_tensors.shape)
        np.save(f'{pref_name}_lambda_tensors.npy', lambda_tensors)


class DatasetWithMetaFolderFlattened(data.Dataset):

    def __init__(self, path: str, features_size: int, instances_size: int, classes_size: int,
                 meta: MetaFeaturesCollector, lambdas: LambdaFeaturesCollector, train_meta: bool,
                 meta_precalc_path: str = None, lambda_precalc_path: str = None,
                 standardscale=False, meta_standardscale=False, transpose=False):
        self.root = path
        self.features = features_size
        self.instances = instances_size
        self.classes = classes_size
        self.meta_precalc_path = meta_precalc_path
        self.lambda_precalc_path = lambda_precalc_path
        if self.meta_precalc_path is not None:
            self.meta_precalculated = np.load(self.meta_precalc_path, allow_pickle=True)
        if self.lambda_precalc_path is not None:
            self.lambda_precalculated = np.load(self.lambda_precalc_path, allow_pickle=True)
        self.standardscale = standardscale
        self.meta_standardscale = meta_standardscale
        self.transpose = transpose
        paths = []
        for fname in os.listdir(self.root):
            path = os.path.join(self.root, fname)
            if not os.path.isdir(path):
                paths.append(path)
        # print(f'paths = {paths}')
        from collections import Counter
        shapes = []
        for i in paths:
            loaded_np_data = np.load(i, allow_pickle=True)
            shapes.append(loaded_np_data.shape)
            if loaded_np_data.shape == (8000, 27):
                print(i)
                # exit(0)
            # print(f'loaded_np_data.shape = {loaded_np_data.}')
        print(f'shapes = {Counter(shapes)}')
        self.data_paths = paths
        self.meta_features = meta
        if train_meta:
            self.meta_features.train(self.root, load_from_fs=True)

        self.lambda_features = lambdas
        # standardscale
        if self.standardscale:
            self.data_scaler = StandardScaler()
            self.all_data = np.concatenate([np.load(i, allow_pickle=True) for i in self.data_paths])
            self.all_data = self.all_data.reshape((-1, self.all_data.shape[-1]))
            print(f"shape = {self.all_data.shape}")
            self.data_scaler.fit(self.all_data)
            del self.all_data

        # meta_standard_scale

        if self.meta_standardscale:
            self.meta_data_scaler = StandardScaler()
            all_meta_data = None
            if self.meta_precalc_path is not None:
                all_meta_data = self.meta_precalculated
            else:
                raise NotImplementedError(":)")
            self.meta_data_scaler.fit(all_meta_data.reshape(-1, self.meta_features.getLength()))

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        data_path = self.data_paths[index]
        data_np = np.load(data_path, allow_pickle=True)
        dataset_tensor = torch.from_numpy(data_np).float().view((self.classes, self.instances, self.features))
        dataset_tensor = dataset_tensor.flatten(0, 1)

        # meta_tensor = self.meta_features.get(data_np)
        meta_tensor = torch.Tensor(self.meta_precalculated[index])
        meta_tensor = meta_tensor.view(self.meta_features.getLength(), 1, 1)
        lambda_tensor = torch.Tensor(self.lambda_precalculated[index])

        if self.standardscale:
            dataset_tensor = torch.Tensor(self.data_scaler.transform(dataset_tensor))

        if self.meta_standardscale:
            meta_tensor = torch.Tensor(
                self.meta_data_scaler.transform(meta_tensor.reshape(-1, self.meta_features.getLength()))).reshape(-1)

        total_elems = self.classes * self.instances
        dataset_tensor = torch.cat([dataset_tensor,
                                    (torch.arange(0, total_elems) < total_elems / 2).reshape((total_elems, 1))
                                    #    (torch.arange(0, total_elems) >= total_elems / 2).reshape((total_elems, 1))
                                    ],
                                   dim=1)
        if self.transpose:
            dataset_tensor = dataset_tensor.T
        return dataset_tensor, meta_tensor, lambda_tensor

    def __len__(self):
        return len(self.data_paths)

    def precalc(self, pref_name):
        from tqdm import tqdm
        meta_tensors = []
        for i in tqdm(range(self.__len__()), total=self.__len__()):
            data_path = self.data_paths[i]
            data_np = np.load(data_path, allow_pickle=True)

            meta_tensor = self.meta_features.get(data_np)
            meta_tensor = meta_tensor.view(self.meta_features.getLength(), 1, 1)
            meta_tensors.append(meta_tensor)

        # write meta_tensors to numpy_file
        meta_tensors = torch.stack(meta_tensors).numpy()
        np.save(f'{pref_name}_meta_tensors.npy', meta_tensors)

    def precalc_lambda(self, pref_name):
        from tqdm import tqdm
        lambda_tensors = []
        for i in tqdm(range(self.__len__()), total=self.__len__()):
            data_path = self.data_paths[i]
            data_np = np.load(data_path, allow_pickle=True)

            lambda_tensor = self.lambda_features.get(data_np)
            lambda_tensors.append(lambda_tensor)

        # write meta_tensors to numpy_file
        lambda_tensors = torch.stack(lambda_tensors).numpy()
        print(lambda_tensors.shape)
        np.save(f'{pref_name}_lambda_tensors.npy', lambda_tensors)

def get_loader(path: str, features_size: int, instances_size: int, classes_size: int, meta: MetaFeaturesCollector,
               lambdas: LambdaFeaturesCollector, batch_size: int, num_workers: int, train_meta: bool = True, precalced_meta_path: str = None,
               precalced_lambda_path: str = None,
               scaler = None, meta_scaler  = None,
               stabilize = None):
    datasets_inner = None
    if precalced_meta_path is not None:
        # datasets_inner = DatasetWithMetaFolder(path, features_size, instances_size, classes_size, meta, lambdas, train_meta,
        #                                       precalced_meta_path, standardscale=False)

        # path: str, features_size: int, instances_size: int, classes_size: int,
        # meta: MetaFeaturesCollector, lambdas: LambdaFeaturesCollector, train_meta: bool,
        # meta_precalc_path: str = None, lambda_precalc_path: str = None,
        # standardscale = False, meta_standardscale = False, transpose = False

        datasets_inner = DatasetWithMetaFolder(path, features_size, instances_size, classes_size, meta, lambdas, train_meta,
                                                  precalced_meta_path, precalced_lambda_path, standardscale=False, meta_standardscale=False, transpose=False,
                                                  stabilize=None)
    else:
        datasets_inner = DatasetFolder(path, features_size, instances_size, classes_size, meta, lambdas, train_meta)

    if scaler is not None:
        from copy import deepcopy
        datasets_inner.data_scaler = deepcopy(scaler)
        datasets_inner.meta_data_scaler = deepcopy(meta_scaler)

    data_loader = data.DataLoader(
        dataset=datasets_inner,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return data_loader, datasets_inner


import torch
from torch.utils import data
import numpy as np
from feature_extraction.MetaFeaturesCollector import MetaFeaturesCollector
from feature_extraction.LambdaFeaturesCollector import LambdaFeaturesCollector
from sklearn.preprocessing import StandardScaler
import os


class DatasetWithTargets(data.Dataset):

    def __init__(self, path: str, features_size: int, instances_size: int, classes_size: int,
                 meta: MetaFeaturesCollector, lambdas: LambdaFeaturesCollector, train_meta: bool,
                 meta_precalc_path: str = None, lambda_precalc_path: str = None,
                 standardscale=False, meta_standardscale=False, transpose=False):
        self.root = path
        self.features = features_size
        self.instances = instances_size
        self.classes = classes_size
        self.meta_precalc_path = meta_precalc_path
        self.lambda_precalc_path = lambda_precalc_path
        if self.meta_precalc_path is not None:
            self.meta_precalculated = np.load(self.meta_precalc_path, allow_pickle=True)
        if self.lambda_precalc_path is not None:
            self.lambda_precalculated = np.load(self.lambda_precalc_path, allow_pickle=True)
        self.standardscale = standardscale
        self.meta_standardscale = meta_standardscale
        self.transpose = transpose
        paths = []
        for fname in os.listdir(self.root):
            path = os.path.join(self.root, fname)
            if not os.path.isdir(path):
                paths.append(path)
        # print(f'paths = {paths}')
        from collections import Counter
        shapes = []
        for i in paths:
            loaded_np_data = np.load(i, allow_pickle=True)
            shapes.append(loaded_np_data.shape)
            if loaded_np_data.shape == (8000, 27):
                print(i)
                # exit(0)
            # print(f'loaded_np_data.shape = {loaded_np_data.}')
        print(f'shapes = {Counter(shapes)}')
        self.data_paths = paths
        self.meta_features = meta
        if train_meta:
            self.meta_features.train(self.root, load_from_fs=True)

        self.lambda_features = lambdas
        # standardscale
        if self.standardscale:
            self.data_scaler = StandardScaler()
            self.all_data = np.concatenate([np.load(i, allow_pickle=True) for i in self.data_paths])
            self.all_data = self.all_data.reshape((-1, self.all_data.shape[-1]))
            print(f"shape = {self.all_data.shape}")
            self.data_scaler.fit(self.all_data)
            del self.all_data

        # meta_standard_scale

        if self.meta_standardscale:
            self.meta_data_scaler = StandardScaler()
            all_meta_data = None
            if self.meta_precalc_path is not None:
                all_meta_data = self.meta_precalculated
            else:
                raise NotImplementedError(":)")
            self.meta_data_scaler.fit(all_meta_data.reshape(-1, self.meta_features.getLength()))

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        data_path = self.data_paths[index]
        data_np = np.load(data_path, allow_pickle=True)
        dataset_tensor = torch.from_numpy(data_np).float().view((self.classes, self.instances, self.features))
        dataset_tensor = dataset_tensor.flatten(0, 1)

        # meta_tensor = self.meta_features.get(data_np)
        meta_tensor = torch.Tensor(self.meta_precalculated[index])
        meta_tensor = meta_tensor.view(self.meta_features.getLength(), 1, 1)
        lambda_tensor = torch.Tensor(self.lambda_precalculated[index])

        if self.standardscale:
            dataset_tensor = torch.Tensor(self.data_scaler.transform(dataset_tensor))

        if self.meta_standardscale:
            meta_tensor = torch.Tensor(
                self.meta_data_scaler.transform(meta_tensor.reshape(-1, self.meta_features.getLength()))).reshape(-1)

        total_elems = self.classes * self.instances
        labels = (torch.arange(0, total_elems) < total_elems / 2).reshape(total_elems)

        dataset_tensor = torch.cat([dataset_tensor,
                                    (torch.arange(0, total_elems) < total_elems / 2).reshape((total_elems, 1)),
                                    # (torch.arange(0, total_elems) >= total_elems / 2).reshape((total_elems, 1))
                                    ],
                                   dim=1)

        pos_ = dataset_tensor[labels]
        neg_ = dataset_tensor[~labels]

        if self.transpose:
            dataset_tensor = dataset_tensor.T
            pos_ = pos_.T
            neg_ = neg_.T
        return pos_, neg_, meta_tensor, lambda_tensor

    def __len__(self):
        return len(self.data_paths)

    def precalc(self, pref_name):
        from tqdm import tqdm
        meta_tensors = []
        for i in tqdm(range(self.__len__()), total=self.__len__()):
            data_path = self.data_paths[i]
            data_np = np.load(data_path, allow_pickle=True)

            meta_tensor = self.meta_features.get(data_np)
            meta_tensor = meta_tensor.view(self.meta_features.getLength(), 1, 1)
            meta_tensors.append(meta_tensor)

        # write meta_tensors to numpy_file
        meta_tensors = torch.stack(meta_tensors).numpy()
        np.save(f'{pref_name}_meta_tensors.npy', meta_tensors)

    def precalc_lambda(self, pref_name):
        from tqdm import tqdm
        lambda_tensors = []
        for i in tqdm(range(self.__len__()), total=self.__len__()):
            data_path = self.data_paths[i]
            data_np = np.load(data_path, allow_pickle=True)

            lambda_tensor = self.lambda_features.get(data_np)
            lambda_tensors.append(lambda_tensor)

        # write meta_tensors to numpy_file
        lambda_tensors = torch.stack(lambda_tensors).numpy()
        print(lambda_tensors.shape)
        np.save(f'{pref_name}_lambda_tensors.npy', lambda_tensors)




def get_deepset_loader():
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
    from copy import deepcopy
    test_dataset.data_scaler = deepcopy(train_dataset.data_scaler)
    test_dataset.meta_data_scaler = deepcopy(test_dataset.meta_data_scaler)

    # train_dataloader = DataLoader(train_dataset,
    #                               batch_size=CONFIG.train_batch_size,
    #                               num_workers=CONFIG.num_workers,
    #                               shuffle=True)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=16,
                                 num_workers=0,
                                 shuffle=False)

    return test_dataloader

def get_deepset_flattened_loader():
    lambdaFeaturesCollector = LambdaFeaturesCollector(CONFIG.features, CONFIG.instances)
    metaFeaturesCollector = MetaFeaturesCollector(CONFIG.features, CONFIG.instances)
    train_dataset = DatasetWithMetaFolderFlattened('../processed_data/processed_16_64_2/',
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

    test_dataset = DatasetWithMetaFolderFlattened('../processed_data/test/',
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
    from copy import deepcopy
    test_dataset.data_scaler = deepcopy(train_dataset.data_scaler)
    test_dataset.meta_data_scaler = deepcopy(test_dataset.meta_data_scaler)

    # train_dataloader = DataLoader(train_dataset,
    #                               batch_size=CONFIG.train_batch_size,
    #                               num_workers=CONFIG.num_workers,
    #                               shuffle=True)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=16,
                                 num_workers=0,
                                 shuffle=False)

    return test_dataloader


def get_lmgan_dataloader():
    def getMeta(data_in: torch.Tensor, metas):
        meta_list = []
        for data in data_in:
            meta_list.append(metas.getShort(data.cpu().detach().numpy()))
        result = torch.stack(meta_list)
        return to_variable(result.view((result.size(0), result.size(1), 1, 1)))


    def to_variable(x):
        return Variable(x)


    exp_num = 1
    datasize = 64
    z_size = 100
    batch_size = 1911
    workers = 0
    lambdas = LambdaFeaturesCollector(16, 64)
    metas = MetaFeaturesCollector(16, 64)


    data_loader, train_dataset = get_loader(
                "../processed_data/corr_stabilized_train/",
                # f"../processed_data/{data_prefix}processed_{self.features}_{self.instances}_{self.classes}/",
                16, 64, 2, metas,
                lambdas, batch_size,
                workers,
                True,
                'precalculated_data/stabilized_corr_train_meta_tensors.npy',
                'precalculated_data/stabilized_corr_train_lambda_tensors.npy', None, None, stabilize=None)

    datatest, _ = get_loader(
                        # f"../processed_data/{data_prefix}test/",
                "../processed_data/corr_stabilized_test/",
                16, 64, 2, metas,
                lambdas, batch_size,
                workers,
                  False,
                  'precalculated_data/stabilized_corr_test_meta_tensors.npy',
                  'precalculated_data/stabilized_corr_test_lambda_tensors.npy', None,
                  None,
                  stabilize=None)
    return datatest

if __name__ == '__main__':
    datasets = DatasetFolder("../processed_data/processed_16_64_2/processed_16_64_2/", 16, 64, 2, MetaFeaturesCollector(16, 64),
                             LambdaFeaturesCollector(16, 64))
    for i in range(len(datasets)):
        print(datasets.__getitem__(i))
