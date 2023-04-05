from os import listdir
from os.path import isfile, join

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from .DecisionTreeMeta import DecisionTreeMeta
from .InformationMeta import InformationMeta
from .StatisticalMeta import StatisticalMeta


class MetaFeaturesCollector:

    def __init__(self, features_size: int, instances_size: int):
        self.features = features_size
        self.instances = instances_size
        self.meta_features = [
            StatisticalMeta(features_size, instances_size),
            InformationMeta(features_size, instances_size),
            DecisionTreeMeta(features_size, instances_size)
        ]
        self.min_max = MinMaxScaler()
        self.length = None

    def getLength(self):
        if self.length is None:
            length = 0
            for meta in self.meta_features:
                length += meta.getLength()
            self.length = length
            return length
        else:
            return self.length

    def train(self, path: str, load_from_fs: bool = False):
        if load_from_fs:
            print(path)
            results = np.load(f'{path}../../-fs-dump.npy')
            print('Loaded from file system')
        else:

            only_files = [f for f in listdir(path) if isfile(join(path, f))]
            results = []
            for name in tqdm(only_files):

                try:
                    stacked = np.load(f'{path}{name}')

                except ValueError:
                    stacked = np.load(f'{path}{name}', allow_pickle=True)

                numpy_res = self.getNumpy(stacked)
                results.append(numpy_res)
            results = np.array(results)

            np.save(f'{path}../../-fs-dump.npy', results)

        self.min_max.fit(results)
        return self.min_max.get_params()

    def get(self, stacked: np.ndarray) -> torch.Tensor:
        zero_in, one_in = stacked[0], stacked[1]
        meta_features = self.meta_features[0].getMeta(zero_in, one_in)
        for meta in self.meta_features[1:]:
            meta_features = np.concatenate((meta_features, meta.getMeta(zero_in, one_in)))
        metas = meta_features.reshape(1, -1)
        metas = self.min_max.transform(metas)
        metas = metas.T
        return torch.from_numpy(metas).float()

    def getShort(self, stacked: np.ndarray) -> torch.Tensor:
        zero_in, one_in = stacked[0], stacked[1]
        meta_features = self.meta_features[0].getMeta(zero_in, one_in)
        for meta in self.meta_features[1:]:
            meta_features = np.concatenate((meta_features, meta.getMeta(zero_in, one_in)))
        metas = meta_features.reshape(1, -1)
        metas = self.min_max.transform(metas)
        metas = metas.T
        return torch.from_numpy(metas).float()

    def getNumpy(self, stacked: np.ndarray) -> np.ndarray:
        zero_in = stacked[0]
        one_in = stacked[1]
        meta_features = self.meta_features[0].getMeta(zero_in, one_in)
        for meta in self.meta_features[1:]:
            new_meta = meta.getMeta(zero_in, one_in)
            meta_features = np.concatenate((meta_features, new_meta))
        metas = meta_features
        return metas


if __name__ == '__main__':
    meta = MetaFeaturesCollector(16, 64)
    print(meta.train(f"../../processed_data/processed_16_64_2/"))
    print(meta.min_max.data_min_)
    print(meta.min_max.data_max_)
    print(meta.min_max.data_range_)
    print(meta.min_max.scale_)
