import numpy as np
from matplotlib import pyplot as plt

import stabilization.NumpyRawToData as nrt

arr = np.load('../processed_data/processed_16_64_2/6_0_1.npy')
arr = arr.reshape(128, 16)
target_column = np.array([i < 64 for i in range(128)]).reshape(128, 1)
arr = np.concatenate([arr, target_column], axis=1)

print(arr.shape)

res = nrt.correlation_method(arr)

print(res[0].shape, res[1].shape)

concated_res = np.concatenate([res[0], res[1]], axis=0)

diff_arr = arr[:, :-1] - concated_res
print(diff_arr)

plt.imshow(diff_arr)
plt.show()