import numpy as np
import pandas as pd
import torch

from matplotlib import pyplot as plt


# torch acc
def torch_acc(y_pred, y_true):
    y_pred_ = (y_pred == y_pred.max(dim=1, keepdim=True)[0]).to(bool)
    y_pred_zeros = (y_pred.sum(dim=1) == 0).sum()
    y_pred__ = ((y_true == 1) & (y_pred_ == 1)).to(int)
    return (y_pred__.sum() - y_pred_zeros) / y_pred__.shape[0]


def torch_acc_numerator(y_pred, y_true):
    y_pred_ = (y_pred == y_pred.max(dim=1, keepdim=True)[0]).to(bool)
    y_pred_zeros = (y_pred.sum(dim=1) == 0).sum()
    y_pred__ = ((y_true == 1) & (y_pred_ == 1)).to(int)
    return (y_pred__.sum() - y_pred_zeros)