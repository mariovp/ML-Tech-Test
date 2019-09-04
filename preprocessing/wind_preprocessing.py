import math

import numpy as np
import pandas as pd
from pandas import Series


def read_time_series(filepath: str, start: int, finish: int):
    raw_series = pd.read_csv(filepath, header=None, parse_dates=True, index_col=0, squeeze=True)
    cut_series = raw_series.values[start:finish]
    return cut_series


def slide_window(data: Series, window_size: int, stride: int):
    x_temp = []
    y_temp = []
    strides = (len(data) - window_size) - 1
    strides = math.floor(strides / stride)
    for i in range(strides):
        s = i * stride
        x_temp.append(data[s:s + window_size:1])
        y_temp.append(data[s + window_size])
    return np.array(x_temp), np.array(y_temp)


def normalize_min_max(data: Series):
    range_min = data.min()
    range_max = data.max()
    normalized = (data - range_min) / (range_max - range_min)
    return normalized, range_min, range_max
