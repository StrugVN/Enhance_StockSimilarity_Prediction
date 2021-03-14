from copy import copy

import pandas as pd
import os
import pickle
from functools import reduce

from financial_features import *
from Const import *


def cal_other_stock_similarity(data, stock_to_compare, other_stocks):
    return


def cal_financial_features(data, norm_func=None):
    feature_df = data[[time_col, name_col, target_col]].copy()

    feature_df['Close_proc'] = PROC(data['Close'])
    # stock_X_finance_df = combine_df(stock_X_finance.values, "_proc", stock_X.columns, TIME, stock_X.index)
    feature_df['rsi'] = rsiFunc(data['Close'])  # Relative strength index
    feature_df['MACD'] = computeMACD(data['Close'])[2]  # Moving Average Convergence/Divergence
    feature_df['Open_Close_diff'] = data['Open'] - data['Close']
    feature_df['High_Low_diff'] = data['High'] - data['Low']

    if norm_func is not None:  # Normalize
        scaler = copy(norm_func)
        numeric_cols = data.select_dtypes(
            include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.tolist()

        scaler.fit(data[numeric_cols])
        data_norm = scaler.transform(data[numeric_cols])

        data_norm_df = pd.DataFrame(data_norm, columns=[s + '_norm' for s in numeric_cols])
        data_norm_df[time_col] = data.index
        data_norm_df = data_norm_df.set_index(time_col)

        feature_df = pd.concat([feature_df, data_norm_df], axis=1, join='inner')

    return feature_df