from copy import copy

import pandas as pd
import os
import pickle
from functools import reduce

from financial_features import *
from Const import *
from length_fixing import *
from similarity_functions import *


def cal_other_stock_similarity(df_stocks, stock_to_compare, stock_names, similarity_func, #similarity_file_path='',
                               fix_len_func=time_join, similarity_col=target_col, force=False, split_time='', **kwargs):
    print("calc similarities for " + stock_to_compare + " func " + str(similarity_func) + \
          " fix len " + str(fix_len_func) + " on column " + similarity_col)

    similarities = [
        similarity_func(df_stocks[df_stocks[name_col] == stock_to_compare],  # [y_col].tolist(),
                        df_stocks[df_stocks[name_col] == stock_name], fix_len_func,
                        similarity_col)
        for stock_name in stock_names
    ]

    return similarities

def normalize_similarity(top_stocks, stock_to_compare):
    stocks_val = list(top_stocks.values())
    top_stock_w = {}
    sum_vals = 0
    for stock_k, v in top_stocks.items():
        if stock_k != stock_to_compare:
            top_stock_w[stock_k] = np.abs(float(v) - max(stocks_val)) / (max(stocks_val) - min(stocks_val))
            sum_vals += top_stock_w[stock_k]

    for stock_k, v in top_stock_w.items():
        top_stock_w[stock_k] = top_stock_w[stock_k] / sum_vals

    return top_stock_w


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

        feature_df = pd.concat([feature_df, data_norm_df], axis=1)

    return feature_df