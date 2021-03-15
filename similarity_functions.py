#from data_preperation import *
import pickle
from dtw import dtw
from sklearn.metrics import mean_squared_error
import statsmodels.tsa.stattools as ts

import random
import pandas as pd
import numpy as np
import os

from Const import *
from length_fixing import *

from util.SAX_FILE import SAX


def apply_dtw(stock1, stock2, fix_len_func=time_join, similarity_col=target_col):
    stock1, stock2 = fix_len_func(stock1, stock2)
    if len(stock1) <= 25 or len(stock2) <= 25:
        return 1000
    return dtw(stock1[similarity_col].tolist(), stock2[similarity_col].tolist(), dist=lambda x, y: abs(x - y))[0]


def apply_pearson(stock1, stock2, fix_len_func=time_join, similarity_col=target_col):
    stock1, stock2 = fix_len_func(stock1, stock2)
    if len(stock1) <= 25 or len(stock2) <= 25:
        return 1000
    pearson = np.corrcoef(np.array(stock1[similarity_col].tolist()), np.array(stock2[similarity_col].tolist()))[0, 1]
    return abs(pearson - 1)


def apply_euclidean(stock1, stock2, fix_len_func=time_join, similarity_col=target_col):
    stock1, stock2 = fix_len_func(stock1, stock2)
    return np.linalg.norm(np.array(stock1[similarity_col].tolist()) - np.array(stock2[similarity_col].tolist()))


def compare_sax(stock1, stock2, fix_len_func=time_join, similarity_col=target_col):
    stock1, stock2 = fix_len_func(stock1, stock2)
    if len(stock1) <= 25 or len(stock2) <= 25:
        return 1000
    sax_obj_ = SAX(wordSize=np.math.ceil(len(stock1)), alphabetSize=12)
    stock1_s = sax_obj_.transform(stock1[similarity_col].tolist())
    stock2_s = sax_obj_.transform(stock2[similarity_col].tolist())
    return sax_obj_.compare_strings(stock1_s, stock2_s)


def cointegration(stock1, stock2, fix_len_func=time_join, similarity_col=target_col):
    stock1, stock2 = fix_len_func(stock1, stock2)
    if len(stock1) <= 25 or len(stock2) <= 25:
        return 1000
    oin_t, p_val, _crit = ts.coint(stock1[similarity_col].tolist(),stock2[similarity_col].tolist())
    return p_val


def get_top_k(stock_names, similarities, k):
    s = np.array(similarities)
    k = k + 1
    idx = np.argpartition(s, k)
    names_top_k = np.array(stock_names)[idx[:k]]
    sim_top_k = s[idx[:k]]
    top_stocks = {}
    for i in range(len(names_top_k)):
        top_stocks[names_top_k[i]] = sim_top_k[i]

    return top_stocks


def get_random_k(stock_names, similarities, k):
    s = np.array(similarities)
    idx = np.argpartition(s, 1)
    name_target_stock = np.array(stock_names)[idx[:1]]
    top_stocks = {}
    top_stocks[name_target_stock[0]] = 0.0

    k -= 1
    random.seed(0)
    for i in range(k - 1):
        ind = random.choice(range(len(stock_names)))
        top_stocks[stock_names[ind]] = similarities[ind]
    return top_stocks


def test():
    all_stocks = pd.read_csv('data/all_stocks_5yr.csv')
    all_stocks['Date'] = pd.to_datetime(all_stocks['Date'], format='%Y-%m-%d', errors='ignore')
    all_stocks = all_stocks.dropna(axis=0)
    all_stocks = all_stocks.set_index('Date', drop=False)

    stock1 = all_stocks[all_stocks['Name'] == 'GOOGL']
    stock1 = stock1[['Close']][:50]

    stock2 = all_stocks[all_stocks['Name'] == 'AAPL']
    stock2 = stock2[['Close']][:50]

    """
    similarity_col = target_col
    stock1 = pd.DataFrame()
    stock1[similarity_col] = np.random.rand(50)
    stock1[time_col] = range(5,55)
    stock1 = stock1.set_index(time_col)

    stock2 = pd.DataFrame()
    stock2[similarity_col] = np.random.rand(50)
    stock2[time_col] = range(10, 60)
    stock2 = stock2.set_index(time_col)
    """

    print('\t\t\t\t\t\tpadding\t\t\t\t\t\ttime_join\t\t\t\t\tdelay_time_join\t\t\t\t\tpip_fix')
    for sim_func in [cointegration, apply_euclidean, compare_sax, apply_dtw, apply_pearson]:

        print(sim_func.__name__, end = ":\t" + ' '*(20 - len(sim_func.__name__)))
        for fix_func in [padding,time_join,delay_time_join,pip_fix]:
            sims = sim_func(stock1, stock2, fix_func, similarity_col=target_col)
            print(sims, end='\t\t\t')
        print('')


if __name__ == "__main__":
    print("Test similarity funcs")
    test()

