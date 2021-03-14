import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util.fastpip import fastpip
from Const import *

def padding(stock1, stock2):
    """
    fix 2 stock to be in same length, by multiplying the first value of the shorter stock
    """
    diff_len = len(stock1) - len(stock2)
    if diff_len > 0:
        add_values = pd.DataFrame([stock2.iloc[0]] * diff_len)
        stock2 = pd.concat([add_values, stock2])
    elif diff_len < 0:
        add_values = pd.DataFrame([stock1.iloc[0]] * abs(diff_len))
        stock1 = pd.concat([add_values, stock1])
    return stock1, stock2


def time_join(stock1, stock2):
    """
    if one stock is missing a time point while the other is not the time point
        is eliminated (equivalent to inner join in SQL) this fixing is
        the most popular but may reduce the data substantially
    """
    stock2_ = stock2.copy()
    stock1_ = stock1.copy()

    stock2_[time_col] = stock2_.index
    stock1_[time_col] = stock1_.index

    stock2_ = stock2_[stock2_[time_col].isin(stock1_[time_col])]
    stock1_ = stock1_[stock1_[time_col].isin(stock2_[time_col])]

    del stock2_[time_col]
    del stock1_[time_col]
    return stock1_, stock2_


def delay_time_join(stock1, stock2, delay=1):
    """
    stock values are pushed t times points backward (delay),
        this correlation meant to identify if one
        stock indicated future behavior of the other one
    """
    stock2_ = stock2.copy()
    stock1_ = stock1.copy()

    stock2_[time_col] = stock2_.index
    stock1_[time_col] = stock1_.index

    stock2_[time_col] = stock2_[time_col].shift(-delay)
    stock2_ = stock2_[stock2_[time_col].isin(stock1_[time_col])]
    stock1_ = stock1_[stock1_[time_col].isin(stock2_[time_col])]

    del stock2_[time_col]
    del stock1_[time_col]

    return stock1_, stock2_


# https://www.witpress.com/Secure/elibrary/papers/CF04/CF04024FU.pdf
def pip_fix(stock1, stock2, factor=10):
    stock1, stock2 = time_join(stock1, stock2)
    min_len = min(len(stock1[target_col]), len(stock2[target_col]))
    pip_size = min_len / factor
    if pip_size < 25 and min_len > 25:
        pip_size = 25
    stock1_pairs = [(t, p) for t, p in zip(range(len(stock1[target_col])), stock1[target_col])]
    stock2_pairs = [(t, p) for t, p in zip(range(len(stock2[target_col])), stock2[target_col])]
    stock1_pairs_pip = fastpip(stock1_pairs, pip_size)
    stock2_pairs_pip = fastpip(stock2_pairs, pip_size)

    locs1 = [i[0] for i in stock1_pairs_pip]
    locs2 = [i[0] for i in stock2_pairs_pip]

    stock1_index = stock1.index[locs1]
    stock2_index = stock2.index[locs2]
    indexes = stock1_index.union(stock2_index)
    return stock1.loc[indexes], stock2.loc[indexes]


def test():
    all_stocks = pd.read_csv('data/all_stocks_5yr.csv')
    all_stocks['Date'] = pd.to_datetime(all_stocks['Date'], format='%Y-%m-%d', errors='ignore')
    all_stocks = all_stocks.dropna(axis=0)
    all_stocks = all_stocks.set_index('Date', drop=False)

    stock1 = all_stocks[all_stocks['Name'] == 'GOOGL']
    stock1 = stock1[['Close']][:40]

    stock2 = all_stocks[all_stocks['Name'] == 'AAPL']
    stock2 = stock2[['Close']][:50]

    """
    stock1 = pd.DataFrame()
    stock1[similarity_col] = range(1, 51)
    stock1[time_col] = range(5, 55)
    stock1 = stock1.set_index(time_col)

    stock2 = pd.DataFrame()
    stock2[similarity_col] = range(1, 51)
    stock2[time_col] = range(10, 60)
    stock2 = stock2.set_index(time_col)
    """
    fix_len_funcs = [padding, time_join, delay_time_join, pip_fix]

    plt.figure(figsize=(16, 6))
    plt.plot(stock1, c='b', marker='.')
    plt.plot(stock2, c='r', marker='.')
    plt.title("Original")
    plt.show()

    for f in fix_len_funcs:
        s1, s2 = f(stock1, stock2)

        plt.figure(figsize=(16, 6))
        plt.plot(s1, c='b', marker='.')
        plt.plot(s2, c='r', marker='.')
        plt.title(f.__name__ + ' ' + str(len(s1)) + ' - ' + str(len(s2)))
        plt.show()


if __name__ == "__main__":
    print("Test fix length")
    test()
