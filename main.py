import pandas as pd

from financial_features import *

def read_data(path):
    all_stocks = pd.read_csv(path)
    all_stocks['Date'] = pd.to_datetime(all_stocks['Date'], format='%Y-%m-%d', errors='ignore')
    all_stocks = all_stocks.dropna(axis=0)
    all_stocks = all_stocks.set_index('Date', drop=False)
    return all_stocks

df = read_data('data/all_stocks_5yr.csv')
print(df)