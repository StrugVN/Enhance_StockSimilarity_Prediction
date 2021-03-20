from Const import *
import pandas as pd


def read_data(path, date_format='%Y-%m-%d'):
    all_stocks = pd.read_csv(path)
    all_stocks[const_time_col] = pd.to_datetime(all_stocks[const_time_col], format=date_format, errors='ignore')
    all_stocks = all_stocks.dropna(axis=0)
    all_stocks = all_stocks.set_index(const_time_col, drop=False)
    return all_stocks


def inverse_scaling(target_col, col_data, scaler_cols, scaler):
    data = pd.DataFrame()

    for c in scaler_cols:
        data[c] = [0.0] * len(col_data)

    data[target_col] = col_data

    inverted_data = scaler.inverse_transform(data)
    inverted_data = pd.DataFrame(inverted_data, columns=scaler_cols)

    return inverted_data[target_col]
