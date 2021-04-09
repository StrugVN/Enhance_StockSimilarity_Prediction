from Const import *
import pandas as pd
import numpy as np

from pandas_datareader import data as pdr
from datetime import date
import yfinance as yf
yf.pdr_override()

import types


def read_data(path, date_format='%Y-%m-%d'):
    all_stocks = pd.read_csv(path)
    all_stocks[const_time_col] = pd.to_datetime(all_stocks[const_time_col], format=date_format, errors='ignore')
    all_stocks = all_stocks.dropna(axis=0)
    all_stocks = all_stocks.set_index(const_time_col, drop=False)
    return all_stocks


def get_sp500_curr_stock_symbols():
    source = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    stock_df = source[0]
    return stock_df['Symbol'].to_list()


def save_stock_pulled(file_name, ticket_lists, start_date, end_date, interval='1h'):
    """
    The requested range [start_day, end_date] must be within:
        - the last 730 days for '1h' interval.
        - the last 60 days for '90m' interval
    """
    final_df = pd.DataFrame()
    attr_list = ['Open', 'High', 'Low', 'Close', 'Volume']

    for ticket in ticket_lists:
        df_ = pdr.get_data_yahoo(ticket, start=start_date, end=end_date, interval=interval)[attr_list]
        df_['Name'] = ticket
        final_df = pd.concat([final_df, df_])

    final_df.index = pd.to_datetime(final_df.index).strftime('%Y/%m/%dT%H:%M:%S')
    final_df.to_csv('../data/' + file_name + '.csv', index_label='Date')
    return


def inverse_scaling(target_col, col_data, scaler_cols, scaler):
    data = pd.DataFrame()

    for c in scaler_cols:
        data[c] = [0.0] * len(col_data)

    data[target_col] = col_data

    inverted_data = scaler.inverse_transform(data)
    inverted_data = pd.DataFrame(inverted_data, columns=scaler_cols)

    return inverted_data[target_col]


def get_y_bin(x, y, window_len, target_col):
    y = y.reshape(y.shape[0], 1)
    y_bin = []
    if window_len <= 1:
        y_bin.append(np.sign(y[0] - x[target_col][0]))
    else:
        col = (target_col, window_len - 1)
        y_bin.append(np.sign(y[0] - x[col][0]))

    for i in range(1, len(y)):
        y_bin.append(np.sign(y[i] - y[i - 1]))

    for i in y_bin:
        if i[0] == 0:
            i[0] = 1

    return y_bin


def long_short_profit_evaluation(curr_price, predicted_price):
    is_long = None
    profit = 0
    last_buy = 0
    profits = []
    position = 0
    start = curr_price[0]
    count = 0
    for i in range(len(curr_price)):
        # go long
        if predicted_price[i] > 0:
            # first time
            if is_long is None:
                last_buy = curr_price[i]
                is_long = True
            # if short position - close it and go long
            elif not is_long:
                profit = profit + (last_buy - curr_price[i])
                position = profit
                last_buy = curr_price[i]
                is_long = True
                count += 1
            elif is_long:
                position = profit + (curr_price[i] - last_buy)

        # go short
        if predicted_price[i] < 0:
            # first time
            if is_long is None:
                last_buy = curr_price[i]
                is_long = False
            # if long position - close it and go short
            elif is_long:
                profit = profit + (curr_price[i] - last_buy)
                position = profit
                last_buy = curr_price[i]
                is_long = False
                count += 1
            elif not is_long:
                position = profit + (last_buy - curr_price[i])

        profits.append(position)

    profit = position
    count += 1

    return profit, profits, profit*100/start, count/len(curr_price)


if __name__ == '__main__':
    # sp500 = get_sp500_curr_stock_symbols()
    # save_stock_pulled('all_stocks_last_1yr', sp500, '2020-04-06', '2021-04-06')
    print(long_short_profit_evaluation([5, 15, 25, 35], [1, -1, -1, 1]))
