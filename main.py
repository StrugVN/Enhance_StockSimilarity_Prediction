import pandas as pd

from data_processing import *
from Const import *
from sklearn.preprocessing import StandardScaler


def read_data(path):
    all_stocks = pd.read_csv(path)
    all_stocks['Date'] = pd.to_datetime(all_stocks['Date'], format='%Y-%m-%d', errors='ignore')
    all_stocks = all_stocks.dropna(axis=0)
    all_stocks = all_stocks.set_index('Date', drop=False)
    return all_stocks


""" TEST """
df = read_data('data/all_stocks_5yr.csv')
stock = 'GOOGL'
k = 10

# get feature all stock
feature_df = cal_financial_features(df, StandardScaler())

# cal sim
stock_df = df[df[name_col] == stock]
all_stock_df = df[df[time_col].isin(stock_df[time_col])]  # time join

stock_count = all_stock_df.groupby([name_col]).count()[time_col].reset_index()  # Số điểm dữ liệu các stock

other_stocks = stock_count[stock_count[time_col] > 30][name_col].tolist()
# stock có ít nhất x điểm dữ liệu sau time join, x dựa vào time window, số ngày dự đoán, ...

""" 
**Chỉ tính tương đồng các stock trong other_stocks**

similarity = dict stock: kq tương đồng 
top_k_stock = chọn ra k stock
normalize kết quả

"""
# split dataset
stock_times = df[df[name_col] == stock][time_col].tolist()  # List of date

train_len = int(len(stock_times) * 70 / 100)

train_start, train_end = stock_times[0], stock_times[train_len]
test_start, test_end = stock_times[train_len + 1], stock_times[-1]

train_df = feature_df[(train_start <= feature_df[time_col])
                      & (feature_df[time_col] < train_end) & feature_df[name_col].isin(other_stocks)]
train_comparing_stock_df = train_df[train_df[name_col] == stock][time_col]
train_df = train_df[train_df[time_col].isin(train_comparing_stock_df)]

test_df = feature_df[(test_start <= feature_df[time_col])
                     & (feature_df[time_col] < test_end) & feature_df[name_col].isin(other_stocks)]
test_comparing_stock_df = test_df[test_df[name_col] == stock][time_col]
test_df = test_df[test_df[time_col].isin(test_comparing_stock_df)]

# Prepare X, Y

## Time window
window_len = 5

## Time point
window_len = 0

""" Nháp
 

"""

print('End')
