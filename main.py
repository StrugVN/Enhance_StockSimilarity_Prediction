import pandas as pd
from sklearn.ensemble import *

from data_processing import *
from similarity_functions import *
from sklearn.preprocessing import StandardScaler


def read_data(path):
    all_stocks = pd.read_csv(path)
    all_stocks[const_time_col] = pd.to_datetime(all_stocks[const_time_col], format='%Y-%m-%d', errors='ignore')
    all_stocks = all_stocks.dropna(axis=0)
    all_stocks = all_stocks.set_index('Date', drop=False)
    return all_stocks


""" TEST FLOW """
df = read_data('data/' + data_name + '.csv')
stock = 'GOOGL'
k = 5

# get feature all stock
feature_df, scaler = cal_financial_features(df, StandardScaler())

# cal sim
stock_df = feature_df[feature_df[const_name_col] == stock]
all_stock_df = feature_df[feature_df[const_time_col].isin(stock_df[const_time_col])]  # time join

stock_count = all_stock_df.groupby([const_name_col]).count()[const_time_col].reset_index()  # Số điểm dữ liệu các stock

all_stock_name = stock_count[stock_count[const_time_col] > 5 + 7 + 30][const_name_col].tolist()
# stock có ít nhất x điểm dữ liệu sau time join, x dựa vào time window, số ngày dự đoán, ...

# calculate similarities
target_col = 'Close_norm'
sim_func = apply_euclidean
fix_len_func = time_join
# use euclidean, time_join on feature Close_norm
similarities = cal_other_stock_similarity(feature_df, stock, all_stock_name[:50],
                                          similarity_func=sim_func,
                                          fix_len_func=fix_len_func,
                                          similarity_col=target_col)
# top k stocks
top_k_stocks = get_top_k(all_stock_name, similarities, k)
# normalize similarity
top_stock_norm = normalize_similarity(top_k_stocks, stock)

# split dataset
train_df, test_df = split_train_test_set(feature_df, stock, all_stock_name, 0.7)

next_t = [1, 3, 7]
# Prepare X, Y
## Time window
window_len = 5
selected_features = ['Close_norm']

train_X_w, train_Y_w = prepare_train_test_data(train_df, selected_features, stock, window_len, next_t,
                                               target_col, top_stock_norm, weighted_sampling=True)
test_X_w, test_Y_w = prepare_train_test_data(test_df, selected_features, stock, window_len, next_t,
                                             target_col, top_stock_norm, is_test=True)
print(len(train_X_w), len(train_Y_w), len(test_X_w), len(test_Y_w))

## Time point
window_len = 0
selected_features = ['Close', 'Close_norm', 'Close_proc', 'rsi_norm', 'MACD_norm']
train_X_p, train_Y_p = prepare_train_test_data(train_df, selected_features, stock, window_len, next_t,
                                               target_col, top_stock_norm, weighted_sampling=True)
test_X_p, test_Y_p = prepare_train_test_data(test_df, selected_features, stock, window_len, next_t,
                                             target_col, top_stock_norm, is_test=True)
print(len(train_X_p), len(train_Y_p), len(test_X_p), len(test_Y_p))

# Modeling
model = RandomForestRegressor(n_estimators=100, random_state=0)

model.fit(train_X_p, train_Y_p)

pred_Y_p = model.predict(test_X_p)



# Error cal
error_p = (pred_Y_p - test_Y_p)

mse_p = np.mean(error_p**2)

close_norm_test_X = pd.DataFrame()
close_norm_test_X['1'], close_norm_test_X['3'], close_norm_test_X['7'] = [test_X_p['Close_norm']]*3
bool_test_Y_p = (test_Y_p - close_norm_test_X) > 0
bool_pred_Y_p = (pred_Y_p - close_norm_test_X) > 0

bool_error_p = bool_test_Y_p & bool_pred_Y_p



print('End')
