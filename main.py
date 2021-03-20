import pandas as pd

from data_processing import *
from util.misc import *
from config import *
from sklearn.preprocessing import StandardScaler


def run_exp(stock, target_col, sim_func, fix_len_func, k, next_t, selected_features, window_len, model_name):
    df = read_data('data/' + data_name + '.csv')

    feature_df, scaler, scaler_cols = cal_financial_features(df, StandardScaler())  # get feature all stock

    stock_df = feature_df[feature_df[const_name_col] == stock]
    all_stock_df = feature_df[feature_df[const_time_col].isin(stock_df[const_time_col])]  # time join

    stock_count = all_stock_df.groupby([const_name_col]).count()[const_time_col].reset_index()

    all_stock_name = stock_count[stock_count[const_time_col] > 5 + next_t + 30][const_name_col].tolist()

    # use euclidean, time_join on feature Close_norm
    sim_path = 'sim_data/' + stock + '_v_' + str(len(all_stock_name)) + 'stocks_' + \
               target_col + '_' + sim_func + '_' + fix_len_func + '.pkl'
    similarities = cal_other_stock_similarity(
        feature_df, stock, all_stock_name,
        similarity_file_path=sim_path,
        similarity_func=similarity_funcs[sim_func],
        fix_len_func=fix_length_funcs[fix_len_func],
        similarity_col=target_col
    )

    # top k stocks
    top_k_stocks = get_top_k(all_stock_name, similarities, k)
    # normalize similarity
    top_stock_norm = normalize_similarity(top_k_stocks, stock)

    # split dataset
    train_df, test_df = split_train_test_set(feature_df, stock, all_stock_name, 0.7)
    # Prepare X, Y
    train_X, train_Y = prepare_train_test_data(train_df, selected_features, stock, window_len, next_t,
                                               target_col, top_stock_norm, weighted_sampling=True)
    test_X, test_Y = prepare_train_test_data(test_df, selected_features, stock, window_len, next_t,
                                             target_col, top_stock_norm, is_test=True)

    if model_name not in fit_model_funcs.keys():
        raise ValueError(model_name + ' is not available')

    model = fit_model_funcs[model_name](train_X, train_Y)

    if model_name == 'LSTM':
        pred_Y = model.predict(test_X.to_numpy().reshape(-1, 1, test_X.shape[1]))
    else:
        pred_Y = model.predict(test_X)

    # Error cal

    print('End')


""" TEST FLOW """
run_exp(**exp_param)

"""
# Error cal
error_p = (pred_Y_p - test_Y_p)

mse_p = np.mean(error_p ** 2)

close_norm_test_X = pd.DataFrame()
close_norm_test_X['1'], close_norm_test_X['3'], close_norm_test_X['7'] = [test_X_p['Close_norm']] * 3
bool_test_Y_p = (test_Y_p - close_norm_test_X) > 0
bool_pred_Y_p = (pred_Y_p - close_norm_test_X) > 0

bool_error_p = bool_test_Y_p & bool_pred_Y_p
"""
