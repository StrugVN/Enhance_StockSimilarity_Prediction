from copy import copy
import os
import pickle

from financial_features import *

from similarity_functions import *


def cal_other_stock_similarity(df_stocks, stock_to_compare, stock_names, similarity_file_path, similarity_func,
                               fix_len_func=time_join, similarity_col=const_target_col):
    if os.path.isfile(similarity_file_path):
        print('Loading existing similarity result: ' + similarity_file_path)
        similarities = pickle.load(open(similarity_file_path, 'rb'))
    else:
        print('Calc similarities: ' + similarity_file_path)
        similarities = [
            similarity_func(df_stocks[df_stocks[const_name_col] == stock_to_compare],
                            df_stocks[df_stocks[const_name_col] == stock_name], fix_len_func,
                            similarity_col)
            for stock_name in stock_names
        ]

        print(' Saving new similarity result')
        pickle.dump(similarities, open(similarity_file_path, 'wb+'))

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
    feature_df = data[[const_time_col, const_name_col, const_target_col]].copy()

    numeric_cols = data.select_dtypes(
        include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.tolist()
    # for c in numeric_cols:
    #    feature_df[c + '_proc'] = PROC(data[c])
    feature_df['Close_proc'] = PROC(data['Close'])

    feature_df['rsi'] = rsiFunc(data['Close'])  # Relative strength index
    feature_df['MACD'] = computeMACD(data['Close'])[2]  # Moving Average Convergence/Divergence
    feature_df['Open_Close_diff'] = data['Open'] - data['Close']
    feature_df['High_Low_diff'] = data['High'] - data['Low']

    if norm_func is not None:  # Normalize
        scaler = copy(norm_func)
        features = ['rsi', 'MACD', 'Open_Close_diff', 'High_Low_diff']

        df = data.copy()
        df[features] = feature_df[features]
        scaler.fit(df[numeric_cols + features])
        data_norm = scaler.transform(df[numeric_cols + features])

        data_norm_df = pd.DataFrame(data_norm, columns=[s + '_norm' for s in numeric_cols + features])
        data_norm_df[const_time_col] = df.index
        data_norm_df = data_norm_df.set_index(const_time_col)

        feature_df = pd.concat([feature_df, data_norm_df], axis=1)

        return feature_df, scaler, [s + '_norm' for s in numeric_cols + features]

    return feature_df


def split_train_test_set(data, stock, stock_names, ratio):
    stock_times = data[data[const_name_col] == stock][const_time_col].tolist()  # List of date

    train_len = int(len(stock_times) * ratio)

    train_start, train_end = stock_times[0], stock_times[train_len]
    test_start, test_end = stock_times[train_len + 1], stock_times[-1]

    train_df = data[(train_start <= data[const_time_col])
                    & (data[const_time_col] < train_end) & data[const_name_col].isin(stock_names)]
    train_comparing_stock_df = train_df[train_df[const_name_col] == stock][const_time_col]
    train_df = train_df[train_df[const_time_col].isin(train_comparing_stock_df)]

    test_df = data[(test_start <= data[const_time_col])
                   & (data[const_time_col] < test_end) & data[const_name_col].isin(stock_names)]
    test_comparing_stock_df = test_df[test_df[const_name_col] == stock][const_time_col]
    test_df = test_df[test_df[const_time_col].isin(test_comparing_stock_df)]

    return train_df, test_df


def prepare_time_point(data, selected_features, next_t, target_col):
    X_df = data[selected_features].iloc[:-next_t].copy()

    Y = []
    y = np.array(data[target_col].tolist())

    for i in range(0, len(data[target_col]) - next_t):
        y_ti = i + next_t
        next_y = y[y_ti].tolist()

        to_dict_y = {}
        to_dict_y[next_t] = next_y
        Y.append(to_dict_y)

    Y_df = pd.DataFrame(Y, index=data.index.values[:len(data[target_col]) - next_t])

    return X_df, Y_df


def prepare_time_window(data, selected_features, w_len, next_t, target_col):
    X = []
    Y = []
    y = np.array(data[target_col].tolist())

    for i in range(0, len(data[target_col]) - w_len + 1 - next_t):
        y_ti = i + w_len - 1 + next_t
        # Y
        next_y = y[y_ti].tolist()
        Y_period = {}
        Y_period[str(next_t)] = next_y
        Y.append(Y_period)

        # X
        X_period = data[i:i + w_len]
        X_period.insert(0, 'i', range(w_len))  # 1 đoạn window_len
        period_time = X_period.index.values[-1]

        X_period = X_period[selected_features + ['i'] + [const_name_col]].pivot(index=const_name_col, columns='i')
        X_period_dict = X_period.iloc[0].to_dict()
        X_period_dict[const_time_col] = period_time
        X.append(X_period_dict)

    X_df = pd.DataFrame(X).set_index(const_time_col)
    Y_df = pd.DataFrame(Y, index=X_df.index)

    return X_df, Y_df


def prepare_train_test_data(data, selected_features, comparing_stock, w_len, next_t, target_col,
                            top_stock, weighted_sampling=False, is_test=False):
    if w_len > 1:
        X_df, Y_df = prepare_time_window(data[data[const_name_col] == comparing_stock],
                                         selected_features, w_len, next_t, target_col)
    else:
        X_df, Y_df = prepare_time_point(data[data[const_name_col] == comparing_stock],
                                        selected_features, next_t, target_col)
    if not is_test and top_stock is not None:
        for stock_name in top_stock.keys():
            stock_df = data[data[const_name_col] == stock_name]
            if stock_df.empty:
                continue
            if w_len > 1:
                sim_stock_X, sim_stock_Y = prepare_time_window(stock_df,
                                                               selected_features, w_len, next_t, target_col)
            else:
                sim_stock_X, sim_stock_Y = prepare_time_point(stock_df,
                                                              selected_features, next_t, target_col)

            if weighted_sampling:
                np.random.seed(0)
                msk = (np.random.rand(len(sim_stock_X)) < top_stock[stock_name])
                sim_stock_X = sim_stock_X[msk]
                sim_stock_Y = sim_stock_Y[msk]

            X_df = pd.concat([X_df, sim_stock_X])
            Y_df = pd.concat([Y_df, sim_stock_Y])

    return X_df, Y_df
