from copy import copy
import os
import pickle

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from financial_features import *

from similarity_functions import *


def cal_other_stock_similarity(df_stocks, stock_to_compare, stock_names, similarity_func, proc_w,
                               fix_len_func=time_join, similarity_col=const_target_col):
    similarities = []
    stock_df = df_stocks[df_stocks[const_name_col] == stock_to_compare]
    stock_df, _, _ = cal_financial_features(stock_df, proc_w, StandardScaler())

    for stock_name in stock_names:
        other_stock_df = df_stocks[df_stocks[const_name_col] == stock_name]
        other_stock_df, _, _ = cal_financial_features(other_stock_df, proc_w, StandardScaler())

        similarities.append(similarity_func(stock_df, other_stock_df, fix_len_func, similarity_col))

    # similarities = [
    #    similarity_func(df_stocks[df_stocks[const_name_col] == stock_to_compare],
    #                    df_stocks[df_stocks[const_name_col] == stock_name], fix_len_func,
    #                    similarity_col)
    #    for stock_name in stock_names
    # ]

    return similarities


def normalize_similarity(top_stocks, stock_to_compare):
    stocks_val = list(top_stocks.values())
    top_stock_w = {}
    sum_vals = 0
    for stock_k, v in top_stocks.items():
        if stock_k != stock_to_compare:
            if v != 0:
                top_stock_w[stock_k] = np.abs(float(v) - max(stocks_val)) / (max(stocks_val) - min(stocks_val))
                sum_vals += top_stock_w[stock_k]
            else:
                top_stock_w[stock_k] = 0

    if sum_vals != 0:
        for stock_k, v in top_stock_w.items():
            top_stock_w[stock_k] = top_stock_w[stock_k] / sum_vals

    return top_stock_w


def cal_financial_features(data, proc_w, norm_func=None, next_t=1, re_fit=True):
    feature_df = data[[const_time_col, const_name_col, const_target_col]].copy()

    numeric_cols = data.select_dtypes(
        include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.tolist()
    # for c in numeric_cols:
    #    feature_df[c + '_proc'] = PROC(data[c])
    feature_df['Close_proc'] = PROC(data['Close'], proc_w,  next_t)

    feature_df['rsi'] = rsiFunc(data['Close'])  # Relative strength index
    feature_df['MACD'] = computeMACD(data['Close'])[2]  # Moving Average Convergence/Divergence
    feature_df['MA'] = movingAverage(data['Close'])  # feature_df['MA'] = data['Close'].rolling(period).mean()
    feature_df['Open_Close_diff'] = data['Open'] - data['Close']
    feature_df['High_Low_diff'] = data['High'] - data['Low']

    if norm_func is not None:  # Normalize
        scaler = copy(norm_func)
        features = ['rsi', 'MACD', 'MA']

        df = data.copy()
        df[features] = feature_df[features]
        if re_fit:
            scaler.fit(df[numeric_cols + features])
        data_norm = scaler.transform(df[numeric_cols + features])

        data_norm_df = pd.DataFrame(data_norm, columns=[s + '_norm' for s in numeric_cols + features])
        data_norm_df[const_time_col] = df.index
        data_norm_df = data_norm_df.set_index(const_time_col)
        data_norm_df['Open_Close_diff_norm'] = data_norm_df['Open_norm'] - data_norm_df['Close_norm']
        data_norm_df['High_Low_diff_norm'] = data_norm_df['High_norm'] - data_norm_df['Low_norm']

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


def prepare_time_point(data, selected_features, next_t, target_col, proc_w,
                       norm_func=None, trans_func=None, re_fit=True):
    data, scaler, scaler_col = cal_financial_features(data, proc_w, norm_func, next_t, re_fit)
    if 'Close_proc' in selected_features:
        data = data.iloc[next_t:]

    X_df = data[selected_features].iloc[:-next_t].copy()

    Y = []
    y = np.array(data[target_col].tolist())
    Price = []
    price = np.array(data['Close'].tolist())
    Proc = []
    proc = np.array(data['Close_proc'].tolist())

    t0_price = data['Close_norm'][0]

    for i in range(0, len(data[target_col]) - next_t):
        y_ti = i + next_t

        Y.append({next_t: y[y_ti].tolist()})

        # price
        Price.append({next_t: price[y_ti].tolist()})

        # bin_proc
        next_b = np.sign(proc[y_ti].tolist())
        if next_b == 0:
            Proc.append({next_t: 1})
        else:
            Proc.append({next_t: next_b})

    Y_df = pd.DataFrame(Y, index=data.index.values[:len(data[target_col]) - next_t])
    Price_df = pd.DataFrame(Price, index=Y_df.index)
    Proc_df = pd.DataFrame(Proc, index=Y_df.index)

    if trans_func is not None:
        transformer = copy(trans_func)
        if re_fit:
            transformer.fit(X_df)
        X_transformed = transformer.transform(X_df)

        cols = [i for i in range(X_transformed.shape[1])]
        if transformer.__class__.__name__ == PCA().__class__.__name__:
            cols = ['pca_{}'.format(i) for i in cols]
        elif transformer.__class__.__name__ == SAX().__class__.__name__:
            cols = X_df.columns

        X_transformed_df = pd.DataFrame(X_transformed, columns=cols,
                                        index=X_df.index)

        X_df = X_transformed_df

        return X_df, Y_df, Price_df, Proc_df, t0_price, scaler, scaler_col, transformer

    return X_df, Y_df, Price_df, Proc_df, t0_price, scaler, scaler_col, None


def prepare_time_window(data, selected_features, w_len, next_t, target_col, proc_w,
                        norm_func=None, trans_func=None, re_fit=True):
    data, scaler, scaler_col = cal_financial_features(data, proc_w, norm_func, next_t, re_fit)

    if 'Close_proc' in selected_features:
        data = data.iloc[next_t:]

    X = []
    Y = []
    y = np.array(data[target_col].tolist())
    Price = []
    price = np.array(data['Close'].tolist())
    Proc = []
    proc = np.array(data['Close_proc'].tolist())

    t0_price = data['Close_norm'][w_len - 1]

    for i in range(0, len(data[target_col]) - w_len + 1 - next_t):
        y_ti = i + w_len - 1 + next_t
        # Y
        next_y = y[y_ti].tolist()
        Y_period = {str(next_t): next_y}
        Y.append(Y_period)

        # X
        X_period = data[i:i + w_len]
        X_period.insert(0, 'i', range(w_len))  # 1 ??o???n window_len
        period_time = X_period.index.values[-1]

        X_period = X_period[selected_features + ['i'] + [const_name_col]].pivot(index=const_name_col, columns='i')
        X_period_dict = X_period.iloc[0].to_dict()
        X_period_dict[const_time_col] = period_time
        X.append(X_period_dict)

        # price
        Price.append({next_t: price[i + w_len].tolist()})

        # bin_proc
        next_b = np.sign(proc[y_ti].tolist())
        if next_b == 0:
            Proc.append({next_t: 1})
        else:
            Proc.append({next_t: next_b})

    X_df = pd.DataFrame(X).set_index(const_time_col)
    Y_df = pd.DataFrame(Y, index=X_df.index)
    Price_df = pd.DataFrame(Price, index=X_df.index)
    Proc_df = pd.DataFrame(Proc, index=X_df.index)

    if trans_func is not None:
        transformer = copy(trans_func)
        if re_fit:
            transformer.fit(X_df)
        X_transformed = transformer.transform(X_df)

        cols = [i for i in range(X_transformed.shape[1])]
        if transformer.__class__.__name__ == PCA().__class__.__name__:
            cols = ['pca_{}'.format(i) for i in cols]
        elif transformer.__class__.__name__ == SAX().__class__.__name__:
            cols = X_df.columns

        X_transformed_df = pd.DataFrame(X_transformed, columns=cols,
                                        index=X_df.index)

        X_df = X_transformed_df

        return X_df, Y_df, Price_df, Proc_df, t0_price, scaler, scaler_col, transformer

    return X_df, Y_df, Price_df, Proc_df, t0_price, scaler, scaler_col, None


def prepare_train_test_data(data, selected_features, comparing_stock, w_len, next_t, target_col,
                            top_stock, proc_w, weighted_sampling=False, is_test=False,
                            norm_func=None, trans_func=None):
    if w_len > 1:
        X_df, Y_df, Prices_df, Proc_df, t_0, scaler, scaler_cols, transformer = prepare_time_window(
            data[data[const_name_col] == comparing_stock],
            selected_features, w_len, next_t, target_col, proc_w, norm_func, trans_func, re_fit=not is_test)
    else:
        X_df, Y_df, Prices_df, Proc_df, t_0, scaler, scaler_cols, transformer = prepare_time_point(
            data[data[const_name_col] == comparing_stock],
            selected_features, next_t, target_col, proc_w, norm_func, trans_func, re_fit=not is_test)

    if not is_test and top_stock is not None:
        _scaler, _transformer = None, None
        if norm_func is not None and norm_func.__class__.__name__ == StandardScaler().__class__.__name__:
            _scaler = StandardScaler()

        if trans_func is not None:
            if trans_func.__class__.__name__ == PCA().__class__.__name__:
                _transformer = PCA(n_components=3, random_state=0)
            elif trans_func.__class__.__name__ == SAX().__class__.__name__:
                _transformer = SAX()

        for stock_name in top_stock.keys():
            stock_df = data[data[const_name_col] == stock_name]
            min_data_point = 1
            if _transformer.__class__.__name__ == PCA().__class__.__name__:
                min_data_point += 3
            if stock_df.empty or stock_df.shape[0] < w_len + next_t + min_data_point:
                continue
            if w_len > 1:
                sim_stock_X, sim_stock_Y, sim_stock_Prices, sim_stock_Proc, _, _, _, _ = \
                    prepare_time_window(stock_df, selected_features, w_len, next_t,
                                        target_col, proc_w, _scaler, _transformer)
            else:
                sim_stock_X, sim_stock_Y, sim_stock_Prices, sim_stock_Proc, _, _, _, _ = \
                    prepare_time_point(stock_df, selected_features, next_t,
                                       target_col, proc_w, _scaler, _transformer)

            if weighted_sampling:
                np.random.seed(0)
                msk = (np.random.rand(len(sim_stock_X)) < top_stock[stock_name])
                sim_stock_X = sim_stock_X[msk]
                sim_stock_Y = sim_stock_Y[msk]
                sim_stock_Prices = sim_stock_Prices[msk]
                sim_stock_Proc = sim_stock_Proc[msk]

            X_df = pd.concat([sim_stock_X, X_df])
            Y_df = pd.concat([sim_stock_Y, Y_df])
            Prices_df = pd.concat([sim_stock_Prices, Prices_df])
            Proc_df = pd.concat([sim_stock_Proc, Proc_df])

    return X_df, Y_df, Prices_df[next_t], Proc_df[next_t].to_numpy(), t_0, scaler, scaler_cols, transformer
