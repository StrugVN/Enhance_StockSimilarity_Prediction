from util.misc import *
from config import *
from data_processing import *

df = read_data('data/' + data_name + '.csv')
stocks = ["GOOGL"]
result_file = 'GOOGL_benchmark.csv'


s = stocks[0]

evals_list = []

n_fold = 5
target_stock_dates = df[df[const_name_col] == s][const_time_col]
folds_df = []
_folds_time = []  # for debugging
threshold = int(len(target_stock_dates) / n_fold)
for i in range(n_fold - 1):
    fold_time = target_stock_dates[i * threshold: (i + 1) * threshold - 1]
    _folds_time.append((i * threshold, (i + 1) * threshold - 1))
    fold_df_ = df[df[const_time_col].isin(fold_time)]
    folds_df.append(fold_df_)

folds_df.append(df[df[const_time_col].isin(target_stock_dates[(n_fold - 1) * threshold:])])
_folds_time.append(((n_fold - 1) * threshold, 'end'))

f_count = 1

for _df in folds_df:
    # feature_df, scaler, scaler_cols = cal_financial_features(_df, StandardScaler())

    stock_df = _df[_df[const_name_col] == s]
    all_stock_df = _df[_df[const_time_col].isin(stock_df[const_time_col])]

    stock_count = all_stock_df.groupby([const_name_col]).count()[const_time_col].reset_index()

    top_stock_norm = None
    all_stock_name = stock_count[stock_count[const_time_col] > 30 + 7 + 5][const_name_col].tolist()

    # split dataset
    train_df, test_df = split_train_test_set(_df, s, all_stock_name, 0.75)
    # Prepare X, Y
    train_X, train_Y, train_price_Y, bin_train_Y, _, scaler, scaler_cols, transformer = \
        prepare_train_test_data(train_df, 'Close_norm', s, 1, 1,
                                'Close_norm', top_stock_norm, weighted_sampling=True,
                                norm_func=StandardScaler(), trans_func=None)

    test_X, test_Y, test_price_Y, bin_test_Y, test_t0_price, _, _, _ = \
        prepare_train_test_data(test_df,
                                'Close_norm', s, 1, 1, 'Close_norm', top_stock_norm,
                                is_test=True, norm_func=scaler, trans_func=transformer)


    bin_pred_Y = np.repeat(1, test_X.shape[0])

    # Trading benchmark
    evals = dict()

    inverted_t0_price = inverse_scaling('Close_norm', [test_t0_price], scaler_cols, scaler).tolist()[0]
    curr_price = test_price_Y.tolist()[:-1]
    curr_price.insert(0, inverted_t0_price)

    evals['fold'] = f_count

    evals["long_short_profit"], profits, evals["profit_%"], evals["order_count"] = \
        long_short_profit_evaluation(curr_price, bin_pred_Y)

    evals["sharpe_ratio"] = np.mean(profits) / (np.std([profits]) + 0.0001)

    print({key: round(evals[key], 3) if not isinstance(evals[key], str) else evals[key] for key in evals})

    evals_list.append(evals)
    f_count += 1

eval_df = pd.DataFrame(evals_list)
print(np.mean(eval_df))
input()
