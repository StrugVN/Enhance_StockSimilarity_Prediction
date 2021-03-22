import time

from data_processing import *
from util.misc import *
from config import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score, mean_squared_error


def run_exp(stock_list, target_col, sim_func, fix_len_func, k, next_t, selected_features,
            window_len, model_name, eval_result_path):
    is_classifier = ('Classifier' in model_name)

    df = read_data('data/' + data_name + '.csv')

    feature_df, scaler, scaler_cols = cal_financial_features(df, StandardScaler())  # get feature all stock

    """
    target_stock_dates = df[df[const_name_col] == stock][const_time_col]
    threshold = int(len(target_stock_dates)/5)

    folds_df = []
    for i in range(n_fold-1):
        fold_time = target_stock_dates[:(i+1)*threshold - 1]
        fold_df_ = df[df[const_time_col].isin(fold_time)]
        folds_df.append(fold_df_)
    folds_df.append(df[df[const_time_col].isin(target_stock_dates)])
    """

    evals_list = []
    for stock in stock_list:
        print('     -------- {0} --------'.format(stock))

        stock_df = feature_df[feature_df[const_name_col] == stock]
        all_stock_df = feature_df[feature_df[const_time_col].isin(stock_df[const_time_col])]  # time join

        stock_count = all_stock_df.groupby([const_name_col]).count()[const_time_col].reset_index()

        # sim
        sim_path = 'similarities_data/5_years_' + stock + '_' + \
                   target_col + '_' + sim_func + '_' + fix_len_func + '.pkl'
        if os.path.isfile(sim_path):
            print('Loading existing similarity result: ' + sim_path)
            _sim_data = pickle.load(open(sim_path, 'rb'))
            all_stock_name, similarities = _sim_data
        else:
            print('Calc similarities: ' + sim_path)
            all_stock_name = stock_count[stock_count[const_time_col] > 30 + 7 + 5][const_name_col].tolist()

            similarities = cal_other_stock_similarity(
                feature_df, stock, all_stock_name,
                similarity_func=similarity_funcs[sim_func],
                fix_len_func=fix_length_funcs[fix_len_func],
                similarity_col=target_col
            )

            print(' Saving new similarity result')
            pickle.dump((all_stock_name, similarities), open(sim_path, 'wb+'))

        # top k stocks
        top_k_stocks = get_top_k(all_stock_name, similarities, k)
        # normalize similarity
        top_stock_norm = normalize_similarity(top_k_stocks, stock)

        # split dataset
        train_df, test_df = split_train_test_set(feature_df, stock, all_stock_name, 0.7)
        # Prepare X, Y
        train_X, train_Y = prepare_train_test_data(train_df, selected_features, stock, window_len, next_t,
                                                   target_col, top_stock_norm, weighted_sampling=True)

        bin_train_Y = get_y_bin(train_X, train_Y.to_numpy(), window_len, target_col)

        test_X, test_Y = prepare_train_test_data(test_df, selected_features, stock, window_len, next_t,
                                                 target_col, top_stock_norm, is_test=True)

        bin_test_Y = get_y_bin(test_X, test_Y.to_numpy(), window_len, target_col)
        inverted_test_Y = inverse_scaling(target_col, test_Y.iloc[:, 0].to_numpy(), scaler_cols, scaler)

        if model_name not in fit_model_funcs.keys():
            raise ValueError(model_name + ' is not available')

        if not is_classifier:
            model = fit_model_funcs[model_name](train_X, train_Y)

            if 'LSTM' in model_name:
                pred_Y = model.predict(test_X.to_numpy().reshape(-1, 1, test_X.shape[1]))
            else:
                pred_Y = model.predict(test_X)
        else:
            train_Y_ = np.array(bin_train_Y)

            model = fit_model_funcs[model_name](train_X, train_Y_)

            pred_Y = model.predict(test_X)

        # Error cal
        evals = dict()
        if not is_classifier:
            inverted_pred_Y = inverse_scaling(target_col, pred_Y, scaler_cols, scaler)
            evals['rmse'] = np.sqrt(mean_squared_error(inverted_test_Y, inverted_pred_Y))

            bin_pred_Y = get_y_bin(test_X, pred_Y, window_len, target_col)
        else:
            bin_test_Y = np.array(bin_test_Y)
            bin_pred_Y = pred_Y

        evals['accuracy_score'] = accuracy_score(bin_test_Y, bin_pred_Y)
        evals['f1_score'] = f1_score(bin_test_Y, bin_pred_Y, average='macro')
        evals['precision_score'] = precision_score(bin_test_Y, bin_pred_Y, average='macro')

        evals["long_short_profit"], profits = long_short_profit_evaluation(inverted_test_Y.tolist(), bin_pred_Y)
        evals["sharp_ratio"] = np.mean(profits) / (np.std([profits]) + 0.0001)
        print(evals)
        evals_list.append(evals)

    # Save evaluation
    eval_df = pd.DataFrame(evals_list)
    """
    No. of features, selected_features, sim_func, fix_len_func, k stock, window_len, next_t, model, 
     mean_accuracy, std_accuracy, mean_f1, std_f1, mean_mse, std_mse, mean_sharp_ratio, mean_profit
    """
    text_selected_ft = str(selected_features).replace(',', ';')

    mean_accuracy, std_accuracy = np.round((np.mean(eval_df['accuracy_score']), np.std(eval_df['accuracy_score'])), 4)
    mean_f1, std_f1 = np.round((np.mean(eval_df['f1_score']), np.std(eval_df['f1_score'])), 4)
    if not is_classifier:
        mean_mse, std_mse = np.round((np.mean(eval_df['rmse']), np.std(eval_df['rmse'])), 3)
    else:
        mean_mse, std_mse = 'NaN', 'NaN'

    mean_sharp_ratio, mean_profit = np.round((np.mean(eval_df['sharp_ratio']),
                                              np.mean(eval_df['long_short_profit'])), 3)

    if not os.path.isfile(eval_result_path):
        with open(eval_result_path, "w") as file:
            file.write("No. of features, selected_features, sim_func, fix_len_func, k stock, window_len, next_t, "
                       "model, mean_accuracy, std_accuracy, mean_f1, std_f1, mean_rmse, std_rmse, mean_sharp_ratio, "
                       "mean_profit\n")
            file.close()

    with open(eval_result_path, "a") as file:
        file.write("{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}, {15}\n".format(
            len(selected_features), text_selected_ft, sim_func, fix_len_func, k, window_len, next_t, model_name,
            mean_accuracy, std_accuracy, mean_f1, std_f1, mean_mse, std_mse, mean_sharp_ratio, mean_profit))
        file.close()


""" Experience """


def sim_func_test1():  # Test các hàm tđ với k = [5 15 25], 10 ngày - 1 feature
    run_param = base_param
    run_param['eval_result_path'] = 'Sim_func_test.csv'
    run_param['window_len'] = 10
    run_param['model_name'] = 'LSTM'
    run_param['selected_features'] = ['Close_norm']
    run_param['next_t'] = 1

    k = [15, 25]

    for _k in k:
        run_param['k'] = _k
        for sim_f_name in similarity_funcs:
            run_param['sim_func'] = sim_f_name
            for fix_f_name in fix_length_funcs:
                run_param['fix_len_func'] = fix_f_name
                print('Run {0}, {1}, {2}'.format(_k, sim_f_name, fix_f_name))
                run_exp(**run_param)


# sim_func_test1()
test = base_param
test['window_len'] = 10
test['eval_result_path'] = 'Model_test.csv'
test['selected_features'] = ['Close_norm']
test['next_t'] = 1

selected_params = [['dtw', 'time_join', 5], ['euclidean', 'time_join', 5],
                 ['dtw', 'time_join', 15], ['euclidean', 'time_join', 15],
                 ['co-integration', 'pip', 25], ['euclidean', 'pip', 25]]

for params in selected_params:
    test['sim_func'] , test['fix_len_func'], test['k'] = params
    for m in fit_model_funcs.keys():
        test['model_name'] = m
        print('====== Run {} with {}, {}, {} ========='.format(m, test['sim_func'] , test['fix_len_func'], test['k']))
        run_exp(**test)
