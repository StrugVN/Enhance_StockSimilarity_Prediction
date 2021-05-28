import time
import os
from datetime import datetime

from data_processing import *
from util.misc import *
from config import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score, mean_squared_error


def run_exp(stock_list, target_col, sim_func, fix_len_func, k, next_t, selected_features,
            window_len, model_name, eval_result_path, n_fold, similarity_col, norm_func, trans_func):
    is_classifier = ('Classifier' in model_name)

    df = read_data('data/' + data_name + '.csv')

    evals_list = []
    for stock in stock_list:
        print('     -------- {0}, {1} folds --------'.format(stock, n_fold))

        # Create fold
        target_stock_dates = df[df[const_name_col] == stock][const_time_col]
        folds_df = []
        _folds_time = []  # for debugging
        # Large fold
        """
        threshold = int(len(target_stock_dates) / (n_fold + 1))
        for i in range(1, n_fold):
            fold_time = target_stock_dates[:(i + 1) * threshold - 1]
            _folds_time.append((i + 1) * threshold - 1)
            fold_df_ = df[df[const_time_col].isin(fold_time)]
            folds_df.append(fold_df_)
        folds_df.append(df[df[const_time_col].isin(target_stock_dates)])
        """

        # Small fold
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

            stock_df = _df[_df[const_name_col] == stock]
            all_stock_df = _df[_df[const_time_col].isin(stock_df[const_time_col])]

            stock_count = all_stock_df.groupby([const_name_col]).count()[const_time_col].reset_index()

            # sim

            if k == 0:
                top_stock_norm = None
                all_stock_name = stock_count[stock_count[const_time_col] > 30 + 7 + 5][const_name_col].tolist()
            else:

                force = False
                if data_name == 'all_stocks_5yr':
                    start_d = str(stock_df[const_time_col][0]).split(' ', 1)[0]
                    end_d = str(stock_df[const_time_col][-1]).split(' ', 1)[0]

                    sim_path = 'similarities_data/' + data_name + '/5_years_' + stock + '_' + similarity_col + '_' + \
                               sim_func + '_' + fix_len_func + '_fold_' + start_d + '_' + end_d + '.pkl'
                else:
                    start_d = str(stock_df[const_time_col][0]).replace(':', '').replace(' ', '_')
                    end_d = str(stock_df[const_time_col][-1]).replace(':', '').replace(' ', '_')

                    sim_path = 'similarities_data/' + data_name + '/' + stock + '_' + similarity_col + '_' + \
                               sim_func + '_' + fix_len_func + '_fold_' + start_d + '_' + end_d + '.pkl'

                if os.path.isfile(sim_path) and not force:
                    # print(' Loading existing similarity result: ' + sim_path)
                    _sim_data = pickle.load(open(sim_path, 'rb'))
                    all_stock_name, similarities = _sim_data
                else:
                    print(' Calc similarities: ' + sim_path)
                    all_stock_name = stock_count[stock_count[const_time_col] > 30 + 7 + 5][const_name_col].tolist()

                    similarities = cal_other_stock_similarity(
                        _df, stock, all_stock_name,
                        similarity_func=similarity_funcs[sim_func],
                        fix_len_func=fix_length_funcs[fix_len_func],
                        similarity_col=similarity_col
                    )

                    print(' Saving new similarity result')
                    pickle.dump((all_stock_name, similarities), open(sim_path, 'wb+'))

                # _sim_df = pd.DataFrame(np.array([all_stock_name, similarities]).transpose(), columns=['Stock', 'Sim'])

                # top k stocks
                top_k_stocks = get_top_k(all_stock_name, similarities, k)
                # normalize similarity
                top_stock_norm = normalize_similarity(top_k_stocks, stock)

            # split dataset
            train_df, test_df = split_train_test_set(_df, stock, all_stock_name, 0.75)

            # Prepare X, Y
            if trans_func is None:
                trans_name = 'None'
            else:
                trans_name = trans_func.__class__.__name__

            if len(selected_features) > 5:
                ft = len(selected_features)
            else:
                ft = str(selected_features).replace(',', ';')

            if not os.path.exists('train_test_data/' + data_name + '/' + sim_func + '/' + fix_len_func):
                os.makedirs('train_test_data/' + data_name + '/' + sim_func + '/' + fix_len_func)

            train_path = 'train_test_data/' + data_name + '/' + sim_func + '/' + fix_len_func + \
                         '/train_{0}_simcol={1}_k={2}_ft={3}_w={4}_pred={5}_t={6}_trans={7}_' \
                         'fold={8}.pkl'.format(stock, similarity_col, k,
                                               ft, window_len, target_col, next_t, trans_name, f_count)

            test_path = 'train_test_data/' + data_name + '/' + sim_func + '/' + fix_len_func + \
                        '/test_{0}_simcol={1}_k={2}_ft={3}_w={4}_pred={5}_t={6}_trans={7}_' \
                        'fold={8}.pkl'.format(stock, similarity_col, k,
                                              ft, window_len, target_col, next_t, trans_name, f_count)

            force = False
            proc_w = 1
            if trans_func.__class__.__name__ == SAX().__class__.__name__:
                #print('      fixing SAX and PROC')
                proc_w = 20
                #force = True

            if os.path.isfile(train_path) and not force:
                _train_data = pickle.load(open(train_path, 'rb'))
                train_X, train_Y, train_price_Y, bin_train_Y, scaler, scaler_cols, transformer = _train_data
            else:
                train_X, train_Y, train_price_Y, bin_train_Y, _, scaler, scaler_cols, transformer = \
                    prepare_train_test_data(train_df, selected_features, stock, window_len, next_t,
                                            target_col, top_stock_norm, proc_w, weighted_sampling=True,
                                            norm_func=norm_func, trans_func=trans_func)
                if k != 0:
                    if os.path.exists(train_path):
                        os.remove(train_path)
                    print('   Saving training data')
                    pickle.dump((train_X, train_Y, train_price_Y, bin_train_Y, scaler, scaler_cols, transformer),
                                open(train_path, 'wb+'))

            # if 'proc' not in target_col:
            #     bin_train_Y = get_y_bin(train_X, train_Y.to_numpy(), window_len, target_col)
            # else:
            #    bin_train_Y = np.sign(train_Y)
            if os.path.isfile(test_path) and not force:
                _test_data = pickle.load(open(test_path, 'rb'))
                test_X, test_Y, test_price_Y, bin_test_Y, test_t0_price = _test_data
            else:
                test_X, test_Y, test_price_Y, bin_test_Y, test_t0_price, _, _, _ = \
                    prepare_train_test_data(test_df,
                                            selected_features, stock, window_len, next_t, target_col, top_stock_norm,
                                            proc_w, is_test=True, norm_func=scaler, trans_func=transformer)
                if k != 0:
                    print('   Saving test data')
                    if os.path.exists(test_path):
                        os.remove(test_path)
                    pickle.dump((test_X, test_Y, test_price_Y, bin_test_Y, test_t0_price),
                                open(test_path, 'wb+'))

            ############# DELETE THIS ###########
            #f_count += 1
            #continue
            #####################################

            # if 'proc' not in target_col:
            #    bin_test_Y = get_y_bin(test_X, test_Y.to_numpy(), window_len, target_col)
            # else:
            #    bin_test_Y = np.sign(test_Y)

            if model_name not in fit_model_funcs.keys():
                raise ValueError(model_name + ' is not available')

            if not is_classifier:
                if 'LSTM' in model_name:
                    # -- Test --
                    # if 'proc' in target_col:
                    #    train_Y['1'] = train_Y['1']*100
                    #    test_Y['1'] = test_Y['1']*100
                    # ----------
                    model = fit_model_funcs[model_name](train_X, train_Y)

                    pred_Y = model.predict(test_X.to_numpy().reshape(-1, 1, test_X.shape[1]))
                else:
                    model = fit_model_funcs[model_name](train_X, train_Y)
                    pred_Y = model.predict(test_X)
            else:
                train_Y_ = np.array(bin_train_Y)

                model = fit_model_funcs[model_name](train_X, train_Y_)

                pred_Y = model.predict(test_X)

            # Error cal
            evals = dict()
            if not is_classifier:
                if 'proc' not in target_col:
                    inverted_pred_Y = inverse_scaling(target_col, pred_Y, scaler_cols, scaler)
                    evals['rmse'] = np.sqrt(mean_squared_error(test_Y, pred_Y))

                    bin_pred_Y = [np.sign(pred_Y[0] - test_t0_price)]
                    for i in range(1, len(pred_Y)):
                        bin_pred_Y.append(np.sign(pred_Y[i] - test_Y.iloc[:, 0][i - 1]))
                    bin_pred_Y = np.array([1 if x == 0 else x for x in bin_pred_Y])
                else:
                    evals['rmse'] = np.sqrt(mean_squared_error(test_Y, pred_Y))

                    bin_pred_Y = [x if x != 0 else 1 for x in np.sign(pred_Y)]

            else:
                bin_pred_Y = pred_Y

            inverted_t0_price = inverse_scaling('Close_norm', [test_t0_price], scaler_cols, scaler).tolist()[0]
            curr_price = test_price_Y.tolist()[:-1]
            curr_price.insert(0, inverted_t0_price)

            evals["long_short_profit"], profits, evals["profit_%"], evals["order_count"] = \
                long_short_profit_evaluation(curr_price, bin_pred_Y)

            evals["sharpe_ratio"] = np.mean(profits) / (np.std([profits]) + 0.0001)

            evals["BLSH_profit"], long_profits, evals["L_profit_%"], evals["L_order_count"] = \
                buy_low_sell_high(curr_price, bin_pred_Y)

            evals["L_sharpe_ratio"] = np.mean(long_profits) / (np.std([long_profits]) + 0.0001)

            evals['accuracy_score'] = accuracy_score(bin_test_Y, bin_pred_Y)
            evals['f1_score'] = f1_score(bin_test_Y, bin_pred_Y, average='macro')
            # evals['precision_score'] = precision_score(bin_test_Y, bin_pred_Y, average='macro')

            if len(np.unique(bin_pred_Y)) < 2:
                if np.unique(bin_pred_Y)[0] > 0:
                    s = 'U'
                else:
                    s = 'D'
                evals['folds result'] = s
            elif evals["long_short_profit"] > 0:
                evals['folds result'] = '+'
            else:
                evals['folds result'] = '-'

            print({key: round(evals[key], 3) if not isinstance(evals[key], str) else evals[key] for key in evals})
            evals_list.append(evals)
            f_count += 1

    ############# DELETE THIS ###########
    #return
    #####################################

    # Save evaluation
    eval_df = pd.DataFrame(evals_list)
    """
    No. of features, selected_features, sim_func, fix_len_func, k stock, window_len, next_t, model, 
     mean_accuracy, std_accuracy, mean_f1, std_f1, mean_mse, std_mse, mean_sharp_ratio, mean_profit, std_profit
    """
    text_selected_ft = str(selected_features).replace(',', ';')

    mean_accuracy, std_accuracy = np.round((np.mean(eval_df['accuracy_score']), np.std(eval_df['accuracy_score'])), 4)
    mean_f1, std_f1 = np.round((np.mean(eval_df['f1_score']), np.std(eval_df['f1_score'])), 4)
    if not is_classifier:
        mean_mse, std_mse = np.round((np.mean(eval_df['rmse']), np.std(eval_df['rmse'])), 3)
    else:
        mean_mse, std_mse = 'NaN', 'NaN'

    mean_sharpe_ratio, mean_profit = np.round((np.mean(eval_df['sharpe_ratio']),
                                               np.mean(eval_df['long_short_profit'])), 3)
    mean_profit_pc, mean_order_count = np.round((np.mean(eval_df['profit_%']),
                                                 np.mean(eval_df['order_count'])), 3)

    mean_L_sharpe_ratio, mean_L_profit = np.round((np.mean(eval_df['L_sharpe_ratio']),
                                                   np.mean(eval_df['BLSH_profit'])), 3)
    mean_L_profit_pc, mean_L_order_count = np.round((np.mean(eval_df['L_profit_%']),
                                                     np.mean(eval_df['L_order_count'])), 3)

    folds_result = ''.join(eval_df['folds result'])

    if trans_func is None:
        trans_func_name = 'None'
    else:
        trans_func_name = trans_func.__class__.__name__

    if not os.path.isfile(eval_result_path):
        with open(eval_result_path, "w") as file:
            file.write("No. of features, selected_features, transformer, sim_func, fix_len_func, k stock, window_len, "
                       "next_t, model, target_col, sim_col, mean_accuracy, std_accuracy, "
                       "mean_f1, std_f1, mean_rmse, std_rmse, "
                       "folds_result, mean_orders_count_per_hours, mean_sharpe_ratio, mean_profit_%, mean_profit, "
                       "mean_L_orders_count_per_hours, mean_L_sharpe_ratio, mean_L_profit_%, mean_L_profit\n")
            file.close()

    with open(eval_result_path, "a") as file:
        file.write("{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}, {15}, {16}, {17}, "
                   "{18}, {19}, {20}, {21}, {22}, {23}, {24}, {25}\n "
                   .format(len(selected_features), text_selected_ft, trans_func_name, sim_func,
                           fix_len_func, k, window_len, next_t, model_name, target_col, similarity_col, mean_accuracy,
                           std_accuracy, mean_f1, std_f1, mean_mse, std_mse,
                           folds_result, mean_order_count, mean_sharpe_ratio, mean_profit_pc, mean_profit,
                           mean_L_order_count, mean_L_sharpe_ratio, mean_L_profit_pc, mean_L_profit))
        file.close()


if __name__ == "__main__":
    # Iterate Experience
    ts = time.time()
    exps = expand_test_param(**model_tunning)
    count, exp_len = 1, len(exps)
    print(' ============= Total: {0} - {1} ============= '.format(exp_len, data_name))
    for d in exps:
        es = time.time()
        print('\nRunning test param: {0}/{1}'.format(count, exp_len))
        print(d)
        count += 1
        if (d['selected_features'] == ['Close_proc'] and d['target_col'] == 'Close_norm') \
                or (d['trans_func'].__class__.__name__ == PCA().__class__.__name__ and len(d['selected_features']) < 4):
            print('     Skipped')
            continue

        run_exp(**d)

        print('Elapsed: ', np.round(time.time() - es, 2), 's, total: ', np.round((time.time() - ts) / 60, 2), 'm',
              sep='')
