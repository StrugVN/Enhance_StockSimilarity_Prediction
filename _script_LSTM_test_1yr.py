from main import *
import main

Const.const_LSTM_saved_weight = 'lstm_w_module1.hdf5'

main.data_name = 'all_stocks_last_1yr'

# Iterate Experience
test = base_test.copy()

test['sim_func'] = ['co-integration']  # 'euclidean', 'pearson', 'co-integration', 'sax', 'dtw'
test['fix_len_func'] = ['padding']  # 'padding', 'time_join', 'delay_time_join', 'pip'
test['eval_result_path'] = ['LSTM_module1.csv']

ts = time.time()
exps = expand_test_param(**test)
count, exp_len = 1, len(exps)
print(' ============= Total: {0} - {1} ============= '.format(exp_len, main.data_name))
for d in exps:
    es = time.time()
    print('\nRunning test param: {0}/{1}'.format(count, exp_len))
    print(d)
    count += 1
    if (d['selected_features'] == ['Close_proc'] and d['target_col'] == 'Close_norm') \
            or (
            d['trans_func'].__class__.__name__ == PCA().__class__.__name__ and len(d['selected_features']) < 4):
        print('     Skipped')
        continue

    run_exp(**d)

    print('Elapsed: ', np.round(time.time() - es, 2), 's, total: ', np.round((time.time() - ts) / 60, 2), 'm',
            sep='')

