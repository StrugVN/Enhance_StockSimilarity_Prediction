from main import *

test = base_test.copy()

test['k'] = [50]
test['model_name'] = ['XGBRegressor']
test['eval_result_path'] = ['XGBRegressor_k50.csv']

print('Running 50 XGBRegressor_k50.csv')

# Iterate Experience
ts = time.time()
exps = expand_test_param(**test)
count, exp_len = 1, len(exps)
print(' ============= Total: {} ============= '.format(exp_len))
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

    print('Elapsed: ', np.round(time.time() - es, 2), 's, total: ', np.round((time.time() - ts) / 60, 2), 'm', sep='')
