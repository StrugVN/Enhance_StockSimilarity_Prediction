from similarity_functions import *
from models import *

# target_col = 'Close_norm'

similarity_funcs = {'dtw': apply_dtw,
                    'pearson': apply_pearson,
                    'euclidean': apply_euclidean,
                    'sax': compare_sax,
                    'co-integration': cointegration}

fix_length_funcs = {#'padding': padding,
                    'time_join': time_join,
                    #'delay_time_join': delay_time_join,
                    'pip': pip_fix}

fit_model_funcs = {'RandomForestRegressor': trainRFR,
                   'GradientBoostingRegressor': trainGBR,
                   'XGBRegressor': trainXGB,
                   'LSTM': trainLSTM,
                   'XGBClassifier': trainXGBClassifier,
                   'GradientBoostingClassifier': trainGBC,
                   'RandomForestClassifier': trainRFC
                   }

"""
stock = 'GOOGL'
k = 5
sim_func = apply_euclidean
fix_len_func = time_join
next_t = 1
window_len = 5
selected_features = ['Close_norm']
window_len = 0
selected_features = ['Close_norm', 'Close_proc', 'rsi_norm', 'MACD_norm']
'stock_list': ['JPM', "GOOGL", "DIS", "JNJ", "MMM", "KO", "GE"],
"""

base_param = {
    'stock_list': ['JPM', "GOOGL", "DIS", "JNJ", "MMM", "KO", "GE"],
    'target_col': 'Close_norm',
    'sim_func': 'dtw',
    'fix_len_func': 'time_join',
    'k': 5,
    'next_t': 1,
    'selected_features': ['Close_norm', 'Close_proc', 'rsi_norm', 'MACD_norm'],
    'window_len': 0,
    'model_name': 'RandomForestClassifier',
    'eval_result_path': 'test.csv'
}
