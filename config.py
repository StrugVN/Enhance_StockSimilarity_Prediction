from similarity_functions import *
from models import *

# target_col = 'Close_norm'

similarity_funcs = {'euclidean': apply_euclidean,
                    #'pearson': apply_pearson,
                    'sax': compare_sax,
                    'co-integration': cointegration,
                    #'dtw': apply_dtw
                    }

fix_length_funcs = {#'padding': padding,
                    'time_join': time_join,
                    #'delay_time_join': delay_time_join,
                    'pip': pip_fix
                    }

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
    'stock_list': ['JPM'],
    'target_col': 'Close_norm',
    'similarity_col': 'Close_norm',
    'sim_func': 'co-integration',
    'fix_len_func': 'time_join',
    'k': 5,
    'next_t': 1,
    'selected_features': ['Close_norm'],
    'window_len': 7,
    'model_name': 'RandomForestRegressor',
    'n_fold': 5,
    'eval_result_path': 'test.csv'
}
