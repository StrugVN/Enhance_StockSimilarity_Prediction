from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from similarity_functions import *
from models import *

# target_col = 'Close_norm'

similarity_funcs = {'euclidean': apply_euclidean,
                    'pearson': apply_pearson,
                    'co-integration': cointegration,
                    'sax': compare_sax,
                    'dtw': apply_dtw
                    }
similarity_name = ['euclidean', 'pearson', 'co-integration', 'sax', 'dtw']

fix_length_funcs = {'padding': padding,
                    'time_join': time_join,
                    'delay_time_join': delay_time_join,
                    'pip': pip_fix
                    }
fix_length_name = ['padding', 'time_join', 'delay_time_join', 'pip']

fit_model_funcs = {'RandomForestRegressor': trainRFR,
                   'GradientBoostingRegressor': trainGBR,
                   'XGBRegressor': trainXGB,
                   'LSTM': trainLSTM,
                   'XGBClassifier': trainXGBClassifier,
                   'GradientBoostingClassifier': trainGBC,
                   'RandomForestClassifier': trainRFC
                   }
fit_model_name = ['RandomForestRegressor', 'GradientBoostingRegressor', 'XGBRegressor', 'LSTM',
                  'XGBClassifier', 'GradientBoostingClassifier', 'RandomForestClassifier']


trans_funcs = [None, PCA(n_components=3, random_state=0), SAX()]
# Phải có ít nhất 4 ft để dùng PCA

base_param = {
    'stock_list': ["GOOGL"],
    'target_col': 'Close_norm',
    'similarity_col': 'Close_norm',
    'sim_func': 'co-integration',
    'fix_len_func': 'time_join',
    'k': 0,
    'next_t': 1,
    'selected_features': ['Close_norm'],
    'window_len': 10,
    'model_name': 'GradientBoostingRegressor',
    'n_fold': 5,
    'eval_result_path': 'test.csv',
    'norm_func': StandardScaler(),
    'trans_func': None
}

# "JPM", "GOOGL", "DIS", "JNJ", "MMM", "KO", "GE"

# Note: w_len -> trans -> k
#
test_create_data = {
    'stock_list': [["GOOGL"]],
    'target_col': ['Close_norm',
                   'Close_proc'],
    'similarity_col': ['Close_norm'],
    'sim_func': ['dtw'],  # 'euclidean', 'pearson', 'co-integration', 'sax', 'dtw'
    'fix_len_func': fix_length_name,  # 'padding', 'time_join', 'delay_time_join', 'pip'
    'k': [50],  # 10, 25, 50 |
    'next_t': [1],
    'selected_features': [
        ['Close_norm'], ['Close_proc'],
        ['Close_norm', 'Close_proc', 'rsi_norm', 'MACD_norm',
         'Open_Close_diff_norm', 'High_Low_diff_norm', 'Volume_norm'],
        ['Close_norm', 'rsi_norm', 'MA_norm']
    ],
    'window_len': [15],  # 5, 10, 15 |
    'model_name': ['GradientBoostingRegressor'],
    'n_fold': [5],
    'eval_result_path': ['create_data_test.csv'],
    'norm_func': [StandardScaler()],
    'trans_func': [None]  # [None, PCA(n_components=3, random_state=0), SAX()] |
}

base_k0_test = {
    'stock_list': [["GOOGL"]],
    'target_col': ['Close_norm',
                   'Close_proc'],
    'similarity_col': ['Close_norm'],
    'sim_func': ['co-integration'],  # k=0 => no effect
    'fix_len_func': ['time_join'],  # k=0 => no effect
    'k': [0],
    'next_t': [1],
    'selected_features': [
        ['Close_norm'], ['Close_proc'],
        ['Close_norm', 'Close_proc', 'rsi_norm', 'MACD_norm',
         'Open_Close_diff_norm', 'High_Low_diff_norm', 'Volume_norm'],
        ['Close_norm', 'rsi_norm', 'MA_norm']
    ],
    'window_len': [5, 10, 15],  # 5, 10, 15
    'model_name': fit_model_name,
    'n_fold': [5],
    'eval_result_path': ['k0_test.csv'],
    'norm_func': [StandardScaler()],
    'trans_func': trans_funcs
}

base_test = {
    'stock_list': [["GOOGL"]],
    'target_col': ['Close_norm',
                   'Close_proc'],
    'similarity_col': ['Close_norm'],
    'sim_func': similarity_name,
    'fix_len_func': fix_length_name,
    'k': [50],
    'next_t': [1],
    'selected_features': [
        ['Close_norm'], ['Close_proc'],
        ['Close_norm', 'Close_proc', 'rsi_norm', 'MACD_norm',
         'Open_Close_diff_norm', 'High_Low_diff_norm', 'Volume_norm'],
        ['Close_norm', 'rsi_norm', 'MA_norm']
    ],
    'window_len': [5, 10, 15],  # 5, 10, 15
    'model_name': [''],
    'n_fold': [5],
    'eval_result_path': [''],
    'norm_func': [StandardScaler()],
    'trans_func': trans_funcs
}
