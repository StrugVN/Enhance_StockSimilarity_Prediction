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
    'stock_list': [["JPM", "GOOGL", "DIS", "JNJ", "MMM", "KO", "GE"]],
    'target_col': ['Close_norm',
                   'Close_proc'],
    'similarity_col': ['Close_norm'],
    'sim_func': ['co-integration'],
    'fix_len_func': ['time_join'],
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

####################

GBC_test = {
    'stock_list': [["GOOGL"]],
    'target_col': ['Close_norm', 'Close_proc'],
    'similarity_col': ['Close_norm'],
    'sim_func': similarity_name,
    'fix_len_func': fix_length_name,
    'k': [10, 25, 50],
    'next_t': [1],
    'selected_features': [
        ['Close_norm', 'Close_proc', 'rsi_norm', 'MACD_norm',
         'Open_Close_diff_norm', 'High_Low_diff_norm', 'Volume_norm']
    ],
    'window_len': [5],
    'model_name': ['GradientBoostingClassifier'],
    'n_fold': [5],
    'eval_result_path': ['GradientBoostingClassifier_test.csv'],
    'norm_func': [StandardScaler()],
    'trans_func': [None, SAX()]
}

GBR_test = {
    'stock_list': [["GOOGL"]],
    'target_col': ['Close_proc'],
    'similarity_col': ['Close_norm'],
    'sim_func': similarity_name,
    'fix_len_func': fix_length_name,
    'k': [10, 25, 50],
    'next_t': [1],
    'selected_features': [
        ['Close_norm'],
        ['Close_norm', 'Close_proc', 'rsi_norm', 'MACD_norm',
         'Open_Close_diff_norm', 'High_Low_diff_norm', 'Volume_norm'],
        ['Close_norm', 'rsi_norm', 'MA_norm']
    ],
    'window_len': [5, 10, 15],
    'model_name': ['GradientBoostingRegressor'],
    'n_fold': [5],
    'eval_result_path': ['GradientBoostingRegressor_test.csv'],
    'norm_func': [StandardScaler()],
    'trans_func': [None, PCA(n_components=3, random_state=0)]
}

RFC_test = {
    'stock_list': [["GOOGL"]],
    'target_col': ['Close_norm', 'Close_proc'],
    'similarity_col': ['Close_norm'],
    'sim_func': similarity_name,
    'fix_len_func': fix_length_name,
    'k': [10, 25, 50],
    'next_t': [1],
    'selected_features': [
        ['Close_norm'], ['Close_proc'],
        ['Close_norm', 'Close_proc', 'rsi_norm', 'MACD_norm',
         'Open_Close_diff_norm', 'High_Low_diff_norm', 'Volume_norm']
    ],
    'window_len': [10, 15],
    'model_name': ['RandomForestClassifier'],
    'n_fold': [5],
    'eval_result_path': ['RandomForestClassifier_test.csv'],
    'norm_func': [StandardScaler()],
    'trans_func': [None, PCA(n_components=3, random_state=0)]
}

RFR_test_old = {
    'stock_list': [["GOOGL"]],
    'target_col': ['Close_norm', 'Close_proc'],
    'similarity_col': ['Close_norm'],
    'sim_func': similarity_name,
    'fix_len_func': fix_length_name,
    'k': [10, 25, 50],
    'next_t': [1],
    'selected_features': [
        ['Close_norm'],
        ['Close_norm', 'Close_proc', 'rsi_norm', 'MACD_norm',
         'Open_Close_diff_norm', 'High_Low_diff_norm', 'Volume_norm'],
        ['Close_norm', 'rsi_norm', 'MA_norm']
    ],
    'window_len': [5, 10, 15],
    'model_name': ['RandomForestRegressor'],
    'n_fold': [5],
    'eval_result_path': ['RandomForestRegressor_test.csv'],
    'norm_func': [StandardScaler()],
    'trans_func': trans_funcs
}

RFR_test_None = {
    'stock_list': [["GOOGL"]],
    'target_col': ['Close_norm', 'Close_proc'],
    'similarity_col': ['Close_norm'],
    'sim_func': similarity_name,
    'fix_len_func': fix_length_name,
    'k': [25],  # to do 25 50 | 50, 25
    'next_t': [1],
    'selected_features': [
        ['Close_norm'],
        ['Close_norm', 'Close_proc', 'rsi_norm', 'MACD_norm',
         'Open_Close_diff_norm', 'High_Low_diff_norm', 'Volume_norm'],
        ['Close_norm', 'rsi_norm', 'MA_norm']
    ],
    'window_len': [5, 10],
    'model_name': ['RandomForestRegressor'],
    'n_fold': [5],
    'eval_result_path': ['RandomForestRegressor_None_k25.csv'],
    'norm_func': [StandardScaler()],
    'trans_func': [None]
}

RFR_test_PCA = {
    'stock_list': [["GOOGL"]],
    'target_col': ['Close_norm', 'Close_proc'],
    'similarity_col': ['Close_norm'],
    'sim_func': similarity_name,
    'fix_len_func': fix_length_name,
    'k': [10, 25, 50],
    'next_t': [1],
    'selected_features': [
        ['Close_norm', 'Close_proc', 'rsi_norm', 'MACD_norm',
         'Open_Close_diff_norm', 'High_Low_diff_norm', 'Volume_norm']
    ],
    'window_len': [5, 15],
    'model_name': ['RandomForestRegressor'],
    'n_fold': [5],
    'eval_result_path': ['RandomForestRegressor_test.csv'],
    'norm_func': [StandardScaler()],
    'trans_func': [PCA(n_components=3, random_state=0)]
}

RFR_test_SAX = {
    'stock_list': [["GOOGL"]],
    'target_col': ['Close_norm'],
    'similarity_col': ['Close_norm'],
    'sim_func': similarity_name,
    'fix_len_func': fix_length_name,
    'k': [10, 25, 50],
    'next_t': [1],
    'selected_features': [
        ['Close_norm'],
        ['Close_norm', 'Close_proc', 'rsi_norm', 'MACD_norm',
         'Open_Close_diff_norm', 'High_Low_diff_norm', 'Volume_norm'],
        ['Close_norm', 'rsi_norm', 'MA_norm']
    ],
    'window_len': [5, 15],
    'model_name': ['RandomForestRegressor'],
    'n_fold': [5],
    'eval_result_path': ['RandomForestRegressor_test.csv'],
    'norm_func': [StandardScaler()],
    'trans_func': [SAX()]
}

XGBC_test = {
    'stock_list': [["GOOGL"]],
    'target_col': ['Close_norm', 'Close_proc'],
    'similarity_col': ['Close_norm'],
    'sim_func': similarity_name,
    'fix_len_func': fix_length_name,
    'k': [10],  # 10, 25, 50 did 50, 25,
    'next_t': [1],
    'selected_features': [
        ['Close_proc'],
        ['Close_norm', 'rsi_norm', 'MA_norm']
    ],
    'window_len': [5, 10],
    'model_name': ['XGBClassifier'],
    'n_fold': [5],
    'eval_result_path': ['XGBClassifier_k10.csv'],
    'norm_func': [StandardScaler()],
    'trans_func': [None]
}

LSTM_test = {
    'stock_list': [["GOOGL"]],
    'target_col': [#'Close_norm',
                   'Close_proc'],
    'similarity_col': ['Close_norm'],
    'sim_func': ['co-integration'],
    'fix_len_func': ['time_join'],
    'k': [50],
    'next_t': [1],
    'selected_features': [
        ['Close_norm'], ['Close_proc'],
        ['Close_norm', 'Close_proc', 'rsi_norm', 'MACD_norm',
         'Open_Close_diff_norm', 'High_Low_diff_norm', 'Volume_norm'],
        ['Close_norm', 'rsi_norm', 'MA_norm']
    ],
    'window_len': [5, 10, 15],  # 5, 10, 15
    'model_name': ['LSTM'],
    'n_fold': [5],
    'eval_result_path': ['_k50_lstm_test2.csv'],
    'norm_func': [StandardScaler()],
    'trans_func': [None]
}
