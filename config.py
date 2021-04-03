from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from similarity_functions import *
from models import *

# target_col = 'Close_norm'

similarity_funcs = {#'euclidean': apply_euclidean,
                    #'pearson': apply_pearson,
                    'co-integration': cointegration,
                    'sax': compare_sax,
                    #'dtw': apply_dtw
                    }

fix_length_funcs = {'padding': padding,
                    'time_join': time_join,
                    'delay_time_join': delay_time_join,
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
selected_features = ['Close_norm', 'Close_proc', 'rsi_norm', 'MACD_norm',
                          'Open_Close_diff_norm', 'High_Low_diff_norm', 'Volume_norm']
'stock_list': ['JPM', "GOOGL", "DIS", "JNJ", "MMM", "KO", "GE"],
"""

trans_funcs = [None, PCA(n_components=3, random_state=0), SAX()]
# Phải có ít nhất 4 ft để dùng PCA

base_param = {
    'stock_list': ['JPM', "GOOGL", "DIS", "JNJ", "MMM", "KO", "GE"],
    'target_col': 'Close_norm',
    'similarity_col': 'Close_norm',
    'sim_func': 'co-integration',
    'fix_len_func': 'time_join',
    'k': 50,
    'next_t': 1,
    'selected_features': ['Close_norm', 'Close_proc', 'rsi_norm', 'MACD_norm',
                          'Open_Close_diff_norm', 'High_Low_diff_norm', 'Volume_norm'],
    'window_len': 0,
    'model_name': 'GradientBoostingRegressor',
    'n_fold': 5,
    'eval_result_path': 'test.csv',
    'norm_func': StandardScaler(),
    'trans_func': None
}
