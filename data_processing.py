from copy import copy

import pandas as pd
import os
import pickle
from functools import reduce

from financial_features import *
from Const import *


def cal_financial_features(data):
    selected_numeric_cols = data.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])\
        .columns.tolist()

