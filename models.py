import os

from keras import callbacks
from sklearn.ensemble import *
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam

""" LSTM Configuration """


def create_LSTM(input_shape, lr=0.01, output_shape=1, loss='mse'):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.25))
    model.add(Dense(units=50))
    model.add(Dense(units=output_shape))
    model.compile(optimizer=Adam(lr=lr), loss=loss)
    return model


LSTM_train_only_config = {'batch_size': 32,
                          'verbose': 0,
                          'epochs': 100,
                          # 'callbacks': callbacks.EarlyStopping(monitor="loss", mode="min", patience=15,
                          #                                      restore_best_weights=True, verbose=2),
                          'callbacks': callbacks.ModelCheckpoint('LSTM_cp.hdf5', monitor='loss', verbose=0,
                                                                 save_best_only=True, save_weights_only=True,
                                                                 mode='min')
                          }

LSTM_with_val_config = {'batch_size': 32,
                        'verbose': 0,
                        'epochs': 100,
                        'validation_split': 0.2,
                        # 'callbacks': callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=15,
                        #                                     restore_best_weights=True, verbose=2),
                        'callbacks': callbacks.ModelCheckpoint('LSTM_cp.hdf5', monitor='val_loss', verbose=0,
                                                               save_best_only=True, save_weights_only=True, mode='min')
                        }
""" ---------------------------------------------------------------------------------------------------- """


def trainRFR(train_X, train_Y, n_estimators=100):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=0)

    model.fit(train_X, train_Y.values.ravel())

    return model


def trainRFC(train_X, train_Y, n_estimators=100):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=0)

    model.fit(train_X, train_Y)

    return model


def trainGBR(train_X, train_Y, n_estimators=100, lr=0.02, es=False):
    if not es:
        model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=lr, random_state=0)
    else:
        model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=lr, random_state=0,
                                          validation_fraction=0.2, n_iter_no_change=15)

    model.fit(train_X, train_Y.values.ravel())

    return model


def trainGBC(train_X, train_Y, n_estimators=100, lr=0.02, es=False):
    if not es:
        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=lr, random_state=0)
    else:
        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=lr, random_state=0,
                                           validation_fraction=0.2, n_iter_no_change=15)

    model.fit(train_X, train_Y)

    return model


def trainXGB(train_X, train_Y, obj='reg:squarederror', lr=0.01, n_estimators=100, es=False):
    model = XGBRegressor(objective=obj, learning_rate=lr, n_estimators=n_estimators, random_state=0)
    if es:
        X_train, X_val, y_train, y_val = train_test_split(train_X, train_Y, test_size=0.2, random_state=0)
        eval_set = [(X_val, y_val)]

        model.fit(X_train, y_train,
                  early_stopping_rounds=10, eval_metric="rmse", eval_set=eval_set, verbose=False)
    else:
        # model.fit(train_X, train_Y,
        #          early_stopping_rounds=10, eval_set=[(train_X, train_Y)], eval_metric="rmse", verbose=False)
        model.fit(train_X, train_Y)

    return model


def trainXGBClassifier(train_X, train_Y, obj='binary:logistic', lr=0.01, n_estimators=100, es=False):
    model = XGBClassifier(objective=obj, learning_rate=lr, n_estimators=n_estimators, random_state=0)
    if es:
        X_train, X_val, y_train, y_val = train_test_split(train_X, train_Y, test_size=0.2, random_state=0)
        eval_set = [(X_val, y_val)]

        model.fit(X_train, y_train,
                  early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=False)
    else:
        # model.fit(train_X, train_Y,
        #          early_stopping_rounds=10, eval_set=[(train_X, train_Y)], eval_metric="logloss", verbose=False)
        model.fit(train_X, train_Y)

    return model


def trainLSTM(train_X, train_Y, config=None):
    if config is None:
        config = LSTM_with_val_config

    model = create_LSTM((1, train_X.shape[1]))

    reshaped_X_p = train_X.to_numpy().reshape(-1, 1, train_X.shape[1])
    reshaped_Y_p = train_Y.to_numpy().reshape(-1, 1, 1)

    model.fit(reshaped_X_p, reshaped_Y_p, **config)

    if os.path.exists("LSTM_cp.hdf5"):
        model.load_weights('LSTM_cp.hdf5')
        os.remove("LSTM_cp.hdf5")

    return model
