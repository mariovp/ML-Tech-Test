from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import preprocessing.wind_preprocessing as ds

import optuna

WINDOW_SIZE = 30

time_series = ds.read_time_series('data/wind/wind_patzcuaro_10m_complete.csv', 24236, 29996)
time_series_normalized, norm_min, norm_max = ds.normalize_min_max(time_series)

x, y = ds.slide_window(time_series_normalized, WINDOW_SIZE, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1337)

def run_model(trial:optuna.Trial) -> float:

    print(study, ' with ', len(study.trials))
    print(trial)

    l1Units = trial.suggest_categorical('l1Units', [2, 4, 8, 16, 32])
    l2Units = trial.suggest_categorical('l2Units', [2, 4, 8, 16, 32])

    print("Training model with: ")
    print("Layer 1 units: ", l1Units)
    print("Layer 2 units: ", l2Units)

    if trial.should_prune():
        raise optuna.structs.TrialPruned()


    model = keras.Sequential([
        keras.layers.Dense(l1Units, activation='relu', input_shape=(WINDOW_SIZE,)),
        keras.layers.Dense(l2Units, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae'])

    model.fit(x_train, y_train, epochs=20, verbose=0)

    _, mse, _ = model.evaluate(x_test, y_test)

    print("Test mse: ", mse)

    return mse


import math

from optuna.pruners import BasePruner
from optuna.storages import BaseStorage  # NOQA
from optuna.structs import TrialState

class RepeatPruner(BasePruner):
    # Based on https://github.com/Minyus/optkeras/blob/master/optkeras/optkeras.py
    
    def prune(self, study, trial):

        print(" *** Calling prunner ***")

        # Get all trials
        all_trials = study.trials

        # Count completed trials
        n_trials = len([t for t in all_trials
                        if t.state == TrialState.COMPLETE])

        # If there are no previous trials
        if n_trials == 0:
            print("Not pruned Trial n_trials==0")
            return False

        # Assert that current trial is running
        assert all_trials[-1].state == TrialState.RUNNING

        # Extract params from previously completed trials
        completed_params_list = \
            [t.params for t in all_trials \
             if t.state == TrialState.COMPLETE]

        # Check if current trial is repeated
        if all_trials[-1].params in completed_params_list:
            print(" ---- Pruned Trial ----")
            return True

        print("Not pruned Trial")
        return False


study = optuna.create_study(pruner=RepeatPruner())
study.optimize(run_model, n_trials=20)

print(study)