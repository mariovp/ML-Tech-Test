import datetime

from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import preprocessing.wind_preprocessing as ds
from keras.models import Sequential

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, rand



WINDOW_SIZE = 30

space = {
    'units1': hp.choice('units1', [2, 4, 8, 16]),
    'activation1': hp.choice('activation1', ['relu', 'sigmoid']),
    'units2': hp.choice('units2', [2, 4, 8, 16]),
    'activation2': hp.choice('activation2', ['relu', 'sigmoid']),
    'optimizer': hp.choice('optimizer', ['adam', 'adadelta']),
    'batch_size': hp.choice('batch_size', [16, 32, 64])
}

time_series = ds.read_time_series('data/wind/wind_patzcuaro_10m_complete.csv', 24236, 29996)
time_series_normalized, norm_min, norm_max = ds.normalize_min_max(time_series)

x, y = ds.slide_window(time_series_normalized, WINDOW_SIZE, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1337)

# last_params = {}

global_trials = Trials()


"""def param_extractor(params):
    global last_params
    last_params = params
    print(params)

    return {'loss': 0.5, 'status': STATUS_OK}"""


def run_model(params) -> float:

    print("Training model with: ")
    print(params)

    model = Sequential()
    model.add(Dense(params['units1'], input_shape=(WINDOW_SIZE,), activation=params['activation1']))
    model.add(Dense(params['units2'], activation=params['activation2']))
    model.add(Dense(1))

    model.compile(optimizer=params['optimizer'], loss='mean_squared_error', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=20, batch_size=params['batch_size'], verbose=0,
                        shuffle=1,
                        validation_data=(x_test, y_test)).history

    y_predicted = model.predict(x_test)
    mse = mean_squared_error(y_test, y_predicted)
    print("Test MSE = ", mse)

    return mse


def get_next_params(current_trials):

    trial_parameters = {}

    def extract_params(params):
        nonlocal trial_parameters
        trial_parameters = params
        return {'loss': 0.5, 'status': STATUS_OK}

    cache_trials = Trials()
    cache_trials.insert_trial_docs(current_trials.trials)
    # cache_trials.refresh()

    fmin(extract_params, space, algo=tpe.suggest, trials=cache_trials, max_evals=1,
         return_argmin=False, show_progressbar=False)

    trial = cache_trials.trials[-1]

    return trial, trial_parameters


for i in range(5):
    t, tp = get_next_params(global_trials)
    mse = run_model(tp)
    t['result']['loss'] = mse
    t['refresh_time'] = datetime.datetime.now()
    global_trials.insert_trial_doc(t)
    global_trials.refresh()

print(global_trials.best_trial)

"""
trials = Trials()
print("First recommendation")
fmin(param_extractor, space, algo=tpe.suggest, trials=trials, max_evals=1, return_argmin=False)

new_trial = trials.trials[-1]

new_trial['result']['loss'] = 0.4

print(new_trial)

global_trials.insert_trial_doc(new_trial)
global_trials.refresh()
print("Trial added to global trials")

test_trials = Trials()
test_trials.insert_trial_docs(global_trials.trials)
print("Inserted all trials from global")"""
#
# # Run model and get evaluation
# mse = run_model(last_params)
#
# # Build dict
#
# data = {
#     'book_time': datetime.datetime.now(),
#     'exp_key': None,
#     'misc': {'cmd': ('domain_attachment', 'FMinIter_Domain'),
#              'idxs': {},
#              'tid': 1,
#              'vals': {},
#              'workdir': None},
#     'owner': None,
#     'refresh_time': datetime.datetime.now(),
#     'result': {'loss': mse, 'status': STATUS_OK},
#     'spec': None,
#     'state': 2,
#     'tid': 1,
#     'version': 0
# }
#
# global_trials.insert_trial_doc(data)
#
# # Add to global trials
#
# print("Second recommendation")
# fmin(param_extractor, space, algo=tpe.suggest, trials=global_trials, max_evals=1)
#
# # best = fmin(function, space, algo=tpe.suggest, trials=trials, max_evals=5)
#
# """print(best)
# print(trials.best_trial)
#
# print("Done")"""
