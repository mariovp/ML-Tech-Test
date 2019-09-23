from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
# import tensorflow as tf
# from tensorflow import keras

import optuna
from optuna import structs

import json

from repeat_pruner import RepeatPruner

# Suggest directly from trial without 'optimize', no multi-threading
"""master_study = optuna.create_study()

print(master_study)

trial_id = master_study.storage.create_new_trial(master_study.study_id)

trial = optuna.trial.Trial(master_study, trial_id)

l1Units = trial.suggest_categorical('l1Units', [2, 4, 8, 16, 32])
l2Units = trial.suggest_categorical('l2Units', [2, 4, 8, 16, 32])

trial.report(0.5)
master_study.storage.set_trial_state(trial.number, structs.TrialState.COMPLETE)

print(master_study)"""


master_study = optuna.create_study()

pruner = RepeatPruner()

trial1_id = master_study.storage.create_new_trial(master_study.study_id)
trial1 = optuna.Trial(master_study, trial1_id)
t1_l1Units = trial1.suggest_categorical('l1Units', [2, 4, 8, 16, 32])
t1_l2Units = trial1.suggest_categorical('l2Units', [2, 4, 8, 16, 32])

if pruner.prune(master_study, trial1):
    print("Prune trial")

print("Don't prune trial")

trial1_params_json = json.dumps(trial1.params)
print(trial1_params_json)

trial2_id = master_study.storage.create_new_trial(master_study.study_id)
trial2 = optuna.Trial(master_study, trial2_id)
t2_l1Units = trial2.suggest_categorical('l1Units', [2, 4, 8, 16, 32])
t2_l2Units = trial2.suggest_categorical('l2Units', [2, 4, 8, 16, 32])

if pruner.prune(master_study, trial1):
    print("Prune trial")

print("Don't prune trial")

print(trial2.params)

trial1.report(0.5)
master_study.storage.set_trial_state(trial1.number, structs.TrialState.COMPLETE)

trial2.report(0.4)
master_study.storage.set_trial_state(trial2.number, structs.TrialState.COMPLETE)

print(master_study)