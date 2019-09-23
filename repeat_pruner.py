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