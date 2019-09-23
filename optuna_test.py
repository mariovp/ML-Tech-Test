import optuna

def objective(trial):

    print(study, ' with ', len(study.trials))
    print(trial)

    x = trial.suggest_uniform('x', -10, 10)
    return (x - 2) ** 2

tpe = optuna.samplers.TPESampler()
study = optuna.create_study()

#print(study)

#sample = tpe.sample_independent(study, trial, 'test', optuna.distributions.UniformDistribution(low=1, high=10))

print(study)
#print(sample)

study.optimize(objective, n_trials=10)

print(study)

#print(study.best_params)