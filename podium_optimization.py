import optuna
import tensorflow as tf
from tensorflow import keras

from repeat_pruner import RepeatPruner

# Load dataset Fashion MNIST (ready)

# Define base architecture

# Define search space

# Optimize using bayesian with many many iterations and few training epochs

# Select n-best candidate models

# Fully train selected candidate models

# Select the best


class PodiumOptimizationPoC(object):

    def __init__(self):
        fashion_mnist = keras.datasets.fashion_mnist

        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()

        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0

    def evaluate_params(self, trial: optuna.Trial):

        params = dict()

        params['l1_units'] = trial.suggest_categorical('l1_units', [2, 4, 8, 16, 32, 64])
        params['l1_activation'] = trial.suggest_categorical('l1_activation', ['relu', 'sigmoid', 'linear'])

        params['l2_units'] = trial.suggest_categorical('l2_units', [2, 4, 8, 16, 32, 64])
        params['l2_activation'] = trial.suggest_categorical('l2_activation', ['relu', 'sigmoid', 'linear'])

        if trial.should_prune():
            raise optuna.structs.TrialPruned()

        return self.run_model(params, trial.number)

    def run_model(self, params, trial_number, n_epochs=1):

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(params['l1_units'], activation=params['l1_activation']),
            keras.layers.Dense(params['l2_units'], activation=params['l2_activation']),
            keras.layers.Dense(10, activation='softmax')
        ])

        model.add

        model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

        model.fit(self.train_images, self.train_labels, epochs=n_epochs, batch_size=64)

        test_loss, test_acc = model.evaluate(self.test_images, self.test_labels)

        print("*** Trial #",trial_number, "test accuracy: ", test_acc, "***")

        return test_acc

    def run(self):
        study = optuna.create_study(pruner=RepeatPruner(), direction='maximize')
        study.optimize(self.evaluate_params, n_trials=30)

        complete_trials = list(filter(lambda trial: trial.state == optuna.structs.TrialState.COMPLETE, study.trials))

        repeated_trials_n = len(study.trials) - len(complete_trials)
        print("Repeated trials: ", repeated_trials_n)

        sorted_trials = sorted(complete_trials, key=lambda trial: trial.value, reverse=True)

        print("Full training for best models...")
        for trial in sorted_trials[:3]:
            print(trial.value, trial.params)
            self.run_model(trial.params, trial.number, n_epochs=20)


if __name__ == "__main__":
    poc = PodiumOptimizationPoC()
    poc.run()
    

