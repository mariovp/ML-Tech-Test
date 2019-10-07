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

        print("Shape of train_images: ", self.train_images.shape)

    def evaluate_params(self, trial: optuna.Trial):

        params = dict()

        params['model_type'] = trial.suggest_categorical('model_type', ['dnn', 'cnn'])

        if params['model_type'] == 'dnn':
            params['l1_units'] = trial.suggest_categorical('l1_units', [8, 16, 32, 64])
            params['l1_activation'] = trial.suggest_categorical('l1_activation', ['relu', 'sigmoid', 'linear'])

            params['l2_units'] = trial.suggest_categorical('l2_units', [8, 16, 32, 64])
            params['l2_activation'] = trial.suggest_categorical('l2_activation', ['relu', 'sigmoid', 'linear'])

        elif params['model_type'] == 'cnn':
            params['cnn_l1_filters'] = trial.suggest_categorical('cnn_l1_filters', [8, 16, 32, 64])
            params['cnn_dense_l1_units'] = trial.suggest_categorical('cnn_dense_l1_units', [8, 16, 32, 64])
            params['cnn_dense_l2_units'] = trial.suggest_categorical('cnn_dense_l2_units', [8, 16, 32, 64])

        if trial.should_prune():
            raise optuna.structs.TrialPruned()

        return self.run_model(params, trial.number)

    def run_model(self, params, trial_number, n_epochs=7):

        if params['model_type'] == 'dnn':

            model = keras.Sequential([
                keras.layers.Flatten(input_shape=(28, 28)),
                keras.layers.Dense(params['l1_units'], activation=params['l1_activation']),
                keras.layers.Dense(params['l2_units'], activation=params['l2_activation']),
                keras.layers.Dense(10, activation='softmax')
            ])

        elif params['model_type'] == 'cnn':

            print('Initializing CNN with params ',params)

            model = keras.Sequential([
                keras.layers.Reshape(input_shape=(28, 28), target_shape=(28,28,1)),
                keras.layers.Conv2D(params['cnn_l1_filters'], (3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                #keras.layers.Conv2D(32, (3, 3), activation='relu'),
                #keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(params['cnn_dense_l1_units'], activation='relu'),
                keras.layers.Dense(params['cnn_dense_l2_units'], activation='relu'),
                keras.layers.Dense(10, activation='softmax')
            ]) 

        model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

        model.fit(self.train_images, self.train_labels, epochs=n_epochs, batch_size=128)

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
            self.run_model(trial.params, trial.number, n_epochs=50)


if __name__ == "__main__":
    poc = PodiumOptimizationPoC()
    poc.run()
    

