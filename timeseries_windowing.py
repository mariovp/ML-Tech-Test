import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sliding_window(dataset: tf.data.Dataset, window_size: int = 10, shift: int = 1, target_size: int = 1):

    def make_window_dataset(ds, window_size, shift):
        windows = ds.window(window_size, shift=shift)

        def sub_to_batch(sub):
          return sub.batch(window_size, drop_remainder=True)

        windows = windows.flat_map(sub_to_batch)
        return windows
    
    def build_pair(batch):
        return batch[:-target_size], batch[-target_size:]

    window_batch_size = window_size + target_size
    ds = make_window_dataset(dataset, window_batch_size, shift)
    pairs_ds = ds.map(build_pair)

    return pairs_ds

WINDOW_SIZE = 72
TARGET_SIZE = 12

df = pd.read_csv("wind_timeseries.csv",  usecols=['value'])
df = df.dropna()
normalized_df = (df-df.min())/(df.max()-df.min())
dataset = tf.data.Dataset.from_tensor_slices((normalized_df.to_numpy()))

# dataset = tf.data.Dataset.range(100000)
dataset_windowed = sliding_window(dataset, window_size=WINDOW_SIZE, target_size=TARGET_SIZE)
# dataset_windowed = dataset_windowed.batch(8)

# print(dataset_windowed.shape)

for inputs, labels in dataset_windowed.take(1):
    input_shape = inputs.shape
    print(inputs.shape, "=>", labels.shape)

dataset_train = dataset_windowed.cache().shuffle(64).batch(64)

single_step_model = tf.keras.models.Sequential()
single_step_model.add(tf.keras.layers.GRU(16, return_sequences=True, input_shape=input_shape))
single_step_model.add(tf.keras.layers.GRU(16))
# Output neurons == predicted steps
single_step_model.add(tf.keras.layers.Dense(TARGET_SIZE))

single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae', metrics=['mse'])

single_step_history = single_step_model.fit(dataset_train, epochs=20)

single_step_model.summary()

for inputs, labels in dataset_windowed.take(5):
    input_array = inputs.numpy().reshape((1,WINDOW_SIZE,1))
    predicted = single_step_model.predict(input_array)
    print("Label shape = ", labels.shape)
    print("Prediction shape = ", predicted.shape)
    input_array = input_array.flatten()
    labels = labels.numpy().flatten()
    predicted = predicted.flatten()
    print(labels, "=>", predicted)
    print(input_array)

    print(input_array.shape)
    print(labels.shape)
    observed = np.concatenate((input_array, labels))
    print(observed)
    df = pd.DataFrame(observed, columns = ["Observed"])

    placeholder = np.empty((WINDOW_SIZE,))
    placeholder[:] = np.NaN
    predicted = np.concatenate((placeholder, predicted))
    print(predicted.shape)
    df.insert(1, "Predicted", predicted)
    print(df)

    df.plot()
    plt.show()


