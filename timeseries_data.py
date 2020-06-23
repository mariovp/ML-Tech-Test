import tensorflow as tf
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

WIND_DATA_PATH = "data/wind/wind_patzcuaro_10m_complete.csv"

"""print(os.path.realpath(__file__))

titanic_file = tf.keras.utils.get_file("train.csv", "data/wind/wind_patzcuaro_10m_complete.csv")

print(titanic_file)

df = pd.read_csv(titanic_file, index_col=None)
print(df.head())"""

"""wind_batches = tf.data.experimental.make_csv_dataset(
    "data/wind/wind_patzcuaro_10m_complete.csv", 
    batch_size=5,
    label_name="time",
    shuffle=False
    )

print(wind_batches)

for feature_batch, label_batch in wind_batches.take(1):
  print("'time': {}".format(label_batch))
  print("features:")
  for key, value in feature_batch.items():
    print("  {!r:20s}: {}".format(key, value))"""

# Read data
wind_dataframe = pd.read_csv(WIND_DATA_PATH,  usecols=['value'])
wind_dataframe = wind_dataframe[:20000]
wind_dataframe = wind_dataframe.clip(lower=0)
print(wind_dataframe.head())

# Rolling mean and plots
rolling = wind_dataframe.rolling(window=8)
rolling_mean = rolling.mean()

rolling_mean.to_csv("wind_timeseries.csv")

wind_dataframe.plot()
rolling_mean.plot(color='red')

wind_dataframe = wind_dataframe.rolling(3).mean()
print(wind_dataframe)

#sns.lineplot(x="time", y="value",
#             data=wind_dataframe)
plt.show()

# Build TF Dataset
# wind_dataset = tf.data.Dataset.from_tensor_slices(wind_dataframe.to_numpy())
"""wind_dataset = tf.data.Dataset.range(100000)

def dense_1_step(batch):
  # Shift features and labels one step relative to each other.
  return batch[:-1], batch[1:]

def build_pair(batch):
  return batch[:-3], batch[-3:]


def make_window_dataset(ds, window_size=5, shift=1, stride=1):
  windows = ds.window(window_size, shift=shift, stride=stride)

  def sub_to_batch(sub):
    return sub.batch(window_size, drop_remainder=True)

  windows = windows.flat_map(sub_to_batch)
  return windows

ds = make_window_dataset(wind_dataset, window_size=13, shift=1, stride=1)

for example in ds.take(10):
  print(example.numpy())

dense_labels_ds = ds.map(build_pair)

for inputs,labels in dense_labels_ds.take(3):
  print(inputs.numpy(), "=>", labels.numpy())"""