import tensorflow as tf


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

dataset = tf.data.Dataset.range(100000)
dataset_windowed = sliding_window(dataset, target_size=5)
# dataset_windowed = dataset_windowed.batch(8)

for inputs, labels in dataset_windowed.take(5):
    print(inputs.numpy(), "=>", labels.numpy())
