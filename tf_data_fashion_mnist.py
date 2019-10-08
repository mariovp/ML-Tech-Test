_EPOCHS      = 5
_NUM_CLASSES = 10
_BATCH_SIZE  = 1000

def tfdata_generator(images, labels, is_training, batch_size=128):
    '''Construct a data generator using tf.Dataset'''

    def preprocess_fn(image, label):
        '''A transformation function to preprocess raw data
        into trainable input. '''
        x = tf.reshape(tf.cast(image, tf.float32), (28, 28, 1))
        #y = tf.one_hot(tf.cast(label, tf.uint8), _NUM_CLASSES)
        y = label
        return x, y

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_training:
            dataset = dataset.shuffle(1000)  # depends on sample size
    # Transform and batch data at the same time
    dataset = dataset.apply(tf.data.experimental.map_and_batch(
        preprocess_fn, batch_size,
        num_parallel_batches=4,  # cpu cores
        drop_remainder=True if is_training else False))
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


train_dataset = tfdata_generator(train_images, train_labels, is_training=True, batch_size=_BATCH_SIZE)
test_dataset = tfdata_generator(test_images, test_labels, is_training=False, batch_size=_BATCH_SIZE)