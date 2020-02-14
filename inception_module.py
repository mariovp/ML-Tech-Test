import tensorflow as tf

from tensorflow.keras import layers
from tensorflow import keras

import time


class InceptionV1Module(layers.Layer):

    def __init__(self,
                 conv1x1_filters=64, conv3x3_reduce_filters=96, conv3x3_filters=128, conv5x5_reduce_filters=16, conv5x5_filters=32, pooling_conv_filters=32,
                 **kwargs):
        super(InceptionV1Module, self).__init__(**kwargs)

        self.conv1x1_filters = conv1x1_filters
        self.conv3x3_reduce_filters = conv3x3_reduce_filters
        self.conv3x3_filters = conv3x3_filters
        self.conv5x5_reduce_filters = conv5x5_reduce_filters
        self.conv5x5_filters = conv5x5_filters
        self.pooling_conv_filters = pooling_conv_filters

        self.conv_1x1 = layers.Conv2D(
            conv1x1_filters, (1, 1), padding='same', activation='relu')
        self.bn_1x1 = layers.BatchNormalization()

        self.conv_3x3_1 = layers.Conv2D(
            conv3x3_reduce_filters, (1, 1), padding='same', activation='relu')
        self.conv_3x3_2 = layers.Conv2D(
            conv3x3_filters, (3, 3), padding='same', activation='relu')
        self.bn_3x3_1 = layers.BatchNormalization()
        self.bn_3x3_2 = layers.BatchNormalization()

        self.conv_5x5_1 = layers.Conv2D(
            conv5x5_reduce_filters, (1, 1), padding='same', activation='relu')
        self.conv_5x5_2 = layers.Conv2D(
            conv5x5_filters, (5, 5), padding='same', activation='relu')

        self.bn_5x5_1 = layers.BatchNormalization()
        self.bn_5x5_2 = layers.BatchNormalization()

        self.pooling_1 = layers.MaxPooling2D(
            (3, 3), strides=(1, 1), padding='same')
        self.pooling_2 = layers.Conv2D(
            pooling_conv_filters, (1, 1), padding='same', activation='relu')

        

    def call(self, inputs):

        res_conv_1x1 = self.conv_1x1(inputs)
        res_conv_1x1 = self.bn_1x1(res_conv_1x1)

        res_conv_3x3 = self.conv_3x3_1(inputs)
        res_conv_3x3 = self.bn_3x3_1(res_conv_3x3)
        res_conv_3x3 = self.conv_3x3_2(res_conv_3x3)
        res_conv_3x3 = self.bn_3x3_2(res_conv_3x3)

        res_conv_5x5 = self.conv_5x5_1(inputs)
        res_conv_5x5 = self.bn_5x5_1(res_conv_5x5)
        res_conv_5x5 = self.conv_5x5_2(res_conv_5x5)
        res_conv_5x5 = self.bn_5x5_2(res_conv_5x5)

        res_pooling = self.pooling_1(inputs)
        res_pooling = self.pooling_2(res_pooling)

        result = layers.concatenate(
            [res_conv_1x1, res_conv_3x3, res_conv_5x5, res_pooling])

        return result

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'conv1x1_filters': self.conv1x1_filters,
            'conv3x3_reduce_filters': self.conv3x3_reduce_filters,
            'conv3x3_filters': self.conv3x3_filters,
            'conv5x5_reduce_filters': self.conv5x5_reduce_filters,
            'conv5x5_filters': self.conv5x5_filters,
            'pooling_conv_filters': self.pooling_conv_filters
        })
        return config

    def from_config(self, cls, config):
        # raise ValueError("From config")
        return cls(**config)



print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))

fashion_mnist = keras.datasets.cifar10

(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = keras.Sequential([
        # keras.layers.Reshape(input_shape=(28, 28), target_shape=(28, 28, 1)),
        keras.layers.InputLayer(input_shape=(32, 32, 3)),
        InceptionV1Module(),
        keras.layers.MaxPool2D(3, 2, padding="same"),
        keras.layers.Dropout(0.3),
        InceptionV1Module(),
        keras.layers.MaxPool2D(3, 2, padding="same"),
        keras.layers.Dropout(0.4),
        InceptionV1Module(),
        keras.layers.Dropout(0.5),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax')
    ])
    # Hasta 85% en Cifar-10 en 85 epochs

    model.summary()

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01, decay=1e-6)

    model.compile(optimizer="adam",
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    """tf.keras.utils.plot_model(
        model,
        to_file='model.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='LR',
        expand_nested=True,
        dpi=96
    )"""

    start_time = time.time()
    model.fit(train_images, train_labels, epochs=250, batch_size=256, validation_data=(test_images, test_labels))
    elapsed_seconds = time.time() - start_time
    elapsed_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_seconds))

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print("Test accuracy: ", test_acc)
    print("Training took", elapsed_time, "(hh:mm:ss)", elapsed_seconds, "(seconds)")
