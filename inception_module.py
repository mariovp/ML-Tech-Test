import tensorflow as tf

from tensorflow.keras import layers
from tensorflow import keras

import time


class InceptionV1Module(layers.Layer):

    def __init__(self):
        super(InceptionV1Module, self).__init__()

        self.conv_1x1 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')

        self.conv_3x3_1 = layers.Conv2D(96, (1, 1), padding='same', activation='relu')
        self.conv_3x3_2 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')

        self.conv_5x5_1 = layers.Conv2D(16, (1, 1), padding='same', activation='relu')
        self.conv_5x5_2 = layers.Conv2D(32, (5, 5), padding='same', activation='relu')

        self.pooling_1 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')
        self.pooling_2 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')

    def call(self, inputs):

        print('Calling Inception module with inputs: ', inputs)

        """conv_1x1 = layers.Conv2D(64, (1, 1), padding='same', activation='relu') (inputs)

        conv_3x3 = layers.Conv2D(96, (1, 1), padding='same', activation='relu') (inputs)
        conv_3x3 = layers.Conv2D(128, (3, 3), padding='same', activation='relu') (conv_3x3)

        conv_5x5 = layers.Conv2D(16, (1, 1), padding='same', activation='relu') (inputs)
        conv_5x5 = layers.Conv2D(32, (5, 5), padding='same', activation='relu') (conv_5x5)

        pooling = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same') (inputs)
        pooling = layers.Conv2D(32, (1, 1), padding='same', activation='relu') (pooling)"""

        res_conv_1x1 = self.conv_1x1(inputs)

        res_conv_3x3 = self.conv_3x3_1(inputs)
        res_conv_3x3 = self.conv_3x3_2(res_conv_3x3)

        res_conv_5x5 = self.conv_5x5_1(inputs)
        res_conv_5x5 = self.conv_5x5_2(res_conv_5x5)

        res_pooling = self.pooling_1(inputs)
        res_pooling = self.pooling_2(res_pooling)

        result = layers.concatenate([res_conv_1x1, res_conv_3x3, res_conv_5x5, res_pooling])

        return result


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images,test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = keras.Sequential([
        keras.layers.Reshape(input_shape=(28, 28), target_shape=(28, 28, 1)),
        #keras.layers.InputLayer(input_shape=(28, 28, 1)),
        InceptionV1Module(),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.summary()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()              
    model.fit(train_images, train_labels, epochs=5, batch_size=512)
    elapsed_time = time.time() - start_time

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print("Test accuracy: ", test_acc)
    print("Training took ", elapsed_time, " seconds")
