import tensorflow as tf

from tensorflow.keras import layers
from tensorflow import keras


class InceptionV1Module(layers.Layer):

    def __init__(self):
        super(InceptionV1Module, self).__init__()

        self.tower_1_1 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')
        self.tower_1_2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')

        self.tower_2_1 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')
        self.tower_2_2 = layers.Conv2D(64, (5, 5), padding='same', activation='relu')

        self.tower_3_1 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')
        self.tower_3_2 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')

    def call(self, inputs):

        print('Calling Inception module with inputs: ', inputs)

        r1 = self.tower_1_1(inputs)
        r1 = self.tower_1_2(r1)

        r2 = self.tower_2_1(inputs)
        r2 = self.tower_2_2(r2)

        r3 = self.tower_3_1(inputs)
        r3 = self.tower_3_2(inputs)

        result = layers.concatenate([r1, r2, r3])

        return result


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0
test_images = test_images / 255.0

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = keras.Sequential([
        keras.layers.Reshape(input_shape=(28, 28), target_shape=(28, 28, 1)),
        InceptionV1Module(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.summary()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=5, batch_size=128)

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print("Test accuracy: ", test_acc)
