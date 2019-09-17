from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import layers

import tensorflow_datasets as tfds

# Construct a tf.data.Dataset
print("Load MNIST dataset")
mnist_train, mnist_test = tfds.load(name="mnist", split=["train", "test"])
print("MNIST dataset loaded")

print("Start MNIST processing")
mnist_train = mnist_train.shuffle(1000).batch(128).prefetch(10)
"""print("Start feature extraction")
for features in mnist_train.take(1):
  image, label = features["image"], features["label"]"""


print(type(mnist_train))

model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(64, activation='relu'))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(layers.Dense(10, activation='softmax'))

print("Compiling model...")
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
print("Model compiled!")

print("Start model training")
model.fit(mnist_train, epochs=10, verbose=1)
