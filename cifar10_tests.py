import tensorflow as tf

from tensorflow.keras import datasets, layers, models

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(96, (5, 5), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((3, 3), (2, 2)))
model.add(layers.Conv2D(192, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((3, 3), (2, 2)))
model.add(layers.Conv2D(192, (3, 3), activation='relu'))
model.add(layers.Conv2D(192, (1, 1), activation='relu'))
model.add(layers.Conv2D(10, (1, 1), activation='relu'))
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=50, batch_size=512,
                    validation_data=(test_images, test_labels))