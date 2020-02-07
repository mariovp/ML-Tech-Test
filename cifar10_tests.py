import tensorflow as tf

from tensorflow.keras import datasets, layers, models, regularizers

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Input((32, 32, 3)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(96, (5, 5), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((3, 3), (2, 2)))
model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(192, (5, 5), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((3, 3), (2, 2)))
model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(192, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(192, (1, 1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(10, (1, 1), activation='relu'))

model.add(layers.GlobalAveragePooling2D())

model.add(layers.Softmax())

model.summary()

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, decay=1e-6)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=500, batch_size=64,
                    validation_data=(test_images, test_labels))