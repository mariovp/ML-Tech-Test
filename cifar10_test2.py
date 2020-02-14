import tensorflow as tf

from tensorflow.keras import datasets, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D, Softmax

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
 
weight_decay = 1e-4
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', activation="relu", kernel_regularizer=regularizers.l2(weight_decay), input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same', activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
 
model.add(Conv2D(64, (3,3), padding='same', activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
 
model.add(Conv2D(128, (3,3), padding='same', activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same', activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(10, (1, 1), activation='relu'))

model.add(GlobalAveragePooling2D())

model.add(Softmax())
 
model.summary()

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, decay=1e-6)

model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=125, batch_size=64,
                    validation_data=(test_images, test_labels))
