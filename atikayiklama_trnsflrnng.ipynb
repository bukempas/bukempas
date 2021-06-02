import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os,shutil
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.layers import Conv2D, BatchNormalization, Dense, Activation, Flatten, MaxPooling2D
from keras import optimizers
import PIL

tflearning_model = tf.keras.applications.ResNet50V2(weights = 'imagenet', include_top = False, input_shape = (224,224,3))
for layer in tflearning_model.layers:
  layer.trainable = False

target_size=(224,224)
train_datagen = ImageDataGenerator(validation_split=0.1,
    rescale=1./255,
    rotation_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2)
train_generator = train_datagen.flow_from_directory(
        'train',
        subset='training',
        target_size=target_size,
        class_mode='sparse',
        batch_size=64)
validation_generator = train_datagen.flow_from_directory(
        'train',
        subset='validation',
        target_size=target_size,
        class_mode='sparse',
        batch_size=64)

test_data = ImageDataGenerator( 
    rescale=1./255)

test_generator = test_data.flow_from_directory(
        'test',
        target_size=target_size,
        class_mode='sparse',
        batch_size=16)

x = layers.Flatten()(tflearning_model.output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(6, activation = 'softmax')(x)

from tensorflow.keras import Model
model = Model( tflearning_model.input, x)

from tensorflow.keras.optimizers import RMSprop, Adam, SGD
model.compile(loss='sparse_categorical_crossentropy',
              optimizer = 'SGD',
              metrics=['accuracy'])
model.summary()

EPOCHS=15
history = model.fit(
      train_generator,
      epochs=EPOCHS,
      validation_data=validation_generator)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./foo.png')
plt.show()

loss, accuracy = model.evaluate(test_generator)

predictions = model.predict(test_generator)
predictions[0]
np.argmax(predictions[0])
test_generator.classes[0]
