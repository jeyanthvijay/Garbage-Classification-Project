from _future_ import division, print_function
import os
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from flask import Flask, request, render_template, jsonify
import numpy as np
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Define directories
train_dir = r"C:\Users\Vijay\PycharmProjects\VijayB\Dataset\trainset"
test_dir = r"C:\Users\Vijay\PycharmProjects\VijayB\Dataset\testset"

# Define image generators
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1. / 255)

# Load datasets
train_transform = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical'
)

test_transform = val_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical'
)

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(kernel_initializer='uniform', activation='relu', units=150))
model.add(Dense(kernel_initializer='uniform', activation='relu', units=68))
model.add(Dense(kernel_initializer='uniform', activation='softmax', units=7))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Train the model
res = model.fit(train_transform, steps_per_epoch=2187 // 64, validation_steps=564 // 64, epochs=30,
                validation_data=test_transform)
model.save('Garbage1.h5')

