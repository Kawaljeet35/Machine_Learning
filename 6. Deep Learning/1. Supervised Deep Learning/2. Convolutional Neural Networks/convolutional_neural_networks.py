# Convolutional Neural Networks

# Importing the libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Preprocessing the Training set
# Creating an instance of the ImageDataGenerator class
# rescale = 1./255 will feature scale by dividing each pixel by 255 to make it between 0 & 1
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
# Connecting the instance to our training dataset
# target_size = (64, 64) is final size of img to be fed into CNN
training_set = train_datagen.flow_from_directory('dataset/training_set',target_size = (64, 64),batch_size = 32,class_mode = 'binary')

# Preprocessing the Test set
# Creating an instance of the ImageDataGenerator class
# rescale = 1./255 will feature scale by dividing each pixel by 255 to make it between 0 & 1
test_datagen = ImageDataGenerator(rescale = 1./255)
# Connecting the instance to our test dataset
# target_size = (64, 64) is final size of img to be fed into CNN
test_set = test_datagen.flow_from_directory('dataset/test_set',target_size = (64, 64),batch_size = 32,class_mode = 'binary')

# Building the CNN
# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Adding an Input layer
# 64X64 is size of images and 3 means colored images in rgb format
cnn.add(tf.keras.layers.Input(shape=(64, 64, 3)))

# Step 1 - Convolution
# filters = 32 is no of feature detectors (kernels) to be applied
# kernel_size = 3 means 3X3 feature detectors (kernels)
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))

# Step 2 - Pooling
# pool_size = 2 means 2X2 frame used to make pooled map
# strides = 2 means shifting by 2 pixels
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
# Filters = 32 is no of feature detectors (kernels) to be applied
# Kernel_size = 3 means 3X3 feature detectors (kernels)
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
# pool_size = 2 means 2X2 frame used to make pooled map
# strides = 2 means shifting by 2 pixels
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
# units = 128 means number of neurons
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
# units = 1 means only 1 neuron
# activation = sigmoid for binary classification
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the CNN

# Compiling the CNN
# optimizer = adam for stochastic gradient descent, loss = binary_crossentropy
# for binary outcome, metrics = accuracy as a list
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

# Part 4 - Making a single prediction

import numpy as np
from tensorflow.keras.utils import load_img, img_to_array # type: ignore

test_image = load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
# predict method expects a numpy array
test_image = img_to_array(test_image)
# adding extra dimension to image as training was done in batches
test_image = np.expand_dims(test_image, axis = 0)

result = cnn.predict(test_image/255.0)
training_set.class_indices

if result[0][0] > 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)

