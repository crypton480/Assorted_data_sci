#https://www.tensorflow.org/tutorials/keras/basic_classification

import tensorflow as tf
from tensorflow import keras
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print("Data Downloaded")

train_images = train_images / 255.0

test_images = test_images / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # change from square matrix of image to flat vector
    keras.layers.Dense(128, activation=tf.nn.relu), # middle layer of 128 neurons using relu activation
    keras.layers.Dense(10, activation=tf.nn.softmax) # final layer mapping to one of 10 image categories
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Fitting Model")
model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

i = 0
print('testing image, ', i)
predictions = model.predict(test_images)
print("predicted value - " , np.argmax(predictions[i]), " ", class_names[np.argmax(predictions[i])])
print("actual value - ", test_labels[i], class_names[test_labels[i]])