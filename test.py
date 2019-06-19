from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras import datasets, layers, models, optimizers

from tensorflow.keras.preprocessing.image import img_to_array, load_img

from tensorflow.keras.regularizers import l2


import tensorflow as tf

from datetime import timedelta

import numpy as np

import os

from numpy.random import seed

seed(1)

from tensorflow import set_random_seed

from res_net import resnet

set_random_seed(2)

batch_size = 32

#Prepare input data
files = os.listdir('dogs-vs-cats/validate_dense')

results = []
num_classes = 2

img_size = 128
num_channels = 3
train_path='dogs-vs-cats/validate_dense'
images = []
labels = []
for image in files:
    img = load_img(train_path + '/' + image, target_size=(img_size, img_size))
    imgarr = img_to_array(img) / 255
    #  imgarr = imgarr.reshape((1,) + imgarr.shape)
    imgarr = imgarr.reshape(imgarr.shape)
    if image.split('.')[0] == 'cat':
        labels.append(0)
    if image.split('.')[0] == 'dog':
        labels.append(1)
    images.append(imgarr)
image_shape = images[0].shape
images = images

model = resnet(image_shape, 20)

model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.categorical_accuracy]
    )

model.load_weights('model_2')

results = model.predict(np.array(images))

accuracy = 0
predictions = []

for i in range(len(results)):

    if results[i][0] >= results[i][1]:
        predictions.append(0)
    if results[i][0] < results[i][1]:
        predictions.append(1)
    if results[i][0] > results[i][1] and labels[i] == 0:
        accuracy += 1
    elif results[i][0] < results[i][1] and labels[i] == 1:
        accuracy += 1
print('accuracy :',accuracy/len(results))
with tf.Session() as sess:
    matrix = tf.confusion_matrix(labels, predictions)
    confusion_matrix_to_Print = sess.run(matrix)
    print(confusion_matrix_to_Print)

