

from __future__ import absolute_import, division, print_function, unicode_literals

from itertools import dropwhile

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

# Prepare input data
files = os.listdir('dogs-vs-cats/train')

results = []
num_classes = 2

img_size = 128
num_channels = 3
train_path = 'dogs-vs-cats/train'
images = []


for image in files:
    # img = Image.open(train_path+'/'+image)
    if image.split('.')[0] == 'cat':
        results.append(0)
    if image.split('.')[0] == 'dog':
        results.append(1)
    img = load_img(train_path + '/' + image, target_size=(img_size, img_size))
    imgarr = img_to_array(img) / 255
    #  imgarr = imgarr.reshape((1,) + imgarr.shape)
    imgarr = imgarr.reshape(imgarr.shape)
    images.append(imgarr)
    '''
    imgarr = resize_image(np.array(img), img_size)/255
    imgarr = imgarr.reshape([-1,imgarr.shape[0],imgarr.shape[1],1])
    images.append(imgarr)
    '''

image_shape = images[0].shape

images_ = np.array(images)

session = tf.Session()

model = resnet(image_shape, 20)

model.compile(optimizer=optimizers.RMSprop(0.001),
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.categorical_accuracy]
              )
cp_callback = tf.keras.callbacks.ModelCheckpoint('model_2',
                                                 save_weights_only=False,
                                                  save_best_only=False,
                                                 verbose=1)
model.fit(images_,  np.array(results), batch_size=batch_size, epochs=20, callbacks=[cp_callback])
model.save_weights('model_weights_2.h5', save_format='h5')
