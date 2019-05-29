### MNIST squeeze ###
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers, activations, optimizers, losses, Input, callbacks
from keras.layers import Activation, Conv2D
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os
from keras import backend as K

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

y_train = to_categorical(train_labels) # one-hot
y_test = to_categorical(test_labels) # one-hot

train_images = train_images.reshape([60000, 28, 28, 1]).astype(np.float32)
test_images = test_images.reshape([10000, 28, 28, 1]).astype(np.float32)
                          


## FIRE module ##
def fire(x, squeeze=16, expand=64):
    '''
    function: fire module
    
    input:
        - x: layer
        - squeeze = squeeze layer
        - expand = expand layer
    
    output:
        - module concat
        - 
    '''
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
        
    x = layers.Conv2D(filters=squeeze, kernel_size=(1, 1), padding='valid',
                      activation='relu')(x)
    e1 = layers.Conv2D(filters=expand, kernel_size=(1, 1), padding='valid',
                       activation='relu')(x)
    e3 = layers.Conv2D(filters=expand, kernel_size=(3, 3), padding='same',
                       activation='relu')(x)
    x = layers.concatenate([e1, e3], axis=channel_axis)
    
    return x


### Layer building ###
input_layer = Input(shape=(28, 28, 1)) 
x = Conv2D(filters=96, kernel_size=(3, 3), padding='valid')(input_layer)
x = Activation('relu')(x)
x = layers.MaxPool2D()(x)

fire2 = fire(x, squeeze=16, expand=64)
fire3 = fire(fire2, squeeze=16, expand=64)
x = layers.add([fire2, fire3])
fire4 = fire(x, squeeze=32, expand=128)
x = layers.MaxPool2D()(fire4)

fire5 = fire(x, squeeze=32, expand=128)
x = layers.add([x, fire5])
fire6 = fire(x, squeeze=48, expand=192)
fire7 = fire(fire6, squeeze=48, expand=192)
x = layers.add([fire6, fire7])
fire8 = fire(x, squeeze=64, expand=256)
x = layers.MaxPool2D()(fire8)

fire9 = fire(x, squeeze=64, expand=256)
x = layers.add([x, fire9])
x = Conv2D(filters=1000, kernel_size=(1, 1), padding='valid')(x)
x = Activation('relu')(x)

x = layers.GlobalAveragePooling2D()(x)
predict = Activation('softmax')(x)

model = models.Model(input_layer, predict)
model.summary()







































