import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Input, BatchNormalization, Subtract, concatenate
from keras.layers import GRU
from keras.activations import relu, softmax
from keras.optimizers import RMSprop
from keras import backend as K
from PIL import Image
from keras.models import load_model
from tqdm import tqdm, trange
from keras.preprocessing import sequence
from keras import losses, optimizers

# sigmoid distance weight inner product #
def diag(inputs):
    h1, h2 = inputs
    diag_part = tf.diag_part(K.dot(K.transpose(h1), h2))
    res = K.reshape(diag_part, (1, 32))
    return res # (32, 1)

def random_num():
    rand_int1 = np.random.randint(low=0, high=9)
    rand_int2 = np.random.randint(low=0, high=9)
    return [rand_int1, rand_int2]

## preprocess ##
def preprocess(directory):
    data_list = []
    digit = []
    file_open = open(directory)
    all_data = file_open.readlines()
    for data in all_data:
        arr = np.array(data.strip().split(), dtype=np.float32)
        if data == '            \n':
            if data_list == []:
                continue
            data_arr = np.array(data_list)
            digit.append(data_arr)
            data_list = []
        else:
            data_list.append(arr)
    data_arr = np.array(data_list)
    digit.append(data_arr)
    file_open.close()
    return digit

digit = preprocess("d:/data/srn/Train_Arabic_Digit.txt")
val = preprocess("d:/data/srn/Test_Arabic_Digit.txt")

y_train = np.zeros(shape=[6600, 1])
j = -1
for i in range(6600):
    if i % 660 == 0:
        j += 1
    y_train[i] = j
                
x_train = np.array(digit)

## normalization ##
counter = 0
total_sum = 0
for i in tqdm(x_train):
    for j in i:
        for k in j:
            counter +=1
            total_sum += k

mean = total_sum / counter

standard_deviation = 0
for i in tqdm(x_train):
    for j in i:
        for k in j:
            standard_deviation += (k-mean)**2
standard_deviation = standard_deviation / counter
standard_deviation = standard_deviation ** 0.5
standard_deviation

x_train -= mean
x_train /= standard_deviation

## test norm ##
x_test = np.array(val)
y_test = np.zeros(shape=[2200, 1])
j = -1
for i in range(2200):
    if i % 220 == 0:
        j += 1
    y_test[i] = j

x_test -= mean
x_test /= standard_deviation

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

## GRU building ##
x_train = sequence.pad_sequences(x_train, maxlen=100,dtype="float32" ) 
x_test = sequence.pad_sequences(x_test, maxlen=100,dtype="float32")

data_input=Input(shape=(x_train.shape[1], x_train.shape[2]))
x = GRU(32, kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
        dropout=0.5, recurrent_dropout=0.5)(data_input)
GRU_model = Model(data_input, x)
GRU_model.summary() # output = (None, 32)

im_in1 = Input(shape=(x_train.shape[1], x_train.shape[2]))
im_in2 = Input(shape=(x_train.shape[1], x_train.shape[2]))

h1 = GRU_model(im_in1)
h2 = GRU_model(im_in2)

lambda_diag = Lambda(diag)([h1, h2])
top = Dense(1, activation=None)(lambda_diag)
out = Activation('sigmoid')(top)
final_model = Model(inputs=[im_in1, im_in2], outputs=out)
final_model.summary()

def generator(batch_size):
  while 1:
    X=[]
    y=[]
    switch=True
    for _ in range(batch_size):
   #   switch += 1
      if switch:
     #   print("correct")
        rand_int = np.random.randint(low=0, high=9)
        range_num = range(rand_int*660, (rand_int+1)*660)
        idx1 = np.random.randint(low=range_num[0], high=range_num[-1])
        idx2 = np.random.randint(low=range_num[0], high=range_num[-1])
        X.append(np.array([x_train[idx1], x_train[idx2]]))
        y.append(np.array([0.]))
      else:
     #   print("wrong")
        rand_int1, rand_int2 = random_num()
        if (rand_int1 == rand_int2):
            rand_int1, rand_int2 = random_num()
        else:
            range_num1 = range(rand_int1*660, (rand_int1+1)*660)
            range_num2 = range(rand_int2*660, (rand_int2+1)*660)
            idx1 = np.random.randint(low=range_num1[0], high=range_num1[-1])
            idx2 = np.random.randint(low=range_num2[0], high=range_num2[-1])
            X.append(np.array([x_train[idx1], x_train[idx2]]))
            y.append(np.array([1.]))
      switch=not switch
    X = np.asarray(X)
    y = np.asarray(y)
    yield [X[:,0],X[:,1]],y
    
def val_generator(batch_size):
  while 1:
    X=[]
    y=[]
    switch=True
    for _ in range(batch_size):
      if switch:
        rand_int = np.random.randint(low=0, high=9)
        range_num = range(rand_int*220, (rand_int+1)*220)
        idx1 = np.random.randint(low=range_num[0], high=range_num[-1])
        idx2 = np.random.randint(low=range_num[0], high=range_num[-1])
        X.append(np.array([x_test[idx1], x_test[idx2]]))
        y.append(np.array([0.]))
      else:
        rand_int1, rand_int2 = random_num()
        if (rand_int1 == rand_int2):
            rand_int1, rand_int2 = random_num()
        else:
            range_num1 = range(rand_int1*220, (rand_int1+1)*220)
            range_num2 = range(rand_int2*220, (rand_int2+1)*220)
            idx1 = np.random.randint(low=range_num1[0], high=range_num1[-1])
            idx2 = np.random.randint(low=range_num2[0], high=range_num2[-1])
            X.append(np.array([x_test[idx1], x_test[idx2]]))
            y.append(np.array([1.]))
      switch=not switch
    X = np.asarray(X)
    y = np.asarray(y)
    yield [X[:,0],X[:,1]],y    
    
gen = generator(32)
val_gen = val_generator(16)
final_model.compile(optimizer=optimizers.adam(), loss=losses.binary_crossentropy, metrics=['accuracy'])
outputs = final_model.fit_generator(gen, steps_per_epoch=200, epochs=50,
                                    validation_data = val_gen, validation_steps=20)










