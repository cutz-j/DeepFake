from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU, concatenate, GlobalAveragePooling2D, Input, BatchNormalization, SeparableConv2D, Subtract, concatenate
from keras.activations import relu, softmax
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras import backend as K
from preprocess import create_couple, create_couple_rgbd, create_wrong, create_wrong_rgbd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def euclidean_distance(inputs):
    assert len(inputs) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    u, v = inputs
    return K.sqrt(K.sum((K.square(u - v)), axis=1, keepdims=True))
        

def contrastive_loss(y_true,y_pred):
    margin=1.
    return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))
   # return K.mean( K.square(y_pred) )

### squeeze NET ###   
def fire(x, squeeze=16, expand=64):
    x = Convolution2D(squeeze, (1,1), padding='valid')(x)
    x = Activation('relu')(x)
    
    left = Convolution2D(expand, (1,1), padding='valid')(x)
    left = Activation('relu')(left)
    
    right = Convolution2D(expand, (3,3), padding='same')(x)
    right = Activation('relu')(right)
    
    x = concatenate([left, right], axis=3)
    return x   

img_input=Input(shape=(200,200,4))

x = Convolution2D(64, (5, 5), strides=(2, 2), padding='valid')(img_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

x = fire(x, squeeze=16, expand=16)
x = fire(x, squeeze=16, expand=16)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

x = fire(x, squeeze=32, expand=32)
x = fire(x, squeeze=32, expand=32)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

x = fire(x, squeeze=48, expand=48)
x = fire(x, squeeze=48, expand=48)
x = fire(x, squeeze=64, expand=64)
x = fire(x, squeeze=64, expand=64)
x = Dropout(0.2)(x)

x = Convolution2D(512, (1, 1), padding='same')(x)
out = Activation('relu')(x)

modelsqueeze= Model(img_input, out)
modelsqueeze.summary()

im_in = Input(shape=(200,200,4))
#wrong = Input(shape=(200,200,3))

x1 = modelsqueeze(im_in)
#x = Convolution2D(64, (5, 5), padding='valid', strides =(2,2))(x)

#x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x1)

"""
x1 = Convolution2D(256, (3,3), padding='valid', activation="relu")(x1)
x1 = Dropout(0.4)(x1)

x1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x1)

x1 = Convolution2D(256, (3,3), padding='valid', activation="relu")(x1)
x1 = BatchNormalization()(x1)
x1 = Dropout(0.4)(x1)

x1 = Convolution2D(64, (1,1), padding='same', activation="relu")(x1)
x1 = BatchNormalization()(x1)
x1 = Dropout(0.4)(x1)
"""

### 2 Models: siamese NET ###

x1 = Flatten()(x1)
x1 = Dense(512, activation="relu")(x1)
x1 = Dropout(0.2)(x1)
#x1 = BatchNormalization()(x1)
feat_x = Dense(128, activation="linear")(x1)
feat_x = Lambda(lambda  x: K.l2_normalize(x,axis=1))(feat_x)

model_top = Model(inputs = [im_in], outputs = feat_x)
model_top.summary()

im_in1 = Input(shape=(200,200,4))
im_in2 = Input(shape=(200,200,4))

feat_x1 = model_top(im_in1)
feat_x2 = model_top(im_in2)

lambda_merge = Lambda(euclidean_distance)([feat_x1, feat_x2])
model_final = Model(inputs = [im_in1, im_in2], outputs = lambda_merge)
model_final.summary()

adam = Adam(lr=0.001)
sgd = SGD(lr=0.001, momentum=0.9)
model_final.compile(optimizer=adam, loss=contrastive_loss)

def generator(batch_size):
  while 1:
    X=[]
    y=[]
    switch=True
    for _ in range(batch_size):
   #   switch += 1
      if switch:
     #   print("correct")
        X.append(create_couple_rgbd("d:/data/faceid_train/").reshape((2,200,200,4)))
        y.append(np.array([0.]))
      else:
     #   print("wrong")
        X.append(create_wrong_rgbd("d:/data/faceid_train/").reshape((2,200,200,4)))
        y.append(np.array([1.]))
      switch=not switch
    X = np.asarray(X)
    y = np.asarray(y)
    XX1=X[0,:]
    XX2=X[1,:]
    yield [X[:,0],X[:,1]],y
    
def val_generator(batch_size):
  while 1:
    X=[]
    y=[]
    switch=True
    for _ in range(batch_size):
      if switch:
        X.append(create_couple_rgbd("d:/data/faceid_val/").reshape((2,200,200,4)))
        y.append(np.array([0.]))
      else:
        X.append(create_wrong_rgbd("d:/data/faceid_val/").reshape((2,200,200,4)))
        y.append(np.array([1.]))
      switch=not switch
    X = np.asarray(X)
    y = np.asarray(y)
    XX1=X[0,:]
    XX2=X[1,:]
    yield [X[:,0],X[:,1]],y
    
    
gen = generator(16)
val_gen = val_generator(4)

outputs = model_final.fit_generator(gen, steps_per_epoch=30, epochs=50, validation_data = val_gen, validation_steps=20)

## raw output ##
im_in1 = Input(shape=(200,200,4))
#im_in2 = Input(shape=(200,200,4))

feat_x1 = model_top(im_in1)
#feat_x2 = model_top(im_in2)



model_output = Model(inputs = im_in1, outputs = feat_x1)

model_output.summary()

adam = Adam(lr=0.001)

sgd = SGD(lr=0.001, momentum=0.9)

model_output.compile(optimizer=adam, loss=contrastive_loss)

model_final.save("faceid_big_rgbd_2.h5")

def create_input_rgbd(file_path):
  #  print(folder)
    mat=np.zeros((480,640), dtype='float32')
    i=0
    j=0
    depth_file = file_path
    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue    
                if int(val) > 1200 or int(val) == -1: val= 1200
                mat[i][j]=float(int(val))
                j+=1
                j=j%640

            i+=1
        mat = np.asarray(mat)
    mat_small=mat[140:340,220:420]
    img = Image.open(depth_file[:-5] + "c.bmp")
    img.thumbnail((640,480))
    img = np.asarray(img)
    img = img[140:340,220:420]
    mat_small=(mat_small-np.mean(mat_small))/np.max(mat_small)
#    plt.figure(figsize=(8,8))
#    plt.grid(True)
#    plt.xticks([])
#    plt.yticks([])
#    plt.imshow(mat_small)
#    plt.show()
#    plt.figure(figsize=(8,8))
#    plt.grid(True)
#    plt.xticks([])
#    plt.yticks([])
#    plt.imshow(img)
#    plt.show()
    
    full1 = np.zeros((200,200,4))
    full1[:,:,:3] = img[:,:,:3]
    full1[:,:,3] = mat_small
    
    return np.array([full1])

## Visualization ###
outputs=[]
n=0
for folder in glob.glob('d:/data/faceid_train/*'):
  i=0
  for file in glob.glob(folder + '/*.dat'):
    i+=1
    outputs.append(model_output.predict(create_input_rgbd(file).reshape((1,200,200,4))))
  print(i)
  n+=1
  print("Folder ", n, " of ", len(glob.glob('d:/data/faceid_train/*')))
print(len(outputs))


outputs= np.asarray(outputs)
outputs = outputs.reshape((-1,128))
outputs.shape

## data compression ##
X_embedded = TSNE(2).fit_transform(outputs)
X_embedded.shape
X_PCA = PCA(3).fit_transform(outputs)
print(X_PCA.shape)

color = 0
for i in range(len((X_embedded))):
  el = X_embedded[i]
  if i % 51 == 0 and not i==0:
    color+=1
    color=color%10
  plt.scatter(el[0], el[1], color="C" + str(color))
  

file1 = ('d:/data/faceid_train/(2012-05-16)(154211)/015_1_d.dat')
inp1 = create_input_rgbd(file1)
file1 = ('d:/data/faceid_train/(2012-05-16)(154211)/011_1_d.dat')
inp2 = create_input_rgbd(file1)

model_final.predict([inp1, inp2])

