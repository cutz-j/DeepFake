### MNIST anomaly data input ###
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers, activations, optimizers, losses, Input, callbacks
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

y_train = to_categorical(train_labels) # one-hot
y_test = to_categorical(test_labels) # one-hot

train_images = train_images.reshape([60000, 28, 28, 1]).astype(np.float32)
test_images = test_images.reshape([10000, 28, 28, 1]).astype(np.float32)
                          
input_layer = Input(shape=(28, 28, 1))             
# layer 1
x = layers.Conv2D(filters=4, kernel_size=3, padding='valid', activation=activations.relu)(input_layer) # 3x3
x = layers.BatchNormalization()(x)
# layer 2
x = layers.Conv2D(filters=4, kernel_size=3, padding='valid', activation=activations.relu)(x)
x = layers.BatchNormalization()(x)
# maxpool
x = layers.MaxPool2D()(x)
# layer 3
x = layers.Conv2D(filters=12, kernel_size=3, padding='valid', activation=activations.relu)(x)
x = layers.BatchNormalization()(x)
# layer 4 --> down channel
x = layers.Conv2D(filters=10, kernel_size=3, padding='valid', activation=activations.relu)(x)
x = layers.BatchNormalization()(x)
# maxpool
x = layers.MaxPool2D()(x)
# Global Averae Pool
x = layers.GlobalAveragePooling2D()(x)
predict = layers.Dense(10, activation='softmax')(x)
model = models.Model(input_layer, predict)
model.summary()

## call back ##
callback_list = [callbacks.EarlyStopping(monitor='val_acc', patience=5),
                 callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)]

model.compile(optimizer=optimizers.adam(), loss=losses.categorical_crossentropy,
              metrics=['acc'])
history = model.fit(x=train_images, y=y_train, batch_size=32, epochs=30, 
                    callbacks=callback_list, validation_split=0.1)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
vall_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Train ACC')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.legend()
plt.show()

model.evaluate(x=test_images, y=y_test) # 97% ACC

### 연예인 흑백사진 데이터 ###
image_dir = "d:/github/keras/MNIST/test"
list_dir = os.listdir(image_dir)
test_data = np.zeros(shape=[6, 28, 28, 1])

for i in range(len(list_dir)):
    image = Image.open(image_dir+"/"+list_dir[i]).convert("L")
    image = image.resize([28, 28])
    image.save("d:/data/mono%d.png" %(i))
    test_data[i, :, :, :] = np.array(image).reshape([28, 28, 1])

predict = model.predict(test_data)
predict_idx = np.argmax(predict, axis=1)


