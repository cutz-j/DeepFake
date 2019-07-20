import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras import backend as K
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

from keras.applications import VGG16
from keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, Lambda
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import glob
from PIL import Image
from tqdm import tqdm, trange

nb_classes = 1  # number of classes
img_width, img_height = 224, 224  # change based on the shape/structure of your images
batch_size = 32  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
nb_epoch = 50  # number of iteration the algorithm gets trained.
learn_rate = 1e-5  # sgd learning rate

train_dir = 'd:/data/preprocessed_dataset/train'
validation_dir = 'd:/data/preprocessed_dataset/validation'
test_dir = 'd:/data/preprocessed_dataset/test'

img_input = Input(shape=(img_height, img_width, 3))

x = Conv2D(96, 11, strides=4, padding='same', use_bias=False)(img_input) # 15
x = Activation('relu')(x)

x = Conv2D(256, 5, strides=1, padding='same', use_bias=False)(x)
x = Activation('relu')(x)

x = MaxPooling2D(pool_size=3, strides=2, padding='valid')(x) # 8

x = Conv2D(384, 3, strides=1, padding='same', use_bias=False)(x) # 15
x = Activation('relu')(x)

x = MaxPooling2D(pool_size=3, strides=2, padding='valid')(x) # 8

x = Conv2D(384, 3, strides=1, padding='same', use_bias=False)(x) # 15
x = Activation('relu')(x)
x = Conv2D(256, 3, strides=1, padding='same', use_bias=False)(x)
x = Activation('relu')(x)

model_out = MaxPooling2D(pool_size=3, strides=2, padding='valid')(x) # 8
# Add fully connected layer
x = GlobalAveragePooling2D()(model_out)
x = Dense(4096, activation=None)(x)
x = Activation('relu')(x)
x = Dense(1, activation=None)(x)
out = Activation('sigmoid')(x)

model = Model(img_input, out)
print(model.summary())
print(len(model.trainable_weights))

model.compile(optimizer=Adam(lr=learn_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(len(model.trainable_weights))

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                        target_size=(img_height, img_width),
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        class_mode='binary')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(img_height, img_width),
                                                  batch_size=32,
                                                  shuffle=True,
                                                  class_mode='binary')


test_classes = test_generator.classes

len(test_classes[test_classes == 0])

callback_list = [EarlyStopping(monitor='val_acc', patience=5),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)]

#history = model.fit_generator(train_generator,
#                             steps_per_epoch=100,
#                             epochs=20,
#                             validation_data=validation_generator,
#                             validation_steps=len(validation_generator),
#                             callbacks=callback_list,
#                             verbose=1)

def generator(directory, batch_size=32):
    folder =  np.sort(os.listdir(directory))
    real_img = np.asarray(glob.glob(directory + '/' + folder[0]+'/*.png'))
    real_idx = np.arange(len(real_img))
    
    while 1:
        X1 = []
        X2 = []
        y = []
        
        if (len(real_idx) < batch_size):
            real_idx = np.arange(len(real_img))
            continue
        
        for _ in range(batch_size):
            if (len(real_idx) < batch_size):
                real_idx = np.arange(len(real_img))
                break
            random1 = np.random.choice(real_idx, 1, replace=False)
            real_idx = real_idx[~np.isin(real_idx, random1)]
            random2 = np.random.choice(real_idx, 1, replace=False)
            real_idx = real_idx[~np.isin(real_idx, random2)]
            X1.append(np.asarray(Image.open(real_img[random1[0]]).convert("RGB"))/255.)
            X2.append(np.asarray(Image.open(real_img[random2[0]]).convert("RGB"))/255.)
            y.append(np.array([0.]))

        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        y = np.asarray(y)
        yield [X1, X2], y
        
def generator_res(ft_dir, directory, batch_size=1):
    folder = np.sort(os.listdir(directory))
    real_img = np.asarray(glob.glob(ft_dir + '/' + '0' +'/*.png'))
    real_idx = np.arange(len(real_img))
    random1 = np.random.choice(real_idx, 1, replace=False)
    img = np.asarray(Image.open(real_img[random1[0]]).convert("RGB"))/255.
    fake_img = np.asarray(glob.glob(directory + '/' + folder[1] + '/*.png'))
    fake_idx = np.arange(len(fake_img))
    test_img = np.asarray(glob.glob(directory + '/' + folder[0] + '/*.png'))
    test_idx = np.arange(len(test_img))
    while 1:
        X1 = []
        X2 = []
        y = []
        if (len(fake_idx) < batch_size):
            break
        if (len(test_idx) < batch_size):
            break
        
        for _ in range(batch_size):
            if np.random.random() < 0.95:
            
                if (len(fake_idx) < batch_size):
                    fake_idx = np.arange(len(fake_img))
                    break
                random2 = np.random.choice(fake_idx, 1, replace=False)
                fake_idx = fake_idx[~np.isin(fake_idx, random2)]
                X1.append(img)
                X2.append(np.asarray(Image.open(fake_img[random2[0]]).convert("RGB"))/255.)
                y.append(np.array([1.]))
            
            else:
                if (len(test_idx) < batch_size):
                    test_idx = np.arange(len(test_img))
                random3 = np.random.choice(test_idx, 1, replace=False)
                test_idx = test_idx[~np.isin(test_idx, random3)]
                X1.append(img)
                X2.append(np.asarray(Image.open(test_img[random3[0]]).convert("RGB"))/255.)
                y.append(np.array([0.]))

        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        y = np.asarray(y)
        yield [X1, X2], y
        
        
def manDist(x):
    result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
    return result

def euclidean_distance(inputs):
    assert len(inputs) == 2, 'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    u, v = inputs
    return K.sqrt(K.sum((K.square(u - v + 1e-7)), axis=1, keepdims=True))  

def contrastive_loss(y_true,y_pred):
    margin=1.4
    return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))

def siamese_acc(y_true, y_pred):
    return K.mean((K.equal(y_true, K.cast(y_pred > 0.4, K.floatx()))), axis=1)

def y_pred_prt(y_true, y_pred):
    return y_pred

input_seq = Input(shape=(224, 224, 3))

ft_dir = 'd:/data/preprocessed_dataset/fine-tune'
ft_datagen = ImageDataGenerator(rescale=1./255)
ft_generator = test_datagen.flow_from_directory(ft_dir,
                                                  target_size=(img_height, img_width),
                                                  batch_size=32,
                                                  shuffle=False,
                                                  class_mode='binary')
ft_model = Model(img_input, out)
ft_model.set_weights(model.get_weights())
for l in range(len(ft_model.layers) - 2):
    ft_model.layers[l].trainable = False

ft_model.summary()
ft_model.compile(optimizer=Adam(lr=learn_rate), loss='binary_crossentropy', metrics=['accuracy'])
#history_ft = ft_model.fit_generator(ft_generator, steps_per_epoch=30, epochs=3,
#                             callbacks=callback_list, verbose=1)

model = load_model("alexnet_95_2.h5")
base_model = Model(img_input, out)
base_model.set_weights(model.get_weights())
for l in range(len(base_model.layers) - 2):
    base_model.layers[l].trainable = False   

im_in = Input(shape=(224, 224, 3))
x1 = base_model([im_in])

model_top = Model(inputs=[im_in], outputs=x1)
model_top.summary()

left_input = Input(shape=(224, 224, 3))
right_input = Input(shape=(224, 224, 3))

h1 = model_top(left_input)
h2 = model_top(right_input)

distance = Lambda(euclidean_distance)([h1, h2])
siam_model = Model(inputs=[left_input, right_input], outputs=distance)
siam_model.compile(loss='mse', optimizer=SGD(0.001), metrics=['acc'])
siam_model.summary()
train_gen = generator(ft_dir)
test_gen = generator_res(ft_dir, test_dir, 1)
callback_list = [EarlyStopping(monitor='acc', patience=3),
                 ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2)]
output = siam_model.fit_generator(train_gen, steps_per_epoch=40, epochs=10,callbacks=callback_list)

## evaluate ##
model = load_model("alexnet_95_2.h5")
model.summary()
predictions = model.predict_generator(test_generator, steps=len(test_generator))
y_pred = predictions.copy()
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0
predictions[np.isnan(predictions)] = 0
true_classes = test_generator.classes
report = metrics.classification_report(true_classes, predictions)
print(report)

fpr, tpr, thresholds = roc_curve(true_classes, y_pred, pos_label=1.)
cm = confusion_matrix(true_classes, predictions)
print(cm)
recall1 = cm[0][0] / (cm[0][0] + cm[0][1])
fallout1 = cm[1][0] / (cm[1][0] + cm[1][1])
eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
thresh = interp1d(fpr, thresholds)(eer)

plt.plot(fpr, tpr, 'o-')
plt.plot([0, 1], [0, 1], 'k--', label="random guess")

plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.show()

roc_auc_score(true_classes, predictions)

print("FPR=FAR", fallout1)
print("FNR=FRR", 1-recall1)
eer
thresh
test_loss, test_acc = model.evaluate_generator(test_generator, steps=len(test_generator))
print('test acc:', test_acc)
print('test_loss:', test_loss)


score = []
answer = []
max_iter = int(20000)
j = 0
for i in tqdm(test_gen):
    y_score = siam_model.predict_on_batch(i[0])
    score.append(y_score)
    answer.append(i[1])
    j += 1
    
score = np.concatenate(score)
answer = np.concatenate(answer)

fpr, tpr, thresholds = roc_curve(answer, score, pos_label=1.)
print(roc_auc_score(answer, score))
y_hat = score.copy()
y_hat[y_hat >= 0.8] = 1.
y_hat[y_hat < 0.8] = 0.
print(metrics.classification_report(answer, y_hat))
print(confusion_matrix(answer, y_hat))
eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
thresh = interp1d(fpr, thresholds)(eer)

cm = confusion_matrix(answer, y_hat)
recall = cm[0][0] / (cm[0][0] + cm[0][1])
fallout = cm[1][0] / (cm[1][0] + cm[1][1])
fpr2, tpr2, thresholds2 = roc_curve(true_classes, y_pred, pos_label=1.)

plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, 'r-', label="Siamese(Ours)")
plt.plot(fpr2, tpr2, 'b-', label="AlexNet")
plt.plot([0, 1], [0, 1], 'k--', label="random guess")
plt.plot([fallout], [recall], 'ro', ms=10)
plt.plot([fallout1], [recall1], 'bo', ms=10)
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.title("Best AUROC: %.3f / Model: Ours" %(roc_auc_score(answer, score)))
plt.legend(loc='lower right')
plt.annotate("%.3f: AlexNet" %(roc_auc_score(true_classes, y_pred)), xy=(0.0, 0.98), xytext=(0.15, 0.75), arrowprops={'color':'blue'})
plt.annotate("%.3f: Ours" %(roc_auc_score(answer, score)), xy=(0.0, 0.99), xytext=(0.15, 0.9), arrowprops={'color':'red'})
plt.show()

print("FPR=FAR", fallout)
print("FNR=FRR", 1-recall)

eer
thresh
len(y_hat[np.equal(y_hat, answer)]) / len(y_hat)












