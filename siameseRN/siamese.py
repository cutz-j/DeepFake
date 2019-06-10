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

## preprocess ##
data_list = []
digit = []
file_open = open("d:/data/srn/Train_Arabic_Digit.txt")
all_data = file_open.readlines()
for data in all_data:
    arr = np.array(data.strip().split(), dtype=np.float32)
    if data == '            \n':
        data_arr = np.array(data_list)
        digit.append(data_arr)
        data_arr = []
    else:
        data_list.append(arr)
    
        

file_open.close()