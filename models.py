import numpy as np
import cv2
import dlib

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.layers.recurrent import LSTM, GRU
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

import logging
import sys

sys.setrecursionlimit(2 ** 20)
np.random.seed(2 ** 10)


def _load_model(filename):
    return load_model(filename)

def pad_sequences(dataSet, seq_len = 24):
    
    from copy import deepcopy

    X = []
    Y = []

    for data in range(len(dataSet) - seq_len):
        temp = []
        x = 0
        while x < seq_len:
            if type(dataSet[data + x]) is not list:
                temp.append([dataSet[data + x]])
            else:    
                temp.append(dataSet[data + x])
                
            x += 1
            
        X.append(deepcopy(temp))
        Y.append(deepcopy(dataSet[data + x]))

    return np.array(X), np.array(Y)


def _shuffle(X,Y):
    
    from random import shuffle
        
    data = list(zip(X, Y))
    shuffle(data)

    dataSet_X, dataSet_Y = zip(*data)

    dataSet_X = np.array(dataSet_X)        
    dataSet_Y = np.array(dataSet_Y)

    return dataSet_X, dataSet_Y


def fitModel(model, train_X, train_Y, filename, val_rate = 0.1, epochs = 1000, batch_size = None):
    
    X,Y = _shuffle(train_X, train_Y)
   
    idx = int((len(X) * val_rate))
    
    val_X, val_Y = X[-idx:], Y[-idx:]
    X, Y = X[:-idx], Y[:-idx]
    
    if batch_size == 0:
        batch_size = len(train_X)
     
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    mc = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    
    return model.fit(X, Y, validation_data = (val_X, val_Y),
              epochs = epochs, batch_size = batch_size,
              callbacks = [es,mc])

def visModel(model,savePath = None, name = 'model_structure'):
        
    if not savePath == None:
        plot_model(model,to_file= savePath + name + '.png' , show_shapes=True)

    return model_to_dot(model, show_shapes=True).create(prog='dot', format='svg')


class IntensRecModelCreater:
    
    def __init__(self, look_back = 24, input_dim = 1, recBlock_type = 'GRU', power = 64):
        
        self.inputSize = (look_back,input_dim)
        self.recBlock_type = recBlock_type
        self.power = power
        
    def __call__(self):
        
        model = Sequential()
    
        if self.recBlock_type == 'GRU':
            model.add(GRU(self.power, return_sequences = False, input_shape = self.inputSize))
            
        elif self.recBlock_type == 'LSTM':   
            model.add(LSTM(self.power, return_sequences = False, input_shape = self.inputSize))

        model.add(Dense(1,activation = 'relu'))

        model.compile(loss = 'mse', optimizer = 'adam')

        return model

    
class FaceDetection:
    
    def __init__(self):
        logging.debug("Load Face Detection model...")
        self.detector = dlib.get_frontal_face_detector()
        
    def __call__(self,img):
        
        if type(img) is str:
            img = cv2.imread(img)
        
        return self.detector(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

class FaceDescription:
    
    def __init__(self, landmarksPath, recModelPath):
        logging.debug("Load Face Description model...")
        self.landmarks = dlib.shape_predictor(landmarksPath)
        self.recModel = dlib.face_recognition_model_v1(recModelPath)

    def __call__(self,img, dets = None):
        
        if type(img) is str:
            img = cv2.imread(img)
            dets = None
            
        if dets == None:
            dets = FaceDetection().faceDetection(img)
            
        faces = []

        for d in dets:
            shape = self.landmarks(img, d)
            faces.append(np.array(self.recModel.compute_face_descriptor(img, shape)))

        return faces
    
class WideResNetCreater:
    
    def __init__(self, image_size = 64, depth=16, k=8):
        self._depth = depth
        self._k = k
        self._dropout_probability = 0
        self._weight_decay = 0.0005
        self._use_bias = False
        self._weight_init = "he_normal"

        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

    def _wide_basic(self, n_input_plane, n_output_plane, stride):
        def f(net):
 
            conv_params = [[3, 3, stride, "same"],
                           [3, 3, (1, 1), "same"]]

            n_bottleneck_plane = n_output_plane
        
            for i, v in enumerate(conv_params):
                if i == 0:
                    if n_input_plane != n_output_plane:
                        net = BatchNormalization(axis=self._channel_axis)(net)
                        net = Activation("relu")(net)
                        convs = net
                    else:
                        convs = BatchNormalization(axis=self._channel_axis)(net)
                        convs = Activation("relu")(convs)
                        
                    convs = Convolution2D(n_bottleneck_plane, nb_col=v[0], nb_row=v[1],
                                          subsample=v[2],
                                          border_mode=v[3],
                                          init=self._weight_init,
                                          W_regularizer=l2(self._weight_decay),
                                          bias=self._use_bias)(convs)
                else:
                    convs = BatchNormalization(axis=self._channel_axis)(convs)
                    convs = Activation("relu")(convs)
                    if self._dropout_probability > 0:
                        convs = Dropout(self._dropout_probability)(convs)
                    convs = Convolution2D(n_bottleneck_plane, nb_col=v[0], nb_row=v[1],
                                          subsample=v[2],
                                          border_mode=v[3],
                                          init=self._weight_init,
                                          W_regularizer=l2(self._weight_decay),
                                          bias=self._use_bias)(convs)

            if n_input_plane != n_output_plane:
                shortcut = Convolution2D(n_output_plane, nb_col=1, nb_row=1,
                                         subsample=stride,
                                         border_mode="same",
                                         init=self._weight_init,
                                         W_regularizer=l2(self._weight_decay),
                                         bias=self._use_bias)(net)
            else:
                shortcut = net

            return add([convs, shortcut])

        return f


    def _layer(self, block, n_input_plane, n_output_plane, count, stride):
        def f(net):
            net = block(n_input_plane, n_output_plane, stride)(net)
            for i in range(2, int(count + 1)):
                net = block(n_output_plane, n_output_plane, stride=(1, 1))(net)
            return net

        return f

    def __call__(self):
        logging.debug("Creating Age and Gender Estimation model...")

        assert ((self._depth - 4) % 6 == 0)
        n = (self._depth - 4) / 6

        inputs = Input(shape=self._input_shape)

        n_stages = [16, 16 * self._k, 32 * self._k, 64 * self._k]

        conv1 = Convolution2D(nb_filter=n_stages[0], nb_row=3, nb_col=3,
                              subsample=(1, 1),
                              border_mode="same",
                              init=self._weight_init,
                              W_regularizer=l2(self._weight_decay),
                              bias=self._use_bias)(inputs)

        
        block_fn = self._wide_basic
        conv2 = self._layer(block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], count=n, stride=(1, 1))(conv1)
        conv3 = self._layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[2], count=n, stride=(2, 2))(conv2)
        conv4 = self._layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[3], count=n, stride=(2, 2))(conv3)
        batch_norm = BatchNormalization(axis=self._channel_axis)(conv4)
        relu = Activation("relu")(batch_norm)

        pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), border_mode="same")(relu)
        flatten = Flatten()(pool)
        predictions_g = Dense(output_dim=2, init=self._weight_init, bias=self._use_bias,
                              W_regularizer=l2(self._weight_decay), activation="softmax")(flatten)
        predictions_a = Dense(output_dim=101, init=self._weight_init, bias=self._use_bias,
                              W_regularizer=l2(self._weight_decay), activation="softmax")(flatten)

        model = Model(input=inputs, output=[predictions_g, predictions_a])

        return model