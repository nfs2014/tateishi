# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:00:00 2018
@author: jpbank.quants
"""
import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD, Adam, Adagrad
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from keras import backend as K
from keras.utils import np_utils
from ml_base import ML_Base

class ML_DNN(ML_Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._nb_epoch = kwargs.get('nb_epoch',10)
        self._batch_size = kwargs.get('batch_size', 30)
        self._params = {'out_dim1': kwargs.get('out_dim1',32),
                        'out_dim2': kwargs.get('out_dim2',32),
                        'label_dim': kwargs.get('label_dim',2),
                        'optimizer': kwargs.get('optimizer','adam'),
                        'dropout1': kwargs.get('dropout1',0.4),
                        'dropout2': kwargs.get('dropout2',0.3),
                        'activation': kwargs.get('activation','relu'),
                        }


    def learn(self, training_data, training_label, tunes_param=False):
        seed = 1234
        np.random.seed(seed)
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="0", # specify GPU number
                                                          allow_growth=True))
        sess = tf.Session(config=config)
        K.set_session(sess)

        #model_file_path = '{model_name}_{value_date}.h5'.format(model_name=self.__class__.__name__,
        #                                                        value_date=training_data.index[-1].strftime('%Y%m%d'))
        #model_file_path = os.path.join('output', 'model', model_file_path)
        
        self._params['label_dim'] = np.max(training_label).iloc[0] + 1
        self._model = KerasClassifier(build_fn=self._create_model,
                                        input_dim=training_data.shape[1],
                                        verbose=0,
                                        **self._params, 
                                        )
        hist = self._model.fit(np.array(training_data)
                                , training_label
                                , batch_size=self._batch_size
                                , epochs=self._nb_epoch
                                )
        #import matplotlib.pyplot as plt
        #plt.plot(hist.history['loss'])
        #import pdb;pdb.set_trace()
            

    def predict_one(self, test_data):
        if type(test_data) != np.array:
            test_data = np.array(test_data)
        if self._is_regression:
            return float(self._model.predict(test_data))
        else:
            return self._model.predict(test_data)[0]

    def predict(self, test_data):
        if type(test_data) != np.array:
            test_data = np.array(test_data)

        if self._is_regression:
            return super().predict(test_data)
        else:
            return self._model.predict(test_data)


    def _create_model(self, 
                      input_dim, 
                      out_dim1,
                      out_dim2,  
                      label_dim,  
                      optimizer, 
                      dropout1,  
                      dropout2,  
                      activation='relu'):
        loss_func = 'categorical_crossentropy'
        model = Sequential()
        model.add(Dense(out_dim1, 
                        input_dim=input_dim))
        model.add(Activation(activation))
        model.add(Dropout(dropout1))

        model.add(Dense(out_dim2))
        model.add(Activation(activation))
        model.add(Dropout(dropout2))

        model.add(Dense(units=label_dim))
        model.add(Activation('softmax'))
        
        model.compile(loss=loss_func
                      , optimizer='rmsprop'
                      , metrics=['accuracy'])
        return model


    def dispose(self):
        super().dispose()
        K.clear_session()


