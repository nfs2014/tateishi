# -*- coding: utf-8 -*-
"""
Created on Wed Dec 5 19:30:00 2018
@author: jpbank.quants
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd

from sklearn.decomposition import KernelPCA, PCA
from abc import ABCMeta, abstractmethod


class ML_Base(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        self._model = None


    def learn(self, training_data, training_label):
        if type(training_label) == pd.DataFrame:
            self._model = self._model.fit(np.array(training_data), np.array(training_label).T[0])
        else:
            self._model = self._model.fit(np.array(training_data), np.array(training_label))


    def predict(self, test_data):
        return self._model.predict(test_data)


    def predict_one(self, test_data):
        return self._model.predict(test_data)[0]

    def predict_proba(self, test_data):
        return self._model.predict_proba(test_data)

    def predict_one_proba(self, test_data):
        return self._model.predict_proba(test_data)[0].tolist()

    def dispose(self):
        self._model = None
        del self._model
    