# -*- coding: utf-8 -*-

import pandas as pd

from ml_dnn_fc import ML_DNN


if __name__ == '__main__':
    x_train_scale = pd.read_csv('x_train_scale.csv').set_index('index')
    y_train = pd.read_csv('y_train.csv')
    
    x_test_scale = pd.read_csv('x_test_scale.csv').set_index('index')
    dnn = ML_DNN()
    
    dnn.learn(x_train_scale, y_train)
    print(dnn.predict_one_proba(x_test_scale))