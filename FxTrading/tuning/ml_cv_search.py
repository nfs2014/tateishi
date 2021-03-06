"""This class is to optimize hyperparameter parameter with Random/Grid Search"""
# -*- coding: utf-8 -*-

import logging
from abc import ABCMeta, abstractmethod
import multiprocessing as mp

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint as sp_randint
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sklearn.svm as svm
from xgboost.sklearn import XGBClassifier, XGBRegressor
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier

class BaseSearchCV(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        self._logger = logging.getLogger("jpbank.quants")
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        self._n_iter = kwargs.get('niter', 300)
        self._cv = kwargs.get('cv', 3)
        self._is_regression = kwargs.get('is_regression', False)
        if kwargs.get('with_grid_cv', True):
            self._search_model = GridSearchCV
        else:
            self._search_model = RandomizedSearchCV

        self._logger.info("tuning parameter with {0}".format(self._search_model.__name__))
        self._logger.info("{0} initialized.".format(self.__class__.__name__))


    def execute(self, training_data, training_label):
        self._logger.info("Executing parameter tuning")
        if self._search_model == GridSearchCV:
            cv_model = GridSearchCV(estimator=self._model,
                                    param_grid=self._param_dic,
                                    cv=self._cv,
                                    n_jobs=-1,
                                    verbose=True)
        else:
            cv_model = RandomizedSearchCV(estimator=self._model,
                                          param_distributions=self._param_dic,
                                          cv=self._cv,
                                          n_iter=self._n_iter,
                                          n_jobs=-1,
                                          verbose=True,
                                          random_state=1)
        cv_model.fit(training_data, training_label)
        return cv_model.best_params_


    def dispose(self):
        self._model = None
        del self._model

class AB_SearchCV(BaseSearchCV):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        self._is_regression = kwargs['is_regression']
        if self._is_regression:
            self._model =AdaBoostRegressor(DecisionTreeRegressor(max_depth=10))
            self._param_dic = {'n_estimators':[50,100,200,300,400,500],
                               'learning_rate':[0.01, 0.05, 0.1, 0.5],
                              }
        else:
            self._model =AdaBoostClassifier(DecisionTreeClassifier(max_depth=10))
            self._param_dic = {'n_estimators':[50,100,200,300,400,500],
                               'learning_rate':[0.01, 0.05, 0.1, 0.5],
                               'algorithm' : ['SAMME', 'SAMME.R'],
                              }
        self._logger.info("{0} initialized.".format(self.__class__.__name__))



class BG_SearchCV(BaseSearchCV):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        self._is_regression = kwargs['is_regression']
        if self._is_regression:
            self._model = BaggingRegressor(DecisionTreeRegressor(max_depth=10))
        else:
            self._model = BaggingClassifier(DecisionTreeClassifier(max_depth=10))

        if self._search_model == GridSearchCV:
            self._param_dic = {'n_estimators':[50,100,200,300,400,500],
                               'max_features': [i for i in range(1, 12)],
                               'max_samples': [i for i in range(2, 12)],
                               'bootstrap': [True, False],
                              }
        else:
            self._param_dic = {'n_estimators':[50,100,200,300,400,500],
                               'max_features': sp_randint(1, 11),
                               'max_samples': sp_randint(2, 11),
                               'bootstrap': [True, False],
                              }
        self._logger.info("{0} initialized.".format(self.__class__.__name__))


class GB_SearchCV(BaseSearchCV):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        if self._search_model == GridSearchCV:
            self._param_dic = {'max_depth': [3, None],
                               'n_estimators':[50,100,200,300,400,500],
                               'learning_rate':[0.01, 0.05, 0.1, 0.5],
                               #'max_features': [i for i in range(1, 12)],
                               #'min_samples_split': [i for i in range(2, 12)],
                               #'min_samples_leaf': [i for i in range(1, 12)],
                               'min_weight_fraction_leaf':[0,0.1,0.2,0.3,0.4,0.5],
                               'criterion': ['friedman_mse', 'mse']
                              }
        else:
            self._param_dic = {'max_depth': [3, None],
                           'n_estimators':[50,100,200,300,400,500],
                           'learning_rate':[0.01, 0.05, 0.1, 0.5],
                           'max_features': sp_randint(1, 11),
                           'min_samples_split': sp_randint(2, 11),
                           'min_samples_leaf': sp_randint(1, 11),
                           'min_weight_fraction_leaf':[0,0.1,0.2,0.3,0.4,0.5],
                           'criterion': ['friedman_mse', 'mse']
                          }

        self._is_regression = kwargs['is_regression']
        if self._is_regression:
            self._model = GradientBoostingRegressor()
        else:
            self._model = GradientBoostingClassifier()
            self._param_dic['loss'] = ['deviance', 'exponential']
        
        self._logger.info("{0} initialized.".format(self.__class__.__name__))


class HGB_SearchCV(BaseSearchCV):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        self._param_dic = {'learning_rate': [0.01, 0.05, 0.1, 0.5],
                           'max_iter': [10*i for i in range(1,4)],
                           'max_leaf_nodes': [i for i in range(10, 50)],
                           'min_samples_leaf': [i for i in range(10, 30)],
                           'l2_regularization': [i*0.1 for i in range(5)],
                           'max_bins': [2**i for i in range(2,8)],
                           'validation_fraction': [0.02 * i for i in range(1,10)],
                           'tol': [10**(-i) for i in range(1,10)]
                           }
        self._is_regression = kwargs['is_regression']
        if self._is_regression:
            self._model = HistGradientBoostingRegressor()
        else:
            self._model = HistGradientBoostingClassifier()
            self._param_dic['loss'] = ['auto', 'binary_crossentropy']#, 'categorical_crossentropy']
            #self._param_dic['loss'] = ['auto', 'binary_crossentropy', 'categorical_crossentropy']
        
        self._logger.info("{0} initialized.".format(self.__class__.__name__))

    
class kNN_SearchCV(BaseSearchCV):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        self._is_regression = kwargs['is_regression']
        if self._is_regression:
            self._model = KNeighborsRegressor()
        else:
            self._model =KNeighborsClassifier()
        self._param_dic = {'weights': ['uniform', 'distance'],
                           'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                           'leaf_size': [i for i in range(10, 60)],
                           'p': [i for i in range(1,5)]
                          }
        self._logger.info("{0} initialized.".format(self.__class__.__name__))


class LGBM_SearchCV(BaseSearchCV):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        
        self._is_regression = kwargs['is_regression']
        if self._is_regression:
            self._model = LGBMRegressor()
        else:
            self._model = LGBMClassifier()

        if self._search_model == GridSearchCV:
            self._param_dic = {'num_leaves':[2**i for i in range(5, 15)],
                               'max_depth': [-1] + [i for i in range(5, 11)],
                               'learning_rate': [0.01, 0.05, 0.1, 0.5],
                               'n_estimators':[50,100,200,300,400,500],
                               #'subsample_for_bin': [5*10**i for i in range(5, 10)],
                               #'min_split_gain': [0] + [10**(i*-1) for i in range(4)],
                               #'min_child_weight': [0] + [10**(i*-1) for i in range(4)],
                               #'min_child_samples': [i for i in range(10, 30)],
                               #'subsample_freq': [i for i in range(5, 10)],
                               'reg_alpha': [0] + [10**(i*-1) for i in range(4)],
                               }
        else:
            self._param_dic = {'num_leaves':[2**i for i in range(5, 15)],
                               'max_depth': [-1] + [i for i in range(5, 11)],
                               'learning_rate': [0.01, 0.05, 0.1, 0.5],
                               'n_estimators':[50,100,200,300,400,500],
                               'subsample_for_bin': [5*10**i for i in range(5, 10)],
                               'min_split_gain': [0] + [10**(i*-1) for i in range(4)],
                               'min_child_weight': [0] + [10**(i*-1) for i in range(4)],
                               'min_child_samples': [i for i in range(10, 30)],
                               'subsample_freq': [i for i in range(5, 10)],
                               'reg_alpha': [0] + [10**(i*-1) for i in range(4)],
                               }
        self._n_jobs = 1
        self._logger.info("{0} initialized.".format(self.__class__.__name__))


class RF_SearchCV(BaseSearchCV):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        
        if self._search_model == GridSearchCV:
            self._param_dic = {'max_depth': [3, None],
                               'n_estimators':[50,100,200,300,400,500],
                               'max_features': [i for i in range(5, 12)],
                               'min_samples_split': [i for i in range(2, 12)],
                               'min_samples_leaf': [i for i in range(5, 12)],
                               'bootstrap': [True, False]
                               }
        else:
            self._param_dic = {'max_depth': [3, None],
                               'n_estimators':[50,100,200,300,400,500],
                               'max_features': sp_randint(1, 11),
                               'min_samples_split': sp_randint(2, 11),
                               'min_samples_leaf': sp_randint(1, 11),
                               'bootstrap': [True, False]
                               }
        self._is_regression = kwargs['is_regression']
        if self._is_regression:
            self._model = RandomForestRegressor()
        else:
            self._model = RandomForestClassifier()
            self._param_dic['criterion'] = ["gini", "entropy"]
        self._logger.info("{0} initialized.".format(self.__class__.__name__))
    
    

class SVM_SearchCV(BaseSearchCV):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        
        self._is_regression = kwargs['is_regression']
        if self._is_regression:
            self._model = svm.SVR()
        else:
            self._model = svm.SVC()
        self._param_dic = {'kernel': ['linear', 'rbf', 'sigmoid'],#'poly',  'precomputed'],
                           'C':[1,10,20,30,40,50,100,1000],
                           'gamma': [10**i for i in range(-3, 4)],
                           'degree': [10**i for i in range(6)],
                           }
        self._logger.info("{0} initialized.".format(self.__class__.__name__))

class XGB_SearchCV(BaseSearchCV):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        self._is_regression = kwargs['is_regression']
        if self._is_regression:
            self._model = XGBRegressor()
        else:
            self._model = XGBClassifier()

        if self._search_model == GridSearchCV:
            self._param_dic = {'n_estimators':[2**i for i in range(5, 10)],
                               'max_depth': [i for i in range(5, 11)],
                               'learning_rate': [0.01, 0.05, 0.1, 0.5],
                               'gamma': [0.01, 0.05, 0.1, 0.5],
                               #'min_child_weight': [i for i in range(5,10)],
                               #'max_delta_step': [i for i in range(5,10)],
                               #'subsample': [i*0.1 for i in range(5)],
                               'reg_alpha': [0] + [10**(i*-1) for i in range(4)],
                               }
        else:
            self._param_dic = {'n_estimators':[2**i for i in range(5, 15)],
                               'max_depth': [i for i in range(1, 11)],
                               'learning_rate': [0.01, 0.05, 0.1, 0.5],
                               'gamma': [0.01, 0.05, 0.1, 0.5],
                               'min_child_weight': [i for i in range(10)],
                               'max_delta_step': [i for i in range(10)],
                               'subsample': [i*0.1 for i in range(11)],
                               'reg_alpha': [0] + [10**(i*-1) for i in range(4)],
                               }
        #self._n_jobs = 1
        self._logger.info("{0} initialized.".format(self.__class__.__name__))


class DNN_SearchCV(BaseSearchCV):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        self._n_iter = kwargs.get('niter', 30)
        self._param_dic = {
                           'out_dim1': [i for i in range(10, 100)],
                           'out_dim2': [i for i in range(10, 100)],
                           'out_dim3': [i for i in range(10, 100)],
                           'optimizer': ['adam', 'adadelta'],
                           'nb_epoch':[100, 500, 1000, 2000],
                           'dropout1': [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                           'dropout2': [0.5],
                           'dropout3': [0.5],
                           'batch_size': [100, 500, 1000, 2000],
                           'activation':['softplus', 'softsign', 'relu', 'tanh', 
                                         'sigmoid', 'hard_sigmoid', 'linear']
                           }
        
        
        self._logger.info("{0} initialized.".format(self.__class__.__name__))


    def execute(self, training_data, training_label):
        self._logger.info("Executing parameter tuning")
        from algorithm.ml_dnn import ML_DNN
        dnn_model = ML_DNN()
        if self._is_regression:
            self._model = KerasRegressor(build_fn=dnn_model._create_model, 
                                         input_dim=training_data.shape[1])
        else:
            self._model = KerasClassifier(build_fn=dnn_model._create_model, 
                                          input_dim=training_data.shape[1])
        if self._search_model == GridSearchCV:
            cv_model = GridSearchCV(estimator=self._model,
                                    param_grid=self._param_dic,
                                    cv=self._cv,
                                    n_jobs=-1,
                                    verbose=True)
        else:
            cv_model = RandomizedSearchCV(estimator=self._model,
                                          param_distributions=self._param_dic,
                                          cv=self._cv,
                                          n_iter=self._n_iter,
                                          n_jobs=8,
                                          verbose=True,
                                          random_state=1)
        cv_model.fit(training_data, training_label, 
                     batch_size=100, nb_epoch=1000)
        return cv_model.best_params_



#class TimeSeries_SearchCV(BaseSearchCV):
#    def __init__(self, **kwargs):
#        super().__init__(**kwargs)
#        self._logger.info("{0} initializing...".format(self.__class__.__name__))
        
#        self._param_dic = {
#                           'out_dim1': [i for i in range(10, 100)],
#                           'nb_epoch':[100, 500, 1000, 2000],
#                           'batch_size': [100, 500, 1000, 2000],
#                           'activation':['softplus', 'softsign', 'relu', 'tanh', 
#                                         'sigmoid', 'hard_sigmoid', 'linear']
#                           }
        
        
#        self._logger.info("{0} initialized.".format(self.__class__.__name__))


#    def execute(self, training_data, training_label):
#        self._logger.info("Executing parameter tuning")
#        from algorithm.ml_time_series import ML_TimeSeries
#        dnn_model = ML_TimeSeries()
#        if self._is_regression:
#            self._model = KerasRegressor(build_fn=dnn_model._create_model, input_dim=training_data.shape[1])
#        else:
#            self._model = KerasClassifier(build_fn=dnn_model._create_model, input_dim=training_data.shape[1])
#        if self._search_model == GridSearchCV:
#            cv_model = GridSearchCV(estimator=self._model,
#                                    param_grid=self._param_dic,
#                                    cv=self._cv,
#                                    n_jobs=-1,
#                                    verbose=True)
#        else:
#            cv_model = RandomizedSearchCV(estimator=self._model,
#                                          param_distributions=self._param_dic,
#                                          cv=self._cv,
#                                          n_iter=self._n_iter,
#                                          n_jobs=-1,
#                                          verbose=True,
#                                          random_state=1)
#        cv_model.fit(training_data, training_label)
#        return cv_model.best_params_