# -*- coding: utf-8 -*-
import os, sys
import logging.config
import gc

import pandas as pd
import numpy as np
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from sklearn.metrics import classification_report

from util.ml_config_parser import MLConfigParser
from exception.exceptions import InvalidFileError
import algorithm as alg

from util.coint_feature_vector_manager import CointFeatureVectorManager
from util.result_manager import ResultManager
import util.common_func as cf
from util.port_label_manager import PortLabelManager
from util.performance_measurer import PerformanceMeasurer

def check_input(target_file):
    if not os.path.exists(target_file): 
        raise InvalidFileError('{0} does not exist.'.format(target_file))


def create_date_list(input_file_name, term_month):

    input_data_df = pd.read_csv(input_file_name)
    #input_data_df['ValueDate'] = cf.convert_date_format(input_data_df.ValueDate)
    input_data_df = cf.convert_date_format(input_data_df)
    input_data_df.set_index('ValueDate', inplace=True)
    start_date = np.min(list(input_data_df.index)) + relativedelta(months=term_month)
    return input_data_df.query("ValueDate >= @start_date").index


def calc_training_result(algo, training_data, training_label, value_date):
    training_result = algo.predict(training_data)
    training_result_df = pd.DataFrame(training_result, columns=['Predict'])
    training_result_df['ValueDate'] = training_data.index
    training_result_df['Algorithm'] = algo.__class__.__name__
    result_manager = ResultManager(PredictedData=training_result_df.set_index('ValueDate'),
                                   PredictedLabel=training_label)
    summary_df = result_manager.create_result(type='Training')
    summary_df['ValueDate'] = value_date
    
    return summary_df

def create_importance(algo_name, importance, columns, value_date):
    importance_df = pd.DataFrame(importance).T
    importance_df.columns = columns
    importance_df['Algorithm'] = algo_name
    importance_df.index = [value_date]

    return importance_df

def create_label(weight_file_name, input_file_name, training_week, is_regression=True):
    price_file = './input/index_price_{0}.csv'.format(training_week)
    port_label_mgr = PortLabelManager(weight_file=weight_file_name, price_file=price_file)
    input_df = pd.read_csv(input_file_name)
    input_df = cf.convert_date_format(input_df)
    start_date = input_df.ValueDate.iloc[0]
    end_date = input_df.ValueDate.iloc[-1]
    #import pdb;pdb.set_trace()
    assert port_label_mgr.port_label.index[0] <= start_date
    #assert port_label_mgr.port_label.index[-1] >= end_date
    
    if is_regression:
        return pd.DataFrame(port_label_mgr.port_label.query("index >= @start_date & index <= @end_date")),\
               pd.DataFrame(port_label_mgr.notional.query("index >= @start_date & index <= @end_date"))
    else:
        return pd.DataFrame(port_label_mgr.port_label.query("index >= @start_date & index <= @end_date")
                                          .Return.apply(lambda x: 1 if x>0 else 0)),\
               pd.DataFrame(port_label_mgr.notional.query("index >= @start_date & index <= @end_date"))


def main():
    logging.config.fileConfig('./logger_config.ini')
    logger = logging.getLogger("jpbank.quants")
    output_suffix = datetime.now().strftime('%Y%m%d%H%M%S')
    try:
        config = MLConfigParser()
        training_month = config.training_term
        training_week = 52
        exec_pca = config.exec_pca
        is_regression = config.is_regression
        feature_file_name = os.path.join(config.input_dir, config.feature_file)
        weight_file_name = os.path.join(config.input_dir, 'coint_vec.csv')
        
        check_input(feature_file_name)
        check_input(weight_file_name)

        
        
        logger.info("Trainig Term {0}".format(training_month))
        logger.info("Excecute PCA {0}".format(exec_pca))
        date_list = create_date_list(feature_file_name, 
                                     training_month)
        port_label, notional = create_label(weight_file_name, feature_file_name, 
                                            training_week,  config.is_regression)
        import pdb;pdb.set_trace()
        algorithm_list = [
                          alg.ML_AR,
                          #alg.ML_ARMA,
                          #alg.ML_ARIMA
                          ]
        #if config.is_regression:
        #    algorithm_list.append(alg.ML_NaiveBayes)
        #algorithm_list = [alg.ML_LightBGM]

        predict_result_df = pd.DataFrame()
        proba_result_df = pd.DataFrame()
        training_result_df = pd.DataFrame()
        importance_df = pd.DataFrame()
        #date_list = date_list[-3:]
        for algo in algorithm_list:
            algo_result_list = []
            proba_result_list = []
            #date_list = date_list[date_list>date(2012,12,1)]
            #date_list = date_list[-100:]
            ml_algo = algo(start_date=date_list[0] - relativedelta(months=training_month), 
                           end_date=date_list[-1] + relativedelta(months=1))
            #date_list = date_list[:10]
            for i in tqdm(range(len(date_list))):
                value_date = date_list[i]
                logger.info("Trainig/Predicting In {0}...".format(value_date))
                start_date = value_date - relativedelta(weeks=training_week)#months=training_month)
                #end_date = value_date-relativedelta(weeks=1)
                #import pdb;pdb.set_trace()
                logger.info("Learing In {0}".format(value_date))
                ml_algo.learn(start_date=start_date, end_date=value_date)
                
                
                algo_result_list.append([value_date,
                                         ml_algo.__class__.__name__,
                                         ml_algo.predict(value_date)])
                gc.collect()

            alg_result_df = pd.DataFrame(algo_result_list,
                                         index=date_list, 
                                         columns=['ValueDate',
                                                 'Algorithm',
                                                 'Predict'])
            

            predict_result_df = predict_result_df.append(alg_result_df)
            #f = open('./output/report_{0}_{1}.txt'.format(ml_algo.__class__.__name__,
            #                                              output_suffix), 'w')
            #common_index = list(set(port_label.index) & set(list(alg_result_df.index)))
            #if not is_regression:
            #    proba_result_df = proba_result_df.append(
            #                            pd.DataFrame(proba_result_list,
            #                                         index=date_list, 
            #                                         columns=['ValueDate',
            #                                                  'Algorithm',
            #                                                  'DownProbability',
            #                                                  'UpProbability']))
            #    #import pdb;pdb.set_trace()
            #    f.write(classification_report(port_label.loc[common_index].Return,
            #                                  alg_result_df.loc[common_index].Predict))
            #else:
                
            #    f.write(classification_report(port_label.loc[common_index].Return.apply(lambda x: 1 if x>0 else 0),
            #                                  alg_result_df.loc[common_index].Predict.apply(lambda x: 1 if x>0 else 0)))
            #f.close()

        #Result Output Process
        predict_result_df.index.name='ValueDate'
        proba_result_df.index.name='ValueDate'
        predicted_return = pd.DataFrame((predict_result_df.Predict - ml_algo.coint_index.loc[predict_result_df.index].Price)\
                                        / notional.Notional.loc[predict_result_df.index],
                                        columns=['Predict'])
        predicted_return['Algorithm'] = predict_result_df.Algorithm.tolist()
        #import pdb;pdb.set_trace()
        predict_result_df.to_csv('./output/predict_result_{0}_{1}_{2}_{3}.csv'\
                                .format('PCA' if exec_pca else 'NoPCA',
                                        int(training_month),
                                        'Reg' if is_regression else 'Class',
                                        output_suffix), 
                                 index=False)
        ml_algo.coint_index.to_csv('coint_index.csv')
        #if not is_regression:
        #    proba_result_df.to_csv('./output/proba_result_{0}_{1}_{2}_{3}.csv'\
        #                            .format('PCA' if exec_pca else 'NoPCA',
        #                                    int(training_month),
        #                                    'Reg' if is_regression else 'Class',
        #                                    output_suffix), 
        #                             index=False)
        #training_result_df.to_csv('./output/training_result_{0}_{1}_{2}_{3}.csv'\
        #                        .format('PCA' if exec_pca else 'NoPCA',
        #                                int(training_month),
        #                                'Reg' if is_regression else 'Class',
        #                                output_suffix), 
        #                         index=False)
    
        #importance_df.to_csv('./output/importance_{0}_{1}_{2}_{3}.csv'\
        #                     .format('PCA' if exec_pca else 'No',
        #                             int(training_month),
        #                             'Reg' if is_regression else 'Class',
        #                             output_suffix), 
        #                     index=True)
        result_manager = ResultManager(PredictedData=predicted_return,
                                       PredictedLabel=port_label)
        #import pdb;pdb.set_trace()
        result_manager.create_result().to_csv('./output/summary_{0}_{1}_{2}_{3}.csv'\
                                               .format('PCA' if exec_pca else 'NoPCA',
                                                       int(training_month),
                                                       'Reg' if is_regression else 'Class',
                                                       output_suffix), 
                                              index=False)
        return_series_df = result_manager.create_return_series()
        return_series_df.to_csv('./output/return_series_{0}_{1}_{2}_{3}.csv'\
                                               .format('PCA' if exec_pca else 'NoPCA',
                                                       int(training_month),
                                                       'Reg' if is_regression else 'Class',
                                                       output_suffix))
        perform_measurer = PerformanceMeasurer()
        perform_measurer.create_result_summary(return_series_df)\
            .to_csv('./output/performance_summary_{0}_{1}_{2}_{3}.csv'\
                                               .format('PCA' if exec_pca else 'NoPCA',
                                                       int(training_month),
                                                       'Reg' if is_regression else 'Class',
                                                       output_suffix))
        sys.exit(0)
    except InvalidFileError as ife:
        logger.error(ife.args)
    else:
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()
