'''
In this file I am going to call all related functions for data cleaning and model
development
'''
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

import seaborn as sns

import logging
from logging_code import setup_logging
logger=setup_logging('main')
from Random_sample_imputation import  handling_missing_value
from var_out import vt_outliers
from filter_methods import fm
from categorical_to_num import c_t_n

from imblearn.over_sampling import SMOTE

from feature_scaling import fs

class CREDIT:
    def __init__(self,path):
        try:
            self.path=path
            self.df=pd.read_csv(self.path) # loading the data into varibale
            logger.info(f'total data size : {self.df.shape}')
            self.df=self.df.drop([150000,150001],axis=0)
            self.df=self.df.drop(['MonthlyIncome.1'],axis=1)
            logger.info(f'null values : \n : {self.df.isnull().sum()}')
            self.x=self.df.iloc[:,:-1] # independent
            self.y=self.df.iloc[:,-1]  # dependent
            self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.x,self.y,test_size=0.2,random_state=42)
            self.y_train = self.y_train.map({'Good': 1, 'Bad': 0}).astype(int)
            self.y_test = self.y_test.map({'Good': 1, 'Bad': 0}).astype(int)
            logger.info(f'train data : {len(self.x_train)} : {len(self.y_train)} total train data : {self.x_train.shape} ')
            logger.info(f'test data : {len(self.x_test)} : {len(self.y_test)} total test data : {self.x_test.shape} ')
        except Exception as e:
            msg,typ,lin=sys.exc_info()
            logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')

    def missing_values(self):
        try:
            logger.info(f'before handling missing value x_train shape and columns : {self.x_train.shape} \n : {self.x_train.columns}  : {self.x_train.isnull().sum()}')
            logger.info(f'before handling missing value x_test shape and columns : {self.x_test.shape} \n : {self.x_test.columns} : {self.x_test.isnull().sum()}')
            self.x_train['NumberOfDependents']=pd.to_numeric(self.x_train['NumberOfDependents'])
            self.x_test['NumberOfDependents'] = pd.to_numeric(self.x_test['NumberOfDependents'])
            self.x_train,self.x_test=handling_missing_value(self.x_train,self.x_test)
            logger.info(f'After handling missing value x_train shape and columns : {self.x_train.shape} \n : {self.x_train.columns}  : {self.x_train.isnull().sum()}')
            logger.info(f'After handling missing value x_test shape and columns : {self.x_test.shape} \n : {self.x_test.columns} : {self.x_test.isnull().sum()}')
        except Exception as e:
            msg,typ,lin=sys.exc_info()
            logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')
    def data_seperation(self):
        try:
            self.x_train_num_col=self.x_train.select_dtypes(exclude='object')
            self.x_test_num_col = self.x_test.select_dtypes(exclude='object')
            self.x_train_cat_col = self.x_train.select_dtypes(include='object')
            self.x_test_cat_col = self.x_test.select_dtypes(include='object')

            logger.info(f'{self.x_train_num_col.columns} : {self.x_train_num_col.shape}')
            logger.info(f'{self.x_test_num_col.columns} : {self.x_test_num_col.shape}')
            logger.info(f'================================================================')
            logger.info(f'{self.x_train_cat_col.columns} : {self.x_train_cat_col.shape}')
            logger.info(f'{self.x_test_cat_col.columns} : {self.x_test_cat_col.shape}')
        except Exception as e:
            msg,typ,lin=sys.exc_info()
            logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')

    def variable_transformation(self):
        try:
            logger.info(f'Before Train columns names : {self.x_train_num_col.columns}')
            logger.info(f'Before test columns names : {self.x_test_num_col.columns}')
            self.x_train_num_col,self.x_test_num_col = vt_outliers(self.x_train_num_col,self.x_test_num_col)
            logger.info(f'After Train columns names : {self.x_train_num_col.columns}')
            logger.info(f'After test columns names : {self.x_test_num_col.columns}')
        except Exception as e:
            msg,typ,lin=sys.exc_info()
            logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')

    def feature_selection(self):
        try:
            self.x_train_num_col,self.x_test_num_col=fm(self.x_train_num_col,self.x_test_num_col,self.y_train,self.y_test)
        except Exception as e:
            msg,typ,lin=sys.exc_info()
            logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')

    def cat_to_num(self):
        try:
            self.x_train_cat_col,self.x_test_cat_col=c_t_n(self.x_train_cat_col,self.x_test_cat_col)

            # combine data
            self.x_train_num_col.reset_index(drop=True, inplace=True)
            self.x_train_cat_col.reset_index(drop=True, inplace=True)
            self.x_test_num_col.reset_index(drop=True, inplace=True)
            self.x_test_cat_col.reset_index(drop=True, inplace=True)

            self.training_data=pd.concat([self.x_train_num_col,self.x_train_cat_col],axis=1)
            self.testing_data=pd.concat([self.x_test_num_col,self.x_test_cat_col],axis=1)

            logger.info(f'========================================================================================')

            logger.info((f'final training data : {self.training_data.shape}'))
            logger.info((f'{self.training_data.columns}'))
            logger.info(f'training data null values : {self.training_data.isnull().sum()}')

            logger.info(f'final testing data : {self.testing_data.shape}')
            logger.info((f'{self.testing_data.columns}'))
            logger.info(f'testing data null values : {self.testing_data.isnull().sum()}')
        except Exception as e:
            msg,typ,lin=sys.exc_info()
            logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')

    def data_balancing(self):
        try:
            logger.info(f'Number of Rows for GOOD customer {1} : {sum(self.y_train==1)}')
            logger.info(f'Number of Rows for BAD customer {0} : {sum(self.y_train==0)}')
            logger.info(f'Training data size : {self.training_data.shape}')

            sm=SMOTE(random_state=42)

            self.training_data_bal,self.y_train_bal=sm.fit_resample(self.training_data,self.y_train)

            logger.info(f'Number of Rows for GOOD customer {1} : {sum(self.y_train_bal == 1)}')
            logger.info(f'Number of Rows for BAD customer {0} : {sum(self.y_train_bal == 0)}')
            logger.info(f'Training data size : {self.training_data_bal.shape}')

            fs(self.training_data_bal,self.y_train_bal,self.testing_data,self.y_test)
        except Exception as e:
            msg,typ,lin=sys.exc_info()
            logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')

if __name__ == '__main__':
    try:
        obj=CREDIT('creditcard.csv')
        obj.missing_values()
        obj.data_seperation()
        obj.variable_transformation()
        obj.feature_selection()
        obj.cat_to_num()
        obj.data_balancing()
    except Exception as e:
        msg, typ, lin = sys.exc_info()
        logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')