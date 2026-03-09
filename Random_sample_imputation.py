import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import logging

import warnings
warnings.filterwarnings("ignore")

from logging_code import  setup_logging
logger = setup_logging('Random_sample_imputation')

def handling_missing_value(x_train,x_test):
    try:
        logger.info(f'before handling missing value x_train shape and columns : {x_train.shape} \n : {x_train.columns} : {x_train.isnull().sum()}')
        logger.info(f'before handling missing value x_test shape and columns : {x_test.shape} \n : {x_test.columns} : {x_test.isnull().sum()}')
        for i in x_train.columns:
            if x_train[i].isnull().sum() > 0:
                x_train[i+'_replaced']=x_train[i].copy()
                x_test[i + '_replaced'] = x_test[i].copy()
                s=x_train[i].dropna().sample(x_train[i].isnull().sum(),random_state=42)
                s1=x_test[i].dropna().sample(x_test[i].isnull().sum(), random_state=42)
                s.index=x_train[x_train[i].isnull()].index
                s1.index = x_test[x_test[i].isnull()].index
                x_train.loc[x_train[i].isnull(), i+'_replaced']=s
                x_test.loc[x_test[i].isnull(), i + '_replaced']=s1
                x_train=x_train.drop([i],axis=1)
                x_test = x_test.drop([i], axis=1)

        logger.info(f'After handling missing value x_train shape and columns : {x_train.shape} \n : {x_train.columns}  : {x_train.isnull().sum()}')
        logger.info(f'After handling missing value x_test shape and columns : {x_test.shape} \n : {x_test.columns} : {x_test.isnull().sum()}')
        return  x_train,x_test
    except Exception as e:
        msg, typ, lin = sys.exc_info()
        logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')