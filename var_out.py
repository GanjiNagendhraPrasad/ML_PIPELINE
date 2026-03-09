import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from scipy.stats import yeojohnson

import logging
from logging_code import setup_logging
logger=setup_logging('var_out')

def vt_outliers(x_train_num,x_test_num):
    try:
        logger.info(f'Before Train columns names : {x_train_num.columns}')
        logger.info(f'Before test columns names : {x_test_num.columns}')
        for i in x_train_num.columns:
            x_train_num[i+'_yeo'],lam_value=yeojohnson((x_train_num[i]))
            x_test_num[i+'_yeo'],lam_value=yeojohnson((x_test_num[i]))
            x_train_num=x_train_num.drop([i],axis=1)
            x_test_num=x_test_num.drop([i],axis=1)
            #trimming
            iqr=x_train_num[i+'_yeo'].quantile(0.75)-x_train_num[i+'_yeo'].quantile(0.25)
            upper_limit=x_train_num[i+'_yeo'].quantile(0.75)+(1.5*iqr)
            lower_limit=x_train_num[i+'_yeo'].quantile(0.25)-(1.5*iqr)
            x_train_num[i+'_trim']=np.where(x_train_num[i+'_yeo']>upper_limit,upper_limit,
                                            np.where(x_train_num[i+'_yeo']<lower_limit,lower_limit,x_train_num[i+'_yeo']))
            x_test_num[i+'_trim']=np.where(x_test_num[i+'_yeo']>upper_limit,upper_limit,
                                           np.where(x_test_num[i+'_yeo']<lower_limit,lower_limit,x_test_num[i+'_yeo']))
            x_train_num=x_train_num.drop([i+'_yeo'],axis=1)
            x_test_num=x_test_num.drop([i+'_yeo'],axis=1)

        logger.info(f'After Train columns names : {x_train_num.columns}')
        logger.info(f'After test columns names : {x_test_num.columns}')

        return  x_train_num,x_test_num
    except Exception as e:
        msg, typ, lin = sys.exc_info()
        logger.info(f'\n{msg}\n{typ}\n{lin.tb_lineno}')
