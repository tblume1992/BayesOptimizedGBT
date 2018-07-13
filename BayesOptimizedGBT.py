#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 20:42:18 2018

@author: tyler
"""
from __future__ import print_function
from __future__ import division
from sklearn.cross_validation import cross_val_score
from xgboost import XGBClassifier as XGBC
from xgboost import XGBRegressor as XGBR
import pandas as pd
from bayes_opt import BayesianOptimization

def show_scores():
    fmt = '{:<8}{:<20}{}'
    print('-'*15 + 'Classification' + '-'*15)	 
    print(fmt.format('', 'Score', 'Comment'))
    score = ['accuracy','average_precision','f1','f1_micro','f1_macro',
             'f1_weighted','f1_samples','neg_log_loss']
    comment = ['-','-','for binary targets','	micro-averaged','macro-averaged',
               'weighted averaged','by multilabel sample','-','-','-','-']
    for i, (score, comment) in enumerate(zip(score, comment)):
            print(fmt.format(i, score, comment))
            
    print('-'*15 + 'Regression' + '-'*15)	 
    print(fmt.format('', 'Score', 'Comment'))
    score = ['explained_variance','neg_mean_absolute_error','neg_mean_squared_error',
             'neg_mean_squared_log_error','neg_median_absolute_error',
             'r2']
    comment = ['-','-','-','-','-','-']
    for i, (score, comment) in enumerate(zip(score, comment)):
            print(fmt.format(i, score,'' ))
def BayesOptimizedBoosting(X,y,rounds = 10, Classification = False, score= 'neg_mean_squared_error', kfold = 5):    
    if Classification is True:
        def xgbcv(num_round, subsample, eta, max_depth):
            val = cross_val_score(
                XGBC(num_round=int(num_round),
                    subsample=float(subsample),
                    eta=min(eta, 0.999),
                    max_depth = int(max_depth),
                    random_state=2
                ),
                X, y, score , cv=kfold
            ).mean()
            return val
    else:
        def xgbcv(num_round, subsample, eta, max_depth):
            val = cross_val_score(
                XGBR(num_round=int(num_round),
                    subsample=float(subsample),
                    eta=min(eta, 0.999),
                    max_depth = int(max_depth),
                    random_state=2
                ),
                X, y,score, cv=5
            ).mean()
            return val
    
    gp_params = {"alpha": 1e-5}
    xgbcBO = BayesianOptimization(
        xgbcv,
        {'num_round': (10,250),
        'subsample': (0.5,0.999),
        'eta': (0.1,.3),
        'max_depth': (3,8)}
    )


    print('-' * 53)
	
    xgbcBO.maximize(n_iter=rounds, **gp_params)

    print('-' * 53)
    print('Final Results')
    print('RFC: %f' % xgbcBO.res['max']['max_val'])
    print(pd.DataFrame(xgbcBO.res['max']['max_params'], index =['Parameters']))
