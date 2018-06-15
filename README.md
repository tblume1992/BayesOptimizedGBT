# BayesOptimizedGBT
This script uses a Bayesian Optimization package in conjunction with Sklearn's CV utilites to train a XGBT model using k-fold cross validation.


BayesOptimizedBoosting(X,y,rounds = 10, Classification = False, score= 'neg_mean_squared_error', kfold = 5)

  X: Your regressors.
  
  y: Your target variable
  
  rounds: The number of rounds to conduct the optimization, default is 10.
  
  Classification: Whether or not you are doing a classification problem.  If 'True' then you must select a classification      score.
  
  score: The metric to judge the CV by and the metric that the Bayesian Optimization is trying to maximize.
  
  ***There are nummerous scores that are supported, for a list simply run BayesOptimizedGBT.show_scores()***
  
  kfold: The number of folds to create for cross validation, default is 5.
  
  
