import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.stats import ks_2samp
from itertools import product


from sklearn.tree import plot_tree
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored , concordance_index_ipcw
from sklearn.impute import SimpleImputer, KNNImputer
from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler,StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split
from sksurv.nonparametric import kaplan_meier_estimator

from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis, GradientBoostingSurvivalAnalysis

from sklearn.ensemble import RandomForestRegressor

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import lightgbm as lgb

from prince import MCA

import optuna

import joblib


class RandomSurvivalForestEvaluator:
    def __init__(self, X, y, n_estimators=200, min_samples_split=30, min_samples_leaf=15, max_features=0.5, max_depth= 5,random_state=42, n_jobs=-1, n_splits=5, imputer=KNNImputer(n_neighbors=6),scaler=MinMaxScaler()):
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.rsf = RandomSurvivalForest(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            max_depth=max_depth,
            n_jobs=n_jobs
        )
        self.rsf_cindex_train = []
        self.rsf_cindex_test = []
        self.imputer = imputer
        self.scaler = scaler

    def train(self):
        for train_index, test_index in self.kf.split(self.X, self.y):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            imputer = self.imputer
            #imputer = SimpleImputer(strategy='mean')
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)
            
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            
            
            self.rsf.fit(X_train, y_train)

            rsf_cindex_train_val = concordance_index_ipcw(y_train, y_train, self.rsf.predict(X_train), tau=7)[0]
            rsf_cindex_test_val = concordance_index_ipcw(y_train, y_test, self.rsf.predict(X_test), tau=7)[0]
            self.rsf_cindex_train.append(rsf_cindex_train_val)
            self.rsf_cindex_test.append(rsf_cindex_test_val)

    def get_scores(self):
        print(f"Average Random Survival Forest IPCW-Index train: {np.mean(self.rsf_cindex_train):.4f}")
        print(f"Average Random Survival Forest IPCW-Index test: {np.mean(self.rsf_cindex_test):.4f}")
    
    def score_reset(self):
        self.rsf_cindex_test = []
        self.rsf_cindex_train = []
        
        

def grid_search_rsf(X, y, param_grid, n_splits=5, tau=7):
    """
    Perform grid search on RandomSurvivalForest with IPCW concordance index.

    Parameters:
    - X: DataFrame or array-like, feature matrix.
    - y: Structured array, survival data in sksurv format.
    - param_grid: Dictionary, parameter grid for RSF.
    - n_splits: Int, number of folds for cross-validation.
    - tau: Float, IPCW truncation time.

    Returns:
    - best_params: Dictionary, best parameter combination.
    - best_score: Float, best test concordance index score.
    """
    # Generate all parameter combinations
    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())

    # for later update of best score and parameters
    best_score = -np.inf
    best_params = None
    n = len(param_combinations)
    i = 0
    for params in param_combinations:
        # Initialize RSF with current parameter set
        rsf_params = dict(zip(param_names, params))
        rsf = RandomSurvivalForest(**rsf_params, random_state=42, n_jobs=-1)

        # Cross-validation setup
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        train_scores = []
        test_scores = []

        for train_index, test_index in kf.split(X,y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Handle missing data
            imputer = SimpleImputer(strategy="mean")
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)
            
            # Fit RSF model
            rsf.fit(X_train, y_train)

            # Evaluate IPCW concordance index
            train_score = concordance_index_ipcw(y_train, y_train, rsf.predict(X_train), tau=tau)[0]
            test_score = concordance_index_ipcw(y_train, y_test, rsf.predict(X_test), tau=tau)[0]

            train_scores.append(train_score)
            test_scores.append(test_score)
        i += 1
        print(f"step {i}/{n}")
        # Average test score for this parameter set
        mean_test_score = np.mean(test_scores)
        print(f"Params: {rsf_params}, Train Score: {np.mean(train_scores):.4f}, Test Score: {mean_test_score:.4f}")

        # Update best parameters if the current score is better
        if mean_test_score > best_score:
            best_score = mean_test_score
            best_params = rsf_params

    return best_params, best_score



# Optuna objective function for RSF hyperparameter optimization
def hyperparameter_optimization_rsf(X, y) -> tuple:
    """
    Optimize RandomSurvivalForest hyperparameters using Optuna.

    Parameters:
    - X: DataFrame or array-like, feature matrix.
    - y: Structured array, survival data in sksurv format.

    Returns:
    - best_params: Dictionary, best hyperparameters found.
    - best_value: Float, best IPCW concordance index score.
    """
    def objective(trial):
        # Define the hyperparameters to tune
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 40)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 40)
        max_features = trial.suggest_float('max_features', 0.1, 1.0)
        max_depth = trial.suggest_int('max_depth', 3, 30)

        # Initialize the model with the suggested hyperparameters
        rsf = RandomSurvivalForest(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )

        fold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for train_index, test_index in fold.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            rsf.fit(X_train, y_train)

            score = concordance_index_ipcw(y_train, y_test, rsf.predict(X_test), tau=7)[0]
            cv_scores.append(score)
        
        # Evaluate the model
        test_score = np.mean(cv_scores)
        print(f"Test score: {test_score} and params: {trial.params}")
        return test_score
    
    # Create a study and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # Print the best hyperparameters
    print('Best hyperparameters: ', study.best_params)
    print('Best score: ', study.best_value)
    
    return study.best_params, study.best_value
    


# Gradient Boosting Survival Analysis

def grid_search_gbs(X, y, param_grid, n_splits=5, tau=7):
    """
    Perform grid search on GradientBoosting Survival Analysis with IPCW concordance index.

    Parameters:
    - X: DataFrame or array-like, feature matrix.
    - y: Structured array, survival data in sksurv format.
    - param_grid: Dictionary, parameter grid for gbs.
    - n_splits: Int, number of folds for cross-validation.
    - tau: Float, IPCW truncation time.

    Returns:
    - best_params: Dictionary, best parameter combination.
    - best_score: Float, best test concordance index score.
    """
    # Generate all parameter combinations
    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())

    # for later update of best score and parameters
    best_score = -np.inf
    best_params = None
    n = len(param_combinations)
    i = 0
    for params in param_combinations:
        gbs_params = dict(zip(param_names, params))
        gbs = GradientBoostingSurvivalAnalysis(**gbs_params, random_state=42)

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        train_scores = []
        test_scores = []

        for train_index, test_index in kf.split(X,y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            imputer = KNNImputer(n_neighbors=15)
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)
            
            gbs.fit(X_train, y_train)

            train_score = concordance_index_ipcw(y_train, y_train, gbs.predict(X_train), tau=tau)[0]
            test_score = concordance_index_ipcw(y_train, y_test, gbs.predict(X_test), tau=tau)[0]

            train_scores.append(train_score)
            test_scores.append(test_score)
        i += 1
        print(f"step {i}/{n}")

        mean_test_score = np.mean(test_scores)
        print(f"Params: {gbs_params}, Train Score: {np.mean(train_scores):.4f}, Test Score: {mean_test_score:.4f}")

        if mean_test_score > best_score:
            best_score = mean_test_score
            best_params = gbs_params

    return best_params, best_score