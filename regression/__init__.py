# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import category_encoders as cs
from sklearn.pipeline import FeatureUnion
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
import scipy


def regression_performance(y_test, y_pred):
    def anon():
        print("MSE: %.2f"
              % mean_squared_error(y_test, y_pred))
        print('r^2 score: %.2f' %
              r2_score(y_test, y_pred))
    return anon


def Regression(X, y):

    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.1, random_state=1)

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)
    # Make predictions using the testing set
    y_train_pred = regr.predict(X_train)

    def predict(X):
        # X_std = sc.transform(X)
        return regr.predict(X)

    return {
        'predict': predict,
        'performance': regression_performance(y_test, y_pred),
        'estimator': regr,
        'X_test': X_test,
        'y_test': y_test,
    }


def Ridge(X, y):

    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.1, random_state=1)

    # Create linear regression object
    regr = linear_model.Ridge()

    random_grid = {
        'alpha': scipy.stats.uniform(),
        'tol': scipy.stats.uniform(),
    }

    regr_random = RandomizedSearchCV(
        estimator=regr, param_distributions=random_grid, n_iter=100, cv=10, verbose=2, random_state=42, n_jobs=-1)

    # Train the model using the training sets
    regr_random.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regr_random.predict(X_test)

    estimator_untuned = linear_model.Ridge()
    estimator_untuned.fit(X_train, y_train)

    def predict(X):
        # X_std = sc.transform(X)
        return regr_random.predict(X)

    return {
        'predict': predict,
        'performance': regression_performance(y_test, y_pred),
        'estimator': regr_random,
        'estimator_untuned': estimator_untuned,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
    }
