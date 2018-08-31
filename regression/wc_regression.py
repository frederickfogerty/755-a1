# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import category_encoders as cs
from sklearn.pipeline import FeatureUnion

worldcup = pd.read_csv("../data/2018 worldcup.csv", index_col=0)
# match date is assumed to be irrelevant for the match results
worldcup.drop(['Date', 'Team1_Ball_Possession(%)'], axis=1, inplace=True)
worldcup.describe()

# world cup attributes
w_features = worldcup.iloc[:, np.arange(26)].copy()
# world cup goal result
w_goals = worldcup.iloc[:, 26].copy()
# wordl cup match result
w_results = worldcup.iloc[:, 27].copy()


# Create a class to select numerical or categorical columns
# since Scikit-Learn doesn't handle DataFrames in this wise manner yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


#  w_features_num: numerical features
#  w_features_cat: categorical features
w_features_num = w_features.drop(['Location', 'Phase', 'Team1', 'Team2',
                                  'Team1_Continent', 'Team2_Continent', 'Normal_Time'], axis=1, inplace=False)
w_features_cat = w_features[['Location', 'Phase', 'Team1', 'Team2',
                             'Team1_Continent', 'Team2_Continent', 'Normal_Time']].copy()


num_pipeline = Pipeline([
    ('selector', DataFrameSelector(list(w_features_num))),
    ('imputer', Imputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(list(w_features_cat))),
    ('cat_encoder', cs.OneHotEncoder(drop_invariant=True)),
])


full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])


feature_prepared = pd.DataFrame(
    data=full_pipeline.fit_transform(w_features), index=np.arange(1, 65))
worldcup_cleaned = pd.concat(
    [feature_prepared, w_goals.to_frame(), w_results.to_frame()], axis=1)

import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

wc_X = feature_prepared

# one target attribute
wc_y = w_goals

# Split the data into training/testing sets
wc_X_train, wc_X_test, wc_y_train, wc_y_test = \
    train_test_split(wc_X, wc_y, test_size=0.1, random_state=1)


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(wc_X_train, wc_y_train)

# Make predictions using the testing set
wc_y_pred = regr.predict(wc_X_test)
# Make predictions using the testing set
wc_y_train_pred = regr.predict(wc_X_train)

print(' ')
# The coefficients
#print('Coefficients and Intercept are: ', regr.coef_,"   ",regr.intercept_,' respectively')
# The mean squared error
print('_________________###################____________________')
print("Mean squared error for testing data: %.2f"
      % mean_squared_error(wc_y_test, wc_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score for testing data: %.2f' % r2_score(wc_y_test, wc_y_pred))
print('******************************************************* ')
print("Mean squared error for training data: %.2f"
      % mean_squared_error(wc_y_train, wc_y_train_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score for training data: %.2f' %
      r2_score(wc_y_train, wc_y_train_pred))

################################
# RIDGE

# Create linear regression object
regr = linear_model.Ridge()

# Train the model using the training sets
regr.fit(wc_X_train, wc_y_train)

# Make predictions using the testing set
wc_y_pred = regr.predict(wc_X_test)
# Make predictions using the testing set
wc_y_train_pred = regr.predict(wc_X_train)

print(' ')
# The coefficients
#print('Coefficients and Intercept are: ', regr.coef_,"   ",regr.intercept_,' respectively')
# The mean squared error
print('_________________###################____________________')
print("Mean squared error for testing data: %.2f"
      % mean_squared_error(wc_y_test, wc_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score for testing data: %.2f' % r2_score(wc_y_test, wc_y_pred))
print('******************************************************* ')
print("Mean squared error for training data: %.2f"
      % mean_squared_error(wc_y_train, wc_y_train_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score for training data: %.2f' %
      r2_score(wc_y_train, wc_y_train_pred))

#from sklearn.manifold import TSNE
#
#tsne = TSNE(n_components=2, random_state=42)
#
#wc_X_reduced = tsne.fit_transform(wc_X)
#
# plt.figure(figsize=(13,10))
#plt.scatter(wc_X_reduced[:, 0], wc_X_reduced[:, 1], c=wc_y, cmap="jet")
# plt.axis('off')
# plt.colorbar()
# plt.show()
#
#
#wc_X_test_reduced = tsne.fit_transform(wc_X_test)
#
# plt.figure(figsize=(13,10))
#plt.scatter(wc_X_test_reduced[:, 0], wc_X_test_reduced[:, 1], c=wc_y_test, cmap="jet")
# plt.axis('off')
# plt.colorbar()
# plt.show()
#
# Plot outputs
#plt.scatter(wc_X_test_reduced, wc_y_test,  color='black')
#plt.plot(wc_X_test_reduced, wc_y_pred, color='blue', linewidth=3)
#
# plt.xticks(np.arange(start=-0.1,stop=0.2,step=0.06))
# plt.yticks(np.arange(500,step=100))
#
# plt.show()
