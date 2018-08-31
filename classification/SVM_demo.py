# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 08:09:10 2018

@author: i7
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import category_encoders as cs
from sklearn.pipeline import FeatureUnion

worldcup=pd.read_csv("../data/2018 worldcup.csv",index_col=0)
#match date is assumed to be irrelevant for the match results
worldcup.drop(['Date','Team1_Ball_Possession(%)'],axis=1,inplace=True)
worldcup.describe()

#world cup attributes
w_features=worldcup.iloc[:,np.arange(26)].copy()
#world cup goal result
w_goals=worldcup.iloc[:,26].copy()
#wordl cup match result
w_results=worldcup.iloc[:,27].copy()


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
w_features_num = w_features.drop(['Location','Phase','Team1','Team2','Team1_Continent','Team2_Continent','Normal_Time'], axis=1,inplace=False)
w_features_cat=w_features[['Location','Phase','Team1','Team2','Team1_Continent','Team2_Continent','Normal_Time']].copy()


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


feature_prepared = pd.DataFrame(data=full_pipeline.fit_transform(w_features),index=np.arange(1,65))
worldcup_cleaned=pd.concat([feature_prepared,w_goals.to_frame(), w_results.to_frame()], axis=1)


#########################demo 1: linear SVM versus SVM with linear Kernel##########################
import matplotlib.pyplot as plt

X = feature_prepared
y = w_goals

#for this demo, we will do a binary classification (only data from class 0 and class 1 will be used)
#all the data will be treated as training data, so no testing samples available in this demo
setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler

C = 5
alpha = 1 / (C * len(X))

lin_clf = LinearSVC(loss="hinge", C=C, random_state=42)
svm_clf = SVC(kernel="linear", C=C)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lin_clf.fit(X_scaled, y)
svm_clf.fit(X_scaled, y)

print("LinearSVC:                   ", lin_clf.intercept_, lin_clf.coef_)
print("SVC:                         ", svm_clf.intercept_, svm_clf.coef_)

## Compute the slope and bias of each decision boundary
#w1 = -lin_clf.coef_[0, 0]/lin_clf.coef_[0, 1]
#b1 = -lin_clf.intercept_[0]/lin_clf.coef_[0, 1]
#w2 = -svm_clf.coef_[0, 0]/svm_clf.coef_[0, 1]
#b2 = -svm_clf.intercept_[0]/svm_clf.coef_[0, 1]
#
## Transform the decision boundary lines back to the original scale
#line1 = scaler.inverse_transform([[-10, -10 * w1 + b1], [10, 10 * w1 + b1]])
#line2 = scaler.inverse_transform([[-10, -10 * w2 + b2], [10, 10 * w2 + b2]])
#
## Plot all three decision boundaries
#plt.figure(figsize=(11, 4))
#plt.plot(line1[:, 0], line1[:, 1], "k:", label="LinearSVC")
#plt.plot(line2[:, 0], line2[:, 1], "b--", linewidth=2, label="SVC")
#plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs") # label="Iris-Versicolor"
#plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo") # label="Iris-Setosa"
#plt.xlabel("Petal length", fontsize=14)
#plt.ylabel("Petal width", fontsize=14)
#plt.legend(loc="upper center", fontsize=14)
#plt.axis([0, 5.5, 0, 2])
#
#plt.show()


#########################demo 2: Tuning hyperparameters for nonlinear SVM##########################
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

MFCC_all=pd.read_csv("SVM_demo.csv",index_col=74)

#Testing and training sentences splitting (stratified + shuffled) based on the index (sentence ID)
Sentences=MFCC_all.index
Sentences_emotion=MFCC_all['Emotion']
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)

for train_index, test_index in sss.split(Sentences, Sentences_emotion):
    train_ind,test_ind =Sentences[train_index],Sentences[test_index]
Test_Matrix=MFCC_all.loc[test_ind]
Train_Matrix=MFCC_all.loc[train_ind]

#data training with hyperparameter tuning for C
clf = Pipeline([
        ('std_scaler', StandardScaler()),
        ("svm", SVC())
])
param_grid = [
        {'svm__kernel': ['rbf'], 'svm__C': [ 2**x for x in range(0,6) ]},
    ]
inner_cv = KFold(n_splits=3, shuffle=True, random_state=1)
grid_search = GridSearchCV(clf, param_grid, cv=inner_cv,  n_jobs=1, scoring='accuracy',verbose=3)
grid_search.fit(Train_Matrix.drop(['Emotion'],axis=1), Train_Matrix['Emotion'])
clf=grid_search.best_estimator_
#data testing
T_predict=clf.predict(Test_Matrix.drop(['Emotion'],axis=1))

print('*******************************************************************')
print("The prediction accuracy (tuned) for all testing sentence is : {:.2f}%.".format(100*accuracy_score(Test_Matrix['Emotion'],T_predict)))


#data training without hyperparameter tuning
clf = Pipeline([
        ('std_scaler', StandardScaler()),
        ("svm", SVC())
])
clf.fit(Train_Matrix.drop(['Emotion'],axis=1), Train_Matrix['Emotion'])
#data testing
T_predict=clf.predict(Test_Matrix.drop(['Emotion'],axis=1))

print('*******************************************************************')
print("The prediction accuracy (untuned) for all testing sentence is : {:.2f}%.".format(100*accuracy_score(Test_Matrix['Emotion'],T_predict)))


