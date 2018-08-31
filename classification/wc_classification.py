# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import category_encoders as cs
from sklearn.pipeline import FeatureUnion

RANDOM_STATE: int = 42


def print_section(section: str) -> None:
    print("\n\n\n##########################")
    print(section.upper())


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


features_prepared = pd.DataFrame(
    data=full_pipeline.fit_transform(w_features), index=np.arange(1, 65))
worldcup_cleaned = pd.concat(
    [features_prepared, w_goals.to_frame(), w_results.to_frame()], axis=1)

X = features_prepared
y = w_results

##################################################
# Perceptron
print_section("perceptron")


def Perceptron(X, y):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Perceptron
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import numpy as np

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    sc = StandardScaler()
    sc.fit(X)

    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    ppn = Perceptron(n_iter=40, eta0=0.1, random_state=42)

    ppn.fit(X_train_std, y_train)

    y_pred = ppn.predict(X_test_std)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "model": ppn,
        "y_pred": y_pred
    }


world_cup_perceptron = Perceptron(X, y)
print('Accuracy: %.2f' % world_cup_perceptron["accuracy"])


##################################################
# linear SVM
print_section("SVM")


def SVM(X, y):
    from sklearn.svm import SVC, LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    C = 5
    alpha = 1 / (C * len(X))

    lin_clf = LinearSVC(loss="hinge", C=C, random_state=42)
    svm_clf = SVC(kernel="linear", C=C)

    sc = StandardScaler()
    sc.fit(X)

    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    lin_clf.fit(X_train_std, y_train)
    svm_clf.fit(X_train_std, y_train)

    y_pred_lin = lin_clf.predict(X_test_std)
    y_pred_svm = svm_clf.predict(X_test_std)

    return {
        "svm": {
            "accuracy": accuracy_score(y_test, y_pred_svm)
        },
        "linear_svc": {
            "accuracy": accuracy_score(y_test, y_pred_lin)
        },
    }


world_cup_svm = SVM(X, y)
print('LinearSVC accuracy: %.2f' % world_cup_svm["linear_svc"]["accuracy"])
print('SVM accuracy: %.2f' % world_cup_svm["svm"]["accuracy"])


#########################demo 2: Tuning hyperparameters for nonlinear SVM##########################
print_section("tuning")


def SVM_with_tuning(X, y):
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.model_selection import train_test_split

    # MFCC_all = pd.read_csv("SVM_demo.csv", index_col=74)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Testing and training sentences splitting (stratified + shuffled) based on the index (sentence ID)
    # Sentences = MFCC_all.index
    # Sentences_emotion = MFCC_all['Emotion']
    # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)

    # for train_index, test_index in sss.split(Sentences, Sentences_emotion):
    #     train_ind, test_ind = Sentences[train_index], Sentences[test_index]
    # Test_Matrix = MFCC_all.loc[test_ind]
    # Train_Matrix = MFCC_all.loc[train_ind]

    # data training with hyperparameter tuning for C
    clf = Pipeline([
        ('std_scaler', StandardScaler()),
        ("svm", SVC())
    ])
    param_grid = [
        {'svm__kernel': ['rbf'], 'svm__C': [2**x for x in range(0, 6)]},
    ]
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(clf, param_grid, cv=inner_cv,
                               n_jobs=1, scoring='accuracy', verbose=3)
    grid_search.fit(X_train, y_train)
    clf = grid_search.best_estimator_
    # data testing
    y_pred = clf.predict(X_test)

    print('*******************************************************************')
    print("The prediction accuracy (tuned) for all testing sentence is : {:.2f}%.".format(
        100*accuracy_score(y_test, y_pred)))
    print(classification_report(y_test, y_pred))

    # data training without hyperparameter tuning
    clf = Pipeline([
        ('std_scaler', StandardScaler()),
        ("svm", SVC())
    ])
    clf.fit(X_train, y_train)
    # data testing
    y_pred = clf.predict(X_test)

    print('*******************************************************************')
    print("The prediction accuracy (untuned) for all testing sentence is : {:.2f}%.".format(
        100*accuracy_score(y_test, y_pred)))


SVM_with_tuning(X, y)
