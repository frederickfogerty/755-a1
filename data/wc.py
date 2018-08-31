import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import category_encoders as cs
from sklearn.pipeline import FeatureUnion
import os


class wc():
    @staticmethod
    def default_data():
        wc_path = os.path.join(os.path.dirname(__file__), '2018 worldcup.csv')
        default_data = pd.read_csv(wc_path, index_col=0)
        return default_data

    @staticmethod
    def extract(train_data, test_data=pd.DataFrame(columns=np.arange(26)), testing_data=False):
        # print('data.shape', data.shape)

        test_data.insert(len(test_data.columns), 'goals', 0)
        test_data.insert(len(test_data.columns), 'result', 0)
        data = pd.concat([train_data, test_data], sort=False)

        # match date is assumed to be irrelevant for the match results
        data.drop(['Date', 'Team1_Ball_Possession(%)'],
                  axis=1, inplace=True)
        data.describe()

        # world cup attributes
        w_features = data.iloc[:, np.arange(26)].copy()
        if (not testing_data):
            # world cup goal result
            w_goals = data.iloc[:, 26].copy()
            print('w_goals', type(w_goals))
            # wordl cup match result
            w_results = data.iloc[:, 27].copy()
        else:
            w_goals = pd.Series()
            w_results = pd.Series()

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
            data=full_pipeline.fit_transform(w_features),
            index=np.arange(1, len(data) + 1)
        )
        # worldcup_cleaned = pd.concat(
        # [features_prepared, w_goals.to_frame(), w_results.to_frame()], axis=1)

        X_train = features_prepared[:train_data.shape[0]]
        X_test = features_prepared[-test_data.shape[0]:]

        y_train_classification = w_results[:train_data.shape[0]]
        y_train_regression = w_goals[:train_data.shape[0]]

        return {
            'train': {
                'X': X_train,
                'y_classification': y_train_classification,
                'y_regression': y_train_regression
            },
            'test': X_test
        }
