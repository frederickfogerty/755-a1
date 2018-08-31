import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import category_encoders as cs
from sklearn.pipeline import FeatureUnion
import os


class landsat():

    @staticmethod
    def default_data():
        path = os.path.join(os.path.dirname(
            __file__), 'Landsat', 'lantsat.csv')
        return landsat.read_file(path)

    @staticmethod
    def read_file(file):
        return pd.read_csv(file, header=None)

    @staticmethod
    def extract(train_data, test_data=pd.DataFrame(columns=np.arange(26))):
        train_data = train_data.copy()
        test_data = test_data.copy()

        X_train = train_data.iloc[:, np.arange(36)].copy()

        y_train_classification = train_data.iloc[:, 36].copy()

        y_train_regression = None

        X_test = test_data

        return {
            'train': {
                'X': X_train,
                'y_classification': y_train_classification,
                'y_regression': y_train_regression
            },
            'test': X_test
        }


class occupancy():
    @staticmethod
    def default_data():
        path = os.path.join(os.path.dirname(
            __file__), 'Occupancy_sensor', 'occupancy_sensor_data.csv')
        return occupancy.read_file(path)

    @staticmethod
    def read_file(file):
        return pd.read_csv(file)

    @staticmethod
    def extract(train_data, test_data=pd.DataFrame(columns=np.arange(26))):
        train_data = train_data.copy()
        test_data = test_data.copy()

        X_train = train_data.drop(
            ['date', 'HumidityRatio', 'Occupancy'], axis=1)

        y_train_classification = train_data['Occupancy']

        y_train_regression = None

        X_test = test_data

        return {
            'train': {
                'X': X_train,
                'y_classification': y_train_classification,
                'y_regression': y_train_regression
            },
            'test': X_test
        }


class traffic():
    @staticmethod
    def default_data():
        path = os.path.join(os.path.dirname(
            __file__), 'Traffic_flow', 'traffic_flow_data.csv')
        return traffic.read_file(path)

    @staticmethod
    def read_file(file):
        return pd.read_csv(file)

    @staticmethod
    def extract(train_data, test_data=pd.DataFrame(columns=np.arange(26))):
        train_data = train_data.copy()
        test_data = test_data.copy()

        X_train = train_data.drop(['Segment23_(t+1)'], axis=1)

        y_train_regression = train_data['Segment23_(t+1)']

        y_train_classification = None

        X_test = test_data

        return {
            'train': {
                'X': X_train,
                'y_classification': y_train_classification,
                'y_regression': y_train_regression
            },
            'test': X_test
        }


class wc():
    @staticmethod
    def default_data():
        path = os.path.join(os.path.dirname(
            __file__), 'World_cup_2018', '2018 worldcup.csv')
        return wc.read_file(path)

    @staticmethod
    def read_file(file):
        return pd.read_csv(file, index_col=0)

    @staticmethod
    def extract(train_data, test_data=pd.DataFrame(columns=np.arange(26))):
        train_data = train_data.copy()
        test_data = test_data.copy()
        test_data.insert(len(test_data.columns), 'Total_Scores', 0)
        test_data.insert(len(test_data.columns), 'Match_result', 0)
        data = pd.concat([train_data, test_data], sort=False)

        # match date is assumed to be irrelevant for the match results

        data.drop([
            'Date',
            'Team1_Attempts',
            'Team1_Corners',
            'Team1_Offsides',
            'Team1_Ball_Possession(%)',
            'Team1_Pass_Accuracy(%)',
            'Team1_Distance_Covered',
            'Team1_Ball_Recovered',
            'Team1_Yellow_Card',
            'Team1_Red_Card',
            'Team1_Fouls',

            'Team2_Attempts',
            'Team2_Corners',
            'Team2_Offsides',
            'Team2_Ball_Possession(%)',
            'Team2_Pass_Accuracy(%)',
            'Team2_Distance_Covered',
            'Team2_Ball_Recovered',
            'Team2_Yellow_Card',
            'Team2_Red_Card',
            'Team2_Fouls',

            'Normal_Time',
        ],
            axis=1,
            inplace=True)
        data.describe()

        # world cup attributes
        w_features = data.iloc[:, np.arange(6)].copy()
        # world cup goal result
        w_goals = data.loc[:, 'Total_Scores'].copy()
        # wordl cup match result
        w_results = data.loc[:, 'Match_result'].copy()

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
        # w_features_num = w_features.drop([
        #     'Location',
        #     'Phase',
        #     'Team1',
        #     'Team2',
        #     'Team1_Continent',
        #     'Team2_Continent',
        # ], axis=1, inplace=False)
        w_features_cat = w_features[[
            'Location',
            'Phase',
            'Team1',
            'Team2',
            'Team1_Continent',
            'Team2_Continent',
        ]].copy()

        # num_pipeline = Pipeline([
        #     ('selector', DataFrameSelector(list(w_features_num))),
        #     ('imputer', Imputer(strategy="median")),
        #     ('std_scaler', StandardScaler()),
        # ])

        cat_pipeline = Pipeline([
            ('selector', DataFrameSelector(list(w_features_cat))),
            ('cat_encoder', cs.OneHotEncoder(drop_invariant=True)),
        ])

        full_pipeline = FeatureUnion(transformer_list=[
            # ("num_pipeline", num_pipeline),
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
