#!/usr/bin/env python3
import argparse
# parser = argparse.ArgumentParser(description='Fit some models.')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='interger list')
# parser.add_argument('--sum', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')
# args = parser.parse_args()

import argparse
import sys
from typing import Dict, Callable, TextIO, Any

from regression import Regression, Ridge
from classification import SVM, Perceptron, Decision_trees, Nearest_neighbour, Bayes

from data import wc, traffic, occupancy, landsat

import pandas as pd
import numpy as np

from pprint import pprint
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report


parser = argparse.ArgumentParser(description='Fit some models.')
parser.add_argument('-t',
                    '--type',
                    help="The type of model to use.", required=True, choices=[
                        'regression',
                        'ridge',
                        'perceptron',
                        'svm',
                        'decision_trees',
                        'nearest_neighbour',
                        'bayes',
                        'all'
                    ]
                    )
parser.add_argument('-d',
                    '--dataset',
                    help="The dataset the model should be trained on. Can be one of: wc, traffic, occupancy, landsat.",
                    required=True, choices=['wc',
                                            'traffic',
                                            'occupancy',
                                            'landsat',
                                            'all'
                                            ]
                    )
parser.add_argument('-i',
                    '--input',
                    type=argparse.FileType('r'),
                    help="File with data to test against."
                    )
parser.add_argument('-o',
                    '--output',
                    type=argparse.FileType('w'),
                    help="File to output to. Default is sys.stdout", nargs='?',
                    default=sys.stdout
                    )
parser.add_argument('-s',
                    '--save-model',
                    dest='save_model',
                    action='store_true',
                    help="Save the trained model to disk."
                    )
parser.add_argument('-r',
                    '--re-train',
                    dest='re_train',
                    action='store_true',
                    help="Re-train the model."
                    )
parser.add_argument('-p',
                    '--performance',
                    dest='display_performance',
                    action='store_true',
                    help="Display performance results for the trained model on the internal validation set."
                    )

type_switcher: Dict[str, Callable[[], Dict[str, Any]]] = {
    'regression': Regression,
    'ridge': Ridge,
    'perceptron': Perceptron,
    'svm': SVM,
    'decision_trees': Decision_trees,
    'nearest_neighbour': Nearest_neighbour,
    'bayes': Bayes
}

dataset_switcher: Dict[str, Callable[[Any, Any], Dict[str, Any]]] = {
    'wc': wc,
    'traffic': traffic,
    'occupancy': occupancy,
    'landsat': landsat
}

import dill as pickle


def pickle_file_path(dataset, type):
    return 'saved_models/{0}-{1}.pickle'.format(dataset, type)


def save_model(dataset, type, model):
    # Store data (serialize)
    with open(pickle_file_path(dataset, type), 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(dataset, type):
    with open(pickle_file_path(dataset, type), 'rb') as handle:
        unserialized_data = pickle.load(handle)
    return unserialized_data


import os.path

TITLES = {
    'regression': 'Ordinary Regression',
    'ridge': 'Ridge Regression',
    'perceptron': 'Perceptron Classification',
    'svm': 'SVM Classification',
    'decision_trees': 'Decision Trees Classification',
    'nearest_neighbour': 'Nearest Neighbour Classification',
    'bayes': 'Naive Bayes Classification',
}


def model_exists_on_disk(dataset, type):
    return os.path.exists(pickle_file_path(dataset, type))


def display_performance(model, dataset, type):
    if (type in ['regression', 'ridge']):
        y_pred_train = model['predict'](model['X_train'])
        y_train = model['y_train']
        MSE_train = mean_squared_error(y_train, y_pred_train)
        r2_train = r2_score(y_train, y_pred_train)
        y_pred = model['predict'](model['X_test'])
        y_test = model['y_test']
        MSE = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        # print("MSE: %.2f"
        #       % MSE)
        # print('r^2 score: %.2f' % r2)
        if ('estimator_untuned' in model):
            y_untuned_pred = model['estimator_untuned'].predict(
                model['X_test'])
            MSE_untuned = mean_squared_error(y_test, y_untuned_pred)
            r2_untuned = r2_score(y_test, y_untuned_pred)

            print("""
\\subsubsection{{{0}}}

The results for {0} are detailed in \\autoref{{{1}-{2}}}

\\begin{{table}}[H]
\\renewcommand{{\\arraystretch}}{{1.3}}
\\caption{{Performance for {0}}}
\\label{{{1}-{2}}}
\\centering
\\begin{{tabular}}{{
@{{}}
l
r
r
r
@{{}}}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Training Value}} & \\textbf{{Tuned Value}} & \\textbf{{Untuned Value}} \\\\
\\midrule
MSE & {3:.2f} & {5:.2f} & {7:.2f} \\\\
$r^2$ & {4:.2f} & {6:.2f} & {8:.2f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""".format(
                TITLES[type],
                dataset,
                type,
                MSE_train,
                r2_train,
                MSE,
                r2,
                MSE_untuned,
                r2_untuned,
            ))
        else:
            print("""
\\subsubsection{{{0}}}

The results for {0} are detailed in \\autoref{{{1}-{2}}}

\\begin{{table}}[H]
\\renewcommand{{\\arraystretch}}{{1.3}}
\\caption{{Performance for {0}}}
\\label{{{1}-{2}}}
\\centering
\\begin{{tabular}}{{
@{{}}
l
r
r
@{{}}}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Training Value}} & \\textbf{{Test Value}} \\\\
\\midrule
MSE & {3:.2f} & {5:.2f} \\\\
$r^2$ & {4:.2f} & {6:.2f}\\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""".format(
                TITLES[type],
                dataset,
                type,
                MSE_train,
                r2_train,
                MSE,
                r2,
            ))

    else:
        y_pred = model['predict'](model['X_test'])
        y_test = model['y_test']
        accuracy = 100*accuracy_score(y_test, y_pred)
        cr = report2dict(classification_report(y_test, y_pred))

        y_pred_train = model['predict'](model['X_train'])
        y_train = model['y_train']
        accuracy_train = 100*accuracy_score(y_train, y_pred_train)
        cr_train = report2dict(classification_report(y_train, y_pred_train))

        if ('estimator_untuned' in model):
            y_untuned_pred = model['estimator_untuned'].predict(
                model['X_test'])
            accuracy_untuned = 100*accuracy_score(y_test, y_untuned_pred)
            cr_untuned = report2dict(
                classification_report(y_test, y_untuned_pred))
            print("""
\\subsubsection{{{0}}}

The results for {0} are detailed in \\autoref{{{1}-{2}}}

\\begin{{table}}[H]
\\renewcommand{{\\arraystretch}}{{1.3}}
\\caption{{Performance for {0}}}
\\label{{{1}-{2}}}
\\centering
\\begin{{tabular}}{{
@{{}}
l
r
r
r
@{{}}}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Training Value}} & \\textbf{{Tuned Value}} & \\textbf{{Untuned Value}}  \\\\
\\midrule
Accuracy & {3:.1f} & {7:.1f} & {11:.1f} \\\\
Precision & {4} & {8} & {12} \\\\
Recall & {5} & {9} & {13} \\\\
f1-score & {6} & {10} & {14} \\\\

\\bottomrule
\\end{{tabular}}
\\end{{table}}
    """.format(
                TITLES[type],
                dataset,
                type,
                accuracy_train,
                cr_train['avg / total']['precision'],
                cr_train['avg / total']['recall'],
                cr_train['avg / total']['f1-score'],
                accuracy,
                cr['avg / total']['precision'],
                cr['avg / total']['recall'],
                cr['avg / total']['f1-score'],
                accuracy_untuned,
                cr_untuned['avg / total']['precision'],
                cr_untuned['avg / total']['recall'],
                cr_untuned['avg / total']['f1-score'],
            ))
        else:
            print("""
\\subsubsection{{{0}}}

The results for {0} are detailed in \\autoref{{{1}-{2}}}

\\begin{{table}}[H]
\\renewcommand{{\\arraystretch}}{{1.3}}
\\caption{{Performance for {0}}}
\\label{{{1}-{2}}}
\\centering
\\begin{{tabular}}{{
@{{}}
l
r
r
@{{}}}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Training Value}} & \\textbf{{Test Value}}  \\\\
\\midrule
Accuracy & {3:.1f} & {7:.1f} \\\\
Precision & {4} & {8} \\\\
Recall & {5} & {9} \\\\
f1-score & {6} & {10} \\\\

\\bottomrule
\\end{{tabular}}
\\end{{table}}
    """.format(
                TITLES[type],
                dataset,
                type,
                accuracy_train,
                cr_train['avg / total']['precision'],
                cr_train['avg / total']['recall'],
                cr_train['avg / total']['f1-score'],
                accuracy,
                cr['avg / total']['precision'],
                cr['avg / total']['recall'],
                cr['avg / total']['f1-score'],
            ))

    if ('estimator_untuned' in model):
        print("""
The tuned hyper-parameters are detailed in \\autoref{{{1}-{2}-hyper}}

\\begin{{table}}[H]
\\renewcommand{{\\arraystretch}}{{1.3}}
\\caption{{Hyperparameters for {0}}}
\\label{{{1}-{2}-hyper}}
\\centering
\\begin{{tabular}}{{@{{}} l r @{{}}}}
\\toprule
\\textbf{{Parameter}} & \\textbf{{Value}} \\\\
\\midrule
""".format(
            TITLES[type],
            dataset,
            type,
        ))
        best_params_ = model['estimator'].best_params_
        for param in best_params_:
            value = best_params_[param]
            # print(param, value)
            print("{0} & {1} \\\\".format(param, best_params_[param]))
            # if (isinstance(value, str)):
            #     print("{0} & {1} \\\\".format(param, best_params_[param]))
            # else:
            #     print("{0} & {1:.2f} \\\\".format(param, best_params_[param]))

        print("""
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")

        # pprint(model['estimator'].best_params_)


def run_for(dataset, type, args):
    # print('========================================')
    # print('Running for: dataset: {0}, type: {1}'.format(dataset, type))
    # print('\\subsection{{ {0} }}'.format(dataset))
    dataset_tools = dataset_switcher.get(dataset)

    default_data = dataset_tools.default_data()
    if (args.input):
        input_data = dataset_tools.read_file(args.input)

        data_extracted = dataset_tools.extract(default_data, input_data)

        X_test = data_extracted['test']
    else:
        data_extracted = dataset_tools.extract(default_data)

    X = data_extracted['train']['X']
    if (type in ['regression', 'ridge']):
        y = data_extracted['train']['y_regression']
    else:
        y = data_extracted['train']['y_classification']

    if (args.re_train or not model_exists_on_disk(dataset, type)):
        # print('Training model...')
        model_factory = type_switcher.get(type)
        model = model_factory(X=X, y=y)
    else:
        # print('Loading model from disk...')
        model = load_model(dataset=dataset, type=type)

    if (args.save_model):
        save_model(dataset=dataset, type=type, model=model)

    # model['performance']()

    if (args.display_performance):
        display_performance(model, dataset, type)
        # pprint(model['estimator'].best_params_)

    if (args.input):
        y_pred = model['predict'](X_test)
        pd.DataFrame(y_pred).to_csv(args.output)


regression_models = ['regression', 'ridge']
classification_models = ['perceptron', 'svm',
                         'decision_trees', 'nearest_neighbour', 'bayes']
dataset_models = {
    'wc': regression_models + classification_models,
    'traffic': regression_models,
    'occupancy': classification_models,
    'landsat': classification_models
}

# Taken from https://github.com/scikit-learn/scikit-learn/issues/7845#issuecomment-273939069


from collections import defaultdict


def report2dict(cr):
    # Parse rows
    tmp = list()
    for row in cr.split("\n"):
        parsed_row = [x for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0:
            tmp.append(parsed_row)

    # Store in dictionary
    measures = tmp[0]

    D_class_data = defaultdict(dict)
    for row in tmp[1:]:
        class_label = row[0]
        for j, m in enumerate(measures):
            D_class_data[class_label][m.strip()] = float(row[j + 1].strip())
    return D_class_data


def run():
    args = parser.parse_args()
    dataset = args.dataset
    type = args.type

    if (dataset == 'all'):
        for temp_dataset in dataset_models.keys():
            print('\\subsection{{ {0} }}'.format(temp_dataset))
            if (type == 'all'):
                for temp_type in dataset_models[temp_dataset]:
                    run_for(temp_dataset, temp_type, args)
            else:
                if (type in dataset_models[temp_dataset]):
                    run_for(temp_dataset, type, args)
    elif (type == 'all'):
        for temp_type in dataset_models[dataset]:
            run_for(dataset, temp_type, args)
    else:
        run_for(dataset, type, args)


run()
