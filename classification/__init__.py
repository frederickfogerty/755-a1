from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import scipy
from pprint import pprint

N_JOBS = -1


def SVM(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    clf_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
        ("svm", SVC())
    ])
    param_grid = {
        'svm__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'svm__C': [(2**x/10) for x in range(0, 10)],
        'svm__gamma': scipy.stats.uniform(),
        'svm__tol': scipy.stats.uniform(),
        'svm__shrinking': [True, False]
    }
    clf = RandomizedSearchCV(
        estimator=clf_pipeline, param_distributions=param_grid, n_iter=10, cv=10, verbose=2, random_state=42, n_jobs=N_JOBS)

    clf.fit(X_train, y_train)

    def performance():
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))

    def predict(X):
        return clf.predict(X)

    estimator_untuned = Pipeline([
        ('std_scaler', StandardScaler()),
        ("svm", SVC())
    ])
    estimator_untuned.fit(X_train, y_train)

    return {
        'performance': performance,
        'predict': predict,
        'estimator': clf,
        'estimator_untuned': estimator_untuned,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }


def Perceptron(X, y):
    from sklearn.linear_model import Perceptron

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    clf_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
        ("perceptron", Perceptron())
    ])
    param_grid = {
        'perceptron__penalty': ['l2', 'l1', 'elasticnet'],
        'perceptron__alpha': [(2**x/10) for x in range(0, 10)],
        'perceptron__tol': scipy.stats.uniform(),
    }
    clf = RandomizedSearchCV(
        estimator=clf_pipeline, param_distributions=param_grid, n_iter=10, cv=10, verbose=2, random_state=42, n_jobs=N_JOBS)

    clf.fit(X_train, y_train)

    def performance():
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))

    def predict(X):
        return clf.predict(X)

    estimator_untuned = Pipeline([
        ('std_scaler', StandardScaler()),
        ("perceptron", Perceptron())
    ])
    estimator_untuned.fit(X_train, y_train)

    return {
        'performance': performance,
        'predict': predict,
        'estimator': clf,
        'estimator_untuned': estimator_untuned,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }


def Decision_trees(X, y):
    from sklearn.tree import DecisionTreeClassifier

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    clf_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
        ("dt", DecisionTreeClassifier())
    ])
    param_grid = {
        'dt__max_features': ['auto', 'sqrt', 'log2', None],
        'dt__min_samples_split': scipy.stats.uniform(),
        'dt__min_samples_leaf': scipy.stats.uniform(0, 0.5),
        'dt__min_weight_fraction_leaf': scipy.stats.uniform(0, 0.5),

        # 'dt__min_impurity_decrease':
        # 'dt__penalty': ['l2', 'l1', 'elasticnet'],
        # 'dt__alpha': [(2**x/10) for x in range(0, 10)],
        # 'dt__tol': scipy.stats.uniform(),
    }
    clf = RandomizedSearchCV(
        estimator=clf_pipeline, param_distributions=param_grid, n_iter=10, cv=10, verbose=2, random_state=42, n_jobs=N_JOBS)

    clf.fit(X_train, y_train)

    def performance():
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))

    def predict(X):
        return clf.predict(X)

    estimator_untuned = Pipeline([
        ('std_scaler', StandardScaler()),
        ("dt", DecisionTreeClassifier())
    ])
    estimator_untuned.fit(X_train, y_train)

    return {
        'performance': performance,
        'predict': predict,
        'estimator': clf,
        'estimator_untuned': estimator_untuned,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }


def Nearest_neighbour(X, y):
    from sklearn.neighbors import KNeighborsClassifier

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    clf_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
        ("kn", KNeighborsClassifier())
    ])
    param_grid = {
        'kn__n_neighbors': scipy.stats.randint(1, 10),
        'kn__weights': ['uniform', 'distance'],
        'kn__algorithm': ['auto', 'ball_tree', 'kd_tree'],
        'kn__leaf_size': scipy.stats.randint(10, 40),
        'kn__p': [1, 2],
    }
    clf = RandomizedSearchCV(
        estimator=clf_pipeline, param_distributions=param_grid, n_iter=10, cv=10, verbose=2, random_state=42, n_jobs=N_JOBS)

    clf.fit(X_train, y_train)

    def performance():
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))

    def predict(X):
        return clf.predict(X)

    estimator_untuned = Pipeline([
        ('std_scaler', StandardScaler()),
        ("kn", KNeighborsClassifier())
    ])
    estimator_untuned.fit(X_train, y_train)

    return {
        'performance': performance,
        'predict': predict,
        'estimator': clf,
        'estimator_untuned': estimator_untuned,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }


def Bayes(X, y):
    from sklearn.naive_bayes import GaussianNB

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    clf = Pipeline([
        ('std_scaler', StandardScaler()),
        ("gn", GaussianNB())
    ])

    clf.fit(X_train, y_train)

    def performance():
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))

    def predict(X):
        return clf.predict(X)

    return {
        'performance': performance,
        'predict': predict,
        'estimator': clf,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }
