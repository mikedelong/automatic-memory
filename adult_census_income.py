# https://www.kaggle.com/bananuhbeatdown/multiple-ml-techniques-and-analysis-of-dataset

import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

start_time = time.time()

if __name__ == '__main__':
    formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    console_handler.setLevel(logging.DEBUG)
    logger.debug('started')

    path = './data/adult.data'
    logger.debug('full input file name: %s' % path)
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
             'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    data = pd.read_csv(path, names=names, header=None)
    logger.debug('load complete; data is %d x %d' % data.shape)
    logger.debug(data.columns)
    logger.debug('before cleanup, head looks like this: \n%s' % data.head(3))

    # we need to strip all the strings so the cleanup code below will work properly
    data = data.applymap(lambda x: x.strip() if type(x) is str else x)

    # remove rows where occupation is unknown
    data = data[data.occupation != '?']
    raw_data = data[data.occupation != '?']

    # create numerical columns representing the categorical data
    data['workclass_num'] = data.workclass.map(
        {'Private': 0, 'State-gov': 1, 'Federal-gov': 2, 'Self-emp-not-inc': 3, 'Self-emp-inc': 4, 'Local-gov': 5,
         'Without-pay': 6})
    data['over50K'] = np.where(data.income == '<=50K', 0, 1)
    data['marital_num'] = data['marital-status'].map(
        {'Widowed': 0, 'Divorced': 1, 'Separated': 2, 'Never-married': 3, 'Married-civ-spouse': 4,
         'Married-AF-spouse': 4, 'Married-spouse-absent': 5})
    data['race_num'] = data.race.map(
        {'White': 0, 'Black': 1, 'Asian-Pac-Islander': 2, 'Amer-Indian-Eskimo': 3, 'Other': 4})
    data['sex_num'] = np.where(data.sex == 'Female', 0, 1)
    data['rel_num'] = data.relationship.map(
        {'Not-in-family': 0, 'Unmarried': 0, 'Own-child': 0, 'Other-relative': 0, 'Husband': 1, 'Wife': 1})
    logger.debug('after cleanup data head looks like this: \n%s' % data.head())

    X = data[['workclass_num', 'education-num', 'marital_num', 'race_num', 'sex_num', 'rel_num', 'capital-gain',
              'capital-loss']]
    y = data.over50K

    # create a base classifier used to evaluate a subset of attributes
    logistic_regression_model = LogisticRegression()

    # create the RFE model and select 3 attributes
    rfe = RFE(logistic_regression_model, 3)
    rfe = rfe.fit(X, y)

    # summarize the selection of the attributes
    logger.debug('RFE model support: %s' % rfe.support_)
    logger.debug('REF model ranking: %s' % rfe.ranking_)

    # fit an Extra Trees model to the data
    extra_trees = ExtraTreesClassifier()
    extra_trees.fit(X, y)

    # display the relative importance of each attribute
    extra_trees_feature_importance = extra_trees.feature_importances_

    # horizontal bar plot of feature importance
    pos = np.arange(8) + 0.5
    plt.barh(pos, extra_trees_feature_importance, align='center')
    plt.title("Feature Importance")
    plt.xlabel("Model Accuracy")
    plt.ylabel("Features")
    y_ticks = ('Working Class', 'Education', 'Marital Status', 'Race', 'Sex', 'Relationship Status', 'Capital Gain',
               'Capital Loss')
    plt.yticks(pos, y_ticks)
    plt.grid(True)
    plt.tight_layout()
    extra_trees_output_file = './output/extra_trees_feature_importances.png'
    logger.debug('writing extra trees classifier feature importances to %s' % extra_trees_output_file)
    plt.savefig(extra_trees_output_file)

    for random_state in range(1, 20):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
        logistic_regression_model = LogisticRegression()
        logistic_regression_model.fit(X_train, y_train)
        y_predicted = logistic_regression_model.predict(X_test)
        logger.debug('random state: %d logistic regression score: %.4f' % (
        random_state, metrics.accuracy_score(y_test, y_predicted)))


    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
