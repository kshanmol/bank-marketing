import get_data
import random
import sys
import os

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import numpy as np
import matplotlib.pyplot as plt


CONST_RANDOM_SEED = 42

def FitModel(train_data, train_labels, cross_val_splits = 5, random_seed = CONST_RANDOM_SEED, max_iter = 1000):

	random.seed(random_seed)

	scaler = StandardScaler(); scaler.fit(train_data)
	train_data = scaler.transform(train_data)

	# # if using SVC
	# param_grid = [
	# 	{'C': map(lambda x: 2 ** x, range(5, 10)), 'kernel': ['rbf'], 
	# 		'gamma': map(lambda x: 2 ** x, range(-10, -5)), 'epsilon': map(lambda x: 2 ** x, range(-5, 0))}
	# ]

	# # if using LinearSVC
	param_grid = [
		{'C': map(lambda x: 2 ** x, range(5, 10)), 'max_iter': [max_iter],
			'dual': [False], 'random_state': [random_seed], 'loss' = ['hinge', 'squared_hinge']}
	]

	best_classifier = GridSearchCV(estimator = LinearSVC(), param_grid = param_grid, cv = cross_val_splits)
	best_classifier.fit(train_data, train_labels)
	opt_hyperparameters = best_classifier.best_params_
	print opt_hyperparameters

	# opt_hyperparameters = {'epsilon': 0.03125, 'C': 64, 'gamma': 0.0009765625, 'kernel': 'rbf'} # for math
	# opt_hyperparameters = {'epsilon': 0.5, 'C': 128, 'gamma': 0.0009765625, 'kernel': 'rbf'} # for por

	classifier = LinearSVC(**opt_hyperparameters)

	classifier.fit(train_data, train_labels)

	return scaler, classifier

def TestModel(test_data, test_labels, scaler, model):
	
	test_data = scaler.transform(test_data)

	predicted_labels = model.predict(test_labels)

	accuracy = model.score(test_data, test_labels)

	print "Accuracy :", accuracy 

	return accuracy


if __name__ == '__main__':

	random.seed(CONST_RANDOM_SEED)

	file_name = os.path.join(os.path.dirname(__file__), 'data/'+ 'transformed-data')