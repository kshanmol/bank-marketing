import get_data
import random
import sys
import os

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score

import numpy as np
import matplotlib.pyplot as plt

import get_data

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
	# param_grid = [
	# 	{'C': map(lambda x: 2 ** x, range(-5, 5)), 'max_iter': [max_iter],
	# 		'dual': [False], 'random_state': [random_seed], 'loss': ['squared_hinge']}
	# ]

	# best_classifier = GridSearchCV(estimator = LinearSVC(), param_grid = param_grid, cv = cross_val_splits)
	# best_classifier.fit(train_data, train_labels)

	# C = map(lambda x: 2 ** x, range(-5, 5))
	# for i in range(len(C)):
	# 	print C[i], best_classifier.cv_results_['mean_train_score'][i], best_classifier.cv_results_['mean_test_score'][i]
	
	# opt_hyperparameters = best_classifier.best_params_
	# print opt_hyperparameters

	opt_hyperparameters = {'loss': 'squared_hinge', 'C': 0.5, 'max_iter': 1000, 'random_state': 42, 'dual': False}

	classifier = LinearSVC(**opt_hyperparameters)
	classifier.fit(train_data, train_labels)

	return scaler, classifier

def TestModel(test_data, test_labels, scaler, model):
	
	test_data = scaler.transform(test_data)

	predicted_labels = model.predict(test_data)

	accuracy = model.score(test_data, test_labels)

	print "Accuracy :", accuracy 

	return accuracy

def roc_statistics(test_data, test_labels, scaler, model):

	test_data = scaler.transform(test_data)

	fpr, tpr, thresholds = roc_curve(test_labels, model.decision_function(test_data))

	# print "AUROC:", auc(fpr, tpr)
	print "ROC AUC SCORE: ", roc_auc_score(test_labels, model.decision_function(test_data))
	
	plt.figure()
	plt.plot(fpr, tpr)
	plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()

if __name__ == '__main__':

	random.seed(CONST_RANDOM_SEED)

	file_name = os.path.join(os.path.dirname(__file__), 'data/'+ 'transformed-bank.csv')

	x, y = get_data.process(file_name)

	train_data, test_data, train_labels, test_labels = train_test_split(x, y, test_size = 0.2, random_state = CONST_RANDOM_SEED)

	# print len(train_data[0]), len(test_data[0])
	scaler, classifier = FitModel(train_data, train_labels)

	# # print classifier
	# TestModel(test_data, test_labels, scaler, classifier)

	roc_statistics(test_data, test_labels, scaler, classifier)