import get_data
import random
import sys
import os

from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score

import numpy as np
import matplotlib.pyplot as plt

CONST_RANDOM_SEED = 42

def FitModel(train_data, train_labels, cross_val_splits = 5, random_seed = CONST_RANDOM_SEED, max_iter = 1000, do_grid_search = False, retrain = False):

	scaler = StandardScaler(); scaler.fit(train_data)
	train_data = scaler.transform(train_data)

	retrain = True
	do_grid_search = True

	if do_grid_search:
		param_grid = [ {# 'alpha': map(lambda x: (10 ** x), range(-1,0)),
						'base_estimator__activation': ['relu'],
						'base_estimator__solver': ['adam'],
						'base_estimator__hidden_layer_sizes':[(6,6), (11,11)],
						'base_estimator__learning_rate_init':[0.01, 0.1, 1],
						'base_estimator__random_state':[random_seed],
						'base_estimator__early_stopping':[True],
						'n_estimators':[7],
						'random_state':[random_seed]
		}]
		clf = GridSearchCV(estimator = BaggingClassifier(MLPClassifier()), param_grid = param_grid, cv = cross_val_splits)
		clf.fit(train_data, train_labels)
	
		means = clf.cv_results_['mean_test_score']
		stds = clf.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, clf.cv_results_['params']):
			print "%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params)

		optimal_params = clf.best_params_
		print optimal_params

		clf = clf.best_estimator_
		clf.fit(train_data, train_labels)
		joblib.dump(clf, './models/model_3.model')
	else:
		if(retrain):
			clf = MLPClassifier(activation = 'relu', solver='lbfgs', hidden_layer_sizes=(11,11), learning_rate_init = 0.1, random_state = random_seed)
			bagging = BaggingClassifier(clf, n_estimators = 7, random_state = random_seed)
			bagging.fit(train_data, train_labels)
			joblib.dump(clf, './models/model_2.model')
		else:
			clf = joblib.load('./models/model_1.model')

	return scaler, bagging

def TestModel(test_data, test_labels, scaler, model):

	test_data = scaler.transform(test_data)

	accuracy = model.score(test_data, test_labels)

	print "Accuracy :", accuracy 

	# cf_mat = np.array(confusion_matrix(y_test, predicted))
	# print 'Confusion matrix:'
	# print cf_mat

	# class_names = range(0,10)
	# title = 'Confusion Matrix (Learning rate = ' + str(lr) + ')'
	# plt.figure()
	# plot_confusion_matrix(cf_mat, class_names, title)	
	# plt.show()

	return accuracy

def roc_statistics(test_data, test_labels, scaler, model):

	test_data = scaler.transform(test_data)

	fpr, tpr, thresholds = roc_curve(test_labels, model.predict_proba(test_data)[:,1])

	# print "AUROC:", auc(fpr, tpr)
	print "ROC AUC SCORE: ", roc_auc_score(test_labels, model.predict_proba(test_data)[:,1])
	
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

def main():

	file_name = os.path.join(os.path.dirname(__file__), 'data/'+ 'transformed-bank-full.csv')

	x, y = get_data.process(file_name)

	train_data, test_data, train_labels, test_labels = train_test_split(x, y, test_size = 0.2, random_state = CONST_RANDOM_SEED)

	print 'Data loaded'

	scaler, classifier = FitModel(train_data, train_labels)
	# TestModel(test_data, test_labels, scaler, classifier)
	# roc_statistics(test_data, test_labels, scaler, classifier)

if __name__ == '__main__':
	main()

