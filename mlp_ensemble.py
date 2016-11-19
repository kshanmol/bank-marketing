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
from sklearn.cross_validation import KFold

import numpy as np
import matplotlib.pyplot as plt

CONST_RANDOM_SEED = 42

def plot_confusion_matrix(cf_mat, classes, title):

    plt.imshow(cf_mat, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thr = cf_mat.max() / 2.
    for i in range(cf_mat.shape[0]):
    	for j in range(cf_mat.shape[1]):
	        plt.text(j, i, cf_mat[i, j], horizontalalignment="center", color="white" if cf_mat[i, j] > thr else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def FitModel(train_data, train_labels, cross_val_splits = 5, random_seed = CONST_RANDOM_SEED, max_iter = 1000, do_grid_search = False, retrain = False):

	scaler = StandardScaler(); scaler.fit(train_data)
	train_data = scaler.transform(train_data)

	retrain = True
	do_grid_search = False

	if do_grid_search:
		param_grid = [ {# 'alpha': map(lambda x: (10 ** x), range(-1,0)),
						'base_estimator__activation': ['relu'],
						'base_estimator__solver': ['sgd'],
						'base_estimator__hidden_layer_sizes':[(6), (11), (6,6), (11,11), (11,6)],
						'base_estimator__learning_rate_init':[0.1],
						'base_estimator__random_state':[random_seed],
						'base_estimator__early_stopping':[False],
						'base_estimator__learning_rate':['adaptive'], 
						'n_estimators':[7],
						'random_state':[random_seed]
		}]
		clf = GridSearchCV(estimator = BaggingClassifier(MLPClassifier()), param_grid = param_grid, cv = cross_val_splits, scoring = 'roc_auc', verbose = 2, n_jobs = 4)
		clf.fit(train_data, train_labels)
	
		means = clf.cv_results_['mean_test_score']
		stds = clf.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, clf.cv_results_['params']):
			print "%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params)

		optimal_params = clf.best_params_
		print optimal_params

		bagging = clf.best_estimator_
		bagging.fit(train_data, train_labels)
#		joblib.dump(bagging, './models/mlp_ensemble.model')
	else:
		if(retrain):
			opt_hyperparameters = {'activation': 'relu', 
			'solver': 'sgd', 
			'early_stopping': False, 
			'hidden_layer_sizes': (6, 6), 
			'learning_rate_init': 0.1, 
			'random_state': 42, 
			'learning_rate': 'adaptive'}

			clf = MLPClassifier(**opt_hyperparameters)
			bagging = BaggingClassifier(clf, n_estimators = 7, random_state = random_seed)
			bagging.fit(train_data, train_labels)
			joblib.dump(clf, './models/mlp_ensemble.model')
		else:
			bagging = joblib.load('./models/mlp_ensemble.model')

	return scaler, bagging

def TestModel(test_data, test_labels, scaler, model):

	test_data = scaler.transform(test_data)

	accuracy = model.score(test_data, test_labels)

	print "Accuracy :", accuracy 

	predicted = model.predict(test_data)
	cf_mat = np.array(confusion_matrix(test_labels, predicted))
	print 'Confusion matrix:'
	print cf_mat

	class_names = ['no', 'yes']
	title = 'Confusion Matrix - Single MLP'
	plt.figure()
	plot_confusion_matrix(cf_mat, class_names, title)	
	plt.show()

	return accuracy

def roc_statistics(test_data, test_labels, scaler, model):

	test_data = scaler.transform(test_data)

	fpr, tpr, thresholds = roc_curve(test_labels, model.predict_proba(test_data)[:,1])

	# print "AUROC:", auc(fpr, tpr)
	print "ROC AUC SCORE: ", roc_auc_score(test_labels, model.predict_proba(test_data)[:,1])
	
	# print fpr.tolist()
	# print tpr.tolist()

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

def plot_learning_curves(data, labels, n_folds = 5, random_seed = CONST_RANDOM_SEED):
	data_perc, train_scores, cv_scores = [], [], []

	train_percs = map(lambda x: 2 ** x, range(-2, 7, 1))
	train_percs.append(100)

	for train_data_perc in train_percs:
		cv_score, train_score = 0, 0
		train_data = data[: int(len(data) * train_data_perc) / 100]
		train_labels = labels[: int(len(labels) * train_data_perc) / 100]

		scaler = StandardScaler(); scaler.fit(train_data)
		train_data = scaler.transform(train_data)

		kf = KFold(len(train_data), n_folds = n_folds, shuffle = True, random_state = random_seed)

		train_data, train_labels = np.asarray(train_data), np.asarray(train_labels)
		for train_index, test_index in kf:
			X_train, X_test = train_data[train_index], train_data[test_index]
			y_train, y_test = train_labels[train_index], train_labels[test_index]

			opt_hyperparameters = {'activation': 'relu', 
			'solver': 'sgd', 
			'early_stopping': False, 
			'hidden_layer_sizes': (6, 6), 
			'learning_rate_init': 0.1, 
			'random_state': 42, 
			'learning_rate': 'adaptive'}

			base_classifier = MLPClassifier(**opt_hyperparameters)

			classifier = BaggingClassifier(base_classifier, n_estimators=7,random_state = random_seed)
			classifier.fit(X_train, y_train)

			train_score += classifier.score(X_train, y_train)
			cv_score += classifier.score(X_test, y_test)

		cv_score /= n_folds
		train_score /= n_folds
		print cv_score, train_score
		data_perc.append(train_data_perc)
		cv_scores.append(cv_score)
		train_scores.append(train_score)

	cv_scores = map(lambda x: 1 - x, cv_scores)
	train_scores = map(lambda x: 1 - x, train_scores)

	plt.figure()
	plt.plot(data_perc, cv_scores, label = "CV-Error")
	plt.plot(data_perc, train_scores, label = "Training Error")
	plt.ylim([0.0, 0.2])
	plt.xlabel('Training Data Percentage')
	plt.ylabel('Classification Error')
	plt.title('Learning Curve')
	plt.legend(loc = 'lower right')
	plt.show()

def plot_learning_curves_AUC(data, labels, n_folds = 5, random_seed = CONST_RANDOM_SEED):
	data_perc, train_scores, cv_scores = [], [], []

	train_percs = map(lambda x: 2 ** x, range(-2, 7, 1))
	train_percs.append(100)

	for train_data_perc in train_percs:
		cv_score, train_score = 0, 0
		train_data = data[: int(len(data) * train_data_perc) / 100]
		train_labels = labels[: int(len(labels) * train_data_perc) / 100]

		scaler = StandardScaler(); scaler.fit(train_data)
		train_data = scaler.transform(train_data)

		kf = KFold(len(train_data), n_folds = n_folds, shuffle = True, random_state = random_seed)

		train_data, train_labels = np.asarray(train_data), np.asarray(train_labels)
		for train_index, test_index in kf:
			X_train, X_test = train_data[train_index], train_data[test_index]
			y_train, y_test = train_labels[train_index], train_labels[test_index]

			opt_hyperparameters = {'activation': 'relu', 
			'solver': 'sgd', 
			'early_stopping': False, 
			'hidden_layer_sizes': (6, 6), 
			'learning_rate_init': 0.1, 
			'random_state': 42, 
			'learning_rate': 'adaptive'}

			base_classifier = MLPClassifier(**opt_hyperparameters)

			classifier = BaggingClassifier(base_classifier, n_estimators=7,random_state = random_seed)
			classifier.fit(X_train, y_train)

			train_score += roc_auc_score(y_train, classifier.predict_proba(X_train)[:,1])
			cv_score += roc_auc_score(y_test, classifier.predict_proba(X_test)[:,1])

		cv_score /= n_folds
		train_score /= n_folds
		print cv_score, train_score
		data_perc.append(train_data_perc)
		cv_scores.append(cv_score)
		train_scores.append(train_score)

	plt.figure()
	plt.plot(data_perc, cv_scores, label = "CV AUROC")
	plt.plot(data_perc, train_scores, label = "Training AUROC")
	plt.ylim([0.75, 1.0])
	plt.xlabel('Training Data Percentage')
	plt.ylabel('AUROC')
	plt.title('Learning Curve')
	plt.legend(loc = 'lower right')
	plt.show()


def main():

	file_name = os.path.join(os.path.dirname(__file__), 'data/'+ 'transformed-bank-full.csv')

	x, y = get_data.process(file_name)

	train_data, test_data, train_labels, test_labels = train_test_split(x, y, test_size = 0.2, random_state = CONST_RANDOM_SEED)

	# print 'Data loaded'

	# scaler, classifier = FitModel(train_data, train_labels)
	# TestModel(test_data, test_labels, scaler, classifier)
	# roc_statistics(test_data, test_labels, scaler, classifier)
	plot_learning_curves_AUC(train_data, train_labels)

if __name__ == '__main__':
	main()

