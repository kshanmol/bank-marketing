import get_data
import random
import sys
import os

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.cross_validation import KFold

import numpy as np
import matplotlib.pyplot as plt

import get_data

CONST_RANDOM_SEED = 42

def FitModel(train_data, train_labels, n_folds = 5, random_seed = CONST_RANDOM_SEED, max_iter = 1000):

	random.seed(random_seed)

	scaler = StandardScaler(); scaler.fit(train_data)
	train_data = scaler.transform(train_data)

	# # if using SVC
	# param_grid = [
	# 	{'C': map(lambda x: 2 ** x, range(1, 8)), 'kernel': ['rbf'], 
	# 		'gamma': map(lambda x: 2 ** x, range(-14, -4))}
	# ]

	# # if using SVC with class weights
	# class_weights = []
	# for x in map(lambda x: 2 ** x, range(2, 3)):
	# 	class_weights.append({0: 1, 1: x})

	# param_grid = [
	# 	{'C': map(lambda x: 2 ** x, range(0, 5)), 'kernel': ['rbf'], 
	# 		'gamma': map(lambda x: 2 ** x, range(-8, -5)), 'random_state': [random_seed], 'class_weight': class_weights}
	# ]

	# best_classifier = GridSearchCV(estimator = SVC(), param_grid = param_grid, cv = n_folds, scoring = 'roc_auc', verbose = 2, n_jobs = 4)
	# best_classifier.fit(train_data, train_labels)

	# print best_classifier.cv_results_['param_gamma']
	# print best_classifier.cv_results_['param_C']
	# print best_classifier.cv_results_['mean_train_score']
	# print best_classifier.cv_results_['mean_test_score']
	
	# opt_hyperparameters = best_classifier.best_params_
	# print opt_hyperparameters

	# print best_classifier.best_score_

	opt_hyperparameters = {'kernel': 'rbf', 'C': 1, 'random_state': 42, 'gamma': 0.00390625, 'class_weight': {0: 1, 1: 4}}

	classifier = SVC(**opt_hyperparameters)
	classifier.fit(train_data, train_labels)

	return scaler, classifier

def plot_confusion_matrix(cf_mat, classes, title):
	plt.figure()
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
	plt.show()

def TestModel(test_data, test_labels, scaler, model):
	
	test_data = scaler.transform(test_data)

	predicted_labels = model.predict(test_data)

	cf_mat = np.array(confusion_matrix(test_labels, predicted_labels))
	print 'Confusion matrix:'
	print cf_mat

	plot_confusion_matrix(cf_mat, classes = ['no', 'yes'], title = 'Confusion Matrix - SVM with gaussian kernel')

	accuracy = model.score(test_data, test_labels)

	print "Accuracy :", accuracy 

	return accuracy

def roc_statistics(test_data, test_labels, scaler, model):

	test_data = scaler.transform(test_data)

	fpr, tpr, thresholds = roc_curve(test_labels, model.decision_function(test_data))

	# print "AUROC:", auc(fpr, tpr)
	print "ROC AUC SCORE: ", roc_auc_score(test_labels, model.decision_function(test_data))
	
	# print fpr.tolist()
	# print tpr.tolist()
	plt.figure()
	plt.plot(fpr, tpr)
	plt.plot([0, 1], [0, 1], color = 'navy', linestyle = '--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc = "lower right")
	plt.show()


def plot_learning_curves(data, labels, n_folds = 5, random_seed = CONST_RANDOM_SEED):
	data_perc, train_scores, cv_scores = [], [], []

	for train_data_perc in (map(lambda x: 2 ** x, range(-2, 7, 1)) + [100]):
		cv_score, train_score = 0, 0
		train_data = data[: int((len(data) * train_data_perc) / 100)]
		train_labels = labels[: int((len(labels) * train_data_perc) / 100)]

		scaler = StandardScaler(); scaler.fit(train_data)
		train_data = scaler.transform(train_data)

		kf = KFold(len(train_data), n_folds = n_folds, shuffle = True, random_state = random_seed)

		train_data, train_labels = np.asarray(train_data), np.asarray(train_labels)
		for train_index, test_index in kf:
			X_train, X_test = train_data[train_index], train_data[test_index]
			y_train, y_test = train_labels[train_index], train_labels[test_index]

			opt_hyperparameters = {'kernel': 'rbf', 'C': 1, 'random_state': 42, 'gamma': 0.00390625, 'class_weight': {0: 1, 1: 4}}

			classifier = SVC(**opt_hyperparameters)
			classifier.fit(X_train, y_train)

			train_score += classifier.score(X_train, y_train)
			cv_score += classifier.score(X_test, y_test)
			# train_score += roc_auc_score(y_train, classifier.decision_function(X_train))
			# cv_score += roc_auc_score(y_test, classifier.decision_function(X_test))

		cv_score /= n_folds
		train_score /= n_folds
		print cv_score, train_score
		data_perc.append(train_data_perc)
		cv_scores.append(cv_score)
		train_scores.append(train_score)

	# data_perc = (map(lambda x: 2 ** x, range(-2, 7, 1)) + [100])
	# cv_scores.append(0.848379120879), train_scores.append(0.987995583819)
	# cv_scores.append(0.809660815939), train_scores.append(0.989605814979)
	# cv_scores.append(0.822855424192), train_scores.append(0.952971463266)
	# cv_scores.append(0.853137198351), train_scores.append(0.927608260069)
	# cv_scores.append(0.885559790098), train_scores.append(0.934622168995)
	# cv_scores.append(0.896031848827), train_scores.append(0.923740420835)
	# cv_scores.append(0.896385374271), train_scores.append(0.923535015858)
	# cv_scores.append(0.906059313633), train_scores.append(0.927964957847)
	# cv_scores.append(0.913662479221), train_scores.append(0.926304196355)
	# cv_scores.append(0.918569764839), train_scores.append(0.926738685065)
		
	cv_scores = map(lambda x: 1 - x, cv_scores)
	train_scores = map(lambda x: 1 - x, train_scores)

	plt.figure()
	plt.plot(data_perc, cv_scores, label = "CV-Error")
	plt.plot(data_perc, train_scores, label = "Training Error")
	# plt.plot(data_perc, cv_scores, label = "CV AUROC")
	# plt.plot(data_perc, train_scores, label = "Training AUROC")
	plt.ylim([0.05, 0.2])
	plt.xlabel('Training Data Percentage')
	plt.ylabel('Classification Error')
	# plt.ylabel('AUROC')
	plt.title('Learning Curve')
	plt.legend(loc = 'lower right')
	plt.show()

if __name__ == '__main__':

	random.seed(CONST_RANDOM_SEED)

	file_name = os.path.join(os.path.dirname(__file__), 'data/'+ 'transformed-bank-full.csv')

	x, y = get_data.process(file_name)

	train_data, test_data, train_labels, test_labels = train_test_split(x, y, test_size = 0.2, random_state = CONST_RANDOM_SEED)

	# # print len(train_data[0]), len(test_data[0])
	scaler, classifier = FitModel(train_data, train_labels)

	# print classifier
	TestModel(test_data, test_labels, scaler, classifier)

	roc_statistics(test_data, test_labels, scaler, classifier)

	# plot_learning_curves(train_data, train_labels)