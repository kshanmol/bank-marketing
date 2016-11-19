import matplotlib.pyplot as plt
import gaussian_svm, linear_svm
import mlp, mlp_ensemble
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

if __name__ == '__main__':

	random.seed(CONST_RANDOM_SEED)

	file_name = os.path.join(os.path.dirname(__file__), 'data/'+ 'transformed-bank-full.csv')

	x, y = get_data.process(file_name)

	train_data, test_data, train_labels, test_labels = train_test_split(x, y, test_size = 0.2, random_state = CONST_RANDOM_SEED)

	scaler, model = linear_svm.FitModel(train_data, train_labels, random_seed = CONST_RANDOM_SEED)
	linear_svm_fpr, linear_svm_tpr = linear_svm.roc_statistics(test_data, test_labels, scaler, model)
	print "linear svm done"

	scaler, model = gaussian_svm.FitModel(train_data, train_labels, random_seed = CONST_RANDOM_SEED)
	gaussian_svm_fpr, gaussian_svm_tpr = gaussian_svm.roc_statistics(test_data, test_labels, scaler, model)
	print "gaussian svm done"

	scaler, model = mlp.FitModel(train_data, train_labels, random_seed = CONST_RANDOM_SEED)
	mlp_fpr, mlp_tpr = mlp.roc_statistics(test_data, test_labels, scaler, model)
	print "mlp done"

	scaler, model = mlp_ensemble.FitModel(train_data, train_labels, random_seed = CONST_RANDOM_SEED)
	mlp_ensemble_fpr, mlp_ensemble_tpr = mlp_ensemble.roc_statistics(test_data, test_labels, scaler, model)
	print "mlp ensemble done"

plt.figure()
plt.plot(mlp_fpr, mlp_tpr, label = "Single MLP")
plt.plot(mlp_ensemble_fpr, mlp_ensemble_tpr, label = "Ensemble of MLPs (Bagging)")
plt.plot(linear_svm_fpr, linear_svm_tpr, label = "SVM with linear kernel")
plt.plot(gaussian_svm_fpr, gaussian_svm_tpr, label = "SVM with gaussian kernel")
plt.plot([0, 1], [0, 1], color = 'navy', linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc = "lower right")
plt.show()