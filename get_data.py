import random

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

CONST_RANDOM_SEED = 42

def process(dataset_file_name, shuffle = False):
	x, y = [], []
	data = []

	with open(dataset_file_name, 'r') as f:
		for line in f:
			data.append(line)

	if shuffle:
		random.shuffle(data)

	for line in data:
		line = line.split(",")
		x.append(map(float, line[:-1]))
		y.append(int(line[-1].split('\n')[0]))

	return x, y

# def reduced_features(x_train, x_test, threshold = 0.99):
# 	pca = PCA()
# 	output_data = pca.fit_transform(x_train)

# 	#Find k which explains 99% of variance

# 	k = 0
# 	explained = 0
# 	while(explained < threshold):
# 		explained += pca.explained_variance_ratio_[k]
# 		k += 1
# 	pca = PCA(k)
# 	x_train = pca.fit_transform(x_train)
# 	x_test = pca.transform(x_test)
	
# 	return x_train, x_test