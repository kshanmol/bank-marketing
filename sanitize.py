import sys
import os
import numpy as np

from sklearn import preprocessing
from collections import defaultdict

"""
   Input variables:
   # bank client data:
   1 - age (numeric)
   2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                       "blue-collar","self-employed","retired","technician","services") 
   3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
   4 - education (categorical: "unknown","secondary","primary","tertiary")
   5 - default: has credit in default? (binary: "yes","no")
   6 - balance: average yearly balance, in euros (numeric) 
   7 - housing: has housing loan? (binary: "yes","no")
   8 - loan: has personal loan? (binary: "yes","no")
   # related with the last contact of the current campaign:
   9 - contact: contact communication type (categorical: "unknown","telephone","cellular") 
  10 - day: last contact day of the month (numeric)
  11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
  12 - duration: last contact duration, in seconds (numeric)
   # other attributes:
  13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
  14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
  15 - previous: number of contacts performed before this campaign and for this client (numeric)
  16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

  Output variable (desired target):
  17 - y - has the client subscribed a term deposit? (binary: "yes","no")

"""

def process(file_name_string):

	file_name = os.path.join(os.path.dirname(__file__), 'data/'+ file_name_string)
	output_file_name = os.path.join(os.path.dirname(__file__), 'data/'+ 'transformed-' + file_name_string)
	output_data = []

	numeric_data = []
	nominal_data  = [2, 3, 4, 5, 7, 8, 9 , 11, 16]

	columnset = defaultdict(list)
	columndata = defaultdict(list)

	with open(file_name, 'r') as f:
		headers = f.readline()
		for line in f:
			data = line.split(';')
			for idx,item in enumerate(data):

				item = item.strip("\"")
				item = item.strip("\n")

				if (idx+1) in nominal_data:
					if item not in columnset[idx]:
						columnset[idx].append(item)
				columndata[idx].append(item)
	
	for key in columnset:
		lb = preprocessing.LabelBinarizer()
		lb.fit(columnset[key])
		columndata[key] = lb.transform(columndata[key])

	print "Binarized data"
	no_items = len(columndata[0])
	print no_items, " items"
	output_data = []
	output_label = {"no":0, "yes":1}
	for item in xrange(no_items):
		transformed_data = []
		for key in columndata:
			if isinstance(columndata[key][item], (np.ndarray, np.generic)):
				transformed_data.extend(columndata[key][item].tolist())
			else:
				if(key == 16):
					columndata[key][item] = output_label[columndata[key][item][:-1]]
				transformed_data.append(columndata[key][item])
		output_data.append(transformed_data)

	with open(output_file_name, 'w') as f:
		for item in output_data:
			f.write(",".join(map(str, item)) + '\n')

def main():
	process('bank-full.csv')

if __name__ == '__main__':
	main()