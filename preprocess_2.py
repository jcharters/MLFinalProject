import csv
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
cat_data = list(csv.DictReader(open('categorical.csv', 'rU')))
# c = 0
# for row in cat_data:
# 	c = c + 1
# 	if c > 10 :
# 		break
# 	else :
# 		print row
cat_data = vec.fit_transform(cat_data).toarray()
# print cat_data[0]
# print vec.inverse_transform(cat_data[0])
with open('./features.csv', 'w') as featurefile:
	writer1 = csv.writer(featurefile)
	with open('./LoanDataExp.csv', 'rU') as loanfile:
		loanreader = csv.reader(loanfile)
		dictionary = []
		for row in loanreader:
			if loanreader.line_num == 1:
				dictionary.extend(row)
				print dictionary
		for i in range(0, len(dictionary)):
			row = []
			row.append(dictionary[i])
			writer1.writerow(row)
		features = vec.get_feature_names()
		for feat in features:
		 	print feat
		 	row = []
		 	row.append(feat)
		 	writer1.writerow(row)

from sklearn import preprocessing
import numpy as np

loan_data = np.genfromtxt('./LoanDataExp.csv', delimiter=',', skiprows=1)

#print 'loan_data has' + str(len(loan_data)) + ' instances.'

loan_target = np.loadtxt('./label.csv', skiprows=1)

instances = len(loan_data)

loan_complete_data = np.concatenate((loan_data, cat_data), axis=1)

print 'loan_data has ' + str(len(loan_complete_data)) + ' instances.'

length = len(loan_complete_data)

counter = 0;

MissingValue = []

# find the missing record indices
for i in range(0, length):
	#print '* ' + str(i) + ' *', np.isnan(sum(loan_complete_data[i]))
	if np.isnan(sum(loan_complete_data[i])) == True:
		counter += 1
		MissingValue.append(i)

# delete records with missing values
train_data = np.delete(loan_complete_data, MissingValue, axis=0)

print "now training data set has " + str(len(train_data)) + " instances."

print "there are in total " + str(counter) + " records that have missing values"

with open('./label.csv', 'rU') as labelfile:
	labelreader = csv.DictReader(labelfile)
	with open('target.csv', 'w') as targetfile:
		writer = csv.DictWriter(targetfile, fieldnames=['loan_status'])
		writer.writeheader()
		rowN = 0
		missN = 0
		for row in labelreader:
			delete = False
			if missN < len(MissingValue):
				if rowN == MissingValue[missN]:
					delete = True
					missN += 1
			if delete == False:
				writer.writerow(row)
			rowN +=1
        # for row in labelreader:
        # 	delete = False
        # 	if missN < len(MissingValue) :
        # 		if rowN == MissingValue[missN]:
        # 			delete = True
        # 			missN += 1
        # 	if delete == False:
        # 		writer.writerow(row)
        # 	# print rowN
        # 	# if missN < len(MissingValue):
        # 	# 	writer.writerow(row)
        # 	# 	if rowN != MissingValue[missN]:
        # 	# 		writer.writerow(row)
        # 	# 	else :
        # 	# 		missN += 1
        # 	# else:
        # 	# 	writer.writerow(row)
        # 	rowN += 1
        # for row in labelreader:
        # 	if missN < len(MissingValue):
        # 		if rowN != MissingValue[missN] :
        # 			writer.writerow(row)
	       #  		missN += 1
	       #  else :
	       #  	writer.writerow(row)
        # 	rowN += 1

min_max_scaler = preprocessing.MinMaxScaler()
train_data = min_max_scaler.fit_transform(train_data)
np.savetxt('train.csv', train_data, delimiter=',')