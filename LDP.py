def perfmetrics(model, test, target):
	# calculate TP, TN, FP, FN
	pred = model.predict(test)
	TP, TN, FP, FN = 0, 0, 0, 0
	for i in range(0, len(test)):
		if 0.9 < pred[i] < 1.1:
			if 0.9 < target[i] < 1.1:
				TP += 1
			else:
				FP += 1
		else:
			if -0.1 < target[i] < 0.1:
				TN += 1
			else:
				FN += 1
	return (TP, TN, FP, FN)

import numpy as np

train_data = np.loadtxt('./train.csv', delimiter=',')

target_data = np.loadtxt('./target.csv', skiprows=1)


# for i in range(0, len(target_data)):
# 	print i+1, target_data[i]

#print train_data[0]

from sklearn.cross_validation import train_test_split
loan_train, loan_test, target_train, target_test = train_test_split(train_data, target_data)

# tmp_train = []
# tmp_target = []
# NofD = 0
# for i in range(0, len(target_train)):
# 	if 0.9 < target_train[i] < 1.1:
# 		tmp_train.append(loan_train[i])
# 		tmp_target.append(target_train[i])
# 		NofD += 1


# for i in range(0, len(loan_train)):
# 	if len(tmp_train) == 2 * NofD:
# 		break
# 	else:
# 		if -0.1 < target_train[i] < 0.1:
# 			tmp_train.append(loan_train[i])
# 			tmp_target.append(target_train[i])

# loan_train = tmp_train
# target_train = tmp_target


print 'Training set: ', len(loan_train), len(target_train)
print 'Test set: ', len(loan_test), len(target_test)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth=5, max_features=10)

truedefault = 0
for i in range(0, len(target_train)):
	if  0.9 < target_train[i] < 1.1:
		truedefault += 1

weight = float((len(target_train) - truedefault)) / float(truedefault)

print 'the weight is', weight

weightarray = []

for i in range(0, len(target_train)):
	if 0.9 < target_train[i] and target_train[i] < 1.1:
		#weightarray.append(10)
		weightarray.append(weight)
		#print target_train[i], weightarray[i]
	else:
		weightarray.append(1)
		#print target_train[i], weightarray[i]

weightarr = np.array(weightarray)
#print weightarr.size

# train the random forest classifier
rf.fit(loan_train, target_train, sample_weight=weightarr)
#rf.fit(loan_train, target_train)
#loan_train = rf.transform(loan_train)
#loan_test = rf.transform(loan_test)
#rf.fit(loan_train, target_train, sample_weight=weightarr)
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(rf, loan_train, target_train)
#print 'cross validation score is', scores.mean()

TP, TN, FP, FN = perfmetrics(rf, loan_test, target_test)

print 'True Positive', TP
print 'False Positive', FP
print 'True Negative', TN
print 'False Negative', FN

#loan_train_r = rf.transform(loan_train, threshold=None)
#loan_test_r = rf.transform(loan_test, threshold=None)
#rf.fit(loan_train_r, target_train, sample_weight=weightarr)
import copy
featim = copy.copy(rf.feature_importances_)
features = []
import csv
with open('./features.csv', 'rU') as featurefile:
	featurereader = csv.reader(featurefile)
	for feature in featurereader:
		features.extend(feature)
importances = []
for i in range(0, len(features)):
	featob = {}
	featob['feature'] = features[i]
	featob['importance'] = featim[i]
	importances.append(featob)
# sort the feature importance
sort = False
while not sort:
	sort = True
	for elem in range(0, len(features)-1):
		if importances[elem]['importance'] > importances[elem+1]['importance']:
			sort = False
			tmp = importances[elem+1]
			importances[elem+1] = importances[elem]
			importances[elem] = tmp
importances.reverse()
for i in range(0, len(features)-1):
	print 'the ' + str(i+1) + ' most important feature is', importances[i]['feature'], 'with', importances[i]['importance']



# test it
print 'the accuracy of Random Forest is ', rf.score(loan_test, target_test)
probs = rf.predict_proba(loan_test)
cla = rf.predict(loan_test)


# plot ROC  #
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(0, 2):
	fpr[i], tpr[i], hold1 = roc_curve(target_test, probs[:, i])
	roc_auc[i] = auc(fpr[i], tpr[i])

print len(target_test)
print len(probs[:, i])

plt.figure()
plt.plot(fpr[1], tpr[1], label='Random Forest (AUC = {0:0.4f})'.format(roc_auc[1]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of Random Forest')
plt.legend(loc="lower right")






print '**** Logistic Regression ****'
#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(class_weight='auto')
lr.fit(loan_train, target_train)
print 'the accuracy of Logistic Regression is ', lr.score(loan_test, target_test)
#print 'the accuracy of balanced test data set is ', lr.score(loan_test_b, target_test_b)


probs = lr.predict_proba(loan_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(0, 2):
	fpr[i], tpr[i], hold1 = roc_curve(target_test, probs[:, i])
	roc_auc[i] = auc(fpr[i], tpr[i])

#plt.figure()
plt.plot(fpr[1], tpr[1], label='Logistic Regression (AUC = {0:0.4f})'.format(roc_auc[1]))

# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
plt.title('ROC Comparison')
plt.legend(loc="lower right")
plt.show()
