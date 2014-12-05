import copy
import csv
with open('LoanData.csv', 'rU') as csvfile:
	loanreader = csv.DictReader(csvfile)
	fieldnames = copy.copy(loanreader.fieldnames)
	fieldnames.remove('fico_range_high')
	fieldnames.remove('fico_range_low')
	fieldnames += ['fico_avg']
	fieldnames.remove('loan_status')
	fieldnames.remove('home_ownership')
	fieldnames.remove('purpose')
	fieldnames.remove('addr_state')
	fieldnames.remove('term')
	print fieldnames

	# create a dictionary for sub_grade, map from string to int
	dictgrade = dict()
	beginNum = ord('A')
	endNum = ord('G')
	count = 0
	for number in range(beginNum, endNum+1):
		for num in range(1, 6):
			count += 1
			dictgrade[str(chr(number)) + str(num)] = count
	#print dictgrade

	# we still have to transform some fields to numeric values
    # dictionary for sub_grade
	# int_rate and 
	with open('./LoanDataExp.csv', 'w') as csvfile2:
		with open('./categorical.csv', 'w') as csvfile3:
			with open('./label.csv', 'w') as csvfile4:
				writer = csv.DictWriter(csvfile2, fieldnames=fieldnames)
				writer.writeheader()
				writer2 = csv.DictWriter(csvfile3, fieldnames=['home_ownership', 'addr_state', 'purpose', 'term'])
				writer2.writeheader()
				writer3 = csv.DictWriter(csvfile4, fieldnames=['loan_status'])
				writer3.writeheader()
				i = 0
				catrow = {}
				default = {}
				for row in loanreader:
					i = i + 1
					if i > 40000 :
						break
					else :
						catrow['home_ownership'] = row['home_ownership']
						catrow['addr_state'] = row['addr_state']
						catrow['purpose'] = row['purpose']
						# print catrow
						ficoavg = 0
						for key in row:
							if key == 'int_rate' or key == 'revol_util':
								row[key] = row[key].replace('%', '')
							elif key == 'term':
								catrow['term'] = row[key].split(' ')[1]
							elif key == 'desc':
								row[key] = len(row[key])
							elif key == 'emp_length':
								emplen = row[key].split(' ')[0]
								if emplen.isdigit():
									row[key] = emplen
								else:
									if emplen == '10+':
										row[key] = 10
									else:
										row[key] = 0
							elif key == 'is_inc_v':
								# 1 for verified, 0 for not verified
								verified = row[key].split(' ')[0]
								if verified.lower() == 'not':
									row[key] = '0'
								else:
									row[key] = '1'
							elif key == 'loan_status':
								status = row[key].lower()
								# 1 for charged off (defualt), 0 for fully paid
								if status == 'charged off':
									default['loan_status'] = 1
								elif status == 'fully paid':
									default['loan_status'] = 0
								else:
									default['loan_status'] = -1
							elif key == 'fico_range_high':
								ficoavg = ficoavg + int(row[key])
							elif key == 'fico_range_low':
								ficoavg = ficoavg + int(row[key])
							elif key == 'sub_grade':
								# transform grade to number, A1 is 1, A2 is 2, .....
								row[key] = dictgrade[row[key]];
							# elif key == 'home_ownership':
							# 	# 1 for OWN, 2 for RENT, 3 for MORTGAGE
							# 	if row[key] == 'OWN':
							# 		row[key] = 1
							# 	elif row[key] == 'RENT':
							# 		row[key] = 2
							# 	elif row[key] == 'MORTGAGE':
							# 		row[key] = 3
						# only keep the instances whose loan status is not "current"
						if default['loan_status'] == 0 or default['loan_status'] == 1:
							ficoavg = ficoavg / 2
							row.pop('loan_status', None)
							row.pop('fico_range_high', None)
							row.pop('fico_range_low', None)
							row.pop('home_ownership', None)
							row.pop('purpose', None)
							row.pop('addr_state', None)
							row.pop('term', None)
							row['fico_avg'] = ficoavg
							#print row
							writer.writerow(row)
							writer2.writerow(catrow)
							writer3.writerow(default)

# from sklearn import datasets
# iris = datasets.load_iris()
# print iris.data[0:10]

# import numpy as np
# loan_data = np.loadtxt('./LoanDataExp.csv', skiprows=1)
# print loan_data 
