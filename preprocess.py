import copy
import csv
with open('LoanData.csv', 'rU') as csvfile:
	loanreader = csv.DictReader(csvfile)
	fieldnames = copy.copy(loanreader.fieldnames)
	fieldnames.remove('fico_range_high')
	fieldnames.remove('fico_range_low')
	fieldnames += ['fico_avg']
    # we still have to transform some fields to numeric values

	# int_rate and 
	with open('LoanDataExp.csv', 'w') as csvfile2:
		writer = csv.DictWriter(csvfile2, fieldnames=fieldnames)
		writer.writeheader()
		i = 0
		for row in loanreader:
			i = i + 1
			if i > 20 :
				break
			else :
				ficoavg = 0
				for key in row:
					if key == 'int_rate' or key == 'revol_util':
						row[key] = row[key].replace('%', '')
					elif key == 'term':
						row[key] = row[key].split(' ')[1]
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
							row[key] = 1
						elif status == 'fully paid':
							row[key] = 0
					elif key == 'fico_range_high':
						ficoavg = ficoavg + int(row[key])
					elif key == 'fico_range_low':
						ficoavg = ficoavg + int(row[key])
				# only keep the instances whose loan status is not "current"
				if row['loan_status'] == 0 or row['loan_status'] == 1:
					ficoavg = ficoavg / 2
					row.pop('fico_range_high', None)
					row.pop('fico_range_low', None)
					row['fico_avg'] = ficoavg
					print row
					writer.writerow(row)
	    	
	    	    