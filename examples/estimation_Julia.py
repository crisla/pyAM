"""
Define functions needed for estimation

@author : Julia

"""
from __future__ import division

from scipy import stats
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import numpy as np
import sympy as sym
import csv

import operator

class Julia_Estimation(object):

	def __init__(self):
		self.mynumber = 5


	def import_data(self,file_name, ID=True, weights=False, logs=False,yearly_w=False, change_weight=False, dummy=False, labels=True):
		'''
		This function imports the data from a csv file, returns ndarrays with it.

	    It can accomodate for having IDs in the first column, for data not in logs already,
	 	for weights (normalized or not) and can transform daily to yearly wages.

    	Parameters
   		-----

 		file_name : (str) path and name of the csv file.
    	ID (optional) : boolean, True if it contais ID in the first collunm
    	weights : True if the data includes a weights column in the end
    	logs: if the data is in logs (True) or in levels (False)
    	yearly_w: (optional) boolean, True if the wages need to be "annualized". 
		change_weight: (optional) boolean, True if data weights need to be "normalized". 
		dummy: Make True if data comes from model simulation
		labels: Make False if first row in csv.file does NOT contain data labels
    	Returns
    	------
    	Following np.ndarrays: profit (float), size (float), wage (float) (all in logs)

    	'''
		#Check
		#if dummy and weights:	
		#	print 'Check the input arguments: dummy data does not have a weight!'
		#	return
		# Opening data
		with open(file_name, 'rb') as f:
			reader = csv.reader(f)
			data = list(reader)

		# Passing data to lists, then to arrays (should change this to make it all in one) 
		size = []
		wage = []
		profit = []
		wgts= []
		c = 0
		if ID==False:
			c += 1
		if labels:
			fr = 1
		else:
			fr = 0
		for row in data[fr:]:
			size.append(float(row[1-c]))
			wage.append(float(row[2-c]))
			profit.append(float(row[3-c]))
			wgts.append(float(row[4-c]))
		if logs==False:
			# Firm size in LOG workers (float)
			size = np.log(np.array(size))
			# Daily LOG average wage for each firm, in euros (float)
			wage = np.log(np.array(wage))
			# Declared LOG average profits for each firm per year, in euros (float)
			profit = np.log(np.array(profit))
		else:
			# Firm size in workers (int)
			size = np.array(size)
			# Daily average wage for each firm, in euros (float)
			wage = np.array(wage)
			# Declared average profits for each firm per year, in euros (float)
			profit = np.array(profit)
		
		# In any case, weights should be the same
		wgts = np.array(wgts)

		if yearly_w:
			wage = np.log(np.exp(wage)*360) # ANNUAL wage
		self.data = (size, wage, profit, wgts)

		#if change_weight:
		#	wgts = wgts/np.sum(wgts)*len(wgts)

		#if dummy:
		#	n_size = dict(zip(list(map(str, range(0,len(size)))),np.exp(size)))
		#	sort_size = sorted(n_size.items(), key=operator.itemgetter(1))
		#	size_range = sorted(size)
		#	pdf_x = self.pdf_workers(self.knots)        	# calculates pdf of xs in one step
		#	n_pdf_x = dict(enumerate(pdf_x)) 			# creates a dictionary where the keys are the #obs of x
		#	wgts = np.empty(0)
		#	for pair in sort_size:
		#		index = int(pair[0])
		#		wgts  = np.hstack((wgts ,(n_pdf_x[index]/pair[1])))		#weights contain the number of firms of this size
#
#			n_wage = dict(enumerate(wage))
#			n_profit = dict(enumerate(profit))
#			wage_range = np.empty(0)
#			profit_range = np.empty(0)
#			for pair in sort_size:
#				index = int(pair[0])
#				wage_range = np.hstack((wage_range,n_wage[index]))		#sort the data by size
#				profit_range = np.hstack((profit_range,n_profit[index]))
			#cdf_theta_hat  = np.cumsum(pdf_theta_hat )			# Backing up model cdf
			#cdf_theta_hat  = cdf_theta_hat /cdf_theta_hat [-1] 	# Normilization of the model cdf

    	# Storing results
	#	if weights:
		#self.data = (size, wage, profit, wgts)
	#	elif dummy:
	#		self.data = (size_range, wage_range, profit_range, wgts) #sorted data
	#	else:
	#		self.data = (size, wage, profit)

	def pdf_workers(self,x):
		'''
		For a given x returns the corresponding pdf value, 
		according to the paramenters especified when creating the instance.

		Parameters:
		-----------

		x: (float or int) a point in x (the distribution of firm size)

		Returns:
		--------
		pdf(x) (float) according to the parameters of the instance.

		'''
		return np.sqrt(2)*np.exp(-(-self.x_pam[0] + np.log(x))**2/(2*self.x_pam[1]**2))/(np.sqrt(np.pi)*x*self.x_pam[1])
