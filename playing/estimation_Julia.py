
"""
Define functions needed for estimation

@author : Julia & Cristina

TO_DOs:

Part I: Re-typing the code

1.1 Check Current functions solve correctly a sample model.
1.2 Build the Residual Function (classical flavour: regressions)
1.3 Test the Residual Function
1.4 Build plotting capabilities (initial guesses, current solution vs data)
1.5 Test RF in minimizer

Part II: One step approach

3.1 Finish reset_inputs--> updates x_pam and y_pam (maybe bounds and scaling too)
3.2 Build a more comprehansive solve model that takes xypams as input too
3.3 Rewrite Residual_Function to incorporate xpams too
3.4 Test new residual function
3.5 Feed it to the minimizer


Part III: Two Step approach

3.1 Build find_optimal_pams --> Finds optimal parameters for x_pam and y_pam
3.2 Build find_optimal_xypams --> Taking the resulting parameters as given, finds optimal x_pam and y_pam
3.3 Build The_Big_Wrap --> wraps 2.1 and 2.2 into a giant solver

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

	def __init__(self, x_pam, x_bounds, y_pam, y_bounds, x_scaling,A_scaling):
		'''
		Puts together an Heterogeneous Workers and Firms model with the parameters especified,
		imports data to carry on the estimation and estimates the parameters of the model:
		                       { omega_A, omega_B, sigma and Big_A }

		Call the Instructions() function to print the condensed user manual.

		Parameters:
		-------

		x_pam: tuple or list, with floats loc1, mu1 sigma1.
		x_bounds: tuple or list, with floats x.lower and x.upper.
		y_pam: tuple or list, with floats loc2, mu2 and sigma2.
		y_bounds: tuple or list, with y.lower and y.upper.
		x_scaling: float, average workers per firm, to scale up the size of x.
				   May change in the future to make it endogenous.
		A_scaling: float, scaling parameter for Big_A. To be chosen from the data.
		
		'''

		self.x_pam = x_pam
		self.x_bounds = x_bounds
		self.y_pam = y_pam
		self.y_bounds = y_bounds
		self.x_scaling = x_scaling
		self.A_scaling = A_scaling 
		
		self.workers = None
		self.firms = None
		self.F

		self.solver = None
		self.initial_guess = None
		self.solution_t  = None

		self.data = None

		self.ready = False	
		self.mynumber = 5

	def set_inputs(self):
		"""
		Sets inputs up with the information provided when creating the instance of the class
		"""
		# define some workers skill
		x, loc1, mu1, sigma1 = sym.var('x, loc1, mu1, sigma1')
		skill_cdf = 0.5 + 0.5 * sym.erf((sym.log(x - loc1) - mu1) / sym.sqrt(2 * sigma1**2))
		skill_params = {'loc1': self.x_pam[0], 'mu1': self.x_pam[1], 'sigma1': self.x_pam[2]}

		self.workers = pyam.Input(var=x,
                     cdf=skill_cdf,
                     params=skill_params,
                     bounds=self.x_bounds,  # guesses for the alpha and (1 - alpha) quantiles!
                     alpha=0.0005,
                     measure=self.x_scaling  # 15x more workers than firms
                     )

		# define some firms - scaling normalized to 1
		y, loc2, mu2, sigma2 = sym.var('y, loc2, mu2, sigma2')
		productivity_cdf = 0.5 + 0.5 * sym.erf((sym.log(y - loc2) - mu2) / sym.sqrt(2 * sigma2**2))
		productivity_params = {'loc2': self.y_pam[0] 'mu2': self.y_pam[1], 'sigma2': self.y_pam[2]}

		self.firms = pyam.Input(var=y,
                   cdf=productivity_cdf,
                   params=productivity_params,
                   bounds=self.x_bounds,  # guesses for the alpha and (1 - alpha) quantiles!
                   alpha=0.0005,
                   measure=1.0
                   )

	def plot_inputs(self):
		""" Under construction. Will return plots for workers or firms."""

	def reset_inputs(self,xypams):
		""" Under construction. Will updated stored parameter values and call set_inputs()"""

	def set_Production_Function(self, Pfunction):
		self.F = PFunction

	def set_up_solver(self,PFunction,F_params,assorty="positive"):
		""" Does orthogonal collocation by default """
		
		problem = pyam.AssortativeMatchingProblem(assortativity=assorty,
                                          input1=self.workers,
                                          input2=self.firms,
                                          F=self.F,
                                          F_params=F_params)

		self.solver = pycollocation.OrthogonalPolynomialSolver(problem)

	def initial_guess(degrees_mu,degrees_theta,kind="Chebyshev"):
		"""Sets up the initial guess"""
		initial_guess = pyam.OrthogonalPolynomialInitialGuess(self.solver)
		initial_polys = initial_guess.compute_initial_guess(kind,
                                                    degrees={'mu':degrees_mu, 'theta': degrees_theta},
                                                    f=lambda x, alpha: x**alpha,
                                                    alpha=0.0001)
		self.initial_guess = {'guess':initial_guess,'polys':initial_polys}

	def areyoureadyforthis(self):
		""" Work in progress: Do a batery of tests before proceesing to detect mistakes"""

	def solve_model(self,kind='Chebyshev'):
		domain = [self.workers.lower, self.workers.upper]
		initial_coefs = {'mu': self.initial_guess['polys']['mu'].coef,
                 'theta': self.initial_guess['polys']['theta'].coef}

		self.solver.solve(kind=kind,
             	coefs_dict=initial_coefs,
             	domain=domain,
             	method='hybr')
		if self.solver.result.success:
			self.solution_t = pyam.Visualizer(solver).solution
		else:
			print "Something went wrong!"


	def diditwork(self):
		"""Helpful shortcut to know if the model worked fine"""
		return self.solver.result.success


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
