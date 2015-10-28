
"""
Define functions needed for estimation

@author : Julia & Cristina

TO_DOs:

Part I: Re-typing the code

1.1 Check Current functions solve correctly a sample model.
1.2 Test the Residual Function
1.3 Build plotting capabilities (initial guesses, current solution vs data)
1.4 Test RF in minimizer

Part II: One step approach

DONE 2.1 Finish reset_inputs--> updates x_pam and y_pam (maybe bounds and scaling too)
DONE 2.2 Build a more comprehansive solve model that takes xypams as input too
IN PROGRESS 2.3 Rewrite Residual_Function to incorporate xpams too
2.4 Test new residual function
2.5 Feed it to the minimizer


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
		self.pdf_t = None
		self.cdf_t = None

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
	def set_Production_Function(self, Pfunction):
		""" To be called after defining inputs """
		self.F = PFunction

	def plot_inputs(self):
		""" Under construction. Will return plots for workers or firms."""

	def reset_inputs(self,xypams):
		"""
		Assuming we only change variance of x and y:

		Input: tuple or list (sigma_x,sigma_y)

		Output: None. Change firms and workers instances stored

		"""
		# 1. Reset parameters
		self.x_pam[2] = xypams[0]
		self.y_pam[2] = xypams[1]
		# 2. Reset firms and workers
		skill_params = {'loc1': self.x_pam[0], 'mu1': self.x_pam[1], 'sigma1': self.x_pam[2]}
		productivity_params = {'loc2': self.y_pam[0] 'mu2': self.y_pam[1], 'sigma2': self.y_pam[2]}
		self.workers = pyam.Input(var=x,
                     cdf=skill_cdf,
                     params=skill_params,
                     bounds=self.x_bounds,  # guesses for the alpha and (1 - alpha) quantiles!
                     alpha=0.0005,
                     measure=self.x_scaling  # 15x more workers than firms
                     )
		self.firms = pyam.Input(var=y,
                   cdf=productivity_cdf,
                   params=productivity_params,
                   bounds=self.x_bounds,  # guesses for the alpha and (1 - alpha) quantiles!
                   alpha=0.0005,
                   measure=1.0
                   )

	

	def set_up_solver(self,F_params,assorty="positive"):
		""" Does orthogonal collocation by default """
		assert self.F != None, "You need to specify a production function first"
		problem = pyam.AssortativeMatchingProblem(assortativity=assorty,
                                          input1=self.workers,
                                          input2=self.firms,
                                          F=self.F,
                                          F_params=F_params)

		self.solver = pycollocation.OrthogonalPolynomialSolver(problem)

	def initial_guess(degrees_mu,degrees_theta,kind="Chebyshev"):
		"""Sets up the initial guess"""
		assert self.solver != None, "You need to set up the solver first. Call set_up_solver"
		initial_guess = pyam.OrthogonalPolynomialInitialGuess(self.solver)
		initial_polys = initial_guess.compute_initial_guess(kind,
                                                    degrees={'mu':degrees_mu, 'theta': degrees_theta},
                                                    f=lambda x, alpha: x**alpha,
                                                    alpha=0.0001)
		self.initial_guess = {'guess':initial_guess,'polys':initial_polys}


	def solve_model(self,kind='Chebyshev'):
		assert self.initial_guess != None, "You need an initial guess first. Call initial_guess"
		domain = [self.workers.lower, self.workers.upper]
		initial_coefs = {'mu': self.initial_guess['polys']['mu'].coef,
                 'theta': self.initial_guess['polys']['theta'].coef}

		self.solver.solve(kind=kind,
             	coefs_dict=initial_coefs,
             	domain=domain,
             	method='hybr')

		assert self.solver.result.success, 'Somethign wnet wrong with the solver. Check parameters.'
		viz = pyam.Visualizer(self.solver)
		self.solution_t = viz.solution
		self.pdf_t = viz.compute_pdf('theta', normalize=True)
		self.cdf_t = viz.compute_cdf(self.pdf_t)


	def Get_Model_Values(self,pams,degrees_mu,degrees_theta):
		""" Work in progress

		Input: pams: tuple or vector with (xypams,F_params)
		Output: tuple with model values for (xs_sol,theta_sol, wage_sol,profit_sol)

	    """
	    # 1. Unpack the parameters
	    xypams = pams[0]
	    F_params=pams[1]
		# 2. Update inputs and solver
		self.reset_inputs(xypams)
		self.set_up_solver(F_params)
		# 3. Get New Initial guess
		self.initial_guess(degrees_mu,degrees_theta)
		# 4. Solve Model
		self.solve_model()
		# 5. Return data of interest
		theta_sol = self.solution_t[['theta']].values
		wage_sol = self.solution_t[['factor_payment_1']].values		
		profit_sol = self.solution_t[['factor_payment_2']].values

		return ((theta_sol,wage_sol, profit_sol),(self.pdf_t,self.cdf_t))

	def import_data(self,file_name, cut_off_point=1, ID=True, weights=False, logs=False,yearly_w=False, change_weight=False, dummy=False, labels=True):
		'''
		This function imports the data from a csv file, returns ndarrays with it.

	    It can accomodate for having IDs in the first column, for data not in logs already,
	 	for weights (normalized or not) and can transform daily to yearly wages.

    	Parameters
   		-----

 		file_name : (str) path and name of the csv file.
 		cut_off_point: (int) minimum firm size to be specified from the data, 1 default value
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
		size = np.array(())
		wage = np.array(())
		profit = np.array(())
		wgts= np.array(())
		c = 0
		if ID==False:
			c += 1
		if labels:
			fr = 1
		else:
			fr = 0
		for row in data[fr:]:
			np.append(size,float(row[1-c]))
			np.append(wage,float(row[2-c]))
			np.append(profit,float(row[3-c]))
			if weights:
				np.append(wgts,float(row[4-c]))
			else:
				wgts = np.ones(len(size))
		if logs==False:
			# Firm size in LOG workers (float)
			size = np.log(size)
			# Daily LOG average wage for each firm, in euros (float)
			wage = np.log(wage)
			# Declared LOG average profits for each firm per year, in euros (float)
			profit = np.log(profit)

		if yearly_w:
			wage = np.log(np.exp(wage)*360) # ANNUAL wage
		
		if change_weight:
			wgts = wgts/np.sum(wgts)*len(wgts)

		# Defining the cutoff
		cutoff = len(size[np.exp(size)<cut_off_point])-1

		# Calculating size distributions
		pdf_size = np.cumsum(wgt_data[cutoff:])/np.sum(wgt_data[cutoff:])
		cdf_size = wgt_data[cutoff:]/np.sum(wgt_data[cutoff:])		

    	# Storing results	
		self.data = ((size[cutoff:], wage[cutoff:], profit[cutoff:], wgts[cutoff:]),(pdf_size,cdf_size))

	def Residual_Function(self,params,degrees_mu,degrees_theta,penalty=100):
		""" Calculates the residuals given the data and parameters """
		assert self.data != None, 'You need to import data first'
		
		# 1. Unpack values
		val_sol,dis_sol = self.Get_Model_Values(params,degrees_mu,degrees_theta)
		# Values from model
		thetas_hat = val_sol[0]
		ws_hat = val_sol[1]
		pis_hat = val_sol[2]
		# Values from data
		thetas = self.data[0][0]
		ws = self.data[0][1]
		pis = self.data[0][2]
 		weights	= self.data[0][3]		
 		# Distributions from model
 		pdf_hat = dis_sol[0]
 		cdf_hat = dis_sol[1]
 		# Distributions from data
 		pdf = self.data[1][0]
 		cdf = self.data[1][1]

		# 2. Calcualte Residuals
		# 2.1 Wage and profit regressions
		# Initializing functions from model	
		theta_w = interp1d(ws_hat, thetas_hat,bounds_error=False,fill_value=0.0)
		theta_pi = interp1d(pis_hat,thetas_hat,bounds_error=False,fill_value=0.0)
		# Storing space
		w_err = np.array(())
		pi_err = np.array(())
		# Wage Regression
		thw_hat = theta_w(np.exp(ws))
		w_err = (thw_hat-thetas)**2
		w_err = np.place(w_err, thw_hat==0, penalty)
		# Profit Regression
		thpi_hat = theta_pi(np.exp(pis))
		pi_err = (thpi_hat-thetas)**2
		pi_err = np.place(pi_err, thpi_hat==0, penalty)
		
		# Adding up the errors
		mse_w = np.sum(w_err)
		mse_pi = np.sum(pi_err)

		# 2.2 Distributions
		# Calculate the error
		theta_err = np.array(())
		""" WARNING: CODE IN PROGRESS """
		

		mse = (w_err + pi_err + theta_err)/len(w_err)

		return mse





