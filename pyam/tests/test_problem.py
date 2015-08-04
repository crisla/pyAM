"""
Testing suite for the problem.py module.

@author : David R. Pugh
@date : 2015-08-02

"""
import unittest

import numpy as np
import sympy as sym

import pycollocation

from .. import inputs
from .. import initial_guesses
from .. import problem
from .. import visualizer


class MultiplicativeSeparabilityCase(unittest.TestCase):

    def setUp(self):
        """Set up code for test fixtures."""
        # define some workers skill
        x, a, b = sym.var('x, a, b')
        skill_cdf = (x - a) / (b - a)
        skill_params = {'a': 1.0, 'b': 2.0}
        skill_bounds = [skill_params['a'], skill_params['b']]

        workers = inputs.Input(var=x,
                               cdf=skill_cdf,
                               params=skill_params,
                               bounds=skill_bounds,
                               )

        # define some firms
        y = sym.var('y')
        productivity_cdf = (y - a) / (b - a)
        productivity_params = skill_params
        productivity_bounds = skill_bounds

        firms = inputs.Input(var=y,
                             cdf=productivity_cdf,
                             params=productivity_params,
                             bounds=productivity_bounds,
                             )

        # define symbolic expression for CES between x and y
        x, y, omega_A, sigma_A = sym.var('x, y, omega_A, sigma_A')
        A = ((omega_A * x**((sigma_A - 1) / sigma_A) +
             (1 - omega_A) * y**((sigma_A - 1) / sigma_A))**(sigma_A / (sigma_A - 1)))

        # define symbolic expression for Cobb-Douglas between l and r
        l, r, omega_B, sigma_B = sym.var('l, r, omega_A, sigma_A')
        B = l**omega_B * r**(1 - omega_B)

        # generate random parameters
        omega, sigma = np.random.random(2)
        F_params = {'omega_A': omega, 'omega_B': omega, 'sigma_A': sigma, 'sigma_B': 1.0}
        F = A * B

        self.problem = problem.AssortativeMatchingProblem(assortativity='positive',
                                                          input1=workers,
                                                          input2=firms,
                                                          F=F,
                                                          F_params=F_params,
                                                          )

        self.solver = pycollocation.OrthogonalPolynomialSolver(self.problem)

    def test_solve(self):
        """Test trivial example for solver."""
        # set up initial guess (which is actually the solution!)
        initial_guess = initial_guesses.OrthogonalPolynomialInitialGuess(self.solver)
        initial_polys = initial_guess.compute_initial_guess("Chebyshev",
                                                            degrees={'mu': 1, 'theta': 1},
                                                            f=lambda x, alpha: x**alpha,
                                                            alpha=1.0)
        initial_coefs = {'mu': initial_polys['mu'].coef,
                         'theta': initial_polys['theta'].coef}

        # solve the model over the desired domain
        domain = [self.problem.input1.lower, self.problem.input1.upper]
        self.solver.solve(kind="Chebyshev",
                          coefs_dict=initial_coefs,
                          domain=domain,
                          method='hybr')

        # visualize the solution at some interpolation points
        viz = visualizer.Visualizer(self.solver)
        interp_knots = np.linspace(domain[0], domain[1], 1000)
        viz.interpolation_knots = interp_knots

        # check that average error is sufficiently small
        np.testing.assert_almost_equal(np.mean(viz.normalized_residuals.mu), 0.0)
        np.testing.assert_almost_equal(np.mean(viz.normalized_residuals.theta), 0.0)
