"""
Test suite for the model.py module.

@author : David R. Pugh
@date : 2014-08-02

"""
import nose

import sympy as sym
from scipy import stats

from .. import inputs
from .. import model


# define endogenous variables
mu, theta = sym.var('mu, theta')

# define some workers skill
x, mu1, sigma1 = sym.var('x, mu1, sigma1')
skill_cdf = 0.5 + 0.5 * sym.erf((sym.log(x) - mu1) / sym.sqrt(2 * sigma1**2))
skill_params = {'mu1': 0.0, 'sigma1': 1.0}
skill_bounds = [1e-3, 5e1]

workers = inputs.Input(var=x,
                       cdf=skill_cdf,
                       params=skill_params,
                       bounds=skill_bounds,
                       )

# define some firms
y, mu2, sigma2 = sym.var('y, mu2, sigma2')
productivity_cdf = 0.5 + 0.5 * sym.erf((sym.log(y) - mu2) / sym.sqrt(2 * sigma2**2))
productivity_params = {'mu2': 0.0, 'sigma2': 1.0}
productivity_bounds = [1e-3, 5e1]

firms = inputs.Input(var=y,
                     cdf=productivity_cdf,
                     params=productivity_params,
                     bounds=productivity_bounds,
                     )

# define some valid model params
valid_F_params = {'nu': 0.89, 'kappa': 1.0, 'gamma': 0.54, 'rho': 0.24, 'A': 1.0}

# define a valid production function
A, kappa, nu, rho, l, gamma, r = sym.var('A, kappa, nu, rho, l, gamma, r')
valid_F = r * A * kappa * (nu * x**rho + (1 - nu) * (y * (l / r))**rho)**(gamma / rho)


def test__validate_assortativity():
    """Testing validation of assortativity attribute."""

    # assortativity must be either 'positive' or 'negative'
    invalid_assortativity = 'invalid_assortativity'

    with nose.tools.assert_raises(AttributeError):
        mod = model.AssortativeMatchingModelLike()
        mod.assortativity = invalid_assortativity

    # assortativity must be a string
    invalid_assortativity = 0.0

    with nose.tools.assert_raises(AttributeError):
        mod = model.AssortativeMatchingModelLike()
        mod.assortativity = invalid_assortativity


def test__validate_production_function():
    """Testing validation of production function attribute."""

    # production function must have type sym.Basic
    def invalid_F(x, y, l, r, A, kappa, nu, rho, gamma):
        """Valid F must return a SymPy expression."""
        return r * A * kappa * (nu * x**rho + (1 - nu) * (y * (l / r))**rho)**(gamma / rho)

    with nose.tools.assert_raises(AttributeError):
        mod = model.AssortativeMatchingModelLike()
        mod.F = invalid_F

    # production function must share vars with workers and firms
    m, n = sym.var('m, n')
    invalid_F = r * A * kappa * (nu * m**rho + (1 - nu) * (n * (l / r))**rho)**(gamma / rho)

    with nose.tools.assert_raises(AttributeError):
        mod = model.AssortativeMatchingModelLike()
        mod.F = invalid_F

    # production function must depend on r and l
    m, n = sym.var('m, n')
    invalid_F = r * A * kappa * (nu * x**rho + (1 - nu) * (y * (m / n))**rho)**(gamma / rho)

    with nose.tools.assert_raises(AttributeError):
        mod = model.AssortativeMatchingModelLike()
        mod.F = invalid_F


def test__validate_F_params():
    """Testing validation of F_params attribute."""

    # valid parameters must be a dict
    invalid_F_params = [1.0, 2.0, 3.0, 4.0, 5.0]

    with nose.tools.assert_raises(AttributeError):
        mod = model.AssortativeMatchingModelLike()
        mod.F_params = invalid_F_params


def test__validate_input():
    """Testing validation of inputs."""

    # valid inputs must be an instance of the Input class
    invalid_input = stats.lognorm(s=1.0)

    with nose.tools.assert_raises(AttributeError):
        mod = model.AssortativeMatchingModelLike()
        mod.input2 = invalid_input
