"""
Numerical maximization of expected log likelihood.

"""
from __future__ import division, print_function, absolute_import

from functools import partial

import numpy as np
from numpy.testing import assert_equal
import scipy.optimize

import algopy
from algopy import log, exp, square, zeros
from algopy.special import logit, expit

from indelmodel import neg_ll


__all__ = [
        'infer_parameter_values',
        ]


def eval_grad(f, theta):
    theta = algopy.UTPM.init_jacobian(theta)
    return algopy.UTPM.extract_jacobian(f(theta))


def eval_hess(f, theta):
    theta = algopy.UTPM.init_hessian(theta)
    return algopy.UTPM.extract_hessian(len(theta), f(theta))


def eval_hessp(f, theta, v):
    theta = algopy.UTPM.init_hess_vec(theta, v)
    return algopy.UTPM.extract_hess_vec(len(theta), f(theta))


def infer_parameter_values(p_guess, mu_guess, data, mask):
    k = p_guess.shape[1]

    # Pack the guess.
    packed_guess = np.concatenate((log(p_guess.flatten()), logit(mu_guess)))

    # Define the objective function, some of its derivatives, and a guess.
    f = partial(penalized_packed_neg_ll, k, data, mask)
    g = partial(eval_grad, f)
    h = partial(eval_hess, f)
    #hessp = partial(eval_hessp, f)

    cg = algopy.CGraph()
    x = algopy.Function(list(range(1, len(packed_guess)+1)))
    y = f(x)
    cg.trace_off()
    cg.independentFunctionList = [x]
    cg.dependentFunctionList = [y]
    hessp = cg.hess_vec

    # Search for the maximum likelihood parameter values.
    # The direct hessp evaluation turns out to be slower, for some reason,
    # than directly calculating the hessian and then multiplying.
    res = scipy.optimize.minimize(
            f, packed_guess, method='trust-ncg', jac=g,
            #hess=h,
            hessp=hessp,
            )
    xopt = res.x

    # unpack the optimal parameters
    p_opt, mu_opt, penalty_opt = unpack_params(xopt, k)

    return p_opt, mu_opt


def unpack_params(params, k):
    """
    Return the unpacked parameters and a normalization penalty.
    
    This is for the numerical likelihood maximization.
    k is the number of contexts

    """
    nprobs = k * 2

    # de-concatenate the packed parameters
    packed_probs = params[:nprobs]
    packed_mu = params[nprobs:nprobs+2]

    # unpack the probabilities and compute a packing penalty
    unnormal_probs = exp(packed_probs.reshape((2, k)))
    denom = unnormal_probs.sum()
    p = unnormal_probs / denom
    penalty = square(log(denom))

    # unpack mu
    mu = expit(packed_mu)

    return p, mu, penalty


def penalized_packed_neg_ll(k, data, mask, packed_params):

    # Check the input format.
    assert_equal(data.shape, (3, k))
    assert_equal(mask.shape, (2, k))

    # unpack the parameters
    p, mu, penalty = unpack_params(packed_params, k)

    # Return the penalized negative log likelihood
    return neg_ll(p, mu, data, mask) + penalty

