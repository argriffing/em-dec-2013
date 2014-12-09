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

    # Search for the maximum likelihood parameter values.
    # The direct hessp evaluation turns out to be slower, for some reason,
    # than directly calculating the hessian and then multiplying.
    res = scipy.optimize.minimize(
            f, packed_guess, method='trust-ncg', jac=g, hess=h)
    xopt = res.x

    # unpack the optimal parameters
    p_opt, mu_opt, penalty_opt = unpack_params(xopt)

    return p_opt, mu_opt, h(xopt)


def unpack_params(params):
    """
    Return the unpacked parameters and a normalization penalty.

    There are four contexts.
    The middle two contexts are constrained to have the same probability.
    
    This is for the numerical likelihood maximization.
    k is the number of contexts

    """
    k = 2
    nprobs = k * 2

    # de-concatenate the packed parameters
    packed_probs = params[:nprobs]
    packed_mu = params[nprobs:nprobs+2]

    # reshape the transformed probabilities
    reshaped_packed_probs = packed_probs.reshape((2, k))

    # force the 0,1 probability to equal the 1,0 probability
    # penalize if P[0, 1] is different than P[1, 0].
    ancestral_diff = reshaped_packed_probs[0, 1] - reshaped_packed_probs[1, 0]
    ancestral_sum = reshaped_packed_probs[0, 1] + reshaped_packed_probs[1, 0]
    ancestral_mean = 0.5 * ancestral_sum
    reshaped_packed_probs[0, 1] = ancestral_mean
    reshaped_packed_probs[1, 0] = ancestral_mean
    ancestral_penalty = square(ancestral_diff)

    # transform the probabilities and compute a normalization penalty
    unnormal_probs = exp(reshaped_packed_probs)
    denom = unnormal_probs.sum()
    simplex_penalty = square(log(denom))

    # compute the normalized transition probabilities
    p = unnormal_probs / denom

    # compute the total penalty
    penalty = ancestral_penalty + simplex_penalty

    # unpack mu
    mu = expit(packed_mu)

    return p, mu, penalty


def penalized_packed_neg_ll(k, data, mask, packed_params):

    # Check the input format.
    assert_equal(data.shape, (3, k))
    assert_equal(mask.shape, (2, k))

    # unpack the parameters
    p, mu, penalty = unpack_params(packed_params)

    # Return the penalized negative log likelihood
    return neg_ll(p, mu, data, mask) + penalty

