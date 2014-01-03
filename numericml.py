"""
Numerical maximization of expected log likelihood.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import assert_equal

from algopy import log, exp, square, zeros
from algopy.special import logit, expit

from indelmodel import neg_ll


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

