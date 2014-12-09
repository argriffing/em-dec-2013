"""
Maximum likelihood estimation for a specific model.

The incorrect "expectation maximization"
code has been removed from this module.

"""
from __future__ import division, print_function, absolute_import

import argparse
import sys
import time

import numpy as np
from numpy.testing import assert_equal, assert_allclose
import scipy.linalg

from indelmodel import get_c, neg_ll
import numericml


OBS_ZERO = 0
OBS_ONE = 1
OBS_STAR = 2

hardcoded_p_1 = np.array([
    [ 0.12826218, 0.24608812, 0.14997917, 0.10612475],
    [ 0.22184387, 0.03004811, 0.05858876, 0.05906504],
    ])

hardcoded_mu_1 = np.array([0.11174128, 0.24989901])


hardcoded_p_2 = np.array([
    [9.99977862e-01, 3.87793408e-06, 1.07720391e-06, 2.36984860e-06],
    [5.56555354e-06, 2.15440781e-06, 4.52425641e-06, 2.56871700e-06],
    ])

hardcoded_mu_2 = np.array([8.97688984e-07, 4.19354839e-01])


def get_observability_mask(k):
    # The mask has three rows corresponding to OBS_ZERO, OBS_ONE
    # The number of columns is k, which is equal to the number of contexts.
    mask = np.ones((2, k))
    mask[OBS_ZERO, 0] = 0
    #mask[OBS_ONE, -1] = 0
    return mask


def main_ml(p_guess, mu_guess, data, mask):
    """
    Use algopy and scipy minimize with trust-ncg for likelihood maximization.

    Use a trick where we parameterize using logs of all probabilities,
    and then convert these to probabilities by using their normalized
    distribution of exps.
    Then compute the neg log likelihood using these probabilities
    and add a penalty term according to how much we had to normalize.
    The penalty could be like the square of log of sum of exps.

    """
    k = data.shape[1]

    # Check that the the number of contexts is agreed upon.
    assert_equal(p_guess.shape[0], 2)
    assert_equal(mu_guess.shape, (2,))
    assert_equal(data.shape[0], 3)
    assert_equal(p_guess.shape[1], data.shape[1])

    # Copy the guesses in case they are modified in place.
    p_guess = p_guess.copy()
    mu_guess = mu_guess.copy()

    # Infer the parameter values using a numerical optimization.
    p_opt, mu_opt, hess_opt = numericml.infer_parameter_values(
            p_guess, mu_guess, data, mask)

    # Compute functions of the estimated parameter values.
    q = get_parameter_transformation(mu_opt[0], mu_opt[1], p_opt)
    nll = neg_ll(p_opt, mu_opt, data, mask)

    # Report the max likelihood parameter value estimates.
    print('ML estimated p parameter values:')
    print(p_opt)
    print()
    print('ML estimated mu parameter values:')
    print(mu_opt)
    print()
    print('ML transformed parameters for guessing ancestral state:')
    print(q)
    print()
    print('ML estimated neg log likelihood:')
    print(nll)
    print()
    print('Hessian at MLE:')
    print(hess_opt)
    print()
    print('Eigenvalues of Hessian:')
    print(scipy.linalg.eigvalsh(hess_opt))
    print('Inverse of Hessian information at MLE:')
    print(np.linalg.inv(hess_opt))
    print()


def get_parameter_transformation(mu01, mu10, p):
    q = np.zeros_like(p)
    q[0] = p[0] * mu01
    q[1] = p[1] * mu10
    qstar = q[0] + q[1]
    return q / qstar


def main(args):

    print('command line:')
    print(' '.join(sys.argv))
    print()

    # Use the same random seed for everything.
    if args.seed > 0:
        np.random.seed(args.seed)
    else:
        np.random.seed(1234)
    nsites = args.sites
    k = args.contexts

    # Choose parameter values for simulation.
    if args.seed == -1:
        p = hardcoded_p_1
        mu = hardcoded_mu_1
        print('using hardcoded parameter values')
        print()
    elif args.seed == -2:
        p = hardcoded_p_2
        mu = hardcoded_mu_2
        print('using hardcoded parameter values')
        print()
    else:
        p = np.exp(np.random.randn(2, k))
        p /= p.sum()
        mu = np.random.rand(2)
        print('using simulated parameter values')
        print()

    # Get the observability mask.
    mask = get_observability_mask(k)
    print('observability mask:')
    print(mask)
    print()

    # Report the parameter values used for simulation.
    print('Actual p parameter values from which data are sampled:')
    print(p)
    print()
    print('Actual mu parameter values from which data are sampled:')
    print(mu)
    print()
    print('Transformed parameters for guessing ancestral state:')
    q = get_parameter_transformation(mu[0], mu[1], p)
    print(q)
    print()

    # Get the full process multinomial probabilities.
    c = get_c(mu, p)

    # Convert these probabilities to a conditional distribution.
    cond = c.copy()
    cond[:2] *= mask
    cond /= cond.sum()

    # Sample from the conditional multinomial distribution.
    n = np.random.multinomial(nsites, cond.flatten()).reshape(cond.shape)

    assert_equal(n.shape, (3, k))

    # Report the sampled data.
    print('sampled data conditional on zero counts for unobservable data')
    print('n:')
    print(n)
    print()

    # Report the neg log likelihood for the actual parameter values.
    print('neg log likelihood for actual parameter values:')
    print(neg_ll(p, mu, n, mask))
    print()

    # Initialize the guess.
    # Use the same guess for ml and em.
    p_guess = np.ones((2, k))
    p_guess = p_guess / p_guess.sum()
    mu_guess = np.array([0.5, 0.5])

    #p_guess = hardcoded_p_2
    #mu_guess = hardcoded_mu_2

    print('neg log likelihood for guess parameter values:')
    print(neg_ll(p_guess, mu_guess, n, mask))
    print()

    # Compute maximum likelihood estimates.
    print('ml...')
    tm_start = time.time()
    main_ml(p_guess, mu_guess, n, mask)
    tm_end = time.time()
    print(tm_end - tm_start, 'seconds')
    print()


if __name__ == '__main__':
    np.set_printoptions(linewidth=200)
    parser = argparse.ArgumentParser()
    parser.add_argument('--contexts',
            type=int, default=4,
            help='number of contexts (default 4)')
    parser.add_argument('--seed',
            type=int, default=1234,
            help='random number seed for simulation')
    parser.add_argument('--sites',
            type=int, default=1000,
            help='number of observed sites')
    main(parser.parse_args())

