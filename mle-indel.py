"""
A more complicated model.

"""
from __future__ import division, print_function

import argparse
from functools import partial
import sys

import numpy as np
from numpy.testing import assert_equal, assert_allclose
import scipy.optimize
import algopy
from algopy import log, exp, square, zeros
from algopy.special import logit, expit

#from fastem import EM as fast_em

from fastem import EM, EM_masked

OBS_ZERO = 0
OBS_ONE = 1
OBS_STAR = 2

STAR_ZERO = 2
STAR_ONE = 3

INITIAL_CONTEXT = 0
FINAL_CONTEXT = -1

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


def fast_em(mu01, mu10, p, data, nsteps, extra):

    # Define the observability mask.
    mask = np.ones_like(p, dtype=int)
    mask[OBS_ZERO, INITIAL_CONTEXT] = 0
    mask[OBS_ONE, FINAL_CONTEXT] = 0

    print('observability mask:')
    print(mask)
    print()

    # Use the masked EM.
    return EM_masked(mu01, mu10, p, mask, data, nsteps, extra)
    #return EM(mu01, mu10, p, data, nsteps, extra)


def eval_grad(f, theta):
    theta = algopy.UTPM.init_jacobian(theta)
    return algopy.UTPM.extract_jacobian(f(theta))


def eval_hess(f, theta):
    theta = algopy.UTPM.init_hessian(theta)
    return algopy.UTPM.extract_hessian(len(theta), f(theta))


def get_c(mu, p):
    """
    p should have shape (2, k)
    c should have shape (3, k)

    """
    assert_equal(len(mu.shape), 1)
    assert_equal(len(p.shape), 2)
    assert_equal(mu.shape, (2,))
    assert_equal(p.shape[0], 2)

    k = p.shape[1]
    m01, m10 = mu
    c = zeros((3, k), dtype=p)
    c[OBS_ZERO] = p[0] * (1 - m01)
    c[OBS_ONE] = p[1] * (1 - m10)
    c[OBS_STAR] = p[0] * m01 + p[1] * m10
    return c


def slow_em(mu01, mu10, p, data, nsteps, extra=1):
    k = data.shape[1]

    for i in range(nsteps):

        assert_allclose(p.sum(), 1)

        # Expand the parameters into a (4, k) table.
        # For the rest of the iteration,
        # we will work with the parameters only through this table.
        q = np.empty((4, k))
        q[0] = p[0] * (1 - mu01)
        q[1] = p[1] * (1 - mu10)
        q[STAR_ZERO] = p[0] * mu01
        q[STAR_ONE] = p[1] * mu10

        assert_allclose(q.sum(), 1)

        # Compute the expectations.
        # This will essentially impute the missing 000 and 111 counts
        # as well as imputing the refinement of the star category,
        # imputing the latent states.
        ntotal = data.sum()
        qstar = q[STAR_ZERO] + q[STAR_ONE]
        qmissing = q[0, INITIAL_CONTEXT] + q[1, FINAL_CONTEXT]
        aug_excess = (ntotal + extra) * qmissing / (1 - qmissing)
        aug = np.empty((4, k))
        aug[0] = data[0]
        aug[1] = data[1]
        aug[STAR_ZERO] = (q[STAR_ZERO] / qstar) * data[OBS_STAR]
        aug[STAR_ONE] = (q[STAR_ONE] / qstar) * data[OBS_STAR]
        aug[0, INITIAL_CONTEXT] = (
                q[0, INITIAL_CONTEXT] / qmissing) * aug_excess
        aug[1, FINAL_CONTEXT] = (
                q[1, FINAL_CONTEXT] / qmissing) * aug_excess
        aug_total = ntotal + aug_excess

        assert_allclose(aug_total, aug.sum())

        # Update the parameter estimates.
        p[0] = (aug[0] + aug[STAR_ZERO]) / aug_total
        p[1] = (aug[1] + aug[STAR_ONE]) / aug_total
        mu01 = aug[STAR_ZERO].sum() / (aug[0] + aug[STAR_ZERO]).sum()
        mu10 = aug[STAR_ONE].sum() / (aug[1] + aug[STAR_ONE]).sum()

    return mu01, mu10


def main_em(p_guess, mu_guess, data, nsteps, em_function, extra=1):

    # Check that the the number of contexts is agreed upon.
    assert_equal(p_guess.shape[0], 2)
    assert_equal(mu_guess.shape, (2,))
    assert_equal(data.shape[0], 3)
    assert_equal(p_guess.shape[1], data.shape[1])

    # Copy the guesses in case they are modified in place.
    p_guess = p_guess.copy()
    mu_guess = mu_guess.copy()

    # Run the em.
    mu01_guess, mu10_guess = mu_guess
    mu01, mu10 = em_function(
            mu01_guess, mu10_guess, p_guess, data, nsteps, extra)
    p = p_guess

    # Summarize the EM output.
    p_opt = p
    mu_opt = np.array([mu01, mu10])
    nll = neg_ll(p_opt, mu_opt, data)

    # Report estimates.
    print('EM estimated p parameter values:')
    print(p_opt)
    print()
    print('EM estimated mu parameter values:')
    print(mu_opt)
    print()
    print('EM transformed parameters for guessing ancestral state:')
    q = get_parameter_transformation(mu_opt[0], mu_opt[1], p_opt)
    print(q)
    print()
    print('EM estimated neg log likelihood:')
    print(nll)
    print()



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


def neg_ll(p, mu, data):
    """
    Report the negative log liklelihood.

    """
    k = data.shape[1]

    # compute the conditional distribution
    c = get_c(mu, p)

    # Convert these probabilities to a conditional distribution.
    c[0, INITIAL_CONTEXT] = 0
    c[1, FINAL_CONTEXT] = 0
    c = c / c.sum()

    # Compute the log of the kernel of the pmf.
    ll = 0
    for j in range(3):
        for i in range(k):
            if data[j, i]:
                ll = data[j, i] * log(c[j, i]) + ll

    return -ll


def penalized_packed_neg_ll(k, data, packed_params):

    # Check the input format.
    assert_equal(data.shape, (3, k))

    # unpack the parameters
    p, mu, penalty = unpack_params(packed_params, k)

    # Return the penalized negative log likelihood
    return neg_ll(p, mu, data) + penalty


def main_ml(p_guess, mu_guess, data):
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

    # Pack the guess.
    packed_guess = np.concatenate((log(p_guess.flatten()), logit(mu_guess)))

    # Define the objective function, some of its derivatives, and a guess.
    f = partial(penalized_packed_neg_ll, k, data)
    g = partial(eval_grad, f)
    h = partial(eval_hess, f)

    # Search for the maximum likelihood parameter values.
    res = scipy.optimize.minimize(
            f, packed_guess, method='trust-ncg', jac=g, hess=h)
    xopt = res.x
    yopt = f(xopt)

    # unpack the optimal parameters
    p_opt, mu_opt, penalty_opt = unpack_params(xopt, k)
    nll = neg_ll(p_opt, mu_opt, data)

    # Report the max likelihood parameter value estimates.
    print('ML estimated p parameter values:')
    print(p_opt)
    print()
    print('ML estimated mu parameter values:')
    print(mu_opt)
    print()
    print('ML transformed parameters for guessing ancestral state:')
    q = get_parameter_transformation(mu_opt[0], mu_opt[1], p_opt)
    print(q)
    print()
    print('ML estimated neg log likelihood:')
    print(nll)
    print()
    print('ML opt penalty (should be near zero):')
    print(penalty_opt)
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
    cond[0, INITIAL_CONTEXT] = 0
    cond[1, FINAL_CONTEXT] = 0
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
    print(neg_ll(p, mu, n))
    print()

    # Initialize the guess.
    # Use the same guess for ml and em.
    p_guess = np.ones((2, k))
    p_guess = p_guess / p_guess.sum()
    mu_guess = np.array([0.5, 0.5])

    #p_guess = hardcoded_p_2
    #mu_guess = hardcoded_mu_2

    print('neg log likelihood for guess parameter values:')
    print(neg_ll(p_guess, mu_guess, n))
    print()

    # Compute maximum likelihood estimates.
    if args.solver == 'ml':
        main_ml(p_guess, mu_guess, n)
    elif args.solver in ('fast-em', 'slow-em'):
        if args.solver == 'fast-em':
            f = fast_em
        elif args.solver == 'slow-em':
            f = slow_em
        main_em(p_guess, mu_guess, n, args.em_iterations, f,
                extra=args.extra)
    elif args.solver is None:
        main_em(p_guess, mu_guess, n, args.em_iterations, fast_em,
                extra=args.extra)
        main_ml(p_guess, mu_guess, n)
    else:
        raise NotImplementedError(args.solver)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver',
            choices=('ml', 'fast-em', 'slow-em'),
            help='ml is numerical MLE, em is expectation maximization')
    parser.add_argument('--contexts',
            type=int, default=4,
            help='number of contexts (default 4)')
    parser.add_argument('--seed',
            type=int, default=1234,
            help='random number seed for simulation')
    parser.add_argument('--sites',
            type=int, default=1000,
            help='number of observed sites')
    parser.add_argument('--em-iterations',
            type=int, default=1000,
            help='number of EM iterations')
    parser.add_argument('--extra',
            type=int, default=1,
            help='extra count added to the observed total, for debugging')
    main(parser.parse_args())

