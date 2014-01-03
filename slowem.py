"""
A pure Python module implementing an EM-like iterative algorithm.

This is for a one-off model.

"""
from __future__ import division, print_function

import numpy as np
from numpy.testing import assert_allclose


__all__ = [
        'EM',
        'EM_masked',
        ]

def _xdivy_helper(x, y):
    return x / y if x else 0

xdivy = np.vectorize(_xdivy_helper)



def EM_masked(mu01, mu10, p, mask, data, nsteps, extra=1):
    """
    Masked-out entries have missing counts.

    Parameters
    ----------
    mu01 : float
        mutation probability
    mu10 : float
        mutation probability
    p : 2d float array with shape (2, k)
        probability distribution over latent states
    mask : 2d binary array with shape (2, k)
        observability mask
    data : 2d integer array with shape (3, k)
        array of observed counts
    nsteps : int
        use this many steps of expectation maximization
    extra : int
        extra counts according to negative binomial expectation

    """
    # Check input shape lengths.
    if len(p.shape) != 2:
        raise ValueError('Expected p to be a 2d array')
    if len(mask.shape) != 2:
        raise ValueError('Expected mask to be a 2d array')
    if len(data.shape) != 2:
        raise ValueError('Expected data to be a 2d array')

    # Initialize some integer constants.
    k = p.shape[1]
    STAR_ZERO = 2
    STAR_ONE = 3
    OBS_ZERO = 0
    OBS_ONE = 1
    OBS_STAR = 2

    # Check input shapes.
    if p.shape[0] != 2:
        raise ValueError(
                'Expected the first axis of the probability matrix '
                'to have length 2')
    if mask.shape[0] != 2:
        raise ValueError(
                'Expected the first axis of the observability mask '
                'to have length 2')
    if mask.shape[1] != k:
        raise ValueError(
                'Expected the length of the second axis '
                'of the observability mask to be equal '
                'to the number of contexts')
    if data.shape[0] != 3:
        raise ValueError(
                'Expected the first axis of the data matrix to have length 3')
    if data.shape[1] != k:
        raise ValueError(
                'Expected the length of the second axis '
                'of the data matrix to be equal '
                'to the number of contexts')

    # Check the observability masking.
    if np.any(data[:2] * (1 - mask)):
        raise ValueError(
                'The data matrix should not include data for '
                'entries that are marked as unobservable')

    # Allocate some intermediate arrays and variables.
    q = np.empty((4, k), dtype=float)
    aug = np.empty((4, k), dtype=float)
    qstar = np.empty(k, dtype=float)
    aug_sums = np.empty(4, dtype=float)

    # Summarize the data.
    ntotal = np.sum(data)

    # Do a few EM steps.
    for iteration_index in range(nsteps):

        # Expand the parameters into a (4, k) table.
        # For the rest of the iteration,
        # we will work with the parameters only through this table.
        q[0] = p[0] * (1 - mu01)
        q[1] = p[1] * (1 - mu10)
        q[STAR_ZERO] = p[0] * mu01
        q[STAR_ONE] = p[1] * mu10

        # Summarize the q table.
        qstar = q[STAR_ZERO] + q[STAR_ONE]

        # Get the total missing probability.
        qmissing = np.sum(q[:2] * (1 - mask))

        # Get the first two rows of augmented data.
        aug_excess = (ntotal + extra) * qmissing / (1 - qmissing)
        aug_total = ntotal + aug_excess
        aug[:2] = data[:2]
        if aug_excess and qmissing:
            aug[:2] += (1 - mask) * q[:2] * (aug_excess / qmissing)

        # Get the next two rows of augmented data.
        aug[2:] = xdivy(q[2:] * data[OBS_STAR], qstar)

        # Compute row sums of augmented data.
        aug_sums = aug.sum(axis=1)

        assert_allclose(aug.sum(), aug_total)
        assert_allclose(aug_sums.sum(), aug_total)

        # Update the parameter estimates.
        p[0] = (aug[OBS_ZERO] + aug[STAR_ZERO]) / aug_total
        p[1] = (aug[OBS_ONE] + aug[STAR_ONE]) / aug_total
        mu01 = aug_sums[STAR_ZERO] / (aug_sums[OBS_ZERO] + aug_sums[STAR_ZERO])
        mu10 = aug_sums[STAR_ONE] / (aug_sums[OBS_ONE] + aug_sums[STAR_ONE])

    # Return the mutation parameters.
    return mu01, mu10


def EM(mu01, mu10, p, data, nsteps, extra=1):
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

