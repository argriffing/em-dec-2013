"""
Speed up EM calculations.

For compilation instructions see
http://docs.cython.org/src/reference/compilation.html

"""

from cython.view cimport array as cvarray
import numpy as np
cimport numpy as np
cimport cython

np.import_array()

__all__ = [
        'EM_masked',
        ]


def _check_input_arrays(
        np.float64_t [:, :] p,
        np.int_t [:, :] mask,
        np.int_t [:, :] data,
        ):
    """
    Check shapes and contents of input arrays.

    """
    cdef int k = p.shape[1]
    cdef int i, j

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

    # Check individual probabilities but to not check the sum.
    for i in range(k):
        for j in range(2):
            if p[j, i] < 0:
                raise ValueError('Expected probabilities to be non-negative')
            if p[j, i] > 1:
                raise ValueError('Expected probabilities to be less than 1')

    # Check the observability masking.
    for i in range(k):
        for j in range(2):
            if data[j, i] and not mask[j, i]:
                raise ValueError(
                        'The data matrix should not include data for '
                        'entries that are marked as unobservable')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def EM_masked(
        double mu01,
        double mu10,
        np.float64_t [:, :] p,
        np.int_t [:, :] mask,
        np.int_t [:, :] data,
        int nsteps,
        int extra,
        ):
    """
    Masked-out entries have missing counts.

    """
    _check_input_arrays(p, mask, data)

    # Initialize some integer constants.
    cdef int k = p.shape[1]
    cdef int STAR_ZERO = 2
    cdef int STAR_ONE = 3
    cdef int OBS_ZERO = 0
    cdef int OBS_ONE = 1
    cdef int OBS_STAR = 2

    # Allocate some intermediate arrays and variables.
    cdef np.float64_t [:, :] q = np.empty((4, k), dtype=float)
    cdef np.float64_t [:, :] aug = np.empty((4, k), dtype=float)
    cdef np.float64_t [:] qstar = np.empty(k, dtype=float)
    cdef np.float64_t [:] aug_sums = np.empty(4, dtype=float)
    cdef double qmissing
    cdef double aug_total
    cdef double aug_excess
    cdef double mu01_num, mu01_den
    cdef double mu10_num, mu10_den

    # Utility variables.
    cdef int i, j
    cdef int iteration_index

    # Summarize the data.
    cdef int ntotal = 0
    for i in range(k):
        for j in range(3):
            ntotal += data[j, i]

    # Do a few EM steps.
    for iteration_index in range(nsteps):

        # Expand the parameters into a (4, k) table.
        # For the rest of the iteration,
        # we will work with the parameters only through this table.
        for i in range(k):
            q[0, i] = p[0, i] * (1 - mu01)
            q[1, i] = p[1, i] * (1 - mu10)
            q[STAR_ZERO, i] = p[0, i] * mu01
            q[STAR_ONE, i] = p[1, i] * mu10

        # Summarize the q table.
        for i in range(k):
            qstar[i] = q[STAR_ZERO, i] + q[STAR_ONE, i]

        # Get the total missing probability.
        qmissing = 0
        for i in range(k):
            for j in range(2):
                if not mask[j, i]:
                    qmissing += q[j, i]

        # Get the first two rows of augmented data.
        aug_excess = (ntotal + extra) * qmissing / (1 - qmissing)
        aug_total = ntotal + aug_excess
        for i in range(k):
            for j in range(2):
                if mask[j, i]:
                    aug[j, i] = data[j, i]
                else:
                    aug[j, i] = (q[j, i] / qmissing) * aug_excess

        # Get the next two rows of augmented data.
        for i in range(k):
            if data[OBS_STAR, i]:
                aug[STAR_ZERO, i] = (
                        q[STAR_ZERO, i] / qstar[i]) * data[OBS_STAR, i]
                aug[STAR_ONE, i] = (
                        q[STAR_ONE, i] / qstar[i]) * data[OBS_STAR, i]
            else:
                aug[STAR_ZERO, i] = 0
                aug[STAR_ONE, i] = 0

        # Summarize the augmented data.
        for j in range(4):
            aug_sums[j] = 0
            for i in range(k):
                aug_sums[j] += aug[j, i]

        # Update the parameter estimates.
        for i in range(k):
            p[0, i] = (aug[0, i] + aug[STAR_ZERO, i]) / aug_total
            p[1, i] = (aug[1, i] + aug[STAR_ONE, i]) / aug_total
        mu01 = aug_sums[STAR_ZERO] / (aug_sums[0] + aug_sums[STAR_ZERO])
        mu10 = aug_sums[STAR_ONE] / (aug_sums[1] + aug_sums[STAR_ONE])

    # Return the mutation parameters.
    return mu01, mu10

