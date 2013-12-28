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
        'EM',
        ]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def EM(
        double mu01,
        double mu10,
        np.float64_t [:, :] p,
        np.int_t [:, :] data,
        int nsteps,
        int extra,
        ):
    """
    mu and p are parameter guesses.

    The extra value should be 1 except for debugging.

    """

    # Initialize some integer constants.
    cdef int k = p.shape[1]
    cdef int STAR_ZERO = 2
    cdef int STAR_ONE = 3
    cdef int OBS_ZERO = 0
    cdef int OBS_ONE = 1
    cdef int OBS_STAR = 2
    cdef int INITIAL_CONTEXT = 0
    cdef int FINAL_CONTEXT = k-1

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
        qmissing = q[0, INITIAL_CONTEXT] + q[1, FINAL_CONTEXT]

        # Compute the expectations.
        # This will essentially impute the missing 000 and 111 counts
        # as well as imputing the refinement of the star category,
        # imputing the latent states.
        aug_excess = (ntotal + extra) * qmissing / (1 - qmissing)
        for i in range(k):
            aug[0, i] = data[0, i]
            aug[1, i] = data[1, i]
            if data[OBS_STAR, i]:
                aug[STAR_ZERO, i] = (
                        q[STAR_ZERO, i] / qstar[i]) * data[OBS_STAR, i]
                aug[STAR_ONE, i] = (
                        q[STAR_ONE, i] / qstar[i]) * data[OBS_STAR, i]
            else:
                aug[STAR_ZERO, i] = 0
                aug[STAR_ONE, i] = 0
        aug[0, INITIAL_CONTEXT] = (
                q[0, INITIAL_CONTEXT] / qmissing) * aug_excess
        aug[1, FINAL_CONTEXT] = (
                q[1, FINAL_CONTEXT] / qmissing) * aug_excess
        aug_total = ntotal + aug_excess
        
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

