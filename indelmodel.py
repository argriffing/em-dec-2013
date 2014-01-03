"""
Define the indel model.

"""
from numpy.testing import assert_equal
from algopy import log, zeros


OBS_ZERO = 0
OBS_ONE = 1
OBS_STAR = 2


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


def neg_ll(p, mu, data, mask):
    """
    Report the negative log liklelihood.

    """
    k = data.shape[1]

    # compute the conditional distribution
    c = get_c(mu, p)

    # Convert these probabilities to a conditional distribution.
    c[:2] *= mask
    c = c / c.sum()

    # Compute the log of the kernel of the pmf.
    ll = 0
    for j in range(3):
        for i in range(k):
            if data[j, i]:
                ll = data[j, i] * log(c[j, i]) + ll

    return -ll

