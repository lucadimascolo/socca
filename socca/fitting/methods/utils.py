"""Utility functions for sampler output processing."""

import scipy.special
import numpy as np


# Support functions
# ========================================================
# Compute importance weights for nested sampling
# --------------------------------------------------------
def get_imp_weights(logw, logz=None):
    r"""
    Compute importance weights from log-weights and log-evidence.

    Converts log-weights to normalized importance weights using the
    log-evidence for numerical stability. The weights are normalized
    such that they sum to 1.0.

    Parameters
    ----------
    logw : array_like
        Log-weights from nested sampling.
    logz : float or array_like, optional
        Log-evidence value(s). If None, uses the maximum log-weight.
        If not None and not iterable, converts to a single-element list.
        Default is None.

    Returns
    -------
    weights : ndarray
        Normalized importance weights in linear space.

    Notes
    -----
    The importance weights are computed as:

    .. math::
        w_i = \exp(\log w_i - \log Z - \log\sum_j \exp(\log w_j - \log Z))

    where :math:`\log Z` is the log-evidence (logz[-1]).
    """
    if logz is None:
        logz = [logw.max()]
    if not hasattr(logz, "__len__"):
        logz = [logz]

    wt = logw - logz[-1]
    wt = wt - scipy.special.logsumexp(wt)
    return np.exp(wt)
