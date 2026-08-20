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
        Log-weights from importance-weighted sampling.
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
        w_i = \exp[(\log w_i - \log Z) - \log\sum_j \exp(\log w_j - \log Z)]

    where :math:`\log Z` is the log-evidence (logz[-1]).
    """
    if logz is None:
        logz = [logw.max()]
    if not hasattr(logz, "__len__"):
        logz = [logz]

    wt = logw - logz[-1]
    wt = wt - scipy.special.logsumexp(wt)
    return np.exp(wt)


# Circular (periodic) statistics
# --------------------------------------------------------
def circular_recenter(samples, weights, period):
    """
    Shift and wrap samples so their circular mean sits at the range center.

    Plain quantile/summary statistics are wrong for a periodic quantity
    (e.g. a position angle) whenever the posterior straddles the wrap
    boundary -- the samples then look like two separate clusters near the
    two edges of the range, rather than one tight cluster. Recentering on
    the circular mean removes the wrap discontinuity, so ordinary
    statistics become valid again on the returned array.

    Parameters
    ----------
    samples : array_like
        Samples of a periodic quantity.
    weights : array_like
        Per-sample weights (e.g. nested-sampling importance weights).
    period : float
        Period of the quantity (e.g. ``jp.pi`` for a position angle
        symmetric under ``theta -> theta + pi``).

    Returns
    -------
    recentered : ndarray
        ``samples``, shifted by the circular mean and wrapped into
        ``[-period/2, period/2)``.
    mean : float
        The circular mean of ``samples``, in the original ``[0, period)``
        range.
    """
    phi = 2.00 * np.pi * np.asarray(samples) / period
    cosbar = np.average(np.cos(phi), weights=weights)
    sinbar = np.average(np.sin(phi), weights=weights)
    mean = (np.arctan2(sinbar, cosbar) * period / (2.00 * np.pi)) % period
    recentered = (samples - mean + period / 2.00) % period - period / 2.00
    return recentered, mean


def circular_quantile(samples, weights, period, quantiles):
    """
    Weighted quantiles of a periodic quantity, correct across its wrap boundary.

    Computed by recentering on the circular mean (see `circular_recenter`),
    taking ordinary weighted quantiles of the recentered (non-wrapping)
    samples, then shifting back and wrapping into ``[0, period)``.

    Parameters
    ----------
    samples : array_like
        Samples of a periodic quantity.
    weights : array_like
        Per-sample weights (e.g. nested-sampling importance weights).
    period : float
        Period of the quantity (e.g. ``jp.pi`` for a position angle).
    quantiles : array_like
        Quantiles to compute, in ``[0, 1]``.

    Returns
    -------
    ndarray
        The requested quantiles, each wrapped into ``[0, period)``.

    Notes
    -----
    Because each returned value is wrapped independently into
    ``[0, period)``, a plain difference between two returned quantiles
    (e.g. for an upper/lower error bar) can come out negative when the
    interval straddles the wrap boundary. Callers needing such a
    difference should wrap it too:
    ``(qhi - qlo + period / 2.00) % period - period / 2.00``.
    """
    recentered, mean = circular_recenter(samples, weights, period)
    q = np.quantile(
        recentered, quantiles, method="inverted_cdf", weights=weights
    )
    return (np.asarray(q) + mean) % period
