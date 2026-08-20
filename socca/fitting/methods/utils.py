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


# Weighted quantiles
# --------------------------------------------------------
def weighted_quantile(a, quantiles, weights, axis=0):
    """
    NumPy>=2.0-independent weighted quantile.

    Equivalent to ``np.quantile(a, quantiles, axis=axis,
    method="inverted_cdf", weights=weights)``. NumPy only gained a
    `weights` keyword for `quantile`/`percentile` in
    2.0, and 2.0 is also the last release to still support Python 3.9 --
    since socca supports Python 3.9+ without pinning a NumPy floor, the
    installed NumPy on 3.9 can be older than 2.0 and lack `weights`
    entirely. This reimplements NumPy's weighted `inverted_cdf` algorithm
    (sort along `axis`, build the normalized weighted CDF, then
    `searchsorted` each quantile into it) so socca doesn't depend on
    NumPy>=2.0.

    Parameters
    ----------
    a : array_like
        Values to compute quantiles of.
    quantiles : array_like
        Quantiles to compute, in ``[0, 1]``.
    weights : array_like
        1D per-sample weights, matching the length of `a` along `axis`.
    axis : int, optional
        Axis along which `a` is sorted/reduced. Default is 0.

    Returns
    -------
    ndarray
        Shape ``quantiles.shape + a.shape`` with `axis` removed. If
        `quantiles` is a scalar, that leading dimension is dropped, just
        as with `np.quantile`.
    """
    a = np.asarray(a, dtype=float)
    weights = np.asarray(weights, dtype=float)
    quantiles_in = np.asarray(quantiles, dtype=float)
    quantiles = np.atleast_1d(quantiles_in)

    if axis != 0:
        a = np.moveaxis(a, axis, 0)

    order = np.argsort(a, axis=0)
    a_sorted = np.take_along_axis(a, order, axis=0)

    w_broadcast = np.broadcast_to(
        weights.reshape((-1,) + (1,) * (a.ndim - 1)), a.shape
    )
    w_sorted = np.take_along_axis(w_broadcast, order, axis=0)

    cdf = np.cumsum(w_sorted, axis=0)
    cdf = cdf / cdf[-1, ...]
    if np.any(cdf[0, ...] == 0):
        cdf = cdf.copy()
        cdf[cdf == 0] = -1

    n = a_sorted.shape[0]
    flat_a = a_sorted.reshape(n, -1)
    flat_cdf = cdf.reshape(n, -1)
    flat_result = np.empty((quantiles.shape[0], flat_a.shape[1]))
    for k in range(flat_a.shape[1]):
        idx = np.searchsorted(flat_cdf[:, k], quantiles, side="left")
        idx = np.minimum(idx, n - 1)
        flat_result[:, k] = flat_a[idx, k]

    result = flat_result.reshape(quantiles.shape + a_sorted.shape[1:])
    if quantiles_in.ndim == 0:
        result = result[0]
    return result


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
    q = weighted_quantile(recentered, quantiles, weights)
    return (np.asarray(q) + mean) % period
