"""Numerical gNFW profile computation for training data generation.

This module provides the reference implementation of the gNFW
Abel integral using adaptive quadrature. These functions are used
to generate training data for the neural network emulator.
"""

from functools import partial

import jax
import jax.numpy as jp
from quadax import quadgk

from .config import Model


def gnfw_radial(x, alpha, beta, gamma):
    r"""Compute the 3D radial gNFW density profile.

    Parameters
    ----------
    x : float or ndarray
        Dimensionless radius r/rc.
    alpha : float
        Intermediate slope parameter (sharpness of transition).
    beta : float
        Outer slope parameter.
    gamma : float
        Inner slope parameter (central cusp).

    Returns
    -------
    float or ndarray
        Normalized density rho/rho0 at radius x.

    Notes
    -----
    The 3D density profile is:

    .. math::

        \\rho(x) = \\frac{1}{x^\\gamma (1 + x^\\alpha)^{(\\beta-\\gamma)/\\alpha}}

    where x = r/rc is the dimensionless radius.
    """
    factor = 1.00 + x**alpha
    factor = factor ** ((beta - gamma) / alpha)
    return 1.00 / factor / x**gamma


def gnfw_integrand(x, xc, alpha, beta, gamma):
    r"""Compute the Abel integrand for the gNFW profile.

    Parameters
    ----------
    x : float
        Integration variable (3D radius).
    xc : float
        Projected radius at which to evaluate the integral.
    alpha : float
        Intermediate slope parameter.
    beta : float
        Outer slope parameter.
    gamma : float
        Inner slope parameter.

    Returns
    -------
    float
        Value of the integrand at x.

    Notes
    -----
    The Abel integrand is:

    .. math::

        \\frac{x}{\\sqrt{x^2 - x_c^2}} \\rho(x)

    where rho(x) is the 3D gNFW density profile.
    """
    factor = x / jp.sqrt(x**2 - xc**2)
    return factor * gnfw_radial(x, alpha, beta, gamma)


@partial(jax.jit, static_argnames=("eps",))
def gnfw_integral(xc, alpha, beta, gamma, eps=Model.eps):
    """Compute the Abel integral for a single projected radius.

    Parameters
    ----------
    xc : float
        Projected dimensionless radius.
    alpha : float
        Intermediate slope parameter.
    beta : float
        Outer slope parameter.
    gamma : float
        Inner slope parameter.
    eps : float, optional
        Tolerance for adaptive quadrature. Default is 1e-8.

    Returns
    -------
    float
        Value of the Abel integral at xc.

    Notes
    -----
    Computes the integral from xc to infinity using adaptive
    Gauss-Kronrod quadrature (quadgk).
    """
    return quadgk(
        gnfw_integrand,
        [xc, jp.inf],
        args=(xc, alpha, beta, gamma),
        epsabs=eps,
        epsrel=eps,
    )[0]


@jax.jit
def gnfw(xc, alpha, beta, gamma):
    """Compute the gNFW Abel integral for an array of projected radii.

    Parameters
    ----------
    xc : ndarray
        Array of projected dimensionless radii.
    alpha : float
        Intermediate slope parameter.
    beta : float
        Outer slope parameter.
    gamma : float
        Inner slope parameter.

    Returns
    -------
    ndarray
        Abel integral values at each projected radius.

    Notes
    -----
    This function vectorizes `gnfw_integral` over the array of
    projected radii using JAX vmap.
    """
    _batch = jax.vmap(partial(gnfw_integral), in_axes=(0, None, None, None))
    return _batch(xc, alpha, beta, gamma)
