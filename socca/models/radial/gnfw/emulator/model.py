"""Reference gNFW Abel integral for training data generation."""

import jax
import jax.numpy as jp
import jax.scipy.special
from quadax import quadgk


@jax.jit
def getzero(alpha, beta, gamma):
    """Compute the exact Abel-projected gNFW integral at x=0.

    Parameters
    ----------
    alpha : float
        Intermediate slope parameter.
    beta : float
        Outer slope parameter.
    gamma : float
        Inner slope parameter.

    Returns
    -------
    float
        Exact value of the projected integral at x=0, computed analytically
        via the Euler beta function.
    """
    a1 = (1.00 - gamma) / alpha
    a2 = (beta - 1.00) / alpha
    log_y = jp.log(2.00 / alpha) + jax.scipy.special.betaln(a1, a2)
    return jp.exp(log_y)


def profile(x, alpha, beta, gamma):
    """Evaluate the 3D gNFW density profile at dimensionless radius x = r/rc.

    Parameters
    ----------
    x : ndarray
        Dimensionless radius (r / rc).
    alpha : float
        Intermediate slope.
    beta : float
        Outer slope.
    gamma : float
        Inner slope.

    Returns
    -------
    ndarray
        Profile values at x.
    """
    factor = 1.00 + x**alpha
    factor = factor ** ((gamma - beta) / alpha)
    return factor / x**gamma


def integrand(x, xc, alpha, beta, gamma):
    """Abel integrand for the line-of-sight projection at projected radius xc."""
    return profile(x, alpha, beta, gamma) * x / jp.sqrt(x**2 - xc**2)


def integral_(xc, alpha, beta, gamma):
    """Compute the Abel-projected gNFW integral at a single projected radius xc."""
    factor = quadgk(
        lambda x: integrand(x, xc, alpha, beta, gamma), (xc, jp.inf)
    )
    return 2.00 * factor[0]


integral = jax.vmap(integral_, in_axes=(0, None, None, None))
