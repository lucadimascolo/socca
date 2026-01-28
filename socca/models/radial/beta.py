"""Beta radial profile."""

import jax
import jax.numpy as jp

from .. import config
from .base import Profile


class Beta(Profile):
    """
    Beta profile for modeling galaxy clusters and elliptical galaxies.

    The Beta profile describes a power-law density distribution commonly used
    for X-ray and radio observations of galaxy clusters. It has the form
    I(r) = Ic * (1 + (r/rc)^2)^(-beta).
    """

    _scale_radius = "rc"
    _scale_amp = "Ic"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rc = kwargs.get("rc", config.Beta.rc)
        self.Ic = kwargs.get("Ic", config.Beta.Ic)
        self.alpha = kwargs.get("alpha", config.Beta.alpha)
        self.beta = kwargs.get("beta", config.Beta.beta)

        self.units.update(dict(rc="deg", alpha="", beta="", Ic="image"))

        self.description.update(
            dict(
                rc="Core radius",
                Ic="Central surface brightness",
                alpha="Radial exponent",
                beta="Slope parameter",
            )
        )
        self._initialized = True

    @staticmethod
    @jax.jit
    def profile(r, Ic, rc, alpha, beta):
        """
        Beta profile surface brightness distribution.

        The Beta profile describes a power-law density distribution commonly
        used for modeling galaxy clusters and elliptical galaxies.

        Parameters
        ----------
        r : ndarray
            Elliptical radius in degrees.
        Ic : float
            Central surface brightness (same units as image).
        rc : float
            Core radius in degrees.
        alpha : float
            Radial exponent parameter (default 2.0 for standard Beta profile).
        beta : float
            Slope parameter (typically 0.4-1.0 for galaxy clusters).

        Returns
        -------
        ndarray
            Surface brightness at radius r.

        Notes
        -----
        The generalized Beta profile is defined as:
        I(r) = Ic * [1 + (r/rc)^alpha]^(-beta)

        With alpha=2, this reduces to the standard Beta profile commonly
        used in X-ray astronomy for modeling hot gas in galaxy clusters.

        References
        ----------
        Cavaliere, A., & Fusco-Femiano, R. 1976, A&A, 49, 137

        Examples
        --------
        >>> import jax.numpy as jp
        >>> from socca.models import Beta
        >>> r = jp.linspace(0, 0.1, 100)
        >>> I = Beta.profile(r, Ic=100.0, rc=0.01, alpha=2.0, beta=0.5)
        """
        return Ic * jp.power(1.00 + (r / rc) ** alpha, -beta)
