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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rc = kwargs.get("rc", config.Beta.rc)
        self.Ic = kwargs.get("Ic", config.Beta.Ic)
        self.beta = kwargs.get("beta", config.Beta.beta)

        self.units.update(dict(rc="deg", beta="", Ic="image"))

        self.description.update(
            dict(
                rc="Core radius",
                Ic="Central surface brightness",
                beta="Slope parameter",
            )
        )

    @staticmethod
    @jax.jit
    def profile(r, Ic, rc, beta):
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
        beta : float
            Slope parameter (typically 0.4-1.0 for galaxy clusters).

        Returns
        -------
        ndarray
            Surface brightness at radius r.

        Notes
        -----
        The Beta profile is defined as:
        I(r) = Ic * [1 + (r/rc)^2]^(-beta)

        This profile is widely used in X-ray astronomy for modeling hot gas
        in galaxy clusters. The parameter beta controls the slope of the
        outer profile.

        References
        ----------
        Cavaliere, A., & Fusco-Femiano, R. 1976, A&A, 49, 137

        Examples
        --------
        >>> import jax.numpy as jp
        >>> from socca.models import Beta
        >>> r = jp.linspace(0, 0.1, 100)
        >>> I = Beta.profile(r, Ic=100.0, rc=0.01, beta=0.5)
        """
        return Ic * jp.power(1.00 + (r / rc) ** 2, -beta)
