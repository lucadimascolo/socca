"""Sersic profile for modeling elliptical galaxies and bulges."""

import jax
import jax.numpy as jp
import numpy as np
from scipy.special import gammaincinv

from .. import config
from .base import Profile


# Precompute Sersic bn values for interpolation
n_ = np.linspace(0.25, 10.00, 1000)
b_ = gammaincinv(2.00 * n_, 0.5)

n_ = jp.array(n_)
b_ = jp.array(b_)


class Sersic(Profile):
    """
    Sersic profile for modeling elliptical galaxies and bulges.

    The Sersic profile is a generalization of de Vaucouleurs' law that describes
    the light distribution in elliptical galaxies and galactic bulges. The profile
    shape is controlled by the Sersic index (ns), with ns=1 corresponding to an
    exponential disk and ns=4 to a de Vaucouleurs profile.
    """

    _scale_radius = "re"
    _scale_amp = "Ie"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.re = kwargs.get("re", config.Sersic.re)
        self.Ie = kwargs.get("Ie", config.Sersic.Ie)
        self.ns = kwargs.get("ns", config.Sersic.ns)

        self.units.update(dict(re="deg", Ie="image", ns=""))

        self.description.update(
            dict(
                re="Effective radius",
                Ie="Surface brightness at re",
                ns="Sersic index",
            )
        )
        self._initialized = True

    @staticmethod
    @jax.jit
    def profile(r, Ie, re, ns):
        """
        Sersic profile surface brightness distribution.

        The Sersic profile is a generalization of de Vaucouleurs' law and
        describes the light distribution in elliptical galaxies and bulges.

        Parameters
        ----------
        r : ndarray
            Elliptical radius in degrees.
        Ie : float
            Surface brightness at the effective radius (same units as image).
        re : float
            Effective (half-light) radius in degrees.
        ns : float
            Sersic index (concentration parameter). Typical values:

            - ns = 0.5-1: disk-like profiles
            - ns = 4: de Vaucouleurs profile (elliptical galaxies)
            - ns > 4: highly concentrated profiles

        Returns
        -------
        ndarray
            Surface brightness at radius r.

        Notes
        -----
        The Sersic profile is defined as:
        I(r) = Ie * exp(-bn * [(r/re)^(1/ns) - 1])

        where bn is chosen such that re encloses half the total light.
        The parameter bn is approximated numerically and interpolated.

        Common special cases:

        - ns = 1: Exponential profile
        - ns = 4: de Vaucouleurs profile (elliptical galaxies)

        The valid range for ns is approximately 0.25 to 10.

        References
        ----------
        Sersic, J. L. 1968, Atlas de Galaxias Australes
        Ciotti, L., & Bertin, G. 1999, A&A, 352, 447

        Examples
        --------
        >>> import jax.numpy as jp
        >>> from socca.models import Sersic
        >>> r = jp.linspace(0, 0.1, 100)
        >>> # de Vaucouleurs profile for elliptical galaxy
        >>> I = Sersic.profile(r, Ie=50.0, re=0.02, ns=4.0)
        """
        bn = jp.interp(ns, n_, b_)
        se = jp.power(r / re, 1.00 / ns) - 1.00
        return Ie * jp.exp(-bn * se)
