"""Gaussian surface brightness profile."""

import jax
import jax.numpy as jp

from .. import config
from .base import Profile


class Gaussian(Profile):
    """
    Gaussian surface brightness profile.

    The Gaussian profile describes a surface brightness distribution that
    follows a Gaussian function of radius: I(r) = Is * exp(-0.5 * (r/rs)^2).
    This profile is equivalent to a Sersic profile with index n=0.5.
    """

    def __init__(self, **kwargs):
        """
        Initialize a Gaussian profile component.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments including:

            rs : float, optional
                Scale radius (standard deviation) in degrees.
            Is : float, optional
                Central surface brightness (same units as image).
            xc, yc : float, optional
                Centroid coordinates (inherited from Profile).
            theta : float, optional
                Position angle in radians (inherited from Profile).
            e : float, optional
                Projected axis ratio (inherited from Profile).
            cbox : float, optional
                Projected boxiness (inherited from Profile).
        """
        super().__init__(**kwargs)
        self.rs = kwargs.get("rs", config.Gaussian.rs)
        self.Is = kwargs.get("Is", config.Gaussian.Is)

        self.units.update(dict(rs="deg", Is="image"))
        self.description.update(
            dict(rs="Scale radius", Is="Central surface brightness")
        )

    @staticmethod
    @jax.jit
    def profile(r, Is, rs):
        """
        Gaussian surface brightness distribution.

        Parameters
        ----------
        r : ndarray
            Elliptical radius in degrees.
        Is : float
            Central surface brightness (same units as image).
        rs : float
            Scale radius (standard deviation) in degrees.

        Returns
        -------
        ndarray
            Surface brightness at radius r.

        Notes
        -----
        The Gaussian profile is defined as:

        I(r) = Is * exp(-0.5 * (r / rs)^2)

        This is equivalent to a Sersic profile with n=0.5. The scale radius
        rs corresponds to the standard deviation of the Gaussian, and the
        half-width at half-maximum (HWHM) is approximately 1.177 * rs.
        """
        return Is * jp.exp(-0.50 * (r / rs) ** 2)
