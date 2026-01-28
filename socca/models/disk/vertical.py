"""Vertical profiles for 3D disk galaxies."""

from abc import abstractmethod

import jax
import jax.numpy as jp

from .. import config
from ..base import Component


class Height(Component):
    """
    Base class for vertical density profiles in 3D disk models.

    Defines the vertical (z-direction) structure of a disk galaxy. Used in
    combination with radial profiles to create 3D disk models via the Disk class.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments including:

        inc : float, optional
            Inclination angle in radians (0 = face-on, pi/2 = edge-on).
        losdepth : float, optional
            Half-extent of line-of-sight integration in degrees.
        losbins : int, optional
            Number of integration points along the line of sight.
        positive : bool, optional
            Whether to enforce positivity constraint.

    Attributes
    ----------
    inc : float
        Inclination angle (rad).
    losdepth : float
        Half line-of-sight depth for numerical integration (deg).
    losbins : int
        Number of points for line-of-sight integration (hyperparameter).

    Notes
    -----
    - Subclasses must implement the abstract profile(z) method
    - The profile should return the density as a function of height z
    - losdepth and losbins control the numerical integration accuracy
    - Larger losdepth needed for extended vertical distributions
    - More losbins increase accuracy but decrease speed
    """

    def __init__(self, **kwargs):
        """
        Initialize a vertical profile component.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments including inc, losdepth, losbins, positive.
        """
        super().__init__(**kwargs)

        self.inc = kwargs.get("inc", config.Height.inc)

        self.losdepth = kwargs.get("losdepth", config.Height.losdepth)
        self.losbins = kwargs.get("losbins", config.Height.losbins)
        self.units = dict(losdepth="deg", losbins="", inc="rad")

        self.hyper = ["losdepth", "losbins"]

        self.description = dict(
            losdepth="Half line-of-sigt extent for integration",
            losbins="Number of points for line-of-sight integration",
            inc="Inclination angle (0=face-on)",
        )

    @abstractmethod
    def profile(z):
        """
        Evaluate vertical density profile at height z.

        Parameters
        ----------
        z : ndarray
            Height above/below disk midplane in degrees.

        Returns
        -------
        ndarray
            Density at height z (normalized).

        Notes
        -----
        This is an abstract method that must be implemented by subclasses.
        """
        pass


# Hyperbolic cosine
# --------------------------------------------------------
class HyperSecantHeight(Height):
    """
    Hyperbolic secant vertical density profile.

    Models the vertical structure of disk galaxies using a hyperbolic secant
    (sech) function raised to a power. Commonly used for thick disks.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments including:

        zs : float, optional
            Scale height in degrees.
        alpha : float, optional
            Exponent applied to sech function (typically 1 or 2).
        inc : float, optional
            Inclination angle (rad).
        losdepth, losbins : float, int, optional
            Integration parameters.

    Attributes
    ----------
    zs : float
        Scale height (deg).
    alpha : float
        Exponent parameter.

    Notes
    -----
    The profile is defined as:

    rho(z) = sech(abs(z)/zs)^alpha

    Common cases:

    - alpha = 1: Simple sech profile
    - alpha = 2: sech^2 profile (isothermal disk)

    The sech profile is smoother than exponential near the midplane and
    has more extended wings, making it suitable for modeling thick disks
    and stellar halos.
    """

    def __init__(self, **kwargs):
        """
        Initialize a hyperbolic secant height profile.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments including zs, alpha, inc, losdepth, losbins.
        """
        super().__init__(**kwargs)
        self.zs = kwargs.get("zs", config.HyperSecantHeight.zs)
        self.alpha = kwargs.get("alpha", config.HyperSecantHeight.alpha)

        self.units.update(dict(zs="deg", alpha=""))
        self.description.update(
            dict(zs="Scale height", alpha="Exponent to the hyperbolic secant")
        )
        self._initialized = True

    @staticmethod
    @jax.jit
    def profile(z, zs, alpha):
        """
        Evaluate hyperbolic secant profile at height z.

        Parameters
        ----------
        z : ndarray
            Height above/below disk midplane (deg).
        zs : float
            Scale height (deg).
        alpha : float
            Exponent parameter.

        Returns
        -------
        ndarray
            Density at height z: sech(abs(z)/zs)^alpha.

        Notes
        -----
        Uses absolute value of z to ensure symmetry about the midplane.
        """
        factor = jp.cosh(jp.abs(z) / zs)
        return 1.00 / factor**alpha


# Exponential height
# --------------------------------------------------------
class ExponentialHeight(Height):
    """
    Exponential vertical density profile.

    Models the vertical structure of thin disk galaxies using an exponential
    function. This is the simplest and most commonly used vertical profile.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments including:

        zs : float, optional
            Scale height in degrees.
        inc : float, optional
            Inclination angle (rad).
        losdepth, losbins : float, int, optional
            Integration parameters.

    Attributes
    ----------
    zs : float
        Scale height (deg).

    Notes
    -----
    The profile is defined as:

    rho(z) = exp(-abs(z)/zs)

    This simple exponential profile is appropriate for thin stellar disks
    and is the vertical analog of the exponential radial profile. The scale
    height zs is typically much smaller than the radial scale length.

    The exponential profile has a sharp peak at the midplane and falls off
    more rapidly than sech profiles, making it suitable for thin disks.
    """

    def __init__(self, **kwargs):
        """
        Initialize an exponential height profile.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments including zs, inc, losdepth, losbins.
        """
        super().__init__(**kwargs)
        self.zs = kwargs.get("zs", config.ExponentialHeight.zs)
        self.units.update(dict(zs="deg"))
        self.description.update(dict(zs="Scale height"))
        self._initialized = True

    @staticmethod
    @jax.jit
    def profile(z, zs):
        """
        Evaluate exponential profile at height z.

        Parameters
        ----------
        z : ndarray
            Height above/below disk midplane (deg).
        zs : float
            Scale height (deg).

        Returns
        -------
        ndarray
            Density at height z: exp(-abs(z)/zs).

        Notes
        -----
        Uses absolute value of z to ensure symmetry about the midplane.
        """
        return jp.exp(-jp.abs(z) / zs)
