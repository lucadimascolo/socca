"""Top-hat profile."""

import jax
import jax.numpy as jp

from .. import config
from .base import Profile


class TopHat(Profile):
    """
    Top-hat surface brightness profile.

    The Top-hat profile describes a uniform surface brightness distribution
    within a cutoff radius: I(r) = 1 for |r| < rc, and I(r) = 0 otherwise.
    This profile is useful for modeling flat-topped emission regions.
    """

    _scale_radius = "rc"
    _scale_amp = "Ic"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.rc = kwargs.get("rc", config.TopHat.rc)
        self.Ic = kwargs.get("Ic", config.TopHat.Ic)

        self.units.update(dict(rc="deg", Ic="image"))
        self.description.update(
            dict(rc="Cutoff distance", Ic="Surface brightness")
        )
        self._initialized = True

    @staticmethod
    @jax.jit
    def profile(r, rc, Ic):
        """
        Top-hat surface brightness distribution.

        Parameters
        ----------
        r : ndarray
            Elliptical radius in degrees.
        rc : float
            Cutoff radius in degrees.

        Returns
        -------
        ndarray
            Surface brightness at radius r.

        Notes
        -----
        The Top-hat profile is defined as:

        I(r) = 1 for |r| < rc, and I(r) = 0 otherwise

        This profile produces a flat, uniform emission within the cutoff
        radius and zero emission outside.
        """
        return jp.where(jp.abs(r) < rc, Ic, 0.0)
