"""Power law radial profile."""

import jax
import jax.numpy as jp

from .. import config
from .base import Profile


class Power(Profile):
    """
    Power law profile for surface brightness modeling.

    The Power profile describes a simple power-law distribution of the form
    I(r) = Ic * (r/rc)^alpha.
    """

    _scale_radius = "rc"
    _scale_amp = "Ic"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rc = kwargs.get("rc", config.Power.rc)
        self.Ic = kwargs.get("Ic", config.Power.Ic)
        self.alpha = kwargs.get("alpha", config.Power.alpha)

        self.units.update(dict(rc="deg", alpha="", Ic="image"))

        self.description.update(
            dict(
                rc="Scale radius",
                Ic="Characteristic surface brightness",
                alpha="Power law slope",
            )
        )
        self._initialized = True

    @staticmethod
    @jax.jit
    def profile(r, Ic, rc, alpha):
        """
        Power law surface brightness distribution.

        Parameters
        ----------
        r : ndarray
            Elliptical radius in degrees.
        Ic : float
            Characteristic surface brightness (same units as image).
        rc : float
            Scale radius in degrees.
        alpha : float
            Power law slope parameter.

        Returns
        -------
        ndarray
            Surface brightness at radius r.

        Notes
        -----
        The Power law profile is defined as:
        I(r) = Ic * (r/rc)^(-alpha)

        For positive alpha values, the profile decreases with radius.

        Examples
        --------
        >>> import jax.numpy as jp
        >>> from socca.models import Power
        >>> r = jp.linspace(0.001, 0.1, 100)
        >>> I = Power.profile(r, Ic=100.0, rc=0.01, alpha=2.0)
        """
        return Ic * jp.power(r / rc, -alpha)
