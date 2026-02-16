"""Generalized Navarro-Frenk-White profile."""

import warnings

import jax
import jax.numpy as jp
import numpyro.distributions
from quadax import quadgk

from .. import config
from .base import Profile


class gNFW(Profile):
    """
    Generalized Navarro-Frenk-White (gNFW) profile.

    The gNFW profile is a flexible three-parameter model commonly used to describe
    the surface brightness distribution of galaxy clusters.
    It generalizes the NFW profile with adjustable inner (gamma), intermediate
    (alpha), and outer (beta) slopes.
    """

    _scale_radius = "rc"
    _scale_amp = "Ic"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rc = kwargs.get("rc", config.gNFW.rc)
        self.Ic = kwargs.get("Ic", config.gNFW.Ic)
        self.alpha = kwargs.get("alpha", config.gNFW.alpha)
        self.beta = kwargs.get("beta", config.gNFW.beta)
        self.gamma = kwargs.get("gamma", config.gNFW.gamma)

        self.rz = kwargs.get("rz", jp.logspace(-7, 2, 1000))
        self.eps = kwargs.get("eps", 1.00e-08)

        self.units.update(
            dict(rc="deg", alpha="", beta="", gamma="", Ic="image")
        )

        self.description.update(
            dict(
                rc="Scale radius",
                Ic="Characteristic surface brightness",
                alpha="Intermediate slope",
                beta="Outer slope",
                gamma="Inner slope",
            )
        )

        def _profile(r, Ic, rc, alpha, beta, gamma):
            return gNFW._profile(
                r, Ic, rc, alpha, beta, gamma, self.rz, self.eps
            )

        self.profile = jax.jit(_profile)
        self._initialized = True

    @property
    def alpha(self):  # noqa: D102
        return self._alpha

    @alpha.setter
    def alpha(self, value):  # noqa: D102
        wstring = None
        if isinstance(value, numpyro.distributions.Distribution):
            if value.support.lower_bound <= 0:
                wstring = "The alpha prior support includes values"
        elif value <= 0:
            wstring = "The alpha parameter is"
        if wstring is not None:
            warnings.warn(
                f"{wstring} less than or equal to 0. "
                "This might lead to unphysical models."
            )
        self._alpha = value

    @property
    def beta(self):  # noqa: D102
        return self._beta

    @beta.setter
    def beta(self, value):  # noqa: D102
        wstring = None
        if isinstance(value, numpyro.distributions.Distribution):
            if value.support.upper_bound >= 3:
                wstring = "The beta prior support includes values"
        elif value >= 3:
            wstring = "The beta parameter is"
        if wstring is not None:
            warnings.warn(
                f"{wstring} greater than or equal to 3. "
                "This might lead to unphysical models."
            )
        self._beta = value

    @property
    def gamma(self):  # noqa: D102
        return self._gamma

    @gamma.setter
    def gamma(self, value):  # noqa: D102
        wstring = None
        if isinstance(value, numpyro.distributions.Distribution):
            if value.support.upper_bound >= 1:
                wstring = "The gamma prior support includes values"
        elif value >= 1:
            wstring = "The gamma parameter is"
        if wstring is not None:
            warnings.warn(
                f"{wstring} greater than or equal to 1. "
                "This might lead to unphysical models."
            )
        self._gamma = value

    @staticmethod
    def _profile(r, Ic, rc, alpha, beta, gamma, rz, eps=1.00e-08):
        """
        Generalized Navarro-Frenk-White (gNFW) profile via Abel deprojection.

        Computes the projected surface brightness profile by Abel transformation
        of a 3D density distribution. Used internally by the gNFW class.

        Parameters
        ----------
        r : ndarray
            Projected elliptical radius in degrees.
        Ic : float
            Characteristic surface brightness (same units as image).
        rc : float
            Scale radius in degrees.
        alpha : float
            Intermediate slope parameter (sharpness of transition).
        beta : float
            Outer slope parameter.
        gamma : float
            Inner slope parameter (central cusp).
        rz : ndarray
            Radial points for numerical integration (in units of rc).
        eps : float, optional
            Absolute and relative error tolerance for integration. Default is 1e-8.

        Returns
        -------
        ndarray
            Projected surface brightness at radius r.

        Notes
        -----
        The 3D profile is:
        s(r) = (Ic/rc) / [(r/rc)^gamma * (1 + (r/rc)^alpha)^((beta-gamma)/alpha)]

        The surface brightness is obtained by Abel projection (line-of-sight
        integration). This is computed numerically using adaptive quadrature.

        The gNFW profile generalizes several important profiles:

        - NFW (alpha=1, beta=3, gamma=1)
        - Hernquist (alpha=1, beta=4, gamma=1)
        - Einasto-like with varying slopes

        This is a computationally expensive profile due to the numerical
        integration required for each evaluation.

        References
        ----------
        Nagai, D., Kravtsov, A. V., & Vikhlinin, A., ApJ, 668, 1 (2007)
        Mroczkowski, T., et al., ApJ, 694, 1034 (2009)
        """

        def radial(u, alpha, beta, gamma):
            factor = 1.00 + u**alpha
            factor = factor ** ((gamma - beta) / alpha)
            return factor / u**gamma

        def integrand(u, uz):
            factor = radial(u, alpha, beta, gamma)
            return 2.00 * factor * u / jp.sqrt(u**2 - uz**2)

        def integrate(rzj):
            return quadgk(
                integrand, [rzj, jp.inf], args=(rzj,), epsabs=eps, epsrel=eps
            )[0]

        mz = Ic * jax.vmap(integrate)(rz)
        return jp.interp(r / rc, rz, mz)
