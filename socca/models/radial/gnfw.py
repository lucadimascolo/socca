"""Generalized Navarro-Frenk-White profile."""

import jax
import jax.numpy as jp
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rc = kwargs.get("rc", config.gNFW.rc)
        self.Ic = kwargs.get("Ic", config.gNFW.Ic)
        self.alpha = kwargs.get("alpha", config.gNFW.alpha)
        self.beta = kwargs.get("beta", config.gNFW.beta)
        self.gamma = kwargs.get("gamma", config.gNFW.gamma)

        self.rz = kwargs.get("rz", jp.logspace(-7, 2, 1000))
        self.eps = kwargs.get("eps", 1.00e-08)

        self.okeys.append("rz")
        self.okeys.append("eps")
        self.okeys.append("profile")

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
        The 3D density profile is:
        rho(r) = rho0 / [(r/rc)^gamma * (1 + (r/rc)^alpha)^((beta-gamma)/alpha)]

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
        Navarro, J. F., Frenk, C. S., & White, S. D. M. 1996, ApJ, 462, 563
        Zhao, H. 1996, MNRAS, 278, 488
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
