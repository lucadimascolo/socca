"""Sersic profile for modeling elliptical galaxies and bulges."""

import jax
import jax.numpy as jp
from jax.scipy.special import gammainc, gammaln

from .. import config
from ..base import _warn_range
from .base import Profile, _safe_power


@jax.custom_jvp
def _bn(ns):
    """Solve P(2 ns, bn) = 1/2 by Newton steps in log bn."""
    a = 2.00 * ns
    big = a - 1.00 / 3.00 + 4.00 / (405.00 * ns) + 46.00 / (25515.00 * ns**2)
    small = jp.exp((jp.log(0.50) + gammaln(a + 1.00)) / a)
    u = jp.log(jp.where(ns < 0.50, small, big))

    def step(u, _):
        f = gammainc(a, jp.exp(u)) - 0.50
        return u - f / jp.exp(a * u - jp.exp(u) - gammaln(a)), None

    u, _ = jax.lax.scan(step, u, None, length=6)
    return jp.exp(u)


@_bn.defjvp
def _bn_jvp(primals, tangents):
    (ns,), (dns,) = primals, tangents
    bn, a = _bn(ns), 2.00 * ns
    dpda = jax.grad(gammainc)(a, bn)
    dpdb = jp.exp((a - 1.00) * jp.log(bn) - bn - gammaln(a))
    return bn, -2.00 * dpda / dpdb * dns


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

    @property
    def re(self):  # noqa: D102
        return self._re

    @re.setter
    def re(self, value):  # noqa: D102
        _warn_range(
            value,
            "re",
            lower=0,
            lower_strict=True,
            note="The effective radius must be strictly positive.",
        )
        self._re = value

    @property
    def ns(self):  # noqa: D102
        return self._ns

    @ns.setter
    def ns(self, value):  # noqa: D102
        _warn_range(
            value,
            "ns",
            lower=0.25,
            upper=10.0,
            note="Sersic indices outside [0.25, 10] are not supported.",
            error=True,
        )
        self._ns = value

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
        bn solves P(2 ns, bn) = 1/2, with P the regularized lower
        incomplete gamma function.

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
        bn = _bn(ns)
        se = _safe_power(r / re, 1.00 / ns) - 1.00
        return Ie * jp.exp(-bn * se)
