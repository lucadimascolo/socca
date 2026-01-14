"""Exponential disk profiles and variants."""

import jax
import jax.numpy as jp

from .. import config
from .base import Profile


class Exponential(Profile):
    """
    Exponential disk profile.

    The exponential profile (Sersic index n=1) is the standard model for disk
    galaxies. It describes a surface brightness distribution that decays
    exponentially with radius: I(r) = Is * exp(-r/rs).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rs = kwargs.get("rs", config.Exponential.rs)
        self.Is = kwargs.get("Is", config.Exponential.Is)

        self.units.update(dict(rs="deg", Is="image"))
        self.description.update(
            dict(rs="Scale radius", Is="Central surface brightness")
        )

    @staticmethod
    @jax.jit
    def profile(r, Is, rs):
        """
        Exponential disk profile surface brightness distribution.

        The exponential profile (Sersic index n=1) describes the light
        distribution in disk galaxies and spiral arms.

        Parameters
        ----------
        r : ndarray
            Elliptical radius in degrees.
        Is : float
            Central surface brightness (same units as image).
        rs : float
            Scale radius (scale length) in degrees.

        Returns
        -------
        ndarray
            Surface brightness at radius r.

        Notes
        -----
        The exponential profile is defined as:
        I(r) = Is * exp(-r / rs)

        This is a special case of the Sersic profile with n=1 and is the
        canonical profile for modeling disk galaxies. The scale radius rs
        contains about 63% of the total light within that radius.

        The exponential profile has no finite effective radius; the half-light
        radius is approximately 1.678 * rs.

        References
        ----------
        Freeman, K. C. 1970, ApJ, 160, 811
        van der Kruit, P. C., & Searle, L. 1981, A&A, 95, 105

        Examples
        --------
        >>> import jax.numpy as jp
        >>> from socca.models import Exponential
        >>> r = jp.linspace(0, 0.1, 100)
        >>> I = Exponential.profile(r, Is=100.0, rs=0.02)
        """
        return Is * jp.exp(-r / rs)


# PolyExponential
# Mancera Pina et al., A&A, 689, A344 (2024)
# --------------------------------------------------------
class PolyExponential(Exponential):
    """
    Exponential profile with polynomial modulation.

    Extended exponential profile that includes polynomial terms to model
    deviations from a pure exponential disk. Based on the formalism from
    Mancera Pina et al., A&A, 689, A344 (2024).
    """

    def __init__(self, **kwargs):
        nk = 4
        super().__init__(**kwargs)
        for k in range(nk):
            setattr(
                self,
                f"c{k + 1}",
                kwargs.get(f"c{k + 1}", config.PolyExponential.ck),
            )
        self.rc = kwargs.get("rc", config.PolyExponential.rc)

        self.units.update(dict(rc="deg"))
        self.units.update({f"c{k + 1}": "" for k in range(nk)})

        self.description.update(
            dict(
                c1="Polynomial coefficient 1",
                c2="Polynomial coefficient 2",
                c3="Polynomial coefficient 3",
                c4="Polynomial coefficient 4",
                rc="Reference radius for polynomial terms",
            )
        )

    @staticmethod
    @jax.jit
    def profile(r, Is, rs, c1, c2, c3, c4, rc):
        """
        Polynomial-exponential profile surface brightness distribution.

        An exponential profile modulated by a 4th-order polynomial, providing
        flexibility to model deviations from pure exponential profiles in
        disk galaxies.

        Parameters
        ----------
        r : ndarray
            Elliptical radius in degrees.
        Is : float
            Central surface brightness (same units as image).
        rs : float
            Exponential scale radius in degrees.
        c1, c2, c3, c4 : float
            Polynomial coefficients for 1st through 4th order terms.
        rc : float
            Reference radius for polynomial terms in degrees.

        Returns
        -------
        ndarray
            Surface brightness at radius r.

        Notes
        -----
        The profile is defined as:

        I(r) = Is * exp(-r/rs) * [1 + c1*(r/rc) + c2*(r/rc)^2
                                    + c3*(r/rc)^3 + c4*(r/rc)^4]

        This profile allows modeling of:

        - Truncations or drops in outer regions
        - Central enhancements or deficits
        - Breaks in disk profiles
        - Type I, II, and III disk breaks

        The polynomial modulation is normalized to the reference radius rc,
        which should typically be comparable to the scale radius rs.

        References
        ----------
        Mancera Pina et al., A&A, 689, A344 (2024)

        Examples
        --------
        >>> import jax.numpy as jp
        >>> from socca.models import PolyExponential
        >>> r = jp.linspace(0, 0.1, 100)
        >>> # Profile with slight central enhancement
        >>> I = PolyExponential.profile(r, Is=100.0, rs=0.02, c1=0.1,
        ...                             c2=-0.05, c3=0.0, c4=0.0, rc=0.02)
        """
        factor = (
            1.00
            + c1 * (r / rc)
            + c2 * ((r / rc) ** 2)
            + c3 * ((r / rc) ** 3)
            + c4 * ((r / rc) ** 4)
        )
        return factor * Is * jp.exp(-r / rs)


# PolyExpoRefact
# Mancera Pina et al., A&A, 689, A344 (2024)
# --------------------------------------------------------
class PolyExpoRefact(Exponential):
    """
    Refactored polynomial exponential profile with intensity coefficients.

    Alternative parameterization of the polynomial exponential profile using
    intensity coefficients instead of polynomial coefficients. Based on the
    formalism from Mancera Pina et al., A&A, 689, A344 (2024).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.I1 = kwargs.get("I1", config.PolyExpoRefact.I1)
        self.I2 = kwargs.get("I2", config.PolyExpoRefact.I2)
        self.I3 = kwargs.get("I3", config.PolyExpoRefact.I3)
        self.I4 = kwargs.get("I4", config.PolyExpoRefact.I4)
        self.rc = kwargs.get("rc", config.PolyExpoRefact.rc)

        self.units.update(dict(rc="deg"))
        self.units.update({f"I{ci}": "image" for ci in range(1, 5)})

        self.description.update(
            dict(
                I1="Polynomial intensity coefficient 1",
                I2="Polynomial intensity coefficient 2",
                I3="Polynomial intensity coefficient 3",
                I4="Polynomial intensity coefficient 4",
                rc="Reference radius for polynomial terms",
            )
        )

    @staticmethod
    @jax.jit
    def profile(r, Is, rs, I1, I2, I3, I4, rc):
        """
        Refactored polynomial-exponential profile using intensity coefficients.

        Alternative parameterization of the polynomial-exponential profile using
        intensity coefficients rather than dimensionless polynomial coefficients.

        Parameters
        ----------
        r : ndarray
            Elliptical radius in degrees.
        Is : float
            Central surface brightness (same units as image).
        rs : float
            Exponential scale radius in degrees.
        I1, I2, I3, I4 : float
            Intensity coefficients for polynomial terms (same units as Is).
        rc : float
            Reference radius for polynomial terms in degrees.

        Returns
        -------
        ndarray
            Surface brightness at radius r.

        Notes
        -----
        The profile is defined as:

        I(r) = exp(-r/rs) * [Is + I1*(r/rc) + I2*(r/rc)^2
                                + I3*(r/rc)^3 + I4*(r/rc)^4]

        This is mathematically equivalent to PolyExponential but uses intensity
        coefficients Ii instead of dimensionless coefficients ci. The relation is:

        Ii = ci * Is

        This parameterization may be more intuitive when the polynomial terms
        represent physical contributions with intensity units.

        References
        ----------
        Mancera Pina et al., A&A, 689, A344 (2024)

        See Also
        --------
        PolyExponential : Alternative parameterization with dimensionless coefficients.
        """
        factor = Is * jp.exp(-r / rs)
        for ci in range(1, 5):
            factor += eval(f"I{ci}") * jp.exp(-r / rs) * ((r / rc) ** ci)
        return factor

    def refactor(self):
        """
        Convert to equivalent PolyExponential parameterization.

        Transforms the intensity-based parameterization (I1, I2, I3, I4) to the
        dimensionless coefficient parameterization (c1, c2, c3, c4) of PolyExponential.

        Returns
        -------
        PolyExponential
            Equivalent profile with dimensionless coefficients ci = Ii / Is.

        Notes
        -----
        This conversion preserves the exact same surface brightness profile
        but expresses it using normalized polynomial coefficients.

        Examples
        --------
        >>> from socca.models import PolyExpoRefact
        >>> prof = PolyExpoRefact(Is=100, I1=10, I2=5, I3=0, I4=0)
        >>> prof_equiv = prof.refactor()
        >>> # prof_equiv is PolyExponential with c1=0.1, c2=0.05, c3=0, c4=0
        """
        kwargs = {
            key: getattr(self, key)
            for key in ["xc", "yc", "theta", "e", "Is", "rs", "rc"]
        }
        for ci in range(1, 5):
            kwargs.update({f"c{ci}": eval(f"I{ci}") / kwargs["Is"]})

        return PolyExponential(**kwargs)


# Modified Exponential
# Mancera Pina et al., A&A, 689, A344 (2024)
# --------------------------------------------------------
class ModExponential(Exponential):
    """
    Modified exponential profile with power-law modulation.

    An exponential profile modulated by a power law to provide additional
    flexibility for modeling disk profiles with deviations from pure exponential
    behavior. Based on the formalism from Mancera Pina et al., A&A, 689, A344
    (2024).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rm = kwargs.get("rm", config.ModExponential.rm)
        self.alpha = kwargs.get("alpha", config.ModExponential.alpha)

        self.units.update(dict(rm="deg", alpha=""))

        self.description.update(
            dict(rm="Modification radius", alpha="Modification exponent")
        )

    @staticmethod
    @jax.jit
    def profile(r, Is, rs, rm, alpha):
        """
        Generate modified exponential profile with power-law modulation.

        An exponential profile modulated by a power law, providing an additional
        degree of freedom to model disk profiles with deviations from pure
        exponential behavior.

        Parameters
        ----------
        r : ndarray
            Elliptical radius in degrees.
        Is : float
            Central surface brightness (same units as image).
        rs : float
            Exponential scale radius in degrees.
        rm : float
            Modification radius where power law becomes important (deg).
        alpha : float
            Power-law exponent (positive for enhancement, negative for suppression).

        Returns
        -------
        ndarray
            Surface brightness at radius r.

        Notes
        -----
        The profile is defined as:
        I(r) = Is * exp(-r/rs) * (1 + r/rm)^alpha

        This profile can model:

        - Truncations (alpha < 0)
        - Enhancements in outer regions (alpha > 0)
        - Smooth transitions between different slopes

        The modification becomes significant at r ~ rm. For r << rm, the
        profile is approximately exponential. For r >> rm, the behavior
        depends on the sign of alpha.

        References
        ----------
        Mancera Pina et al., A&A, 689, A344 (2024)

        Examples
        --------
        >>> import jax.numpy as jp
        >>> from socca.models import ModExponential
        >>> r = jp.linspace(0, 0.1, 100)
        >>> # Profile with outer truncation
        >>> I = ModExponential.profile(r, Is=100.0, rs=0.02, rm=0.05, alpha=-2.0)
        """
        return Is * jp.exp(-r / rs) * (1.00 + r / rm) ** alpha
