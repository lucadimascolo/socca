"""
Generalized Navarro-Frenk-White (gNFW) profile module.

This module provides two implementations of the gNFW profile:

- **integral**: Numerical Abel deprojection using adaptive quadrature (default).
  Accurate but computationally expensive.
- **emulator**: Neural network approximation of the Abel integral.
  Faster but requires a pre-trained model and is limited to the training domain.

Examples
--------
Using the default numerical integration method:

>>> from socca.models import gNFW
>>> profile = gNFW(xc=180.5, yc=45.2, rc=0.01, Ic=100)

Using the emulator method:

>>> profile = gNFW(xc=180.5, yc=45.2, rc=0.01, Ic=100, method='emulator')
"""

from .integral import gNFWIntegral
from .emulator import gNFWEmulator


__all__ = ["gNFW"]


class gNFW:
    """
    Generalized Navarro-Frenk-White (gNFW) profile.

    The gNFW profile is a flexible three-parameter model commonly used to
    describe the surface brightness distribution of galaxy clusters.
    It generalizes the NFW profile with adjustable inner (gamma),
    intermediate (alpha), and outer (beta) slopes.

    Parameters
    ----------
    method : str, optional
        Evaluation method to use. Options are:

        - ``'integral'`` (default): Numerical Abel deprojection using
          adaptive quadrature. Accurate but computationally expensive.
        - ``'emulator'``: Neural network approximation. Faster but
          requires a pre-trained model and is limited to the training
          parameter domain.

    rc : float
        Scale radius in degrees.
    Ic : float
        Characteristic surface brightness (same units as image).
    alpha : float, optional
        Intermediate slope parameter. Default is 1.0510.
    beta : float, optional
        Outer slope parameter. Default is 5.4905.
    gamma : float, optional
        Inner slope parameter. Default is 0.3081.
    xc, yc : float, optional
        Centroid coordinates in degrees.
    theta : float, optional
        Position angle in radians.
    e : float, optional
        Ellipticity (1 - b/a).
    cbox : float, optional
        Boxiness parameter.

    For method='emulator':
        model : MLP or str, optional
            Pre-trained emulator model or path to .dill file.

    For method='integral':
        rz : ndarray, optional
            Radial integration points (units of rc).
        eps : float, optional
            Integration error tolerance.

    Notes
    -----
    The 3D density profile is:
    rho(r) = rho0 / [(r/rc)^gamma * (1 + (r/rc)^alpha)^((beta-gamma)/alpha)]

    The surface brightness is obtained by Abel projection (line-of-sight
    integration).

    The gNFW profile generalizes several important profiles:

    - NFW (alpha=1, beta=3, gamma=1)
    - Hernquist (alpha=1, beta=4, gamma=1)
    - Einasto-like with varying slopes

    References
    ----------
    Nagai, D., Kravtsov, A. V., & Vikhlinin, A., ApJ, 668, 1 (2007)
    Mroczkowski, T., et al., ApJ, 694, 1034 (2009)

    Examples
    --------
    >>> from socca.models import gNFW
    >>> # Default numerical integration
    >>> profile_int = gNFW(rc=0.01, Ic=100)
    >>> # Emulator-based evaluation
    >>> profile_emu = gNFW(method='emulator', rc=0.01, Ic=100)
    """

    def __new__(cls, method="integral", **kwargs):
        """Create a gNFW profile instance with the specified method."""
        method = method.lower()

        if method == "integral":
            return gNFWIntegral(**kwargs)
        elif method in ("emulator", "emulated", "emu"):
            return gNFWEmulator(**kwargs)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Use 'integral' or 'emulator'."
            )
