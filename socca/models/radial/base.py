"""Base class for 2D radial surface brightness profiles."""

from abc import abstractmethod
from functools import partial
import inspect
import warnings

import jax
import jax.numpy as jp
import numpyro.distributions

from .. import config
from ..base import Component


class Profile(Component):
    """
    Base class for 2D surface brightness profiles.

    This class extends Component to provide common coordinate transformation
    and projection parameters (position, orientation, ellipticity, boxiness)
    for all 2D surface brightness profiles.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments for profile initialization including:

        xc : float, optional
            Right ascension of profile centroid (deg).
        yc : float, optional
            Declination of profile centroid (deg).
        theta : float, optional
            Position angle measured east from north (rad).
        e : float, optional
            Ellipticity, defined as 1 - b/a where b/a is axis ratio (0 <= e < 1).
        cbox : float, optional
            Boxiness parameter (0 = elliptical, >0 = boxy, <0 = disky).
        positive : bool, optional
            Whether to enforce positivity constraint.

    Attributes
    ----------
    xc, yc : float
        Centroid coordinates in celestial degrees.
    theta : float
        Position angle in radians.
    e : float
        Ellipticity parameter.
    cbox : float
        Boxiness/diskyness parameter.

    Notes
    -----
    All Profile subclasses must implement the abstract profile() method that
    defines the radial surface brightness distribution.
    """

    _scale_radius = None
    _scale_amp = None

    def __init__(self, **kwargs):
        """
        Initialize a profile with standard coordinate and shape parameters.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments including xc, yc, theta, e, cbox, positive.
            See class docstring for parameter descriptions.
        """
        super().__init__(**kwargs)
        self.xc = kwargs.get("xc", config.Profile.xc)
        self.yc = kwargs.get("yc", config.Profile.yc)

        self.theta = kwargs.get("theta", config.Profile.theta)
        self.e = kwargs.get("e", config.Profile.e)
        self.cbox = kwargs.get("cbox", config.Profile.cbox)

        self.units.update(dict(xc="deg", yc="deg", theta="rad", e="", cbox=""))

        self.description.update(
            dict(
                xc="Right ascension of centroid",
                yc="Declination of centroid",
                theta="Position angle (east from north)",
                e="Projected eccentricity (1 - axis ratio)",
                cbox="Projected boxiness",
            )
        )

    @property
    def e(self):  # noqa: D102
        return self._e

    @e.setter
    def e(self, value):  # noqa: D102
        wsuffix = "Ellipticity must be in the range [0, 1)."
        wstring = None
        if isinstance(value, numpyro.distributions.Distribution):
            if value.support.upper_bound > 1:
                wstring = "The ellipticity prior support includes values"
        elif value >= 1:
            wstring = "The ellipticity parameter is"
        if wstring is not None:
            raise ValueError(f"{wstring} greater than 1. {wsuffix}")

        if isinstance(value, numpyro.distributions.Distribution):
            if value.support.lower_bound < 0:
                wstring = "The ellipticity prior support includes values"
        elif value < 0:
            wstring = "The ellipticity parameter is"
        if wstring is not None:
            raise ValueError(f"{wstring} lower than 0. {wsuffix}")
        self._e = value

    @abstractmethod
    def profile(self, r):
        """
        Abstract method defining the radial profile function.

        Subclasses must implement this method to define the surface brightness
        distribution as a function of radius and other profile-specific parameters.

        Parameters
        ----------
        r : ndarray
            Radial coordinate in degrees.

        Returns
        -------
        ndarray
            Surface brightness values at each radius.
        """
        pass

    def getmap(self, img, convolve=False):
        """
        Generate an image map from the profile on the given grid.

        Evaluates the profile on the image grid and optionally convolves with
        the PSF. All parameters must have fixed values.

        Parameters
        ----------
        img : Image
            Image object containing grid, PSF, and WCS information.
        convolve : bool, optional
            If True, convolve the model with the PSF. Default is False.

        Returns
        -------
        ndarray
            2D array of surface brightness values on the image grid.
            Shape matches img.data.shape.

        Raises
        ------
        ValueError
            If any parameter is a prior distribution or set to None.

        Warns
        -----
        UserWarning
            If convolve=True but no PSF is defined in img.

        Notes
        -----
        - All profile parameters must be fixed values (not distributions).
        - The elliptical grid is computed using position, ellipticity, and PA.
        - Convolution is performed in Fourier space for efficiency.

        Examples
        --------
        >>> from socca.models import Beta
        >>> from socca.data import Image
        >>> beta = Beta(xc=180.5, yc=45.2, rc=0.01, Ic=100, beta=0.5)
        >>> img = Image('observation.fits')
        >>> model_map = beta.getmap(img, convolve=True)
        """
        kwarg = {
            key: getattr(self, key)
            for key in list(inspect.signature(self.profile).parameters.keys())
            + ["xc", "yc", "theta", "e", "cbox"]
            if key != "r"
        }

        for key in kwarg.keys():
            if isinstance(kwarg[key], numpyro.distributions.Distribution):
                raise ValueError(
                    "Priors must be fixed values, not distributions."
                )
            if kwarg[key] is None:
                raise ValueError(
                    f"keyword {key} is set to None. "
                    f"Please provide a valid value."
                )

        mgrid = self._evaluate(img, **kwarg)

        if convolve:
            if img.psf is None:
                warnings.warn(
                    "No PSF defined, so no convolution will be performed."
                )
            else:
                mgrid = img.convolve(mgrid)
        return mgrid

    def _build_kwargs(self, pars, comp_prefix):
        """
        Build keyword arguments for _evaluate from the full parameters dict.

        Parameters
        ----------
        pars : dict
            Full parameters dictionary with prefixed keys.
        comp_prefix : str
            Component prefix (e.g., 'comp_00').

        Returns
        -------
        dict
            Keyword arguments for _evaluate.
        """
        return {
            key.replace(f"{comp_prefix}_", ""): pars[key]
            for key in pars
            if key.startswith(f"{comp_prefix}_")
            and key.replace(f"{comp_prefix}_", "")
            in list(inspect.signature(self.profile).parameters.keys())
            + ["xc", "yc", "theta", "e", "cbox"]
        }

    def _evaluate(self, img, **kwarg):
        """
        Evaluate the profile on the image grid with explicit parameters.

        This internal method computes the profile on a coordinate grid using
        the provided geometric and profile-specific parameters. It is used by
        both getmap() and Model.getmodel() to avoid code duplication.

        Parameters
        ----------
        img : Image
            Image object containing grid, PSF, and WCS information.
        **kwarg : dict
            All parameters including xc, yc, theta, e, cbox and profile-specific
            parameters (e.g., rc, Ic, beta for Beta profile).

        Returns
        -------
        ndarray
            2D array of surface brightness values, averaged over subpixels.
        """
        xc = kwarg.pop("xc")
        yc = kwarg.pop("yc")
        theta = kwarg.pop("theta")
        e = kwarg.pop("e")
        cbox = kwarg.pop("cbox")
        rgrid = self.getgrid(img.grid, xc, yc, theta, e, cbox)
        mgrid = self.profile(rgrid, **kwarg)
        return jp.mean(mgrid, axis=0)

    @staticmethod
    @partial(jax.jit, static_argnames=["grid"])
    def getgrid(grid, xc, yc, theta=0.00, e=0.00, cbox=0.00):
        """
        Compute elliptical radius grid with rotation and projection.

        This static JIT-compiled method transforms celestial coordinates to
        elliptical radius values accounting for position angle, ellipticity,
        and boxiness. Used internally by profile evaluation.

        Parameters
        ----------
        grid : Grid
            Grid object with .x and .y celestial coordinate arrays (deg).
        xc : float
            Right ascension of centroid (deg).
        yc : float
            Declination of centroid (deg).
        theta : float, optional
            Position angle east from north (rad). Default is 0.
        e : float, optional
            Ellipticity (1 - b/a). Default is 0 (circular).
        cbox : float, optional
            Boxiness parameter. Default is 0 (elliptical).

        Returns
        -------
        ndarray
            Elliptical radius grid in degrees. Same shape as grid.x and grid.y.

        Notes
        -----
        The transformation accounts for:

        - Spherical geometry (cos(dec) correction)
        - Position angle rotation
        - Ellipticity via axis ratio correction
        - Generalized elliptical isophotes with boxiness

        The elliptical radius is computed as:
        r = [(abs(x')^(2+c) + abs(y'/(1-e))^(2+c)]^(1/(2+c))
        where c is the boxiness parameter.

        This function is JIT-compiled for performance during model evaluation.
        """
        sint = jp.sin(theta)
        cost = jp.cos(theta)

        xgrid = (
            -(grid.x - xc) * jp.cos(jp.deg2rad(yc)) * sint
            - (grid.y - yc) * cost
        )
        ygrid = (grid.x - xc) * jp.cos(jp.deg2rad(yc)) * cost - (
            grid.y - yc
        ) * sint

        xgrid = jp.abs(xgrid) ** (cbox + 2.00)
        ygrid = jp.abs(ygrid / (1.00 - e)) ** (cbox + 2.00)
        return jp.power(xgrid + ygrid, 1.00 / (cbox + 2.00))

    def refactor(self):
        """
        Return a refactored version of the profile with equivalent parameterization.

        For most profiles, this returns a copy of the profile with the same
        parameters. Some profiles (e.g., PolyExpoRefact) override this to
        convert to an alternative parameterization.

        Returns
        -------
        Profile
            A new profile instance with refactored parameterization.

        Warns
        -----
        UserWarning
            Warns that this profile has no alternative parameterization.

        Examples
        --------
        >>> from socca.models import Beta
        >>> beta = Beta(xc=180.5, yc=45.2)
        >>> beta_refactored = beta.refactor()
        """
        warnings.warn("Nothing to refactor here.")
        return self.__class__(**self.__dict__)


class CustomProfile(Profile):
    r"""
    User-defined custom surface brightness profile.

    Allows users to define arbitrary profile functions with custom parameters,
    enabling modeling of non-standard surface brightness distributions.

    Parameters
    ----------
    parameters : list of dict
        List of parameter specifications. Each dict should contain:

        - 'name': str, parameter name
        - 'unit': str, optional, physical unit (default: 'not specified')
        - 'description': str, optional, parameter description
    profile : callable
        Function defining the profile. Should have signature profile(r, \*\*params)
        where r is the elliptical radius and \*\*params are the custom parameters.
    **kwargs : dict
        Standard profile parameters (xc, yc, theta, e, cbox, positive).

    Notes
    -----
    - The profile function is automatically JIT-compiled for performance.
    - All parameters in the parameters list are initialized to None and must
      be set before use.
    - The profile function should be compatible with JAX (use jax.numpy operations).

    Examples
    --------
    >>> from socca.models import CustomProfile
    >>> import jax.numpy as jp
    >>> # Define a custom Gaussian profile
    >>> def gaussian_profile(r, amplitude, sigma):
    ...     return amplitude * jp.exp(-0.5 * (r / sigma)**2)
    >>> params = [
    ...     {'name': 'amplitude', 'unit': 'image', 'description': 'Peak value'},
    ...     {'name': 'sigma', 'unit': 'deg', 'description': 'Gaussian width'}
    ... ]
    >>> profile = CustomProfile(params, gaussian_profile, xc=180.5, yc=45.2)
    >>> profile.amplitude = 100.0
    >>> profile.sigma = 0.01
    """

    def __init__(self, parameters, profile, **kwargs):
        r"""
        Initialize a custom profile with user-defined parameters and function.

        Parameters
        ----------
        parameters : list of dict
            Parameter specifications with 'name', 'unit', and 'description'.
        profile : callable
            Profile function with signature profile(r, \*\*params).
        **kwargs : dict
            Standard profile parameters (xc, yc, theta, e, cbox).
        """
        super().__init__(**kwargs)

        for p in parameters:
            setattr(self, p["name"], None)
            self.units.update({p["name"]: p.get("unit", "not specified")})
            self.description.update(
                {p["name"]: p.get("description", "No description provided")}
            )

        self.profile = jax.jit(profile)
