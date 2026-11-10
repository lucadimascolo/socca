"""Base class for 2D radial surface brightness profiles."""

from abc import abstractmethod
from functools import partial
import inspect
import warnings

import jax
import jax.numpy as jp
import numpy as np

from .. import config
from ..base import Component, _warn_range
from .truncation import Truncation, HyperTangent


_TRUNCATION = dict(rt="radius", wt="width")


def _check_names(names):
    """Reject parameter names reserved for the truncation."""
    for name in names:
        if name in ("rt", "wt"):
            raise ValueError(
                f"'{name}' is reserved for the profile's truncation: give "
                "this parameter another name."
            )


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
        _check_names(inspect.signature(type(self).profile).parameters)
        super().__init__(**kwargs)
        self.xc = kwargs.get("xc", config.Profile.xc)
        self.yc = kwargs.get("yc", config.Profile.yc)

        self.theta = kwargs.get("theta", config.Profile.theta)
        self.e = kwargs.get("e", config.Profile.e)
        self.cbox = kwargs.get("cbox", config.Profile.cbox)

        self.units.update(dict(xc="deg", yc="deg", theta="rad", e="", cbox=""))
        self.periodic.update(dict(theta=jp.pi))

        self.description.update(
            dict(
                xc="Right ascension of centroid",
                yc="Declination of centroid",
                theta="Position angle (east from north)",
                e="Projected ellipticity (1 - axis ratio)",
                cbox="Projected boxiness",
            )
        )

        self._truncation = None
        self.truncation = kwargs.get("truncation", None)

    def addparameter(self, name, value=None, units="", description=""):  # noqa: D102
        _check_names([name])
        super().addparameter(name, value, units, description)

    addparameter.__doc__ = Component.addparameter.__doc__

    @property
    def truncation(self):  # noqa: D102
        return self._truncation

    @truncation.setter
    def truncation(self, value):  # noqa: D102
        if isinstance(value, bool):
            value = HyperTangent() if value else None
        if value is not None:
            if not isinstance(value, Truncation):
                raise ValueError(
                    "Truncation must be an instance of the Truncation class."
                )
            _init = self._initialized
            self._initialized = False
            self.rt = value.rt
            self.wt = value.wt
            self._initialized = _init
            self.units.update(dict(rt=value.units["rt"], wt=value.units["wt"]))
            # the new values are native, whatever units the old ones had
            self._input_units.pop("rt", None)
            self._input_units.pop("wt", None)
            self.description.update(
                dict(
                    rt=value.description["rt"],
                    wt=value.description["wt"],
                )
            )
        else:
            self._truncation = None
            if hasattr(self, "rt"):
                self.removeparameter("rt")
            if hasattr(self, "wt"):
                self.removeparameter("wt")
        self._truncation = value
        if self._wrapper is not None:
            self._wrapper._retruncate(self)

    @property
    def theta(self):  # noqa: D102
        return self._theta

    @theta.setter
    def theta(self, value):  # noqa: D102
        self._theta = value

    def __setattr__(self, name, value):
        """Reject non-positive truncation radii and widths, priors included."""
        if name in ("rt", "wt"):
            _warn_range(
                value,
                name,
                lower=0,
                lower_strict=True,
                upper=np.inf,
                upper_strict=True,
                note=f"The truncation {_TRUNCATION[name]} must be strictly "
                "positive and finite; set truncation = None for none.",
                error=True,
            )
        super().__setattr__(name, value)

    @property
    def e(self):  # noqa: D102
        return self._e

    @e.setter
    def e(self, value):  # noqa: D102
        _warn_range(
            value,
            "e",
            lower=0,
            upper=1,
            upper_strict=True,
            note="Ellipticity must be in the range [0, 1).",
            error=True,
        )
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
        _keys = list(inspect.signature(self.profile).parameters.keys()) + [
            "xc",
            "yc",
            "theta",
            "e",
            "cbox",
        ]
        if self.truncation is not None:
            _keys += ["rt", "wt"]

        kwarg = self._getpars([key for key in _keys if key != "r"])

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
        _keys = list(inspect.signature(self.profile).parameters.keys()) + [
            "xc",
            "yc",
            "theta",
            "e",
            "cbox",
        ]
        if self.truncation is not None:
            _keys += ["rt", "wt"]

        return {
            key.replace(f"{comp_prefix}_", ""): pars[key]
            for key in pars
            if key.startswith(f"{comp_prefix}_")
            and key.replace(f"{comp_prefix}_", "") in _keys
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

        if self.truncation is not None:
            rt = kwarg.pop("rt")
            wt = kwarg.pop("wt")

        rgrid = self.getgrid(img.grid, xc, yc, theta, e, cbox)
        mgrid = self.profile(rgrid, **kwarg)

        if self.truncation is not None:
            mgrid = mgrid * self.truncation.profile(rgrid, rt, wt)

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

        For most profiles, this returns a copy keeping every parameter,
        unit and setting; ties to the profile itself point to the copy.
        Some profiles (e.g., PolyExpoRefact) override this to convert to an
        alternative parameterization.

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
        return self._copy()


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
        _check_names([p["name"] for p in parameters])
        super().__init__(**kwargs)

        for p in parameters:
            setattr(self, p["name"], None)
            self.units.update({p["name"]: p.get("unit", "not specified")})
            self.description.update(
                {p["name"]: p.get("description", "No description provided")}
            )

        self._function = profile
        self.profile = jax.jit(profile)

    def __getstate__(self):
        """Drop the compiled profile, rebuilt on unpickling."""
        state = super().__getstate__()
        state.pop("profile", None)
        return state

    def __setstate__(self, state):
        """Restore state and recompile the profile."""
        super().__setstate__(state)
        self.__dict__["profile"] = jax.jit(self._function)
