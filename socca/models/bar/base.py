"""Bar model with 3D line-of-sight integration."""

from functools import partial
import inspect
import types

import warnings

import jax
import jax.numpy as jp
import numpyro.distributions

import numpy as np

from .. import config
from ..base import Component
from ..radial import Sersic
from ...priors import _BoundTo


class Bar(Component):
    """
    3D bar model evaluated via line-of-sight integration.

    Projects an ellipsoidal Sersic-like density onto the sky by integrating
    along the line of sight, in the same spirit as Disk but using a single
    radial profile for shape rather than separate radial/vertical profiles.
    xc, yc, theta, e, inc, rot, losdepth, and losbins live directly on Bar;
    radial only supplies the brightness-shape parameters (re, Ie, ns for
    Sersic).

    Parameters
    ----------
    radial : Profile, optional
        Profile defining the bar's brightness-shape parameters (re, Ie, ns
        for Sersic). Default is Sersic(). Any xc/yc/theta/e already set on
        it are inherited by Bar unless also passed as a Bar keyword
        argument, in which case Bar's value wins and a warning is raised.
    xc, yc : float, optional
        Right ascension and declination of the bar centroid (deg).
    theta : float, optional
        Position angle, east from north (rad).
    e : float, optional
        Projected ellipticity (1 - axis ratio).
    inc : float, optional
        Inclination angle (0 = face-on). Typically tied to a host Disk's
        inclination via boundto().
    rot : float, optional
        Intrinsic bar rotation, added on top of theta.
    losdepth : float, optional
        Half line-of-sight extent for integration.
    losbins : int, optional
        Number of points for line-of-sight integration.
    **kwargs : dict
        Additional keyword arguments passed to Component.

    Attributes
    ----------
    radial : Profile
        Radial brightness-shape profile component.
    profile : callable
        JIT-compiled 3D density profile evaluated on the (x, y, z) grid.

    Examples
    --------
    >>> from socca.models import Bar, Sersic
    >>> bar = Bar(radial=Sersic(re=2e-4, Ie=10.0, ns=0.25))
    >>> bar.xc, bar.yc, bar.theta, bar.e = 180.5, 45.2, 0.5, 0.7
    >>> bar.inc, bar.rot = 1.0, 0.2
    """

    def __init__(self, radial=Sersic(), **kwargs):
        super().__init__(**kwargs)

        self.radial = radial

        # xc/yc/theta/e live on Bar itself, not on radial. If the caller set
        # one of them on radial (before passing it in) and didn't also pass
        # it to Bar, inherit radial's value; if both were set, Bar's value
        # wins and radial's is ignored (with a warning).
        for key in ["xc", "yc", "theta", "e"]:
            radial_val = getattr(self.radial, key)
            radial_isset = radial_val != getattr(config.Profile, key)

            if key in kwargs:
                if radial_isset:
                    warnings.warn(
                        f"Bar's '{key}' and radial's '{key}' were both "
                        f"set; radial's value will be ignored in favor of "
                        f"the value passed to Bar.",
                        UserWarning,
                        stacklevel=2,
                    )
                setattr(self, key, kwargs[key])
            elif radial_isset:
                setattr(self, key, radial_val)
            else:
                setattr(self, key, getattr(config.Bar, key))

        # cbox isn't used by Bar's profile at all; xc/yc/theta/e are gone
        # too now that Bar owns them -- strip all five off the sub-component
        # so it doesn't carry unused, confusing duplicates. Mirrors
        # Bridge.__init__ for its own radial/parallel components.
        for key in ["xc", "yc", "theta", "e", "cbox"]:
            bkey = f"_{key}" if f"_{key}" in self.radial.__dict__ else key
            if bkey in self.radial.__dict__:
                delattr(self.radial, bkey)
            self.radial.units.pop(key, None)
            self.radial.description.pop(key, None)

        self._namespaces = {"radial": self.radial}

        self.inc = kwargs.get("inc", config.Bar.inc)
        self.rot = kwargs.get("rot", config.Bar.rot)
        self.losdepth = kwargs.get("losdepth", config.Bar.losdepth)
        self.losbins = kwargs.get("losbins", config.Bar.losbins)

        for param in ["losdepth", "losbins"]:
            if param not in self.hyper:
                self.hyper.append(param)

        if self.radial.id != self.id:
            type(self).idcls -= 1
            idmin = np.minimum(
                int(self.radial.id.replace("comp_", "")),
                int(self.id.replace("comp_", "")),
            )
            self.id = f"comp_{idmin:02d}"
            self.radial.id = self.id

        self.profile = jax.jit(Bar._bar_profile)

        self.units.update(
            {
                f"radial.{key}": self.radial.units[key]
                for key in self.radial.units.keys()
            }
        )
        self.units.update(
            dict(
                xc="deg",
                yc="deg",
                theta="rad",
                e="",
                inc="rad",
                rot="rad",
                losdepth="deg",
                losbins="",
            )
        )

        self.description.update(
            {
                f"radial.{key}": self.radial.description[key]
                for key in self.radial.description.keys()
            }
        )
        self.description.update(
            dict(
                xc="Right ascension of centroid",
                yc="Declination of centroid",
                theta="Position angle (east from north)",
                e="Projected ellipticity (1 - axis ratio)",
                inc="Inclination angle (0=face-on); Typically inherited from a Disk object",
                rot="Intrinsic bar rotation relative to theta (added to position angle theta)",
                losdepth="Half line-of-sigt extent for integration",
                losbins="Number of points for line-of-sight integration",
            )
        )

        self._initialized = True

    def getmap(self, img, convolve=False):
        """
        Generate bar image via 3D line-of-sight integration.

        Computes the projected surface brightness by integrating the 3D
        ellipsoidal density along the line of sight, accounting for
        inclination and intrinsic rotation.

        Parameters
        ----------
        img : Image
            Image object containing grid, PSF, and WCS information.
        convolve : bool, optional
            If True, convolve the model with the PSF. Default is False.

        Returns
        -------
        ndarray
            Projected bar image on the image grid.

        Raises
        ------
        ValueError
            If any parameter is a prior distribution or set to None.

        Warns
        -----
        UserWarning
            If convolve=True but no PSF is defined.

        Notes
        -----
        Integration accuracy is controlled by losdepth (extent) and losbins
        (number of points); increase both for highly inclined bars.

        Examples
        --------
        >>> from socca.models import Bar, Sersic
        >>> from socca.data import Image
        >>> bar = Bar(radial=Sersic(re=2e-4, Ie=10.0, ns=0.25))
        >>> bar.xc, bar.yc, bar.theta, bar.e = 180.5, 45.2, 0.5, 0.7
        >>> bar.inc, bar.rot = 1.0, 0.2
        >>> img = Image('observation.fits')
        >>> bar_map = bar.getmap(img, convolve=True)
        """
        kwarg = {}

        # Profile shape parameters from radial (re, Ie, ns for Sersic)
        for key in inspect.signature(self.radial.profile).parameters.keys():
            if key == "r":
                continue  # skip r
            val = getattr(self.radial, key)
            if callable(val):
                sig = inspect.signature(val)
                params = list(sig.parameters.keys())
                if params:
                    args = [
                        getattr(self, p.replace(f"{self.id}_", ""))
                        for p in params
                    ]
                    val = val(*args)
            kwarg[key] = val

        # Bar's own position, orientation, and 3D geometry
        for key in [
            "xc",
            "yc",
            "theta",
            "e",
            "inc",
            "rot",
            "losdepth",
            "losbins",
        ]:
            val = getattr(self, key)
            if callable(val):
                sig = inspect.signature(val)
                params = list(sig.parameters.keys())
                if params:
                    args = [
                        getattr(self, p.replace(f"{self.id}_", ""))
                        for p in params
                    ]
                    val = val(*args)
            kwarg[key] = val

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
            Keyword arguments for _evaluate: radial's shape parameters
            (re, Ie, ns) plus xc, yc, theta, e, inc, rot, losdepth, losbins.
        """
        # radial profile shape parameters (re, Ie, ns)
        kwarg = {
            key.replace(f"{comp_prefix}_radial.", ""): pars[key]
            for key in pars
            if key.startswith(f"{comp_prefix}_radial.")
        }
        # Bar's own position, orientation, and 3D geometry
        for key in [
            "xc",
            "yc",
            "theta",
            "e",
            "inc",
            "rot",
            "losdepth",
            "losbins",
        ]:
            kwarg[key] = pars[f"{comp_prefix}_{key}"]
        return kwarg

    @staticmethod
    def _bar_profile(xt, yt, zt, Ie, re, e, ns):
        """Evaluate an ellipsoidal Sersic-like density on a 3D grid."""
        rs = re * (1 - e)
        m = jp.sqrt((xt / rs) ** 2 + (yt / re) ** 2 + (zt / rs) ** 2)
        return Sersic.profile(m, Ie, 1.0, ns)

    def _evaluate(self, img, **kwarg):
        """
        Evaluate bar model on the given grid with explicit parameters.

        This internal method computes the projected bar surface brightness
        via line-of-sight integration using the provided geometric and
        profile parameters. It is used by both getmap() and
        Model.getmodel() to avoid code duplication.

        Parameters
        ----------
        img : Image
            Image object containing grid and WCS information.
        **kwarg : dict
            All parameters, including geometric (xc, yc, theta, inc, rot,
            losdepth, losbins) and radial profile-specific ones (re, Ie, ns,
            e).

        Returns
        -------
        ndarray
            2D array of projected surface brightness, averaged over
            subpixels.
        """
        xc = kwarg.pop("xc")
        yc = kwarg.pop("yc")
        theta = kwarg.pop("theta")
        inc = kwarg.pop("inc")
        rot = kwarg.pop("rot")
        losdepth = kwarg.pop("losdepth")
        losbins = kwarg.pop("losbins")

        xt, yt, zt = self.getgrid(
            img.grid, xc, yc, losdepth, losbins, theta, inc, rot
        )

        dx = 2.0 * losdepth / (losbins - 1)
        mgrid = self.profile(xt, yt, zt, **kwarg)
        mgrid = jp.trapezoid(mgrid, dx=dx, axis=1)
        return jp.mean(mgrid, axis=0)

    @staticmethod
    @partial(jax.jit, static_argnames=["grid", "losbins"])
    def getgrid(
        grid, xc, yc, losdepth, losbins=200, theta=0.0, inc=0.0, rot=0.0
    ):
        """
        Compute 3D bar coordinates with position angle, inclination, and rotation.

        Generates 3D coordinate grids (x, y, z) for bar model evaluation,
        accounting for the position angle, inclination, and intrinsic
        rotation transformations needed for line-of-sight integration
        through an inclined, rotated bar.

        Parameters
        ----------
        grid : Grid
            Grid object with .x and .y celestial coordinate arrays (deg).
        xc : float
            Right ascension of bar center (deg).
        yc : float
            Declination of bar center (deg).
        losdepth : float
            Half-extent of line-of-sight integration (deg).
        losbins : int, optional
            Number of integration points along line of sight. Default is 200.
        theta : float, optional
            Position angle, east from north (rad). Default is 0.
        inc : float, optional
            Inclination angle (0 = face-on, pi/2 = edge-on) (rad). Default is 0.
        rot : float, optional
            Intrinsic bar rotation, added on top of theta (rad). Default is 0.

        Returns
        -------
        xt, yt, zt : ndarray
            4D arrays of bar-frame coordinates (deg).
            Shape: (ssize, losbins, ysize, xsize).

        Notes
        -----
        The transformation sequence:
        1. Center coordinates on (xc, yc) and apply spherical geometry correction (cos(dec))
        2. Rotate by position angle theta, with a 90-degree offset to match Disk's convention
        3. Create the line-of-sight grid from -losdepth to +losdepth and broadcast to 4D
        4. Incline around the y-axis
        5. Apply the intrinsic bar rotation rot, adding onto theta
        """
        ssize, ysize, xsize = grid.x.shape

        xt = (grid.x - xc) * jp.cos(jp.deg2rad(yc))  # Build x
        yt = grid.y - yc  # Build y
        zt = jp.linspace(-losdepth, losdepth, losbins)  # Build z

        # Rotate by Position Angle (CCW, with 90° offset to match Disk convention).
        sint = jp.sin(theta - 0.5 * jp.pi)
        cost = jp.cos(theta - 0.5 * jp.pi)
        xt, yt = -xt * sint - yt * cost, xt * cost - yt * sint

        # Make the cube 4d.
        xt = jp.broadcast_to(
            xt[:, None, :, :], (ssize, losbins, ysize, xsize)
        ).copy()
        yt = jp.broadcast_to(
            yt[:, None, :, :], (ssize, losbins, ysize, xsize)
        ).copy()
        zt = jp.broadcast_to(
            zt[None, :, None, None], (ssize, losbins, ysize, xsize)
        ).copy()

        # Incline around y-axis.
        sini = jp.sin(inc - 0.5 * jp.pi)
        cosi = jp.cos(inc - 0.5 * jp.pi)
        zt, xt = xt * cosi - zt * sini, xt * sini + zt * cosi

        # Rotate by rot (plain CCW, adds to theta).
        sinr = jp.sin(rot)
        cosr = jp.cos(rot)
        xt, yt = xt * cosr + yt * sinr, -xt * sinr + yt * cosr

        return xt, yt, zt

    def parameters(self):
        """
        Print formatted table of bar parameters from the radial component.

        Displays parameters from both the radial sub-component (prefixed
        as 'radial.parameter') and Bar's own xc/yc/theta/e/inc/rot.
        Separates regular parameters from hyperparameters (integration
        settings).

        Notes
        -----
        Output format:

        Model parameters
        ================
        radial.re    [deg]   : value | Effective radius
        radial.Ie    [image] : value | Surface brightness at re
        radial.ns    []      : value | Sersic index
        xc           [deg]   : value | Right ascension of centroid
        theta        [rad]   : value | Position angle (east from north)
        inc          [rad]   : value | Inclination angle (0=face-on)

        Hyperparameters
        ===============
        losdepth     [deg]   : value | Half line-of-sight extent
        losbins      []      : value | Number of integration points

        Examples
        --------
        >>> from socca.models import Bar, Sersic
        >>> bar = Bar(radial=Sersic(re=2e-4, Ie=10.0, ns=0.25))
        >>> bar.xc, bar.yc, bar.theta, bar.e = 180.5, 45.2, 0.5, 0.7
        >>> bar.parameters()
        Model parameters
        ================
        radial.re         [deg]    : 2.0000E-04 | Effective radius
        ...
        """
        keyout = [key for key in self.units.keys() if key not in self.hyper]

        if len(keyout) > 0:
            maxlen = np.max(
                np.array(
                    [
                        len(f"{key} [{self.units[key]}]")
                        for key in keyout + self.hyper
                    ]
                )
            )

            print("\nModel parameters")
            print("=" * 16)
            for key in keyout:
                keylen = maxlen - len(f" [{self.units[key]}]")
                if key.startswith("radial."):
                    kvalue = getattr(self.radial, key.replace("radial.", ""))
                else:
                    kvalue = getattr(self, key)

                if kvalue is None:
                    kvalue = None
                elif isinstance(kvalue, numpyro.distributions.Distribution):
                    kvalue = f"Distribution: {kvalue.__class__.__name__}"
                elif isinstance(
                    kvalue, (types.LambdaType, types.FunctionType, _BoundTo)
                ):
                    kvalue = "Tied parameter"
                else:
                    kvalue = f"{kvalue:.4E}"

                print(
                    f"{key:<{keylen}} [{self.units[key]}] : "
                    + f"{kvalue}".ljust(10)
                    + f" | {self.description[key]}"
                )

            if len(self.hyper) > 0:
                print("\nHyperparameters")
                print("=" * 15)
                for key in self.hyper:
                    keylen = maxlen - len(f" [{self.units[key]}]")
                    if key.startswith("radial."):
                        kvalue = getattr(
                            self.radial, key.replace("radial.", "")
                        )
                    else:
                        kvalue = getattr(self, key)

                    if kvalue is None:
                        kvalue = None
                    elif isinstance(
                        kvalue, numpyro.distributions.Distribution
                    ):
                        kvalue = f"Distribution: {kvalue.__class__.__name__}"
                    elif isinstance(
                        kvalue,
                        (types.LambdaType, types.FunctionType, _BoundTo),
                    ):
                        kvalue = "Tied parameter"
                    else:
                        kvalue = f"{kvalue:.4E}"

                    print(
                        f"{key:<{keylen}} [{self.units[key]}] : "
                        + f"{kvalue}".ljust(10)
                        + f" | {self.description[key]}"
                    )

    def parlist(self):
        """
        Return list of parameter names from the radial component and Bar itself.

        Returns
        -------
        list of str
            Combined list of parameter names: radial's shape parameters
            (prefixed 'radial.') plus Bar's own xc, yc, theta, e, inc, rot,
            losdepth, losbins.

        Notes
        -----
        This method is used internally by Model.addcomponent() when adding
        a Bar component to a composite model, ensuring all parameters are
        registered.

        Examples
        --------
        >>> from socca.models import Bar, Sersic
        >>> bar = Bar(radial=Sersic(re=2e-4, Ie=10.0, ns=0.25))
        >>> bar.parlist()
        ['radial.re', 'radial.Ie', 'radial.ns', 'xc', 'yc', 'theta', 'e',
         'inc', 'rot', 'losdepth', 'losbins']
        """
        return list(self.units.keys())
