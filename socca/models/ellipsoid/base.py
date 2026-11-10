"""Ellipsoid model with 3D line-of-sight integration."""

from functools import partial

import warnings

import jax
import jax.numpy as jp


from .. import config
from ..base import (
    Component,
    _claim,
    _strip,
    _subkeymap,
    _subprofile,
    _warn_range,
)
from ..radial import Sersic
from ..radial.base import Profile, _safe_power


class Ellipsoid(Component):
    """
    3D triaxial-ellipsoid model evaluated via line-of-sight integration.

    Projects an ellipsoidal density onto the sky by integrating
    along the line of sight, in the same spirit as Disk but using a single
    radial profile for shape rather than separate radial/vertical profiles.
    A common use case is modeling a galactic bar embedded within a host
    Disk, with its position, position angle, and inclination tied to the
    disk's via boundto() -- see the Examples below and
    socca.priors.boundto(). xc, yc, theta, e, eratio, inc, rot, losdepth,
    and losbins live directly on Ellipsoid; radial supplies the density
    profile, evaluated on the ellipsoidal radius
    sqrt(x^2 + (y / (1 - e))^2 + (z / (1 - eratio * e))^2), and its shape
    parameters (e.g. re, Ie, ns for Sersic).

    The ellipsoid has one major semi-axis (along x) and two minor
    semi-axes (along y and z), which need not be equal: e sets the
    ellipticity in the (x, y) plane -- the plane that corresponds to the
    projected ellipse when the ellipsoid is seen face-on (inc = 0) -- and
    eratio sets the ratio of the z axis's ellipticity to e, so the two
    minor axes can differ.

    Parameters
    ----------
    radial : Profile, optional
        Radial profile defining the ellipsoid's density and its shape
        parameters (e.g. re, Ie, ns for Sersic). Any radial profile can be
        used, with its truncation if any. Default is Sersic(). Any
        xc/yc/theta/e already set on it are inherited by Ellipsoid unless
        also passed as an Ellipsoid keyword argument, in which case
        Ellipsoid's value wins and a warning is raised.
    xc, yc : float, optional
        Right ascension and declination of the ellipsoid centroid (deg).
    theta : float, optional
        Position angle of the line of nodes, east from north (rad), as for
        Disk. With rot = 0 it is also the position angle of the major axis.
    e : float, optional
        Ellipticity (1 - axis ratio) between the major (x) axis and the
        first minor (y) axis -- i.e. the ellipticity of the face-on
        projected ellipse.
    eratio : float, optional
        Ratio in [0, 1] of the second minor (z) axis's ellipticity to e:
        z-axis ellipticity = eratio * e. Default is 1, which makes both
        minor axes equal (a prolate/oblate spheroid rather than a fully
        triaxial ellipsoid).
    inc : float, optional
        Inclination angle (0 = face-on). Typically tied to a host Disk's
        inclination via boundto().
    rot : float, optional
        Rotation of the major axis away from the line of nodes, applied
        after inclining, within the ellipsoid's own tilted frame. With
        rot = 0 the major axis lies along the line of nodes, so theta is
        also its position angle. Only equivalent to a position-angle
        offset (i.e. theta + rot) when inc = 0.
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
    >>> from socca.models import Ellipsoid, Sersic
    >>> ellipsoid = Ellipsoid(radial=Sersic(re=2e-4, Ie=10.0, ns=0.25))
    >>> ellipsoid.xc, ellipsoid.yc, ellipsoid.theta, ellipsoid.e = (
    ...     180.5, 45.2, 0.5, 0.7
    ... )
    >>> ellipsoid.inc, ellipsoid.rot = 1.0, 0.2

    Tying a galactic bar's geometry to that of a host Disk:

    >>> from socca.models import Disk
    >>> from socca.priors import boundto
    >>> disk = Disk()
    >>> bar = Ellipsoid()
    >>> bar.xc = boundto(disk, "xc")
    >>> bar.yc = boundto(disk, "yc")
    >>> bar.theta = boundto(disk, "theta")
    >>> bar.inc = boundto(disk, "inc")
    """

    _periods = {"theta": jp.pi, "rot": jp.pi}

    def __init__(self, radial=None, **kwargs):
        super().__init__(**kwargs)

        # radial defaults to a fresh Sersic() per call, not a shared
        # mutable default -- __init__ strips xc/yc/theta/e/cbox off
        # radial below, which would otherwise corrupt a single
        # module-level-evaluated default shared across every Ellipsoid()
        # call that doesn't pass its own radial.
        radial = Sersic() if radial is None else radial
        if not isinstance(radial, Profile):
            raise TypeError(
                f"{type(radial).__name__} is not supported for Ellipsoid: "
                "use a radial profile (e.g. Sersic)."
            )
        _claim(radial, free=["cbox"])
        # checked before the radial profile is changed
        self.eratio = kwargs.get("eratio", config.Ellipsoid.eratio)
        self.inc = kwargs.get("inc", config.Ellipsoid.inc)
        self.rot = kwargs.get("rot", config.Ellipsoid.rot)
        self.losdepth = kwargs.get("losdepth", config.Ellipsoid.losdepth)
        self.losbins = kwargs.get("losbins", config.Ellipsoid.losbins)
        self.radial = radial

        # xc/yc/theta/e live on Ellipsoid itself, not on radial. If the
        # caller set one of them on radial (before passing it in) and
        # didn't also pass it to Ellipsoid, inherit radial's value; if
        # both were set, Ellipsoid's value wins and radial's is ignored
        # (with a warning).
        for key in ["xc", "yc", "theta", "e"]:
            radial_val = getattr(self.radial, key)
            radial_isset = radial_val != getattr(config.Profile, key)

            if key in kwargs:
                if radial_isset:
                    warnings.warn(
                        f"Ellipsoid's '{key}' and radial's '{key}' were "
                        f"both set; radial's value will be ignored in "
                        f"favor of the value passed to Ellipsoid.",
                        UserWarning,
                        stacklevel=2,
                    )
                setattr(self, key, kwargs[key])
            elif radial_isset:
                setattr(self, key, radial_val)
                if key in self.radial._input_units:
                    self._input_units[key] = self.radial._input_units[key]
            else:
                setattr(self, key, getattr(config.Ellipsoid, key))

        _strip(self.radial, ["xc", "yc", "theta", "e", "cbox"])

        self._namespaces = {"radial": self.radial}
        self._wrap()

        self.hyper += ["losdepth", "losbins"]

        self._adopt()
        self.units.update(
            dict(
                xc="deg",
                yc="deg",
                theta="rad",
                e="",
                eratio="",
                inc="rad",
                rot="rad",
                losdepth="deg",
                losbins="",
            )
        )

        self.description.update(
            dict(
                xc="Right ascension of centroid",
                yc="Declination of centroid",
                theta="Position angle of the line of nodes (east from north)",
                e="Ellipticity between the major axis and the first (in-plane) minor axis",
                eratio="Ratio of the second (vertical) minor axis's ellipticity to e",
                inc="Inclination angle (0=face-on); Typically inherited from a Disk object",
                rot="Rotation of the major axis from the line of nodes, in the tilted frame",
                losdepth="Half line-of-sight extent for integration",
                losbins="Number of points for line-of-sight integration",
            )
        )

        self._initialized = True

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
            note="This might lead to unphysical models.",
        )
        self._e = value

    @property
    def eratio(self):  # noqa: D102
        return self._eratio

    @eratio.setter
    def eratio(self, value):  # noqa: D102
        _warn_range(
            value,
            "eratio",
            lower=0,
            upper=1,
            note="This might lead to unphysical models.",
        )
        self._eratio = value

    @property
    def theta(self):  # noqa: D102
        return self._theta

    @theta.setter
    def theta(self, value):  # noqa: D102
        self._theta = value

    @property
    def rot(self):  # noqa: D102
        return self._rot

    @rot.setter
    def rot(self, value):  # noqa: D102
        self._rot = value

    def getmap(self, img, convolve=False):
        """
        Generate ellipsoid image via 3D line-of-sight integration.

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
            Projected ellipsoid image on the image grid.

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
        (number of points); increase both for highly inclined ellipsoids.

        Examples
        --------
        >>> from socca.models import Ellipsoid, Sersic
        >>> from socca.data import Image
        >>> ellipsoid = Ellipsoid(radial=Sersic(re=2e-4, Ie=10.0, ns=0.25))
        >>> ellipsoid.xc, ellipsoid.yc, ellipsoid.theta, ellipsoid.e = (
        ...     180.5, 45.2, 0.5, 0.7
        ... )
        >>> ellipsoid.inc, ellipsoid.rot = 1.0, 0.2
        >>> img = Image('observation.fits')
        >>> ellipsoid_map = ellipsoid.getmap(img, convolve=True)
        """
        kwarg = self._getpars(self._keymap())
        return self._convolve(img, self._evaluate(img, **kwarg), convolve)

    def _keymap(self):
        """Map the keywords of _evaluate to parameter names."""
        keys = _subkeymap(self.radial, "r", "radial")
        keys.update(
            {
                key: key
                for key in [
                    "xc",
                    "yc",
                    "theta",
                    "e",
                    "eratio",
                    "inc",
                    "rot",
                    "losdepth",
                    "losbins",
                ]
            }
        )
        return keys

    def profile(self, xt, yt, zt, e, eratio, **kw):
        """Evaluate the radial profile on the ellipsoidal radius."""
        r = _safe_power(
            xt**2 + (yt / (1 - e)) ** 2 + (zt / (1 - eratio * e)) ** 2, 0.50
        )
        return _subprofile(self.radial, r, kw, "r")

    def _evaluate(self, img, **kwarg):
        """
        Evaluate ellipsoid model on the given grid with explicit parameters.

        This internal method computes the projected ellipsoid surface
        brightness via line-of-sight integration using the provided
        geometric and profile parameters. It is used by both getmap() and
        Model.getmodel() to avoid code duplication.

        Parameters
        ----------
        img : Image
            Image object containing grid and WCS information.
        **kwarg : dict
            All parameters, including geometric (xc, yc, theta, e, eratio,
            inc, rot, losdepth, losbins) and radial profile-specific ones
            (re, Ie, ns).

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
        mgrid = self._compiled()(xt, yt, zt, **kwarg)
        mgrid = jp.trapezoid(mgrid, dx=dx, axis=1)
        return jp.mean(mgrid, axis=0)

    @staticmethod
    @partial(jax.jit, static_argnames=["grid", "losbins"])
    def getgrid(
        grid, xc, yc, losdepth, losbins=200, theta=0.0, inc=0.0, rot=0.0
    ):
        """
        Compute 3D ellipsoid coordinates with position angle, inclination, and rotation.

        Generates 3D coordinate grids (x, y, z) for ellipsoid model
        evaluation, accounting for the position angle, inclination, and
        intrinsic rotation transformations needed for line-of-sight
        integration through an inclined, rotated ellipsoid.

        Parameters
        ----------
        grid : Grid
            Grid object with .x and .y celestial coordinate arrays (deg).
        xc : float
            Right ascension of ellipsoid center (deg).
        yc : float
            Declination of ellipsoid center (deg).
        losdepth : float
            Half-extent of line-of-sight integration (deg).
        losbins : int, optional
            Number of integration points along line of sight. Default is 200.
        theta : float, optional
            Position angle of the line of nodes, east from north (rad), as
            for Disk. Default is 0.
        inc : float, optional
            Inclination angle (0 = face-on, pi/2 = edge-on) (rad). Default is 0.
        rot : float, optional
            Rotation of the major axis away from the line of nodes, within
            the already-inclined frame (rad). Default is 0. Only equivalent
            to a position-angle offset (i.e. theta + rot) when inc = 0 --
            see Notes.

        Returns
        -------
        xt, yt, zt : ndarray
            4D arrays of ellipsoid-frame coordinates (deg).
            Shape: (ssize, losbins, ysize, xsize).

        Notes
        -----
        The transformation sequence:
        1. Center coordinates on (xc, yc) and apply spherical geometry correction (cos(dec))
        2. Rotate by position angle theta, so that the tilt (y) axis is the line of nodes, as for Disk
        3. Create the line-of-sight grid from -losdepth to +losdepth and broadcast to 4D
        4. Incline around the y-axis, mixing the position-angle-rotated x with the line-of-sight coordinate
        5. Rotate by rot + pi/2 within that already-inclined (x, y) frame, so that rot = 0 aligns the major axis with the line of nodes

        Because step 4 mixes the line-of-sight coordinate into x before
        step 5 runs, rot is not simply an additive offset to theta except
        in the face-on case (inc = 0): for inc != 0, rotating by rot after
        inclining is a genuinely different transform from inclining after
        rotating by theta + rot.
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

        # Rotate by rot in the already-inclined (x, y) frame -- xt already
        # carries a line-of-sight (z) contribution from the inclination
        # step above, so this is only equivalent to a theta offset when
        # inc == 0.
        sinr = jp.sin(rot + 0.5 * jp.pi)
        cosr = jp.cos(rot + 0.5 * jp.pi)
        xt, yt = xt * cosr + yt * sinr, -xt * sinr + yt * cosr

        return xt, yt, zt
