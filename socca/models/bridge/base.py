"""Bridge/filament model for intracluster structures.

This module provides models for describing elongated emission structures
such as intracluster bridges and filaments connecting galaxy clusters.
"""

import numbers
from functools import partial


import jax
import jax.numpy as jp


from ..base import (
    Component,
    _claim,
    _strip,
    _subkeymap,
    _subprofile,
    _warn_range,
)
from ..radial import Beta, PolyExpoRefact, Power, TopHat
from ..radial.base import Profile
from .. import config
from ...priors import _Function, _identity, function


def _minor(rs, e):
    return rs * (1.00 - e)


class Bridge(Component):
    """
    Base class for bridge/filament emission models.

    The Bridge model describes elongated emission structures by combining
    a radial profile (perpendicular to the bridge axis) with a parallel
    profile (along the bridge axis). This factorized approach allows for
    flexible modeling of intracluster bridges and large-scale filaments.

    Parameters
    ----------
    radial : Profile, optional
        Profile describing the surface brightness distribution perpendicular
        to the bridge major axis. Default is Beta().
    parallel : Profile, optional
        Profile describing the surface brightness distribution along the
        bridge major axis. Default is TopHat().

        Both need a scale radius and an amplitude (e.g. rc and Ic for
        Beta), which the Bridge sets from rs, e and Is; their geometric
        parameters (xc, yc, theta, e, cbox) are dropped.
    **kwargs : dict
        Additional keyword arguments including:

        xc : float, optional
            Right ascension of the bridge centroid in degrees.
        yc : float, optional
            Declination of the bridge centroid in degrees.
        rs : float, optional
            Scale radius for both radial and parallel profiles in degrees.
        Is : float, optional
            Scale intensity of the bridge (same units as image).
        theta : float, optional
            Position angle of the bridge major axis in radians
            (measured east from north).
        e : float, optional
            Projected ellipticity of the bridge profiles: the radial
            profile's scale radius is rs * (1 - e). Default is 0.5.

    Attributes
    ----------
    radial : Profile
        The radial profile component.
    parallel : Profile
        The parallel profile component.

    Raises
    ------
    TypeError
        If a profile is a PolyExpoRefact, is not a Profile, or has no
        scale radius and amplitude.
    ValueError
        If a profile is already in a model or part of another Disk,
        Ellipsoid or Bridge, or the same profile is passed twice.

    Notes
    -----
    The Bridge class is an abstract base that defines the interface and
    parameter handling. Use SimpleBridge or MesaBridge for concrete
    implementations with specific profile combination rules.

    The scale radius `rs` is shared between the parallel profile (where
    it is used directly) and the radial profile (where it is scaled by
    the factor `1 - e` to control the axis ratio). Setting a sub-profile's
    scale radius or amplitude raises at getmap and when fitting.

    The profiles are evaluated on the absolute distances from the bridge
    axis and along it, and their truncations, if any, act on these
    distances. The sub-profiles cannot be replaced: create a new Bridge.

    See Also
    --------
    SimpleBridge : Multiplicative combination of radial and parallel profiles.
    MesaBridge : Harmonic mean combination for mesa-like profiles.
    """

    _periods = {"theta": jp.pi}

    def __init__(self, radial=None, parallel=None, **kwargs):
        super().__init__(**kwargs)

        # radial/parallel default to fresh instances per call, not shared
        # mutable defaults -- __init__ strips their native geometric and
        # scale-radius/amplitude attributes below, which would otherwise
        # corrupt a single module-level-evaluated default shared across
        # every Bridge() call that doesn't pass its own radial/parallel.
        if radial is None:
            radial = Beta()
        if parallel is None:
            parallel = TopHat()
        for attr in [radial, parallel]:
            if isinstance(attr, PolyExpoRefact):
                raise TypeError(
                    "PolyExpoRefact is not supported for Bridge. "
                    "Use PolyExponential or another profile type instead."
                )
            if not isinstance(attr, Profile):
                raise TypeError(
                    f"{type(attr).__name__} is not supported for Bridge."
                )
            if attr._scale_radius is None or attr._scale_amp is None:
                raise TypeError(
                    f"{type(attr).__name__} has no scale radius and "
                    "amplitude, so it is not supported for Bridge."
                )

        _claim(
            radial,
            parallel,
            free=["xc", "yc", "theta", "e", "cbox"]
            + [sub._scale_radius for sub in (radial, parallel)]
            + [sub._scale_amp for sub in (radial, parallel)],
        )

        self.xc = kwargs.get("xc", config.Bridge.xc)
        self.yc = kwargs.get("yc", config.Bridge.yc)
        self.rs = kwargs.get("rs", config.Bridge.rs)
        self.Is = kwargs.get("Is", config.Bridge.Is)
        self.theta = kwargs.get("theta", config.Bridge.theta)
        self.e = kwargs.get("e", config.Bridge.e)

        self.units.update(
            dict(
                xc="deg",
                yc="deg",
                rs="deg",
                Is="image",
                theta="rad",
                e="",
            )
        )
        self.description.update(
            dict(
                xc="Right ascension of the bridge centroid",
                yc="Declination of the bridge centroid",
                rs="Scale radius of the bridge profiles",
                Is="Scale intensity of the bridge profiles",
                theta="Position angle of the bridge major axis",
                e="Projected ellipticity of the bridge profiles",
            )
        )

        self.radial = radial
        self.parallel = parallel

        for attr in [self.radial, self.parallel]:
            _strip(
                attr,
                ["xc", "yc", "theta", "e", "cbox"]
                + [attr._scale_amp, attr._scale_radius],
            )

        for attr in [self.parallel, self.radial]:
            attr.addparameter(attr._scale_amp, 1.00)

        self.parallel.addparameter(self.parallel._scale_radius)
        self.radial.addparameter(self.radial._scale_radius)
        self._scale_ties()
        self._namespaces = {"radial": self.radial, "parallel": self.parallel}
        self._wrap()
        self._adopt(hidden=["_scale_radius", "_scale_amp"])

    def _keymap(self):
        """Map the keywords of _evaluate to parameter names."""
        keys = _subkeymap(self.radial, "r", "radial")
        keys.update(_subkeymap(self.parallel, "z", "parallel"))
        keys.update({key: key for key in ["xc", "yc", "Is", "theta"]})
        return keys

    @property
    def rs(self):  # noqa: D102
        return self._rs

    @rs.setter
    def rs(self, value):  # noqa: D102
        _warn_range(
            value,
            "rs",
            lower=0,
            lower_strict=True,
            note="The scale radius must be strictly positive.",
        )
        self._rs = value

    @property
    def theta(self):  # noqa: D102
        return self._theta

    @theta.setter
    def theta(self, value):  # noqa: D102
        self._theta = value

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
            note="Ellipticity is typically in [0, 1).",
        )
        self._e = value

    def getmap(self, img, convolve=False):
        """
        Generate a two-dimensional model image of the bridge.

        Parameters
        ----------
        img : Image
            Image object defining the coordinate grid and optional PSF.
        convolve : bool, optional
            If True, convolve the model with the image PSF. Default is False.

        Returns
        -------
        ndarray
            Two-dimensional model image evaluated on the image grid.

        Raises
        ------
        ValueError
            If any model parameter is set to None or contains a prior
            distribution instead of a fixed value, or if a sub-profile's
            scale radius or amplitude was set (set rs, e and Is instead).

        Notes
        -----
        The model is computed by evaluating the combined radial and parallel
        profiles on a rotated coordinate grid centered on the bridge position.
        """
        self._check_internal()
        kwarg = self._getpars(self._keymap())
        return self._convolve(img, self._evaluate(img, **kwarg), convolve)

    def _check_internal(self):
        """Reject values set on the sub-profiles' scale radius or amplitude."""
        for sub in (self.radial, self.parallel):
            rs = getattr(sub, sub._scale_radius)
            amp = getattr(sub, sub._scale_amp)
            tied = isinstance(rs, _Function) and rs.native
            if not tied or not (isinstance(amp, numbers.Real) and amp == 1):
                raise ValueError(
                    f"{type(sub).__name__}.{sub._scale_radius} and "
                    f".{sub._scale_amp} are set by the {type(self).__name__} "
                    "from rs, e and Is: set those instead."
                )

    @staticmethod
    @partial(jax.jit, static_argnames=["grid"])
    def getgrid(grid, xc, yc, theta):
        """
        Compute rotated coordinate grids for bridge evaluation.

        Parameters
        ----------
        grid : WCSgrid
            Coordinate grid from the Image object.
        xc : float
            Right ascension of bridge centroid in degrees.
        yc : float
            Declination of bridge centroid in degrees.
        theta : float
            Position angle of bridge major axis in radians.

        Returns
        -------
        rgrid : ndarray
            Absolute distance from the bridge axis, |r| (deg).
        zgrid : ndarray
            Absolute distance along the bridge axis from the centroid,
            |z| (deg).

        Notes
        -----
        The transformation accounts for spherical coordinate projection
        using the cosine of the declination at the bridge centroid.
        """
        sint = jp.sin(theta)
        cost = jp.cos(theta)

        zgrid = (
            -(grid.x - xc) * jp.cos(jp.deg2rad(yc)) * sint
            - (grid.y - yc) * cost
        )
        rgrid = (grid.x - xc) * jp.cos(jp.deg2rad(yc)) * cost - (
            grid.y - yc
        ) * sint

        return jp.abs(rgrid), jp.abs(zgrid)

    def _scale_ties(self):
        """Tie the sub-profiles' scale radii to rs and e, in native units."""
        ties = {
            self.parallel: function(_identity, (self, "rs")),
            self.radial: function(_minor, (self, "rs"), (self, "e")),
        }
        for sub, tie in ties.items():
            tie.native = True
            setattr(sub, sub._scale_radius, tie)

    @staticmethod
    def _internal(sub, name, pars, comp_prefix):
        """Value of a sub-profile's internal scale radius or amplitude."""
        if name not in (sub._scale_radius, sub._scale_amp):
            raise KeyError(f"{comp_prefix}: {name} is not in the model.")
        value = getattr(sub, name)
        if not isinstance(value, _Function):
            return value
        return value(
            *[pars[f"{comp_prefix}_{key}"] for _, key in value.sources]
        )

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
            Keyword arguments for _evaluate including geometric and profile
            parameters with r_ and z_ prefixes.
        """
        self._check_internal()
        kwarg = {}
        for out, key in self._keymap().items():
            if f"{comp_prefix}_{key}" in pars:
                kwarg[out] = pars[f"{comp_prefix}_{key}"]
            else:
                sub, name = key.split(".")
                kwarg[out] = self._internal(
                    getattr(self, sub), name, pars, comp_prefix
                )
        return kwarg

    def _evaluate(self, img, **kwarg):
        """
        Evaluate bridge model on the given grid with explicit parameters.

        This internal method computes the bridge surface brightness using
        the provided geometric and profile parameters. It is used by both
        getmap() and Model.getmodel() to avoid code duplication.

        Parameters
        ----------
        img : Image
            Image object containing grid and WCS information.
        **kwarg : dict
            All parameters including geometric (xc, yc, theta) and
            profile-specific parameters with r_ and z_ prefixes.

        Returns
        -------
        ndarray
            2D array of surface brightness values, averaged over subpixels.
        """
        xc = kwarg.pop("xc")
        yc = kwarg.pop("yc")
        theta = kwarg.pop("theta")

        rgrid, zgrid = self.getgrid(img.grid, xc, yc, theta)
        mgrid = self._compiled()(rgrid, zgrid, **kwarg)
        return jp.mean(mgrid, axis=0)


class SimpleBridge(Bridge):
    """
    Simple bridge model with multiplicative profile combination.

    The SimpleBridge combines radial and parallel profiles multiplicatively,
    producing a surface brightness distribution of the form:

        I(r, z) = Is * f_radial(r) * f_parallel(z)

    This creates structures where the emission is the product of the two
    independent profile functions.

    Parameters
    ----------
    radial : Profile, optional
        Profile describing emission perpendicular to the bridge axis.
        Default is Beta().
    parallel : Profile, optional
        Profile describing emission along the bridge axis.
        Default is TopHat(), producing a uniform distribution along
        the bridge length.
    **kwargs : dict
        Additional keyword arguments passed to Bridge base class.

    See Also
    --------
    Bridge : Base class with parameter descriptions.
    MesaBridge : Alternative with mesa-like profile combination.

    Examples
    --------
    >>> from socca.models import SimpleBridge
    >>> bridge = SimpleBridge()
    >>> bridge.parameters()
    <BLANKLINE>
    SimpleBridge
    ============
    xc        [deg] : None       | Right ascension of the bridge centroid
    yc        [deg] : None       | Declination of the bridge centroid
    rs        [deg] : None       | Scale radius of the bridge profiles
    Is      [image] : None       | Scale intensity of the bridge profiles
    theta     [rad] : 0.0000E+00 | Position angle of the bridge major axis
    e            [] : 5.0000E-01 | Projected ellipticity of the bridge profiles
    radial.alpha [] : 2.0000E+00 | Radial exponent
    radial.beta  [] : 5.5000E-01 | Slope parameter
    """

    def __init__(self, radial=None, parallel=None, **kwargs):
        super().__init__(radial=radial, parallel=parallel, **kwargs)

        self._initialized = True

    def profile(self, r, z, Is, **kw):
        """Evaluate Is * f_r(r) * f_z(z)."""
        return (
            Is
            * _subprofile(self.radial, r, kw, "r")
            * _subprofile(self.parallel, z, kw, "z")
        )


class MesaBridge(Bridge):
    """
    Mesa bridge model with harmonic mean profile combination.

    The MesaBridge combines radial and parallel profiles using a harmonic
    mean, producing a mesa-like (flat-topped) surface brightness distribution:

        I(r, z) = Is / (1/f_radial(r) + 1/f_parallel(z))

    This creates smooth transitions between the flat central region and
    the declining edges, resembling a mesa or table-top shape.

    Parameters
    ----------
    radial : Profile, optional
        Profile for perpendicular direction. Default is Beta with
        alpha=8.0 and beta=1.0 for steep edges.
    parallel : Profile, optional
        Profile for parallel direction. Default is Power with
        alpha=8.0 for steep drop-off.
    **kwargs : dict
        Additional keyword arguments passed to Bridge base class.

    Notes
    -----
    The default parameters are chosen to produce a characteristic
    mesa-like shape with steep edges, suitable for modeling intracluster
    bridges with relatively uniform central emission.

    See Also
    --------
    Bridge : Base class with parameter descriptions.
    SimpleBridge : Alternative with multiplicative profile combination.

    References
    ----------
    Hincks, A. D., et al., MNRAS, 510, 3335 (2022)
    https://scixplorer.org/abs/2022MNRAS.510.3335H/abstract

    Examples
    --------
    >>> from socca.models import MesaBridge
    >>> bridge = MesaBridge()
    >>> bridge.parameters()
    <BLANKLINE>
    MesaBridge
    ==========
    xc          [deg] : None       | Right ascension of the bridge centroid
    yc          [deg] : None       | Declination of the bridge centroid
    rs          [deg] : None       | Scale radius of the bridge profiles
    Is        [image] : None       | Scale intensity of the bridge profiles
    theta       [rad] : 0.0000E+00 | Position angle of the bridge major axis
    e              [] : 5.0000E-01 | Projected ellipticity of the bridge profiles
    radial.alpha   [] : 8.0000E+00 | Radial exponent
    radial.beta    [] : 1.0000E+00 | Slope parameter
    parallel.alpha [] : 8.0000E+00 | Power law slope
    """

    def __init__(self, radial=None, parallel=None, **kwargs):
        if radial is None:
            radial = Beta(
                alpha=config.MesaBridge.r_alpha,
                beta=config.MesaBridge.r_beta,
            )

        if parallel is None:
            parallel = Power(
                alpha=config.MesaBridge.z_alpha,
            )

        super().__init__(radial=radial, parallel=parallel, **kwargs)

        self._initialized = True

    def profile(self, r, z, Is, **kw):
        """Evaluate Is / (1 / f_r(r) + 1 / f_z(z))."""
        return Is / (
            1.00 / _subprofile(self.radial, r, kw, "r")
            + 1.00 / _subprofile(self.parallel, z, kw, "z")
        )
