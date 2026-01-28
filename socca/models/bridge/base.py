"""Bridge/filament model for intracluster structures.

This module provides models for describing elongated emission structures
such as intracluster bridges and filaments connecting galaxy clusters.
"""

from functools import partial
import inspect
import types

import warnings

import jax
import jax.numpy as jp
import numpyro.distributions

import numpy as np

from ..base import Component
from ..misc import Background, Point
from ..radial import Beta, PolyExpoRefact, Power, TopHat
from .. import config


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
            Axis ratio controlling the relative extent of radial vs
            parallel profiles. Default is 0.5.

    Attributes
    ----------
    radial : Profile
        The radial profile component.
    parallel : Profile
        The parallel profile component.

    Notes
    -----
    The Bridge class is an abstract base that defines the interface and
    parameter handling. Use SimpleBridge or MesaBridge for concrete
    implementations with specific profile combination rules.

    The scale radius `rs` is shared between the parallel profile (where
    it is used directly) and the radial profile (where it is scaled by
    the factor `1 - e` to control the axis ratio).

    See Also
    --------
    SimpleBridge : Multiplicative combination of radial and parallel profiles.
    MesaBridge : Harmonic mean combination for mesa-like profiles.
    """

    def __init__(self, radial=Beta(), parallel=TopHat(), **kwargs):
        super().__init__(**kwargs)
        for key in ["radial", "parallel", "profile", "_rkw", "_zkw"]:
            self.okeys.append(key)

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
                e="Axis ratio of the bridge profiles",
            )
        )

        # Inherited parameters
        # ------------------------------

        self.radial = radial
        self.parallel = parallel

        for attr in [self.radial, self.parallel]:
            if isinstance(attr, PolyExpoRefact):
                raise TypeError(
                    "PolyExpoRefact is not supported for Bridge. "
                    "Use PolyExponential or another profile type instead."
                )
            if isinstance(attr, (Background, Point)):
                raise TypeError(
                    f"{type(attr).__name__} is not supported for Bridge."
                )

        for key in ["xc", "yc", "theta", "e", "cbox"]:
            if hasattr(self.radial, key):
                delattr(self.radial, key)
            if hasattr(self.parallel, key):
                delattr(self.parallel, key)

        for key in ["_scale_amp", "_scale_radius"]:
            delattr(self.radial, getattr(self.radial, key))
            delattr(self.parallel, getattr(self.parallel, key))

        if self.radial.id != self.id:
            type(self).idcls -= 1
            idmin = np.minimum(
                int(self.radial.id.replace("comp_", "")),
                int(self.id.replace("comp_", "")),
            )
            self.id = f"comp_{idmin:02d}"
            self.radial.id = self.id

        for attr in [self.parallel, self.radial]:
            attr.addparameter(attr._scale_amp, 1.00)

        self.parallel.addparameter(
            self.parallel._scale_radius,
            eval(f"lambda {self.id}_rs: {self.id}_rs"),
        )

        self.radial.addparameter(
            self.radial._scale_radius,
            eval(
                f"lambda {self.id}_rs, {self.id}_e: "
                f"{self.id}_rs * (1.00 - {self.id}_e)"
            ),
        )

        self._rkw = [
            f"r_{key}"
            for key in list(
                inspect.signature(self.radial.profile).parameters.keys()
            )
            if key not in ["r", "z"]
        ]
        self._zkw = [
            f"z_{key}"
            for key in list(
                inspect.signature(self.parallel.profile).parameters.keys()
            )
            if key not in ["r", "z"]
        ]

        for key in self.radial.hyper:
            self.hyper.append(f"radial.{key}")

        for key in self.parallel.hyper:
            self.hyper.append(f"parallel.{key}")

        self.units.update(
            {
                f"radial.{key}": self.radial.units[key]
                for key in self.radial.units.keys()
                if key
                not in [self.radial._scale_radius, self.radial._scale_amp]
            }
        )
        self.units.update(
            {
                f"parallel.{key}": self.parallel.units[key]
                for key in self.parallel.units.keys()
                if key
                not in [self.parallel._scale_radius, self.parallel._scale_amp]
            }
        )

        self.description.update(
            {
                f"radial.{key}": self.radial.description[key]
                for key in self.radial.description.keys()
                if key
                not in [self.radial._scale_radius, self.radial._scale_amp]
            }
        )
        self.description.update(
            {
                f"parallel.{key}": self.parallel.description[key]
                for key in self.parallel.description.keys()
                if key
                not in [self.parallel._scale_radius, self.parallel._scale_amp]
            }
        )

        self.profile = None

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
            distribution instead of a fixed value.

        Notes
        -----
        The model is computed by evaluating the combined radial and parallel
        profiles on a rotated coordinate grid centered on the bridge position.
        """
        kwarg = {}
        for key in list(inspect.signature(self.profile).parameters.keys()):
            if key not in ["r", "z"]:
                if key.startswith("r_"):
                    val = getattr(self.radial, key.replace("r_", ""))
                elif key.startswith("z_"):
                    val = getattr(self.parallel, key.replace("z_", ""))
                else:
                    continue

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

        rgrid, zgrid = self.getgrid(img.grid, self.xc, self.yc, self.theta)

        mgrid = self.profile(rgrid, zgrid, **kwarg)
        mgrid = jp.mean(mgrid, axis=0)

        if convolve:
            if img.psf is None:
                warnings.warn(
                    "No PSF defined, so no convolution will be performed."
                )
            else:
                mgrid = (
                    jp.fft.rfft2(jp.fft.fftshift(mgrid), s=img.data.shape)
                    * img.psf_fft
                )
                mgrid = jp.fft.ifftshift(
                    jp.fft.irfft2(mgrid, s=img.data.shape)
                ).real
        return mgrid

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
            Coordinate grid along the radial (perpendicular) direction.
        zgrid : ndarray
            Coordinate grid along the parallel (major axis) direction.

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

        return rgrid, zgrid

    def parameters(self):
        """
        Print a summary of all model parameters.

        Displays model parameters organized by category (base, radial,
        parallel) along with their units, current values, and descriptions.
        Hyperparameters are shown in a separate section.
        """
        keyout = []
        for key in self.__dict__.keys():
            if (
                key not in self.okeys
                and key not in self.hyper
                and key != "okeys"
            ):
                keyout.append(key)
        for key in self.radial.__dict__.keys():
            if (
                key not in self.okeys
                and f"radial.{key}" not in self.hyper
                and key != "okeys"
                and key
                not in [self.radial._scale_radius, self.radial._scale_amp]
            ):
                keyout.append(f"radial.{key}")
        for key in self.parallel.__dict__.keys():
            if (
                key not in self.okeys
                and f"parallel.{key}" not in self.hyper
                and key != "okeys"
                and key
                not in [self.parallel._scale_radius, self.parallel._scale_amp]
            ):
                keyout.append(f"parallel.{key}")

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
                    kvalue = self.radial.__dict__[key.replace("radial.", "")]
                elif key.startswith("parallel."):
                    kvalue = self.parallel.__dict__[
                        key.replace("parallel.", "")
                    ]
                else:
                    kvalue = self.__dict__[key]

                if kvalue is None:
                    kvalue = None
                elif isinstance(kvalue, numpyro.distributions.Distribution):
                    kvalue = f"Distribution: {kvalue.__class__.__name__}"
                elif isinstance(
                    kvalue, (types.LambdaType, types.FunctionType)
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
                        kvalue = self.radial.__dict__[
                            key.replace("radial.", "")
                        ]
                    elif key.startswith("parallel."):
                        kvalue = self.parallel.__dict__[
                            key.replace("parallel.", "")
                        ]
                    else:
                        kvalue = self.__dict__[key]
                    kvalue = None if kvalue is None else f"{kvalue:.4E}"
                    print(
                        f"{key:<{keylen}} [{self.units[key]}] : "
                        + f"{kvalue}".ljust(10)
                        + f" | {self.description[key]}"
                    )
        else:
            print("No parameters defined.")

    def parlist(self):
        """
        Return a list of all parameter names.

        Returns
        -------
        list of str
            Parameter names including base, radial, and parallel parameters.
        """
        pars_ = [key for key in self.__dict__.keys() if key not in self.okeys]
        pars_ += [
            f"radial.{key}"
            for key in self.radial.__dict__.keys()
            if key not in self.okeys and key != "okeys"
        ]
        pars_ += [
            f"parallel.{key}"
            for key in self.parallel.__dict__.keys()
            if key not in self.okeys and key != "okeys"
        ]
        return pars_


class SimpleBridge(Bridge):
    """
    Simple bridge model with multiplicative profile combination.

    The SimpleBridge combines radial and parallel profiles multiplicatively,
    producing a surface brightness distribution of the form:

        I(r, z) = f_radial(r) * f_parallel(z)

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

    Model parameters
    ================
    xc              [deg] : None       | Right ascension of bridge centroid
    yc              [deg] : None       | Declination of bridge centroid
    rs              [deg] : None       | Scale radius of the bridge profiles
    Is            [image] : None       | Scale intensity of the bridge
    theta           [rad] : 0.0000E+00 | Position angle of bridge major axis
    e                  [] : 5.0000E-01 | Axis ratio of the bridge profiles
    radial.alpha       [] : 2.0000E+00 | Radial exponent
    radial.beta        [] : 5.5000E-01 | Slope parameter
    """

    def __init__(self, radial=Beta(), parallel=TopHat(), **kwargs):
        super().__init__(radial=radial, parallel=parallel, **kwargs)

        _profile = [
            "rfoo.profile(r,{0})".format(",".join(self._rkw)),
            "zfoo.profile(z,{0})".format(",".join(self._zkw)),
        ]
        _profile = "*".join(_profile)
        _profile = "lambda rfoo,zfoo,r,z,{0},{1}: {2}".format(
            ",".join(self._rkw), ",".join(self._zkw), _profile
        )
        self.profile = jax.jit(
            partial(eval(_profile), self.radial, self.parallel)
        )
        self._initialized = True


class MesaBridge(Bridge):
    """
    Mesa bridge model with harmonic mean profile combination.

    The MesaBridge combines radial and parallel profiles using a harmonic
    mean, producing a mesa-like (flat-topped) surface brightness distribution:

        I(r, z) = 1 / (1/f_radial(r) + 1/f_parallel(z))

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

    Examples
    --------
    >>> from socca.models import MesaBridge
    >>> bridge = MesaBridge()
    >>> bridge.parameters()

    Model parameters
    ================
    xc              [deg] : None       | Right ascension of bridge centroid
    yc              [deg] : None       | Declination of bridge centroid
    rs              [deg] : None       | Scale radius of the bridge profiles
    Is            [image] : None       | Scale intensity of the bridge
    theta           [rad] : 0.0000E+00 | Position angle of bridge major axis
    e                  [] : 5.0000E-01 | Axis ratio of the bridge profiles
    radial.alpha       [] : 8.0000E+00 | Radial exponent
    radial.beta        [] : 1.0000E+00 | Slope parameter
    parallel.alpha     [] : 8.0000E+00 | Power law slope
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

        _profile = [
            "1.00/rfoo.profile(r,{0})".format(",".join(self._rkw)),
            "1.00/zfoo.profile(z,{0})".format(",".join(self._zkw)),
        ]
        _profile = "+".join(_profile)
        _profile = "lambda rfoo,zfoo,r,z,{0},{1}: 1.00/({2})".format(
            ",".join(self._rkw), ",".join(self._zkw), _profile
        )
        self.profile = jax.jit(
            partial(eval(_profile), self.radial, self.parallel)
        )
        self._initialized = True
