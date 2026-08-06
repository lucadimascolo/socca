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


class BarGeometry():
    def __init__(self, inc=0, rot=0, losdepth=1.0, losbins=10):
        self.inc = inc        # inclination
        self.rot = rot        # bar intrinsic rotation
        self.losdepth = losdepth
        self.losbins = losbins


class Bar(Component):
    """
    Explanation TBD.
    """

    def __init__(self, radial=Sersic(), geometry=BarGeometry(), **kwargs):
        super().__init__(**kwargs)

        # Inherit some parameters from the radial profile;
        # Default is Sersic, where we inherit re, Ie, ns.
        # Hosts and auto-initializes the parameters xc, yc, and theta according to the config file.
        self.radial = radial

        # 3D Geometry
        # We initialize to the standard values in the config file for Height, to match the 3d Disk component. 
        # Rotation is bar-specific, so we just let it live here.
        self.inc = kwargs.get("inc", config.Height.inc)
        self.rot = 0
        self.losdepth = kwargs.get("losdepth", config.Height.losdepth)
        self.losbins = kwargs.get("losbins", config.Height.losbins)

        for param in ["losdepth", "losbins"]:
            if param not in self.hyper:
                self.hyper.append(param)

        self.profile = jax.jit(Bar._bar_profile)

        self.units.update(
            {
                f"radial.{key}": self.radial.units[key]
                for key in self.radial.units.keys()
            }
        )
        self.units.update(dict(inc="rad", rot="rad", losdepth="deg", losbins=""))

        self.description.update(
            {
                f"radial.{key}": self.radial.description[key]
                for key in self.radial.description.keys()
            }
        )
        self.description.update(
            dict(
                losdepth="Half line-of-sigt extent for integration",
                losbins="Number of points for line-of-sight integration",
                inc="Inclination angle (0=face-on); Typically inherited from a Disk object",
                theta="Position angle (east from north); Typically inherited from a Disk Object",
                rot="Intrinsic bar rotation relative to theta (added to position angle theta)"
            )
        )

        self._initialized = True

    def getmap(self, img, convolve=False):
        """
        Docstring TBD.
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

        # Geometric parameters for a bar
        kwarg["xc"]       = self.radial.xc
        kwarg["yc"]       = self.radial.yc
        kwarg["theta"]    = self.radial.theta

        kwarg["inc"]      = self.inc
        kwarg["rot"]      = self.rot
        kwarg["losdepth"] = self.losdepth
        kwarg["losbins"]  = self.losbins

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
                warnings.warn("No PSF defined, so no convolution will be performed.")
            else:
                mgrid = img.convolve(mgrid)
        return mgrid

    def _build_kwargs(self, pars, comp_prefix):
        """
        Docstring TBD.
        """
        # profile parameters
        kwarg = {
            key.replace(f"{comp_prefix}.", ""): pars[key]
            for key in pars
            if key.startswith(f"{comp_prefix}.")
        }

        # geometric parameters
        kwarg["xc"] = pars[f"{comp_prefix}.xc"]
        kwarg["yc"] = pars[f"{comp_prefix}.yc"]
        kwarg["theta"] = pars[f"{comp_prefix}.theta"]
        kwarg["inc"] = pars[f"{comp_prefix}.inc"]
        kwarg["rot"] = pars[f"{comp_prefix}.rot"]
        kwarg["losdepth"] = pars[f"{comp_prefix}.losdepth"]
        kwarg["losbins"] = pars[f"{comp_prefix}.losbins"]

        return kwarg

    @staticmethod
    def _bar_profile(xt, yt, zt, Ie, re, rs, ns):
        m = jp.sqrt((xt / rs) ** 2 + (yt / re) ** 2 + (zt / rs) ** 2)
        return Sersic.profile(m, Ie, re, ns)

    def _evaluate(self, img, **kwarg):
        """
        Docstring TBD.
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
        Docstring TBD.
        """
        ssize, ysize, xsize = grid.x.shape

        xt = (grid.x - xc) * jp.cos(jp.deg2rad(yc))  # Build x
        yt = grid.y - yc  # Build y
        zt = jp.linspace(-losdepth, losdepth, losbins)  # Build z

        # Rotate with Position Angle - around z.
        sint = jp.sin(theta)
        cost = jp.cos(theta)
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

        # Rotate with Inclination - around x.
        sini = jp.sin(inc - 0.5 * jp.pi)
        cosi = jp.cos(inc - 0.5 * jp.pi)
        zt, yt = (yt * cosi - zt * sini, yt * sini + zt * cosi)

        # Rotate with Rotation - around disk-frame z.
        sinr = jp.sin(rot)
        cosr = jp.cos(rot)
        xt, yt = xt * cosr + yt * sinr, -xt * sinr + yt * cosr

        return xt, yt, zt

    def parameters(self):
        """
        Docstring TBD.
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
                        kvalue = getattr(
                            self.radial, key.replace("radial.", "")
                        )
                    elif key.startswith("vertical."):
                        kvalue = getattr(
                            self.vertical, key.replace("vertical.", "")
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
    
    def parlist(self):
        """
        Docstring TBD.
        """
        return list(self.units.keys())
