from functools import partial
import inspect
import types
from dataclasses import dataclass

import warnings

import jax
import jax.numpy as jp
import numpyro.distributions

import numpy as np

from .. import config
from ..base import Component
from ..radial import Sersic

@dataclass(frozen=True)
class BarGeometry:  # I have placed a copy of this into ..config.py
    xc: float = None
    yc: float = None
    inc: float = 0.00
    theta: float = 0.00
    rot: float = 0.00
    e: float = 0.00
    losdepth: float = 10.00 / 60.00 / 60.00
    losbins: int = 200

class BarGeometry(Component):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xc       = kwargs.get("xc", config.BarGeometry.xc)
        self.yc       = kwargs.get("yc", config.BarGeometry.yc)
        self.theta    = kwargs.get("theta", config.BarGeometry.theta)
        self.inc      = kwargs.get("inc", config.BarGeometry.inc)
        self.rot      = kwargs.get("rot", config.BarGeometry.rot)
        self.e        = kwargs.get("e", config.BarGeometry.e)
        self.losdepth = kwargs.get("losdepth", config.BarGeometry.losdepth)
        self.losbins  = kwargs.get("losbins", config.BarGeometry.losbins)

        for param in ["losdepth", "losbins"]:
            if param not in self.hyper:
                self.hyper.append(param)

        self.units.update(
            dict(
                xc="deg",
                yc="deg",
                inc="rad",
                theta="rad",
                rot="rad",
                e="",
                losdepth="deg",
                losbins=""
            )
        )

        self.description.update(
            dict(
                xc="Right ascension of centroid",
                yc="Declination of centroid",
                inc="Inclination angle (0=face-on); Typically inherited from a Disk object",
                theta="Position angle (east from north); Typically inherited from a Disk Object",
                rot="Intrinsic bar rotation relative to theta (added to position angle theta)",
                e="Projected ellipticity (1 - axis ratio)",
                losdepth="Half line-of-sigt extent for integration",
                losbins="Number of points for line-of-sight integration",
            )
        )

        self._initialized = True


class Bar(Component):
    """
    Docsring TBD.
    """

    def __init__(self, radial=Sersic(), geometry=BarGeometry(), **kwargs):
        super().__init__(**kwargs)

        self.radial = radial
        self.geometry = geometry

        self._namespaces = {"radial": self.radial, "geometry": self.geometry}

        for param in self.geometry.hyper:
            self.hyper.append(f"geometry.{param}")
        
        if self.radial.id != self.id:
            type(self).idcls -= 1
            idmin = np.minimum(
                int(self.radial.id.replace("comp_", "")),
                int(self.id.replace("comp_", "")),
            )
            self.id = f"comp_{idmin:02d}"
            self.radial.id = self.id
            self.geometry.id = self.id

        self.profile = jax.jit(Bar._bar_profile)

        self.units.update(
            {
                f"radial.{key}": self.radial.units[key]
                for key in self.radial.units.keys()
                if key not in ["xc", "yc", "theta", "e", "cbox"]
            }
        )
        self.units.update(
            {
                f"geometry.{key}": self.geometry.units[key]
                for key in self.geometry.units.keys()
            }
        )

        self.description.update(
            {
                f"radial.{key}": self.radial.description[key]
                for key in self.radial.description.keys()
                if key not in ["xc", "yc", "theta", "e", "cbox"]
            }
        )
        self.description.update(
            {
                f"geometry.{key}": self.geometry.description[key]
                for key in self.geometry.units.keys()
            }
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
        for key in self.geometry.units.keys():
            val = getattr(self.geometry, key)
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
                warnings.warn("No PSF defined, so no convolution will be performed.")
            else:
                mgrid = img.convolve(mgrid)
        return mgrid

    def _build_kwargs(self, pars, comp_prefix):
        """
        Docstring TBD.
        """
        # radial profile parameters (re, Ie, ns)
        kwarg = {
            key.replace(f"{comp_prefix}_radial.", ""): pars[key]
            for key in pars
            if key.startswith(f"{comp_prefix}_radial.")
        }
        # geometry parameters (xc, yc, theta, inc, rot, e, losdepth, losbins)
        kwarg.update({
            key.replace(f"{comp_prefix}_geometry.", ""): pars[key]
            for key in pars
            if key.startswith(f"{comp_prefix}_geometry.")
        })
        return kwarg

    @staticmethod
    def _bar_profile(xt, yt, zt, Ie, re, e, ns):
        rs = re * (1-e)
        m = jp.sqrt((xt/rs)**2 + (yt/re)**2 + (zt/rs)**2)
        return Sersic.profile(m, Ie, 1.0, ns)

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
        xt, yt = xt * cost - yt * sint, xt * sint + yt * cost

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
                elif key.startswith("geometry."):
                    kvalue = getattr(self.geometry, key.replace("geometry.", ""))
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
                    if key.startswith("geometry."):
                        kvalue = getattr(
                            self.radial, key.replace("geometry.", "")
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
