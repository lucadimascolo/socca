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
    Docsring TBD.
    """

    def __init__(self, radial=Sersic(), **kwargs):
        super().__init__(**kwargs)

        self.radial = radial

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
                if key not in ["cbox"]
            }
        )
        self.units.update(
            dict(inc="rad", rot="rad", losdepth="deg", losbins="")
        )

        self.description.update(
            {
                f"radial.{key}": self.radial.description[key]
                for key in self.radial.description.keys()
                if key not in ["cbox"]
            }
        )
        self.description.update(
            dict(
                inc="Inclination angle (0=face-on); Typically inherited from a Disk object",
                rot="Intrinsic bar rotation relative to theta (added to position angle theta)",
                losdepth="Half line-of-sigt extent for integration",
                losbins="Number of points for line-of-sight integration",
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

        # Position/orientation parameters, inherited from radial
        for key in ["xc", "yc", "theta", "e"]:
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

        # Bar's own 3D geometry
        for key in ["inc", "rot", "losdepth", "losbins"]:
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
        Docstring TBD.
        """
        # radial profile + position/orientation parameters (re, Ie, ns,
        # xc, yc, theta, e)
        kwarg = {
            key.replace(f"{comp_prefix}_radial.", ""): pars[key]
            for key in pars
            if key.startswith(f"{comp_prefix}_radial.")
        }
        # Bar's own 3D geometry (inc, rot, losdepth, losbins)
        for key in ["inc", "rot", "losdepth", "losbins"]:
            kwarg[key] = pars[f"{comp_prefix}_{key}"]
        return kwarg

    @staticmethod
    def _bar_profile(xt, yt, zt, Ie, re, e, ns):
        rs = re * (1 - e)
        m = jp.sqrt((xt / rs) ** 2 + (yt / re) ** 2 + (zt / rs) ** 2)
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
        Docstring TBD.
        """
        return list(self.units.keys())
