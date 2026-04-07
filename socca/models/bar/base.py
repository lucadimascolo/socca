from functools import partial
import inspect
import types

import warnings

import jax
import jax.numpy as jp
import numpyro.distributions

import numpy as np

from ..base import Component
from ..radial import Sersic

class Bar(Component):
    """
    Explanation TBD.
    """

    def __init__(self, radial=Sersic(), **kwargs):
        super().__init__(**kwargs)

        # Inherit some parameters from the radial profile;
        # default is Sersic, where we inherit re, Ie, ns. 
        self.radial = radial

        # Sky Plane Geometry; the Sersic profile is capable of hosting xc, yc, and theta.
        self.radial.xc = 0
        self.radial.yc = 0
        self.radial.theta = 0

        # 3D Geometry
        self.inc = 0
        self.rs = 0.1
        self.rot = 0

        self.profile = jax.jit(Bar._bar_profile)

    def getmap(self, img, convolve=False):
        """
        Docstring TBD.
        
        ------

        Note: Kwargs currently assume a Sersic radial profile.
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
        kwarg["rs"]       = self.rs
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
        kwarg["xc"]       = pars[f"{comp_prefix}.xc"]
        kwarg["yc"]       = pars[f"{comp_prefix}.yc"]
        kwarg["theta"]    = pars[f"{comp_prefix}.theta"]
        kwarg["inc"]      = pars[f"{comp_prefix}.inc"]
        kwarg["rot"]      = pars[f"{comp_prefix}.rot"]
        kwarg["losdepth"] = pars[f"{comp_prefix}.losdepth"]
        kwarg["losbins"]  = pars[f"{comp_prefix}.losbins"]

        return kwarg

    @staticmethod
    def _bar_profile(xt, yt, zt, Ie, re, rs, ns):
        m = jp.sqrt((xt/rs)**2 + (yt/re)**2 + (zt/rs)**2)
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

        xt, yt, zt  = self.getgrid(
            img.grid, xc, yc, losdepth, losbins, theta, inc, rot
        )

        dx    = 2.0 * losdepth / (losbins - 1)
        mgrid = self.profile(xt, yt, zt, **kwarg)
        mgrid = jp.trapezoid(mgrid, dx=dx, axis=1)
        return jp.mean(mgrid, axis=0)

    @staticmethod
    @partial(jax.jit, static_argnames=["grid", "losbins"])
    def getgrid(grid, xc, yc, losdepth, losbins=200, theta=0.0, inc=0.0, rot=0.0):
        """
        Docstring TBD.
        """
        ssize, ysize, xsize = grid.x.shape

        xt = (grid.x - xc) * jp.cos(jp.deg2rad(yc))     # Build x
        yt = grid.y - yc                                # Build y
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
        sini = jp.sin(inc - 0.5*jp.pi)
        cosi = jp.cos(inc - 0.5*jp.pi)
        zt, yt = (
            yt*cosi - zt*sini,
            yt*sini + zt*cosi
        )

        # Rotate with Rotation - around disk-frame z. 
        sinr = jp.sin(rot)
        cosr = jp.cos(rot)
        xt, yt = xt*cosr + yt*sinr, -xt*sinr + yt*cosr

        return xt, yt, zt
    
