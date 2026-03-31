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

    # Needs a self.profile which is just going to be Sersic.

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