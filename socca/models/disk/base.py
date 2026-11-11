"""Disk model with finite thickness."""

from functools import partial


import jax
import jax.numpy as jp


from ..base import Component, _claim, _strip, _subkeymap, _subprofile
from ..radial import Sersic
from ..radial.base import Profile, _safe_power
from .vertical import Height, HyperSecantHeight

_LOS_CHUNK = 16


def _los_integrate(density, pars, z, w, chunk=_LOS_CHUNK):
    """
    Integrate density(pars, z) along z with the weights w.

    The density has shape (subgrid, len(z), ny, nx). The value is one
    fused weighted sum; its gradient recomputes the density in chunks of
    z nodes, so the full cube is never stored.
    """

    def total(pars, z, w):
        return jp.einsum("slyx,l->syx", density(pars, z), w)

    @jax.custom_vjp
    def integrate(pars, z, w):
        return total(pars, z, w)

    def fwd(pars, z, w):
        return total(pars, z, w), (pars, z, w)

    def bwd(res, g):
        pars, z, w = res
        n = z.size
        size = min(chunk, n)
        nchunk = -(-n // size)
        pad = nchunk * size - n
        zc = jp.concatenate([z, jp.full(pad, z[-1])]).reshape(nchunk, size)
        wc = jp.concatenate([w, jp.zeros(pad)]).reshape(nchunk, size)

        def body(acc, zw):
            _, vjp = jax.vjp(total, pars, *zw)
            dp, dz, dw = vjp(g)
            return jax.tree.map(jp.add, acc, dp), (dz, dw)

        zero = jax.tree.map(jp.zeros_like, pars)
        dp, (dz, dw) = jax.lax.scan(body, zero, (zc, wc))
        return dp, dz.reshape(-1)[:n], dw.reshape(-1)[:n]

    integrate.defvjp(fwd, bwd)
    return integrate(pars, z, w)


class Disk(Component):
    """
    3D disk model combining radial and vertical profiles with inclination.

    Creates a realistic disk galaxy model by combining a radial surface brightness
    profile with a vertical density distribution. The model accounts for disk
    inclination via line-of-sight integration.

    Parameters
    ----------
    radial : Profile, optional
        Radial profile defining in-plane surface brightness distribution.
        Default is Sersic(). Can be any 2D profile (Beta, Exponential, etc.).
    vertical : Height, optional
        Vertical profile defining scale height and vertical structure.
        Default is HyperSecantHeight(). Can be HyperSecantHeight or ExponentialHeight.
    **kwargs : dict
        Additional keyword arguments passed to Component.

    Attributes
    ----------
    radial : Profile
        Radial profile component.
    vertical : Height
        Vertical profile component.
    profile : callable
        Combined 3D density profile rho(r, z) = radial(r) * vertical(z).

    Notes
    -----
    The 3D disk model:
    1. Defines density as rho(r,z) = radial_profile(r) * vertical_profile(z)
    2. Integrates along the line of sight to obtain projected surface brightness
    3. Accounts for inclination by rotating the disk before integration
    4. Uses numerical integration (trapezoidal rule) along z-direction

    The inclination angle is stored in vertical.inc:

    - inc = 0: face-on (looking down on the disk)
    - inc = pi/2: edge-on (viewing disk from the side)

    Line-of-sight integration parameters (vertical.losdepth, vertical.losbins)
    control accuracy and computation time. Larger losdepth and more losbins
    improve accuracy for highly inclined disks.

    Examples
    --------
    >>> from socca.models import Disk, Exponential
    >>> from socca.models.disk.vertical import ExponentialHeight
    >>> # Create edge-on exponential disk
    >>> radial = Exponential(xc=180.5, yc=45.2, rs=0.02, Is=100)
    >>> vertical = ExponentialHeight(zs=0.002, inc=np.pi/2)
    >>> disk = Disk(radial=radial, vertical=vertical)
    """

    def __init__(self, radial=None, vertical=None, **kwargs):
        """
        Initialize a 3D disk model from radial and vertical profiles.

        Parameters
        ----------
        radial : Profile, optional
            Radial profile component. Default is Sersic().
        vertical : Height, optional
            Vertical profile component. Default is HyperSecantHeight().
        **kwargs : dict
            Additional keyword arguments.

        Notes
        -----
        The component ID is synchronized between the Disk, radial, and vertical
        components to ensure consistent parameter naming in composite models.
        """
        super().__init__(**kwargs)

        # radial/vertical default to fresh instances per call, not shared
        # mutable defaults -- otherwise every Disk() call that doesn't pass
        # its own radial/vertical would silently share the same
        # sub-component objects, so setting one Disk's radial.xc would
        # leak into every other default-constructed Disk.
        if radial is None:
            radial = Sersic()
        if vertical is None:
            vertical = HyperSecantHeight()
        if not isinstance(radial, Profile) or not isinstance(vertical, Height):
            raise TypeError(
                "Disk needs a radial profile (e.g. Sersic) and a vertical "
                "Height profile."
            )
        _claim(radial, vertical, free=["e", "cbox"])
        _strip(radial, ["e", "cbox"])

        self.radial = radial
        self.vertical = vertical

        self._namespaces = {"radial": self.radial, "vertical": self.vertical}
        self._wrap()
        self._adopt()
        self._initialized = True

    def _keymap(self):
        """Map the keywords of _evaluate to parameter names."""
        keys = _subkeymap(self.radial, "r", "radial")
        keys.update(_subkeymap(self.vertical, "z", "vertical"))
        keys.update(
            xc="radial.xc",
            yc="radial.yc",
            theta="radial.theta",
            inc="vertical.inc",
            losdepth="vertical.losdepth",
            losbins="vertical.losbins",
        )
        return keys

    def profile(self, r, z, **kw):
        """Evaluate the density radial(r) * vertical(z)."""
        return _subprofile(self.radial, r, kw, "r", True) * _subprofile(
            self.vertical, z, kw, "z"
        )

    def getmap(self, img, convolve=False):
        """
        Generate disk image via 3D line-of-sight integration.

        Computes the projected surface brightness by integrating the 3D density
        distribution along the line of sight, accounting for inclination.

        Parameters
        ----------
        img : Image
            Image object containing grid, PSF, and WCS information.
        convolve : bool, optional
            If True, convolve the model with the PSF. Default is False.

        Returns
        -------
        ndarray
            Projected disk image on the image grid.

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
        The algorithm:
        1. Creates 3D coordinate grids (r, z) accounting for inclination
        2. Evaluates 3D density rho(r, z) = radial(r) * vertical(z)
        3. Integrates along line of sight using trapezoidal rule
        4. Averages over sub-pixel sampling
        5. Optionally convolves with PSF

        Integration accuracy controlled by:

        - vertical.losdepth: extent of integration
        - vertical.losbins: number of integration points

        For edge-on disks, use larger losdepth (≥ 5*zs) and more losbins (≥ 200).

        Examples
        --------
        >>> from socca.models import Disk, Exponential
        >>> from socca.models.disk.vertical import ExponentialHeight
        >>> import numpy as np
        >>> from socca.data import Image
        >>> radial = Exponential(xc=180.5, yc=45.2, rs=0.02, Is=100)
        >>> vertical = ExponentialHeight(zs=0.002, inc=np.pi/4)  # 45 deg
        >>> disk = Disk(radial=radial, vertical=vertical)
        >>> img = Image('observation.fits')
        >>> disk_map = disk.getmap(img, convolve=True)
        """
        kwarg = self._getpars(self._keymap())
        return self._convolve(img, self._evaluate(img, **kwarg), convolve)

    def _evaluate(self, img, **kwarg):
        """
        Evaluate disk model on the given grid with explicit parameters.

        This internal method computes the projected disk surface brightness
        via line-of-sight integration using the provided geometric and profile
        parameters. It is used by both getmap() and Model.getmodel() to avoid
        code duplication.

        Parameters
        ----------
        img : Image
            Image object containing grid and WCS information.
        **kwarg : dict
            All parameters including geometric (xc, yc, theta, inc, losdepth,
            losbins) and profile-specific parameters with r_ and z_ prefixes.

        Returns
        -------
        ndarray
            2D array of projected surface brightness, averaged over subpixels.
        """
        xc = kwarg.pop("xc")
        yc = kwarg.pop("yc")
        theta = kwarg.pop("theta")
        inc = kwarg.pop("inc")
        losdepth = kwarg.pop("losdepth")
        losbins = kwarg.pop("losbins")

        # one weighted sum fuses with the density; trapezoid would slice it
        z = jp.linspace(-losdepth, losdepth, losbins)
        dx = 2.00 * losdepth / (losbins - 1)
        wlos = jp.full(losbins, dx).at[jp.array([0, -1])].multiply(0.50)
        pars = dict(xc=xc, yc=yc, theta=theta, inc=inc, kw=kwarg)
        pars = jax.tree.map(lambda v: jp.asarray(v, dtype=float), pars)

        def density(pars, z):
            r, zt = self.getgrid(
                img.grid, pars["xc"], pars["yc"], z, pars["theta"], pars["inc"]
            )
            return self._compiled()(r, zt, **pars["kw"])

        mgrid = _los_integrate(density, pars, z, wlos)
        return jp.mean(mgrid, axis=0)

    @staticmethod
    @partial(jax.jit, static_argnames=["grid"])
    def getgrid(grid, xc, yc, z, theta=0.00, inc=0.00):
        """
        Compute 3D disk coordinates with rotation and inclination.

        Generates the cylindrical radius and height of every pixel at the
        line-of-sight positions z, accounting for the position angle and
        the inclination of the disk.

        Parameters
        ----------
        grid : Grid
            Grid object with .x and .y celestial coordinate arrays (deg).
        xc : float
            Right ascension of disk center (deg).
        yc : float
            Declination of disk center (deg).
        z : ndarray
            Line-of-sight positions, from the sky plane through the center
            (deg).
        theta : float, optional
            Position angle of disk major axis, east from north (rad). Default is 0.
        inc : float, optional
            Inclination angle (0 = face-on, pi/2 = edge-on) (rad). Default is 0.

        Returns
        -------
        rcube : ndarray
            Radial distances from the disk axis (deg).
            Shape: (ssize, len(z), ysize, xsize).
        zcube : ndarray
            Heights above/below disk midplane (deg).
            Shape: (ssize, len(z), ysize, xsize).

        Notes
        -----
        The transformation sequence:
        1. Center coordinates on (xc, yc)
        2. Apply spherical geometry correction (cos(dec))
        3. Rotate by position angle theta
        4. Place each pixel at the line-of-sight positions z
        5. Apply inclination rotation
        6. Compute cylindrical radius r and height z

        The inclination is measured from face-on:

        - inc = 0: face-on, z-axis points toward observer
        - inc = pi/2: edge-on, disk in plane of sky
        """
        sint = jp.sin(theta)
        cost = jp.cos(theta)

        xt = (grid.x - xc) * jp.cos(jp.deg2rad(yc))
        yt = grid.y - yc
        xt, yt = -xt * sint - yt * cost, xt * cost - yt * sint
        xt, yt = xt[:, None], yt[:, None]
        z = z[None, :, None, None]

        sini = jp.sin(inc - 0.50 * jp.pi)
        cosi = jp.cos(inc - 0.50 * jp.pi)
        zt, yt = yt * cosi - z * sini, yt * sini + z * cosi

        return _safe_power(xt**2 + yt**2, 0.50), zt
