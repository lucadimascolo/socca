"""Ellipsoid model with 3D line-of-sight integration."""

from functools import partial

import warnings

import jax
import jax.numpy as jp
import numpy as np


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


_ABEL_NODES = 512
_ABEL_RANGE = 1.00e-08
_ABEL_SPLIT = 1.00e01
_ABEL_CAP = 1.00e05
_ABEL_TAIL = 32
_ABEL_BREAK = 32
_ABEL_WIDTHS = 5.00  # truncation widths resolved on each side of rt
_ABEL_EDGE = 1.00e-06


def _abel_table(f, rhi, losdepth, losbins, rscale=0.00):
    """
    Tabulate the projection P(R) = 2 int_0^inf f(sqrt(R^2 + s^2)) ds.

    Uses s = R sinh(t) on a log-spaced R grid in [_ABEL_RANGE * rhi, rhi],
    with losbins Gauss-Legendre nodes in t up to _ABEL_SPLIT times the
    larger of rhi and the scale radius rscale, and _ABEL_TAIL more up to
    _ABEL_CAP times it. The remainder is added assuming a power-law tail,
    which is infinite if f falls off no faster than 1/r. A finite
    losdepth instead caps the integral at that radius, and the table holds
    P / _abel_window(R), which stays smooth up to the cap. The table
    assumes a smooth f: truncated profiles are integrated per pixel
    instead (_abel_direct).
    """
    inf = jp.isinf(losdepth)
    rout = jp.maximum(rhi, rscale)
    rcap = jp.where(inf, _ABEL_CAP * rout, jp.where(inf, 1.00, losdepth))
    rmid = jp.minimum(_ABEL_SPLIT * rout, rcap)

    def nodes(n, lo, hi):
        x, w = np.polynomial.legendre.leggauss(n)
        return lo + 0.50 * (hi - lo) * (x + 1.00), 0.50 * (hi - lo) * w

    def acosh(x):
        return jp.where(x > 1, jp.arccosh(jp.where(x > 1, x, 2.00)), 0.00)

    def project(lr):
        r = jp.exp(lr)[:, None]
        tmid = acosh(rmid / r)
        tcap = acosh(rcap / r)
        ta, wa = nodes(int(losbins), 0.00, tmid)
        tb, wb = nodes(_ABEL_TAIL, tmid, tcap)
        t = jp.concatenate([ta, tb], axis=1)
        w = jp.concatenate([wa, wb], axis=1)
        u = jp.cosh(t)
        p = 2.00 * jp.sum(f(r * u) * r * u * w, axis=1)
        return p / _abel_window(jp.exp(2.00 * lr), losdepth)

    lr = jp.linspace(np.log(_ABEL_RANGE), 0.00, _ABEL_NODES) + jp.log(rhi)
    pr, dp = jax.jvp(project, (lr,), (jp.ones_like(lr),))

    fc, df = jax.jvp(lambda q: f(jp.exp(q)), (jp.log(rcap),), (1.00,))
    pmax = jax.lax.stop_gradient(jp.max(jp.abs(pr)))
    ok = inf & (jp.abs(fc) * rcap > 1.00e-12 * pmax)
    fc, df = jp.where(ok, fc, 1.00), jp.where(ok, df, -2.00)
    slope = -df / fc
    conv = slope > 1.00
    p = jp.where(conv, slope, 2.00)
    x2 = jp.exp(2.00 * lr) / rcap**2
    tail = (
        1.00 / (p - 1.00)
        + x2 / (2.00 * (p + 1.00))
        + 3.00 * x2**2 / (8.00 * (p + 3.00))
    )
    tail = 2.00 * fc * rcap * tail
    dtail = 2.00 * fc * rcap * (x2 / (p + 1.00) + 1.50 * x2**2 / (p + 3.00))
    tail = jp.where(ok, jp.where(conv, tail, jp.inf), 0.00)
    dtail = jp.where(ok & conv, dtail, 0.00)
    return lr, pr + tail, dp + dtail


def _abel_window(r2, losdepth):
    """Fraction sqrt(1 - R^2 / losdepth^2) of a capped chord; 1 if uncapped."""
    x = 1.00 - r2 / losdepth**2
    return jp.where(x > 0, jp.sqrt(jp.where(x > 0, x, 1.00)), 0.00)


def _hermite(lq, lr, y, d):
    """Cubic Hermite interpolation of y, with slopes d, against log R."""
    h = lr[1] - lr[0]
    lq = jp.clip(lq, lr[0], lr[-1])
    i = jp.clip(jp.floor((lq - lr[0]) / h).astype(int), 0, lr.size - 2)
    s = (lq - lr[i]) / h
    v = (
        (2 * s**3 - 3 * s**2 + 1) * y[i]
        + (s**3 - 2 * s**2 + s) * h * d[i]
        + (-2 * s**3 + 3 * s**2) * y[i + 1]
        + (s**3 - s**2) * h * d[i + 1]
    )
    # a zero weight times an infinite node would give nan
    return jp.where(jp.isinf(y[i]) | jp.isinf(y[i + 1]), jp.inf, v)


def _interp(lq, lr, pr, dp):
    """
    Interpolate the table at log R = lq.

    log|P| for single-signed tables, P itself if P changes sign. Nodes
    below tiny^(1/4) count as zero, so log slopes and their derivatives
    up to second order cannot overflow.
    """
    floor = jp.finfo(pr.dtype).tiny ** 0.25
    mixed = jp.any(pr > floor) & jp.any(pr < -floor)
    sign = jp.where(jp.sum(pr) < 0, -1.00, 1.00)
    pos = sign * pr > floor
    safe = jp.where(pos, pr, sign)
    ok = jp.isfinite(pr)
    y = jp.where(
        mixed,
        jp.where(ok, pr, 0.00),
        jp.where(pos, jp.log(sign * safe), jp.log(floor)),
    )
    d = jp.where(mixed, jp.where(ok, dp, 0.00), jp.where(pos, dp / safe, 0.00))
    v = _hermite(lq, lr, y, d)
    # exp of a linear P would overflow, and its zero gradient give nan
    return jp.where(
        mixed, v, sign * (jp.exp(jp.where(mixed, 0.00, v)) - floor)
    )


def _abel_direct(density, pars, radii, losbins):
    """
    Project a density with steps, P(R) = 2 int_0^chord f(sqrt(R^2 + s^2)) ds.

    Evaluated at every pixel's rmin = pars["rmin"], with s = R sinh(t)
    split exactly at the increasing radii (the last is the cap): losbins
    Gauss-Legendre nodes up to the first and _ABEL_BREAK in each next
    segment, shared by all pixels as a segment index plus a position in
    [0, 1].
    """
    counts = [int(losbins)] + [_ABEL_BREAK] * (len(radii) - 1)
    z, w = [], []
    for k, n in enumerate(counts):
        x, wk = np.polynomial.legendre.leggauss(n)
        z.append(k + 0.50 * (x + 1.00))
        w.append(0.50 * wk)
    # floats throughout: integer leaves have no gradient to accumulate
    pars = jax.tree.map(lambda v: jp.asarray(v, dtype=float), pars)
    pars = dict(pars, radii=jp.stack(radii))

    def chunk(pars, z):
        # tiny radii instead of 0, where acosh(b / R) diverges
        r = jp.maximum(pars["rmin"], _ABEL_RANGE * pars["radii"][-1])
        x = pars["radii"][:, None, None, None] / r[None]
        edges = jp.where(x > 1, jp.arccosh(jp.where(x > 1, x, 2.00)), 0.00)
        edges = jp.concatenate([jp.zeros_like(edges[:1]), edges])
        k = jp.floor(z).astype(int)
        lo = jp.moveaxis(edges[k], 0, 1)
        hi = jp.moveaxis(edges[k + 1], 0, 1)
        t = lo + (hi - lo) * (z - k)[None, :, None, None]
        c = jp.cosh(t)
        r = r[:, None]
        return 2.00 * density(pars, r * c) * r * c * (hi - lo)

    z, w = jp.asarray(np.concatenate(z)), jp.asarray(np.concatenate(w))
    return jp.einsum("slyx,l->syx", chunk(pars, z), w)


class Ellipsoid(Component):
    """
    3D triaxial-ellipsoid model evaluated via line-of-sight integration.

    Projects an ellipsoidal density onto the sky by integrating
    along the line of sight, in the same spirit as Disk but using a single
    radial profile for shape rather than separate radial/vertical profiles.
    Since the ellipsoidal radius along any line of sight is
    r^2 = a (z - z0)^2 + rmin^2, with a fixed by the orientation, the
    projection reduces exactly to the Abel projection of the radial profile
    evaluated at rmin and divided by sqrt(a).
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
        Maximum ellipsoidal radius included in the projection (deg).
        Default is inf, i.e. the full line of sight. Must be finite for
        profiles falling off no faster than 1/r (e.g. Power with
        alpha <= 1), whose projection otherwise diverges.
    losbins : int, optional
        Number of Gauss-Legendre nodes for the line-of-sight integral.
        Default is 128.
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
                losdepth="Maximum ellipsoidal radius for integration",
                losbins="Number of nodes for line-of-sight integration",
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
        The line-of-sight integral is computed once as a 1D Abel projection
        of the radial profile, so its cost does not scale with the image
        size; losbins sets the number of quadrature nodes.

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
        """
        Evaluate the density on the ellipsoidal radius.

        The radial profile is divided by twice the shortest semi-axis at
        its scale radius, 2 (1 - eratio * e) r_s, as a Disk divides it by
        the thickness of its vertical profile.
        """
        r = _safe_power(
            xt**2 + (yt / (1 - e)) ** 2 + (zt / (1 - eratio * e)) ** 2, 0.50
        )
        return self._density(r, e, eratio, kw)

    def _density(self, r, e, eratio, kw):
        """Normalised density at the ellipsoidal radius r."""
        m = _subprofile(self.radial, r, kw, "r", True)
        return m / self._thickness(e, eratio, kw)

    def _thickness(self, e, eratio, kw):
        """Twice the shortest semi-axis at the scale radius, 1 if none."""
        if self.radial._scale_radius is None:
            return 1.00
        rs = kw[f"r_{self.radial._scale_radius}"]
        return 2.00 * (1.00 - eratio * e) * rs

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

        e = kwarg.pop("e")
        eratio = kwarg.pop("eratio")

        rmin, a = self.getgrid(img.grid, xc, yc, theta, e, eratio, inc, rot)

        proj = getattr(self.radial, "_projection", None)
        truncation = getattr(self.radial, "truncation", None)
        if truncation is not None:
            # steps defeat the interpolated table: integrate each pixel,
            # split exactly at the truncation and at the profile's steps
            rt = jax.lax.stop_gradient(jp.abs(kwarg["r_rt"]))
            wt = jax.lax.stop_gradient(jp.abs(kwarg["r_wt"]))
            cap = jp.minimum(rt + 2.00 * _ABEL_WIDTHS * wt, losdepth)
            fixed = jp.stack(
                [
                    jp.maximum(rt - _ABEL_WIDTHS * wt, 0.00),
                    rt,
                    rt + _ABEL_WIDTHS * wt,
                ]
            )
            # a step's radius keeps its gradient: the jump moves with it
            steps = [
                jp.abs(kwarg[f"r_{name}"])
                for name in getattr(self.radial, "_steps", ())
            ]
            if steps:
                # a fixed split on a step would take its gradient: move it
                at = jax.lax.stop_gradient(jp.stack(steps))
                near = jp.abs(fixed[:, None] - at[None]) <= 1e-9 * at[None]
                fixed = jp.where(
                    jp.any(near, axis=1), fixed * 1.0000001, fixed
                )
            radii = jp.sort(jp.concatenate([fixed, *[s[None] for s in steps]]))
            radii = list(jp.minimum(radii, cap)) + [cap]
            pars = dict(rmin=rmin, e=e, eratio=eratio, kw=kwarg)

            def density(pars, r):
                return self._density(r, pars["e"], pars["eratio"], pars["kw"])

            mgrid = _abel_direct(density, pars, radii, losbins)
            return jp.mean(mgrid / jp.sqrt(a), axis=0)

        if proj is not None:
            # exact chord, which the table would smooth at a sharp edge
            pars = {k[2:]: v for k, v in kwarg.items() if k.startswith("r_")}
            mgrid = proj(rmin, rcap=losdepth, **pars)
            mgrid = mgrid / self._thickness(e, eratio, kwarg) / jp.sqrt(a)
            return jp.mean(mgrid, axis=0)

        # tabulate at unit amplitude: the log table loses amplitudes near 0
        amp = 1.00
        if self.radial._scale_amp is not None:
            key = f"r_{self.radial._scale_amp}"
            amp, kwarg = kwarg[key], {**kwarg, key: 1.00}

        def f(r):
            return self._density(r, e, eratio, kwarg)

        rhi = jax.lax.stop_gradient(jp.max(rmin))
        # keep the pixels strictly inside the table, where the clip has a
        # gradient, and the table just inside a finite cap
        rhi = jp.where(rhi > 0, rhi, 1.00) * (1.00 + _ABEL_EDGE)
        rhi = jp.minimum(rhi, (1.00 - _ABEL_EDGE) * losdepth)
        rscale = 0.00
        if self.radial._scale_radius is not None:
            rscale = jp.abs(kwarg[f"r_{self.radial._scale_radius}"])
            rscale = jax.lax.stop_gradient(rscale)
        lr, pr, dp = _abel_table(f, rhi, losdepth, losbins, rscale)
        lq = jp.log(jp.maximum(rmin, jp.exp(lr[0])))
        mgrid = _interp(lq, lr, pr, dp)
        mgrid = mgrid * _abel_window(rmin**2, losdepth)
        mgrid = amp * mgrid / jp.sqrt(a)
        return jp.mean(mgrid, axis=0)

    @staticmethod
    @partial(jax.jit, static_argnames=["grid"])
    def getgrid(grid, xc, yc, theta=0.0, e=0.0, eratio=1.0, inc=0.0, rot=0.0):
        """
        Compute the line-of-sight geometry of an inclined, rotated ellipsoid.

        Along the line of sight through each pixel, the ellipsoidal radius
        is r^2 = a (z - z0)^2 + rmin^2, so the projection of the density is
        its Abel projection at rmin divided by sqrt(a).

        Parameters
        ----------
        grid : Grid
            Grid object with .x and .y celestial coordinate arrays (deg).
        xc : float
            Right ascension of ellipsoid center (deg).
        yc : float
            Declination of ellipsoid center (deg).
        theta : float, optional
            Position angle of the line of nodes, east from north (rad), as
            for Disk. Default is 0.
        e : float, optional
            Ellipticity of the first minor axis. Default is 0.
        eratio : float, optional
            Ratio of the second minor axis's ellipticity to e. Default is 1.
        inc : float, optional
            Inclination angle (0 = face-on, pi/2 = edge-on) (rad). Default is 0.
        rot : float, optional
            Rotation of the major axis away from the line of nodes, within
            the already-inclined frame (rad). Default is 0. Only equivalent
            to a position-angle offset (i.e. theta + rot) when inc = 0 --
            see Notes.

        Returns
        -------
        rmin : ndarray
            Smallest ellipsoidal radius along each line of sight (deg).
            Shape: (ssize, ysize, xsize).
        a : float
            Squared rate of change of the ellipsoidal radius along the line
            of sight, the same for every pixel.

        Notes
        -----
        The transformation sequence:
        1. Center coordinates on (xc, yc) and apply spherical geometry correction (cos(dec))
        2. Rotate by position angle theta, so that the tilt (y) axis is the line of nodes, as for Disk
        3. Incline around the y-axis, mixing the position-angle-rotated x with the line-of-sight coordinate
        4. Rotate by rot + pi/2 within that already-inclined (x, y) frame, so that rot = 0 aligns the major axis with the line of nodes

        Because step 3 mixes the line-of-sight coordinate into x before
        step 4 runs, rot is not simply an additive offset to theta except
        in the face-on case (inc = 0): for inc != 0, rotating by rot after
        inclining is a genuinely different transform from inclining after
        rotating by theta + rot.
        """
        xp = (grid.x - xc) * jp.cos(jp.deg2rad(yc))
        yp = grid.y - yc
        sint = jp.sin(theta - 0.5 * jp.pi)
        cost = jp.cos(theta - 0.5 * jp.pi)
        xp, yp = -xp * sint - yp * cost, xp * cost - yp * sint
        sini = jp.sin(inc - 0.5 * jp.pi)
        cosi = jp.cos(inc - 0.5 * jp.pi)
        sinr = jp.sin(rot + 0.5 * jp.pi)
        cosr = jp.cos(rot + 0.5 * jp.pi)

        # frame coordinates at z = 0 (p) and their z derivative (v)
        pz, px = xp * cosi, xp * sini
        px, py = px * cosr + yp * sinr, -px * sinr + yp * cosr
        vz, vx = -sini, cosi
        vx, vy = vx * cosr, -vx * sinr
        qy, qz = 1.00 / (1 - e) ** 2, 1.00 / (1 - eratio * e) ** 2

        a = vx**2 + qy * vy**2 + qz * vz**2
        b = px * vx + qy * py * vy + qz * pz * vz
        c = px**2 + qy * py**2 + qz * pz**2
        return _safe_power(jp.maximum(c - b**2 / a, 0.00), 0.50), a
