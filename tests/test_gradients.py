"""Gradients of profiles and models must stay finite, including at r = 0."""

import importlib.util
import inspect
import itertools

import jax
import jax.numpy as jp
import numpy as np
import pytest
from astropy.io import fits

import socca.data as data
import socca.noise as noise
import socca.priors as priors
from socca import fitter
from socca.models import (
    Beta,
    Disk,
    Ellipsoid,
    Exponential,
    Gaussian,
    MesaBridge,
    Model,
    ModExponential,
    PolyExponential,
    PolyExpoRefact,
    Power,
    Sersic,
    SimpleBridge,
    TopHat,
    gNFW,
)
from socca.models.disk.vertical import ExponentialHeight, HyperSecantHeight
from socca.models.radial import truncation

AS = 1.0 / 3600.0
S0 = 3.0 * AS
RADII = S0 * jp.array([0.0, 1e-6, 1e-2, 1.0, 10.0, 1e4])
HEIGHTS = jp.concatenate([-RADII[::-1], RADII])
SCALES = dict(re=S0, rc=S0, rs=S0, rm=0.7 * S0, zs=0.2 * S0, rt=2 * S0)
SCALES.update(wt=0.2 * S0)

PROFILES = [
    ("Sersic", Sersic.profile, dict(Ie=1.0), dict(ns=[0.3, 1.0, 4.0, 10.0])),
    (
        "Beta",
        Beta.profile,
        dict(Ic=1.0),
        dict(alpha=[0.3, 1.0, 3.0], beta=[0.3, 1.0, 3.0]),
    ),
    ("Power", Power.profile, dict(Ic=1.0), dict(alpha=[0.3, 1.0, 3.0])),
    ("Gaussian", Gaussian.profile, dict(Is=1.0), {}),
    ("Exponential", Exponential.profile, dict(Is=1.0), {}),
    (
        "PolyExponential",
        PolyExponential.profile,
        dict(Is=1.0, c1=0.3, c2=-0.2, c3=0.1, c4=0.05),
        {},
    ),
    (
        "PolyExpoRefact",
        PolyExpoRefact.profile,
        dict(Is=1.0, I1=0.3, I2=0.2, I3=0.1, I4=0.05),
        {},
    ),
    (
        "ModExponential",
        ModExponential.profile,
        dict(Is=1.0),
        dict(alpha=[-2.0, 0.5, 2.0]),
    ),
    ("TopHat", TopHat.profile, dict(Ic=1.0), {}),
    (
        "HyperSecantHeight",
        HyperSecantHeight.profile,
        {},
        dict(alpha=[0.5, 2.0, 4.0]),
    ),
    ("ExponentialHeight", ExponentialHeight.profile, {}, {}),
    ("HyperTangent", truncation.HyperTangent.profile, {}, {}),
    ("ExpTruncation", truncation.Exponential.profile, {}, {}),
]

GNFW = dict(alpha=[0.1, 1.05, 10.0], beta=[2.0, 10.0], gamma=[-1.0, 1.0, 2.4])


def _kwargs(fun, base):
    """Fill in the scale parameters of fun's signature."""
    keys = [
        k for k in inspect.signature(fun).parameters if k not in ("r", "z")
    ]
    return {k: base.get(k, SCALES.get(k)) for k in keys}


def _check(fun, kwargs, x):
    """Assert finite values and parameter gradients over the points x."""
    assert bool(jp.all(~jp.isnan(fun(x, **kwargs))))
    for key, val in kwargs.items():
        grad = jax.grad(lambda v: jp.sum(fun(x, **{**kwargs, key: v})))(val)
        assert np.isfinite(float(grad)), (key, kwargs)


@pytest.mark.parametrize(
    "name,fun,base,grid", PROFILES, ids=[p[0] for p in PROFILES]
)
def test_profile_gradients(name, fun, base, grid):
    """Profile gradients are finite at r = 0 and far out, over the ranges."""
    x = HEIGHTS if name.endswith("Height") else RADII
    for combo in itertools.product(*grid.values()):
        kwargs = _kwargs(fun, {**base, **dict(zip(grid, combo))})
        _check(fun, kwargs, x)


_GNFW_QUAD = pytest.mark.xfail(
    reason="gNFW quadrature gradients are unreliable until fix/gnfw",
)


@pytest.mark.parametrize(
    "emulator",
    [
        pytest.param(None, marks=_GNFW_QUAD),
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                importlib.util.find_spec("flax") is None,
                reason="the gNFW emulator needs flax",
            ),
        ),
    ],
    ids=["quad", "emulator"],
)
def test_gnfw_gradients(emulator):
    """The gNFW gradients are finite over the supported slope ranges."""
    profile = gNFW(emulator=emulator).profile
    for combo in itertools.product(*GNFW.values()):
        kwargs = dict(Ic=1.0, rc=S0, **dict(zip(GNFW, combo)))
        _check(profile, kwargs, RADII)


def _image(n=16):
    """Image whose central pixel is centred on (180, 45)."""
    header = fits.Header()
    for key, val in dict(
        NAXIS=2,
        NAXIS1=n,
        NAXIS2=n,
        CTYPE1="RA---TAN",
        CTYPE2="DEC--TAN",
        CRPIX1=n // 2 + 1,
        CRPIX2=n // 2 + 1,
        CRVAL1=180.0,
        CRVAL2=45.0,
        CDELT1=-0.5 * AS,
        CDELT2=0.5 * AS,
    ).items():
        header[key] = val
    rng = np.random.default_rng(0)
    hdu = fits.PrimaryHDU(0.1 * rng.normal(size=(n, n)), header=header)
    return data.Image(hdu, noise=noise.Normal(sigma=0.1))


def _uniform(val, width=0.5):
    """Uniform prior around val."""
    return priors.uniform(val * (1 - width) - 1e-3, val * (1 + width) + 1e-3)


def _center(comp):
    """Free centre exactly on the central pixel."""
    comp.xc = priors.uniform(180.0 - AS, 180.0 + AS)
    comp.yc = priors.uniform(45.0 - AS, 45.0 + AS)


def _sersic2d():
    comp = Sersic(re=_uniform(S0), Ie=_uniform(1.0), ns=_uniform(4.0))
    comp.theta, comp.e = _uniform(0.3), _uniform(0.3)
    _center(comp)
    return comp, dict(xc=180.0, yc=45.0, theta=0.3, e=0.3, ns=4.0)


def _beta2d():
    comp = Beta(rc=_uniform(S0), Ic=_uniform(1.0), alpha=_uniform(0.6))
    comp.beta, comp.theta, comp.e = 1.0, _uniform(0.3), _uniform(0.3)
    _center(comp)
    return comp, dict(xc=180.0, yc=45.0, alpha=0.6, theta=0.3, e=0.3)


def _gnfw2d():
    comp = gNFW(rc=_uniform(S0), Ic=_uniform(1.0), emulator=None)
    comp.alpha, comp.beta, comp.gamma = (
        _uniform(3.0),
        _uniform(5.0),
        _uniform(0.8),
    )
    _center(comp)
    return comp, dict(xc=180.0, yc=45.0, alpha=3.0, beta=5.0, gamma=0.8)


def _disk(inc):
    def build():
        radial = Sersic(re=_uniform(S0), Ie=_uniform(1.0), ns=_uniform(2.0))
        radial.theta = _uniform(0.3)
        _center(radial)
        vertical = HyperSecantHeight(zs=_uniform(0.2 * S0))
        vertical.inc = priors.uniform(inc - 0.3, inc + 0.3)
        values = {
            "radial.xc": 180.0,
            "radial.yc": 45.0,
            "radial.ns": 2.0,
            "radial.theta": 0.3,
            "vertical.inc": inc,
        }
        return Disk(radial=radial, vertical=vertical), values

    return build


def _ellipsoid():
    comp = Ellipsoid(
        radial=Sersic(re=_uniform(S0), Ie=_uniform(1.0), ns=_uniform(2.0))
    )
    comp.theta, comp.e, comp.eratio = (
        _uniform(0.3),
        _uniform(0.5),
        _uniform(0.6),
    )
    comp.inc, comp.rot = _uniform(0.7), _uniform(0.3)
    _center(comp)
    values = dict(xc=180.0, yc=45.0, theta=0.3, e=0.5, eratio=0.6)
    values.update({"inc": 0.7, "rot": 0.3, "radial.ns": 2.0})
    return comp, values


def _bridge(cls):
    def build():
        comp = cls(rs=_uniform(S0), Is=_uniform(1.0))
        comp.theta, comp.e = _uniform(0.3), _uniform(0.5)
        _center(comp)
        return comp, dict(xc=180.0, yc=45.0, theta=0.3, e=0.5)

    return build


COMPONENTS = [
    ("Sersic2D", _sersic2d),
    ("Beta2D", _beta2d),
    pytest.param("gNFW2D", _gnfw2d, marks=_GNFW_QUAD, id="gNFW2D"),
    ("Disk face-on", _disk(0.0)),
    ("Disk inclined", _disk(1.0)),
    ("Disk edge-on", _disk(np.pi / 2)),
    ("Ellipsoid", _ellipsoid),
    ("SimpleBridge", _bridge(SimpleBridge)),
    ("MesaBridge", _bridge(MesaBridge)),
]


@pytest.mark.parametrize(
    "name,build",
    COMPONENTS,
    ids=[c[0] if isinstance(c, tuple) else c.id for c in COMPONENTS],
)
def test_model_gradients(name, build):
    """Likelihood gradients are finite with the centre on a pixel centre."""
    comp, values = build()
    fit = fitter(_image(), Model(comp))
    scale = {"re": S0, "rc": S0, "rs": S0, "zs": 0.2 * S0}
    p = []
    for label in fit.labels:
        key = label.split("_", 2)[-1]
        short = key.split(".")[-1]
        p.append(values.get(key, scale.get(short, 1.0)))
    p = np.array(p)
    assert np.isfinite(float(fit._log_likelihood(p)))
    grad = np.asarray(jax.grad(lambda q: fit._log_likelihood(q))(p))
    assert np.all(np.isfinite(grad)), dict(zip(fit.labels, grad))
