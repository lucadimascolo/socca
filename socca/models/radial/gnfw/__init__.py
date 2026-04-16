"""Generalized Navarro-Frenk-White profile."""

import warnings

import jax
import jax.numpy as jp
import numpy as np
import numpyro.distributions

from ... import config
from ..base import Profile
from .model import getzero, integral

__all__ = ["gNFW"]


class gNFW(Profile):
    """
    Generalized Navarro-Frenk-White (gNFW) profile.

    The gNFW profile is a flexible three-parameter model commonly used to
    describe the surface brightness distribution of galaxy clusters.
    It generalizes the NFW profile with adjustable inner (gamma),
    intermediate (alpha), and outer (beta) slopes.

    By default the projected surface brightness is computed via numerical
    Abel integration on a 1D radial grid that is then interpolated onto the
    image pixels (accurate and fast). Direct per-pixel integration is also
    available, as well as a pre-trained MLP emulator.

    Parameters
    ----------
    emulator : bool or str or None, optional
        Controls whether to use the MLP emulator instead of numerical
        integration. Pass ``True`` to use the bundled pre-trained model,
        a file-system path (``str`` or :class:`~pathlib.Path`) to load a
        custom checkpoint, or ``None`` (default) to use numerical
        integration. Requires ``flax`` to be installed.
    interpolate : bool, optional
        Selects the evaluation strategy. If ``True`` (default) the profile
        is computed on a 1D radial grid and the result is interpolated to
        the image pixels. If ``False`` the profile is evaluated directly at
        every pixel (slower but free from interpolation error). Applies to
        both the numerical integration and emulator paths.
    rz : array-like, optional
        Radial grid (in units of rc) for the interpolation path. Only used
        when ``interpolate`` is True. Default is logspace(-7, 2, 1000).
    eps : float, optional
        Absolute and relative quadrature tolerance. Only used when
        ``emulator`` is None. Default is 1e-8.
    """

    _scale_radius = "rc"
    _scale_amp = "Ic"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rc = kwargs.get("rc", config.gNFW.rc)
        self.Ic = kwargs.get("Ic", config.gNFW.Ic)
        self.alpha = kwargs.get("alpha", config.gNFW.alpha)
        self.beta = kwargs.get("beta", config.gNFW.beta)
        self.gamma = kwargs.get("gamma", config.gNFW.gamma)

        self.rz = np.asarray(
            kwargs.get("rz", np.logspace(-7, 2, 1000)), dtype=np.float64
        )
        self.eps = kwargs.get("eps", 1.00e-08)
        self.interpolate = kwargs.get("interpolate", True)

        self.emulator = kwargs.get("emulator", None)
        self._emulator_model = None
        if self.emulator is not None:
            self._emulator_model = gNFW._load_emulator(self.emulator)

        self.units.update(
            dict(rc="deg", alpha="", beta="", gamma="", Ic="image")
        )

        self.description.update(
            dict(
                rc="Scale radius",
                Ic="Characteristic surface brightness",
                alpha="Intermediate slope",
                beta="Outer slope",
                gamma="Inner slope",
            )
        )

        self._rebuild_profile()
        self._initialized = True

    @staticmethod
    def _bundled_path():
        """Return the path to the bundled pre-trained checkpoint."""
        from pathlib import Path

        return Path(__file__).parent / "emulator" / "pretrained" / "gnfw.dill"

    @staticmethod
    def _load_emulator(path):
        """Load a pre-trained gNFW emulator checkpoint from a dill file.

        Parameters
        ----------
        path : bool or str or pathlib.Path
            Pass ``True`` to load the bundled pre-trained checkpoint, or
            a file-system path to load a custom checkpoint produced by
            :func:`socca.models.radial.gnfw.emulator.train`.

        Returns
        -------
        MLP
            Trained emulator model ready for inference.

        Notes
        -----
        Module aliases for ``emulate.*`` are temporarily injected into
        ``sys.modules`` so that checkpoints created by the standalone
        gnfw training script (which used ``import emulate``) can be
        loaded transparently.
        """
        import sys
        import types

        try:
            from flax import nnx  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The gNFW emulator requires 'flax'. "
                "Install with: pip install flax"
            ) from e

        from . import emulator as _pkg
        from .emulator import emulator as _emulator_mod
        from .emulator import config as _config_mod
        from . import model as _model_mod
        from .emulator.io import load_checkpoint

        # Temporarily expose old 'emulate.*' module paths so that dill
        # can deserialise checkpoints that were saved before the code was
        # moved into socca.
        _fake = types.ModuleType("emulate")
        _fake.emulator = _emulator_mod
        _fake.config = _config_mod
        _fake.model = _model_mod
        _fake.io = _pkg.io
        _fake.data = _pkg.data

        _aliases = {
            "emulate": _fake,
            "emulate.emulator": _emulator_mod,
            "emulate.config": _config_mod,
            "emulate.model": _model_mod,
        }
        if path is True:
            path = gNFW._bundled_path()

        _saved = {k: sys.modules.get(k) for k in _aliases}
        sys.modules.update(_aliases)
        try:
            checkpoint = load_checkpoint(path)
        finally:
            for k, v in _saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

        return checkpoint["model"]

    def _rebuild_profile(self):
        """Build (or rebuild) the JIT-compiled profile from current code."""
        model = self._emulator_model

        def _profile(r, Ic, rc, alpha, beta, gamma):
            return gNFW._profile_switch(
                r, Ic, rc, alpha, beta, gamma, emul_model=model, eps=self.eps
            )

        if self.interpolate:
            self.profile = jax.jit(
                lambda r, Ic, rc, alpha, beta, gamma: jp.interp(
                    r / rc,
                    self.rz,
                    _profile(self.rz * rc, Ic, rc, alpha, beta, gamma),
                )
            )
        else:
            self.profile = jax.jit(_profile)

    def __setstate__(self, state):
        """Restore state and rebuild profile from current source."""
        self.__dict__.update(state)
        self.rz = np.asarray(self.rz, dtype=np.float64)
        self._emulator_model = None
        if self.emulator is not None:
            self._emulator_model = gNFW._load_emulator(self.emulator)
        self._rebuild_profile()

    @property
    def alpha(self):  # noqa: D102
        return self._alpha

    @alpha.setter
    def alpha(self, value):  # noqa: D102
        wstring = None
        if isinstance(value, numpyro.distributions.Distribution):
            if value.support.lower_bound <= 0:
                wstring = "The alpha prior support includes values"
        elif value <= 0:
            wstring = "The alpha parameter is"
        if wstring is not None:
            warnings.warn(
                f"{wstring} less than or equal to 0. "
                "This might lead to unphysical models."
            )
        self._alpha = value

    @property
    def beta(self):  # noqa: D102
        return self._beta

    @beta.setter
    def beta(self, value):  # noqa: D102
        wstring = None
        if isinstance(value, numpyro.distributions.Distribution):
            if value.support.lower_bound < 3:
                wstring = "The beta prior support includes values"
        elif value < 3:
            wstring = "The beta parameter is"
        if wstring is not None:
            warnings.warn(
                f"{wstring} less than or equal to 3. "
                "This might lead to unphysical models."
            )
        self._beta = value

    @property
    def gamma(self):  # noqa: D102
        return self._gamma

    @gamma.setter
    def gamma(self, value):  # noqa: D102
        wstring = None
        if isinstance(value, numpyro.distributions.Distribution):
            if value.support.upper_bound >= 1:
                wstring = "The gamma prior support includes values"
        elif value >= 1:
            wstring = "The gamma parameter is"
        if wstring is not None:
            warnings.warn(
                f"{wstring} greater than or equal to 1. "
                "This might lead to unphysical models."
            )
        self._gamma = value

    @staticmethod
    def _profile_switch(
        r, Ic, rc, alpha, beta, gamma, emul_model=None, eps=1.00e-08
    ):
        """
        Compute projected gNFW surface brightness.

        Parameters
        ----------
        r : ndarray
            Projected elliptical radius in degrees.
        Ic : float
            Characteristic surface brightness (same units as image).
        rc : float
            Scale radius in degrees.
        alpha : float
            Intermediate slope parameter.
        beta : float
            Outer slope parameter.
        gamma : float
            Inner slope parameter.
        emul_model : MLP, optional
            Pre-trained emulator instance. If None, the profile is computed
            via numerical Abel integration. Default is None.
        eps : float, optional
            Absolute and relative quadrature tolerance. Only used when
            ``emul_model`` is None. Default is 1e-8.

        Returns
        -------
        ndarray
            Projected surface brightness at radius r, same shape as r.

        Notes
        -----
        The r=0 case is handled analytically via the Euler beta function
        (see :func:`~socca.models.radial.gnfw.model.getzero`).
        """
        shape = r.shape
        x = (r / rc).ravel()
        safe_x = jp.where(x > 0, x, jp.ones_like(x))
        if emul_model is None:
            y0 = getzero(alpha, beta, gamma)
            mz = integral(safe_x, alpha, beta, gamma, eps) / y0
        else:
            mz = emul_model(safe_x, alpha, beta, gamma, log=False)
        return (Ic * jp.where(x > 0, mz, jp.ones_like(x))).reshape(shape)
