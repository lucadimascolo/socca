"""Generalized Navarro-Frenk-White profile via neural network emulator."""

import os
import warnings

import jax
import jax.numpy as jp

from ... import config
from ..base import Profile

# Lazy imports for optional dependencies
_emulator_available = None
_create_model = None
_predict = None
_ModelConfig = None


def _check_emulator_available():
    """Check if emulator dependencies are available and load them."""
    global _emulator_available, _create_model, _predict, _ModelConfig
    if _emulator_available is None:
        try:
            from .emulate.emulator import create_model, predict
            from .emulate.config import Model as ModelConfig

            _create_model = create_model
            _predict = predict
            _ModelConfig = ModelConfig
            _emulator_available = True
        except ImportError as e:
            _emulator_available = False
            warnings.warn(
                f"Emulator dependencies not available: {e}. "
                "Install flax to use the emulator: pip install flax"
            )
    return _emulator_available


class gNFWEmulator(Profile):
    """
    Generalized Navarro-Frenk-White (gNFW) profile via neural network emulator.

    This implementation uses a pre-trained neural network to approximate the
    Abel deprojection integral, providing faster evaluation at the cost of
    some accuracy loss within the emulator's training domain.

    Parameters
    ----------
    rc : float
        Scale radius in degrees.
    Ic : float
        Characteristic surface brightness (same units as image).
    alpha : float, optional
        Intermediate slope parameter. Default is 1.0510.
    beta : float, optional
        Outer slope parameter. Default is 5.4905.
    gamma : float, optional
        Inner slope parameter. Default is 0.3081.
    model : MLP or str, optional
        Pre-trained emulator model or path to a .dill file containing one.
        If None, will attempt to load from the default location.
    **kwargs
        Additional parameters passed to Profile (xc, yc, theta, e, cbox).

    Notes
    -----
    The emulator is trained on the parameter ranges defined in
    `emulate.config.Model`:

    - alpha: [0.25, 10.0]
    - beta: [0.25, 10.0]
    - gamma: [-5.0, 5.0]
    - x (r/rc): [1e-6, 100]

    Using parameters outside these ranges may produce unreliable results.

    The 3D density profile is:
    rho(r) = rho0 / [(r/rc)^gamma * (1 + (r/rc)^alpha)^((beta-gamma)/alpha)]

    Requires
    --------
    flax : Neural network library for JAX.
        Install with: pip install flax
    """

    _default_model = None
    _default_model_path = None

    def __init__(self, **kwargs):
        if not _check_emulator_available():
            raise ImportError(
                "The gNFW emulator requires flax. "
                "Install it with: pip install flax"
            )

        super().__init__(**kwargs)
        self.rc = kwargs.get("rc", config.gNFW.rc)
        self.Ic = kwargs.get("Ic", config.gNFW.Ic)
        self.alpha = kwargs.get("alpha", config.gNFW.alpha)
        self.beta = kwargs.get("beta", config.gNFW.beta)
        self.gamma = kwargs.get("gamma", config.gNFW.gamma)

        model = kwargs.get("model", None)
        self._model = self._load_model(model)

        self.okeys.append("model")
        self.okeys.append("profile")

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

        def _profile(r, Ic, rc, alpha, beta, gamma):
            return gNFWEmulator._profile(
                r, Ic, rc, alpha, beta, gamma, self._model
            )

        self.profile = jax.jit(_profile)

    def _load_model(self, model):
        """Load or retrieve the emulator model."""
        if model is None:
            if gNFWEmulator._default_model is not None:
                return gNFWEmulator._default_model
            model_path = self._find_default_model()
            if model_path is not None:
                return self._load_from_file(model_path)
            raise ValueError(
                "No emulator model provided and no default model found. "
                "Either pass a model or train one using "
                "`socca.models.radial.gnfw.emulate.rerun()`."
            )
        elif isinstance(model, str):
            return self._load_from_file(model)
        else:
            return model

    def _find_default_model(self):
        """Find the default model file if it exists."""
        module_dir = os.path.dirname(__file__)
        candidates = [
            os.path.join(module_dir, "emulate", "gnfw_emulator.dill"),
            os.path.join(module_dir, "gnfw_emulator.dill"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _load_from_file(self, path):
        """Load model from a dill file."""
        import dill

        with open(path, "rb") as f:
            data = dill.load(f)
        if isinstance(data, dict) and "model" in data:
            return data["model"]
        return data

    @classmethod
    def set_default_model(cls, model):
        """
        Set the default emulator model for all instances.

        Parameters
        ----------
        model : MLP or str
            Pre-trained emulator model or path to a .dill file.
        """
        import dill

        if isinstance(model, str):
            with open(model, "rb") as f:
                data = dill.load(f)
            if isinstance(data, dict) and "model" in data:
                cls._default_model = data["model"]
            else:
                cls._default_model = data
            cls._default_model_path = model
        else:
            cls._default_model = model
            cls._default_model_path = None

    @staticmethod
    def _profile(r, Ic, rc, alpha, beta, gamma, model):
        """
        Generalized Navarro-Frenk-White (gNFW) profile via emulator.

        Computes the projected surface brightness profile using a neural
        network approximation of the Abel transformation.

        Parameters
        ----------
        r : ndarray
            Projected elliptical radius in degrees.
        Ic : float
            Characteristic surface brightness (same units as image).
        rc : float
            Scale radius in degrees.
        alpha : float
            Intermediate slope parameter (sharpness of transition).
        beta : float
            Outer slope parameter.
        gamma : float
            Inner slope parameter (central cusp).
        model : MLP
            Pre-trained neural network emulator.

        Returns
        -------
        ndarray
            Projected surface brightness at radius r.
        """
        x = r / rc
        x = jp.clip(x, _ModelConfig.x[0], _ModelConfig.x[1])
        normalized = _predict(model, x, alpha, beta, gamma)
        return Ic * normalized
