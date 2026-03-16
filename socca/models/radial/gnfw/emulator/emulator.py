"""MLP emulator for the Abel-projected gNFW profile."""

import jax.numpy as jp
from flax import nnx

from .config import Emulator, Profile
from ..model import getzero


class MLP(nnx.Module):
    """Multi-layer perceptron emulator for the projected gNFW integral.

    Predicts log10(I(x; alpha, beta, gamma) / I(0; alpha, beta, gamma))
    as a function of the dimensionless radius x = r/rc and the three gNFW
    slope parameters. Boundary masking inputs allow accurate evaluation at
    the edges of the training parameter ranges.

    Parameters
    ----------
    features : list of int
        Hidden layer widths.
    activation : callable
        Element-wise activation function (e.g. nnx.gelu).
    rngs : nnx.Rngs
        PRNG key container for parameter initialisation.
    """

    def __init__(self, features, activation, rngs):
        self.features = features
        self.activation = activation

        # 4 inputs (x, alpha, beta, gamma) +
        # 3 boundary flags (one per slope parameter)
        self.input_layer = nnx.Linear(7, features[0], rngs=rngs)

        hidden_layers = []
        for i in range(1, len(features)):
            hidden_layers.append(
                nnx.Linear(features[i - 1], features[i], rngs=rngs)
            )
        self.hidden_layers = nnx.List(hidden_layers)

        self.output_layer = nnx.Linear(features[-1], 1, rngs=rngs)

    def __call__(
        self,
        x,
        alpha,
        beta,
        gamma,
        boundary_alpha=0,
        boundary_beta=0,
        boundary_gamma=0,
        log=False,
    ):
        """Evaluate the emulator.

        Parameters
        ----------
        x : ndarray
            Dimensionless projected radius (r / rc). Must be positive.
        alpha : float
            Intermediate slope.
        beta : float
            Outer slope.
        gamma : float
            Inner slope.
        boundary_alpha : int, optional
            Boundary flag for alpha: -1 (lower), 0 (interior), +1 (upper).
        boundary_beta : int, optional
            Boundary flag for beta.
        boundary_gamma : int, optional
            Boundary flag for gamma.
        log : bool, optional
            If True, return log10 of the normalized profile. Default is False.

        Returns
        -------
        ndarray
            Emulated profile values (or their log10) normalized by I(0).
        """
        x = jp.asarray(x, dtype=jp.float64)
        alpha = jp.broadcast_to(jp.float64(alpha), x.shape)
        beta = jp.broadcast_to(jp.float64(beta), x.shape)
        gamma = jp.broadcast_to(jp.float64(gamma), x.shape)

        x_ = _normalize(x, Profile.x[0], Profile.x[1], take_log=True)
        alpha_ = _normalize(
            alpha, Profile.alpha[0], Profile.alpha[1], take_log=False
        )
        beta_ = _normalize(
            beta, Profile.beta[0], Profile.beta[1], take_log=False
        )
        gamma_ = _normalize(
            gamma, Profile.gamma[0], Profile.gamma[1], take_log=False
        )

        fa = jp.broadcast_to(jp.float64(boundary_alpha), x.shape)
        fb = jp.broadcast_to(jp.float64(boundary_beta), x.shape)
        fg = jp.broadcast_to(jp.float64(boundary_gamma), x.shape)

        # Zero out inputs pinned to a boundary; the flag carries direction.
        alpha_ = alpha_ * jp.where(fa == 0, 1.0, 0.0)
        beta_ = beta_ * jp.where(fb == 0, 1.0, 0.0)
        gamma_ = gamma_ * jp.where(fg == 0, 1.0, 0.0)

        z = jp.stack([x_, alpha_, beta_, gamma_, fa, fb, fg], axis=-1)

        z = self.input_layer(z)
        z = self.activation(z)

        for layer in self.hidden_layers:
            z_ = self.activation(layer(z))
            z = z + z_ if z.shape[-1] == z_.shape[-1] else z_

        logy = self.output_layer(z).squeeze(-1)
        return logy if log else jp.power(10.0, logy)


def _normalize(p, pmin, pmax, take_log=False):
    """Normalise p to [0, 1]; optionally take log10 first."""
    if take_log:
        pmin = jp.log10(pmin)
        pmax = jp.log10(pmax)
        p = jp.log10(p)
    return (p - pmin) / (pmax - pmin)


def create_model(seed=42, **kwargs):
    """Instantiate a new MLP emulator with random weights.

    Parameters
    ----------
    seed : int, optional
        PRNG seed for weight initialisation. Default is 42.
    **kwargs : dict
        Optional overrides for ``features`` and ``activation``.

    Returns
    -------
    MLP
        Untrained emulator instance.
    """
    rngs = nnx.Rngs(seed)
    features = kwargs.get("features", Emulator.features)
    activation = kwargs.get("activation", Emulator.activation)
    return MLP(features=features, activation=activation, rngs=rngs)


@nnx.jit
def predict(
    model,
    x,
    alpha,
    beta,
    gamma,
    boundary_alpha=0,
    boundary_beta=0,
    boundary_gamma=0,
):
    """Evaluate the emulator and rescale to physical units.

    Returns the Abel-projected integral I(x; alpha, beta, gamma), normalised
    so that I(0) equals the exact analytical value from :func:`model.getzero`.
    The x=0 case is handled analytically.

    Parameters
    ----------
    model : MLP
        Trained emulator instance.
    x : ndarray
        Dimensionless projected radius (r / rc). May include x=0.
    alpha, beta, gamma : float
        gNFW slope parameters.
    boundary_alpha, boundary_beta, boundary_gamma : int, optional
        Boundary flags (-1, 0, +1). Use 0 for interior evaluation.

    Returns
    -------
    ndarray
        Projected integral values, same shape as x.
    """
    safe_x = jp.where(x > 0, x, jp.float64(Profile.x[0]))
    y0 = getzero(alpha, beta, gamma)
    return jp.where(
        x > 0,
        model(
            safe_x,
            alpha,
            beta,
            gamma,
            boundary_alpha,
            boundary_beta,
            boundary_gamma,
            log=False,
        )
        * y0,
        y0,
    )
