"""Data generation utilities for training the gNFW emulator."""

import jax
import jax.numpy as jp
from tqdm import tqdm

from .config import Model
from .model import gnfw


def generate_training_data(n_samples, n_radial, show_progress=True, seed=42):
    """Generate training data for the gNFW emulator.

    Creates a dataset of (x, alpha, beta, gamma, log10(y)) pairs by
    evaluating the numerical gNFW Abel integral for randomly sampled
    parameter combinations.

    Parameters
    ----------
    n_samples : int
        Number of parameter combinations to sample.
    n_radial : int
        Number of radial points to evaluate per parameter combination.
    show_progress : bool, optional
        Whether to show a progress bar. Default is True.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'x': Dimensionless radius values (r/rc)
        - 'alpha': Intermediate slope parameter values
        - 'beta': Outer slope parameter values
        - 'gamma': Inner slope parameter values
        - 'logy': Log10 of the Abel integral values

        Each value is a JAX array of shape (n_valid,) where n_valid
        is the number of finite output values (some parameter
        combinations may produce NaN/Inf).

    Notes
    -----
    The parameters are sampled uniformly within the ranges defined
    in `config.Model`:

    - alpha: [0.25, 10.0]
    - beta: [0.25, 10.0]
    - gamma: [-5.0, 5.0]
    - x: [1e-6, 50.0] (log-uniform)

    Non-finite values (NaN, Inf) are filtered out from the final dataset.
    """
    rng = jax.random.PRNGKey(seed)
    rng, *subkeys = jax.random.split(rng, 5)

    alpha = jax.random.uniform(
        subkeys[0], (n_samples,), minval=Model.alpha[0], maxval=Model.alpha[1]
    )
    beta = jax.random.uniform(
        subkeys[1], (n_samples,), minval=Model.beta[0], maxval=Model.beta[1]
    )
    gamma = jax.random.uniform(
        subkeys[2], (n_samples,), minval=Model.gamma[0], maxval=Model.gamma[1]
    )

    logx = jax.random.uniform(
        subkeys[3],
        (n_samples, n_radial),
        minval=jp.log10(Model.x[0]),
        maxval=jp.log10(Model.x[1]),
    )
    x = jp.power(10.00, logx)

    iterator = range(n_samples)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating training data")

    data = dict(x=[], alpha=[], beta=[], gamma=[], logy=[])

    for i in iterator:
        y = gnfw(x[i], alpha=alpha[i], beta=beta[i], gamma=gamma[i])

        for ri in range(n_radial):
            data["x"].append(x[i, ri])
            data["alpha"].append(alpha[i])
            data["beta"].append(beta[i])
            data["gamma"].append(gamma[i])
            data["logy"].append(jp.log10(y[ri]))

    for key in data.keys():
        data[key] = jp.array(data[key])

    valid = jp.isfinite(data["logy"])

    return {key: val[valid] for key, val in data.items()}
