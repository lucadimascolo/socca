"""Visualisation utilities for the gNFW emulator."""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jp

from .config import Profile
from .emulator import predict
from . import model as gnfw_model


def _ema(values, alpha=0.05):
    """Exponential moving average with smoothing factor alpha."""
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def plot_history(history, ax=None, smooth=True):
    """Plot training and validation loss curves vs epoch.

    Parameters
    ----------
    history : dict
        Dictionary with ``train_loss`` and ``valid_loss`` lists.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. A new figure is created if None.
    smooth : bool, optional
        If True, overlay an EMA-smoothed curve on the raw loss. Default True.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    train_loss = np.array(history["train_loss"])
    valid_loss = np.array(history["valid_loss"])

    if smooth:
        ax.plot(epochs, train_loss, alpha=0.25, color="C0")
        ax.plot(epochs, valid_loss, alpha=0.25, color="C1", linestyle="--")
        ax.plot(epochs, _ema(train_loss), label="train", color="C0")
        ax.plot(
            epochs,
            _ema(valid_loss),
            label="valid",
            color="C1",
            linestyle="--",
        )
    else:
        ax.plot(epochs, train_loss, label="train")
        ax.plot(epochs, valid_loss, label="valid", linestyle="--")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_prediction(model, alpha, beta, gamma, n_radial=100, axs=None):
    """Compare emulator predictions against the reference integral.

    Parameters
    ----------
    model : MLP
        Trained emulator instance.
    alpha, beta, gamma : float
        gNFW slope parameters.
    n_radial : int, optional
        Number of radial points. Default is 100.
    axs : array-like of Axes or None, optional
        Two-element array of Axes (profile, residuals). A new figure with two
        panels is created if None.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : ndarray of matplotlib.axes.Axes
    """
    if axs is None:
        fig, axs = plt.subplots(
            2,
            1,
            figsize=(6, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )
    else:
        fig = axs[0].get_figure()

    xr = jp.logspace(
        jp.log10(Profile.x[0]), jp.log10(Profile.x[1]), n_radial
    )

    yemu = np.array(predict(model, xr, alpha, beta, gamma))
    yref = np.array(gnfw_model.integral(xr, alpha, beta, gamma))

    rel = (yemu - yref) / yref

    axs[0].plot(xr, yref, label="reference")
    axs[0].plot(xr, yemu, label="emulator", linestyle="--")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_ylabel("y")
    axs[0].set_title(f"α={alpha}, β={beta}, γ={gamma}")
    axs[0].legend()

    axs[1].axhline(0, color="k", linewidth=0.8, linestyle="--")
    axs[1].plot(xr, rel)
    axs[1].set_xscale("log")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("(emu − ref) / ref")

    fig.tight_layout()
    return fig, axs


def plot_residuals(model, valid_data, bins=60, ax=None):
    """Histogram of per-point relative residuals over the validation set.

    Residuals are computed in log10 space: pred_logy − true_logy.

    Parameters
    ----------
    model : MLP
        Trained emulator instance.
    valid_data : dict
        Validation dataset returned by :func:`data.generate_training_data`.
    bins : int, optional
        Number of histogram bins. Default is 60.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. A new figure is created if None.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    pred_logy = np.array(
        model(
            valid_data["x"],
            valid_data["alpha"],
            valid_data["beta"],
            valid_data["gamma"],
            log=True,
        )
    )
    true_logy = np.array(valid_data["logy"])

    residuals = pred_logy - true_logy

    ax.hist(residuals, bins=bins, density=True)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("pred log y − true log y")
    ax.set_ylabel("density")

    rms = np.sqrt(np.mean(residuals**2))
    ax.set_title(f"RMS residual = {rms:.4f} dex")

    fig.tight_layout()
    return fig, ax
