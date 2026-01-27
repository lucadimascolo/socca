"""Plotting utilities for gNFW emulator training and evaluation.

This module provides functions for visualizing training progress
and evaluating emulator performance against the numerical reference.
"""

import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np

from .config import Model


def plot_training_history(history, ax=None, **kwargs):
    """Plot training and validation loss curves.

    Parameters
    ----------
    history : dict
        Training history dictionary with 'train_loss' and 'valid_loss' keys.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    **kwargs : dict
        Additional keyword arguments passed to plt.plot().

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.

    Examples
    --------
    >>> from socca.models.radial.gnfw.emulate import plotting
    >>> plotting.plot_training_history(history)
    >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    ax.plot(epochs, history["train_loss"], label="Train", **kwargs)
    ax.plot(epochs, history["valid_loss"], label="Validation", **kwargs)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log space)")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Training History")

    return ax


def plot_loss_comparison(history, ax=None):
    """Plot final train vs validation loss comparison.

    Parameters
    ----------
    history : dict
        Training history dictionary.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    final_train = history["train_loss"][-1]
    final_valid = history["valid_loss"][-1]

    bars = ax.bar(["Train", "Validation"], [final_train, final_valid])
    ax.set_ylabel("Final MSE Loss")
    ax.set_title("Final Loss Comparison")

    for bar, val in zip(bars, [final_train, final_valid]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2e}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    return ax


def plot_predictions(model, n_samples=1000, seed=42, axes=None):
    """Plot emulator predictions vs numerical reference.

    Generates random parameter samples and compares emulator
    predictions against the true numerical integral values.

    Parameters
    ----------
    model : MLP
        Trained emulator model.
    n_samples : int, optional
        Number of random samples to evaluate. Default is 1000.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    axes : array of matplotlib.axes.Axes, optional
        Array of 2 axes for scatter and residual plots.
        If None, creates a new figure.

    Returns
    -------
    axes : array of matplotlib.axes.Axes
        The axes with the plots.
    stats : dict
        Dictionary with performance statistics.
    """
    from .model import gnfw

    import jax

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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
        (n_samples,),
        minval=jp.log10(Model.x[0]),
        maxval=jp.log10(Model.x[1]),
    )
    x = jp.power(10.0, logx)

    # Compute true and predicted values
    y_true = []
    y_pred = []

    for i in range(n_samples):
        true_val = gnfw(x[i : i + 1], alpha[i], beta[i], gamma[i])[0]
        pred_val = model(x[i : i + 1], alpha[i], beta[i], gamma[i], log=False)[
            0
        ]
        if jp.isfinite(true_val) and jp.isfinite(pred_val):
            y_true.append(float(true_val))
            y_pred.append(float(pred_val))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.5, s=10)
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax1.plot(lims, lims, "k--", lw=1, label="1:1")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("True (numerical)")
    ax1.set_ylabel("Predicted (emulator)")
    ax1.set_title("Prediction vs Truth")
    ax1.legend()

    # Residual plot
    ax2 = axes[1]
    residuals = (y_pred - y_true) / y_true * 100  # Percent error
    ax2.scatter(y_true, residuals, alpha=0.5, s=10)
    ax2.axhline(0, color="k", linestyle="--", lw=1)
    ax2.set_xscale("log")
    ax2.set_xlabel("True (numerical)")
    ax2.set_ylabel("Relative error (%)")
    ax2.set_title("Residuals")

    # Compute statistics
    stats = {
        "n_samples": len(y_true),
        "mean_rel_error": np.mean(np.abs(residuals)),
        "median_rel_error": np.median(np.abs(residuals)),
        "max_rel_error": np.max(np.abs(residuals)),
        "std_rel_error": np.std(residuals),
        "rmse": np.sqrt(np.mean((y_pred - y_true) ** 2)),
        "rmse_log": np.sqrt(
            np.mean((np.log10(y_pred) - np.log10(y_true)) ** 2)
        ),
    }

    return axes, stats


def plot_parameter_slices(model, n_points=100, axes=None):
    """Plot emulator predictions along parameter slices.

    Shows how the emulator responds to variations in each parameter
    while holding others at their midpoint values.

    Parameters
    ----------
    model : MLP
        Trained emulator model.
    n_points : int, optional
        Number of points per parameter slice. Default is 100.
    axes : array of matplotlib.axes.Axes, optional
        Array of 3 axes for alpha, beta, gamma slices.
        If None, creates a new figure.

    Returns
    -------
    axes : array of matplotlib.axes.Axes
        The axes with the plots.
    """
    from .model import gnfw

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Fixed values at midpoints
    alpha_mid = (Model.alpha[0] + Model.alpha[1]) / 2
    beta_mid = (Model.beta[0] + Model.beta[1]) / 2
    gamma_mid = (Model.gamma[0] + Model.gamma[1]) / 2

    # Fixed x value
    x_test = jp.array([1.0])

    # Alpha slice
    ax = axes[0]
    alphas = np.linspace(Model.alpha[0], Model.alpha[1], n_points)
    y_true = [float(gnfw(x_test, a, beta_mid, gamma_mid)[0]) for a in alphas]
    y_pred = [
        float(model(x_test, a, beta_mid, gamma_mid, log=False)[0])
        for a in alphas
    ]
    ax.plot(alphas, y_true, "k-", label="True", lw=2)
    ax.plot(alphas, y_pred, "r--", label="Emulator", lw=2)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Profile value")
    ax.set_title(rf"$\beta$={beta_mid:.1f}, $\gamma$={gamma_mid:.1f}, x=1")
    ax.legend()

    # Beta slice
    ax = axes[1]
    betas = np.linspace(Model.beta[0], Model.beta[1], n_points)
    y_true = [float(gnfw(x_test, alpha_mid, b, gamma_mid)[0]) for b in betas]
    y_pred = [
        float(model(x_test, alpha_mid, b, gamma_mid, log=False)[0])
        for b in betas
    ]
    ax.plot(betas, y_true, "k-", label="True", lw=2)
    ax.plot(betas, y_pred, "r--", label="Emulator", lw=2)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("Profile value")
    ax.set_title(rf"$\alpha$={alpha_mid:.1f}, $\gamma$={gamma_mid:.1f}, x=1")
    ax.legend()

    # Gamma slice
    ax = axes[2]
    gammas = np.linspace(Model.gamma[0], Model.gamma[1], n_points)
    y_true = [float(gnfw(x_test, alpha_mid, beta_mid, g)[0]) for g in gammas]
    y_pred = [
        float(model(x_test, alpha_mid, beta_mid, g, log=False)[0])
        for g in gammas
    ]
    ax.plot(gammas, y_true, "k-", label="True", lw=2)
    ax.plot(gammas, y_pred, "r--", label="Emulator", lw=2)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("Profile value")
    ax.set_title(rf"$\alpha$={alpha_mid:.1f}, $\beta$={beta_mid:.1f}, x=1")
    ax.legend()

    return axes


def plot_radial_profiles(model, n_profiles=10, seed=42, ax=None):
    """Plot radial profiles for random parameter combinations.

    Parameters
    ----------
    model : MLP
        Trained emulator model.
    n_profiles : int, optional
        Number of random profiles to plot. Default is 5.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    from .model import gnfw

    import jax

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    rng = jax.random.PRNGKey(seed)
    rng, *subkeys = jax.random.split(rng, 4)

    alphas = jax.random.uniform(
        subkeys[0],
        (n_profiles,),
        minval=Model.alpha[0],
        maxval=Model.alpha[1],
    )
    betas = jax.random.uniform(
        subkeys[1], (n_profiles,), minval=Model.beta[0], maxval=Model.beta[1]
    )
    gammas = jax.random.uniform(
        subkeys[2], (n_profiles,), minval=Model.gamma[0], maxval=Model.gamma[1]
    )

    x = jp.logspace(jp.log10(Model.x[0]), jp.log10(Model.x[1]), 102)

    colors = plt.cm.viridis(np.linspace(0, 1, n_profiles))

    for i in range(n_profiles):
        y_true = gnfw(x, alphas[i], betas[i], gammas[i])
        y_pred = model(x, alphas[i], betas[i], gammas[i], log=False)

        label = rf"$\alpha$={float(alphas[i]):.1f}, $\beta$={float(betas[i]):.1f}, $\gamma$={float(gammas[i]):.1f}"
        ax.plot(x, y_true, "-", color=colors[i], lw=2, label=f"True: {label}")
        ax.plot(x, y_pred, "--", color=colors[i], lw=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("x = r/rc")
    ax.set_ylabel("Profile value")
    ax.set_title("Radial Profiles (solid=true, dashed=emulator)")
    ax.legend(fontsize=8, loc="best")

    return ax


def print_statistics(stats):
    """Print performance statistics in a formatted table.

    Parameters
    ----------
    stats : dict
        Dictionary with performance statistics from plot_predictions().
    """
    print("=" * 50)
    print("Emulator Performance Statistics")
    print("=" * 50)
    print(f"  Number of samples:      {stats['n_samples']}")
    print(f"  Mean |rel. error|:      {stats['mean_rel_error']:.4f} %")
    print(f"  Median |rel. error|:    {stats['median_rel_error']:.4f} %")
    print(f"  Max |rel. error|:       {stats['max_rel_error']:.4f} %")
    print(f"  Std rel. error:         {stats['std_rel_error']:.4f} %")
    print(f"  RMSE:                   {stats['rmse']:.6e}")
    print(f"  RMSE (log space):       {stats['rmse_log']:.6f}")
    print("=" * 50)


def summary_plot(model, history, seed=42):
    """Create a comprehensive summary plot of training and performance.

    Parameters
    ----------
    model : MLP
        Trained emulator model.
    history : dict
        Training history dictionary.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure with all plots.
    stats : dict
        Dictionary with performance statistics.
    """
    fig = plt.figure(figsize=(16, 12))

    # Training history
    ax1 = fig.add_subplot(2, 3, 1)
    plot_training_history(history, ax=ax1)

    # Final loss comparison
    ax2 = fig.add_subplot(2, 3, 2)
    plot_loss_comparison(history, ax=ax2)

    # Predictions vs truth
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 6)
    _, stats = plot_predictions(model, seed=seed, axes=[ax3, ax4])

    # Radial profiles
    ax5 = fig.add_subplot(2, 3, 4)
    plot_radial_profiles(model, seed=seed, ax=ax5)

    # Parameter slices (use remaining space)
    ax6 = fig.add_subplot(2, 3, 5)
    # Just show alpha slice in the summary
    from .model import gnfw

    alpha_mid = (Model.alpha[0] + Model.alpha[1]) / 2
    beta_mid = (Model.beta[0] + Model.beta[1]) / 2
    gamma_mid = (Model.gamma[0] + Model.gamma[1]) / 2
    x_test = jp.array([1.0])

    alphas = np.linspace(Model.alpha[0], Model.alpha[1], 100)
    y_true = [float(gnfw(x_test, a, beta_mid, gamma_mid)[0]) for a in alphas]
    y_pred = [
        float(model(x_test, a, beta_mid, gamma_mid, log=False)[0])
        for a in alphas
    ]
    ax6.plot(alphas, y_true, "k-", label="True", lw=2)
    ax6.plot(alphas, y_pred, "r--", label="Emulator", lw=2)

    betas = np.linspace(Model.beta[0], Model.beta[1], 100)
    y_true = [float(gnfw(x_test, alpha_mid, b, gamma_mid)[0]) for b in betas]
    y_pred = [
        float(model(x_test, alpha_mid, b, gamma_mid, log=False)[0])
        for b in betas
    ]
    ax6.plot(alphas, y_true, "k-", label="True", lw=2)
    ax6.plot(alphas, y_pred, "g--", label="Emulator Beta", lw=2)

    gammas = np.linspace(Model.gamma[0], Model.gamma[1], 100)
    y_true = [float(gnfw(x_test, alpha_mid, beta_mid, g)[0]) for g in gammas]
    y_pred = [
        float(model(x_test, alpha_mid, beta_mid, g, log=False)[0])
        for g in gammas
    ]
    ax6.plot(alphas, y_true, "k-", label="True", lw=2)
    ax6.plot(alphas, y_pred, "b--", label="Emulator", lw=2)

    ax6.set_yscale("log")

    ax6.set_xlabel(r"$\alpha$")
    ax6.set_ylabel("Profile value")
    ax6.set_title(
        rf"Alpha slice ($\beta$={beta_mid:.1f}, $\gamma$={gamma_mid:.1f})"
    )
    ax6.legend()

    fig.tight_layout()

    return fig, stats
