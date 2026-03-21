"""Training loop for the gNFW MLP emulator."""

import os

import jax
import jax.numpy as jp
import optax
from flax import nnx
from tqdm import tqdm

from .config import Training
from .emulator import create_model
from .io import load_checkpoint, save_checkpoint


def mse_loss(model, batch):
    """Mean squared error in log10-normalised space."""
    pred = model(
        batch["x"],
        batch["alpha"],
        batch["beta"],
        batch["gamma"],
        boundary_alpha=batch.get("boundary_alpha", 0),
        boundary_beta=batch.get("boundary_beta", 0),
        boundary_gamma=batch.get("boundary_gamma", 0),
        log=True,
    )
    return jp.mean((pred - batch["logy"]) ** 2)


@nnx.jit
def train_step(model, optimizer, batch):
    """Perform a single gradient-descent update and return the batch loss."""
    loss, grads = nnx.value_and_grad(mse_loss)(model, batch)
    optimizer.update(model, grads)
    return loss


def _make_stratified_batches(train_data, rng, interior_fraction=0.70):
    """Build stratified batch index arrays with a fixed interior/boundary ratio.

    Parameters
    ----------
    train_data : dict
        Training dataset as returned by :func:`data.generate_training_data`.
    rng : jax.random.PRNGKey
        PRNG key for shuffling.
    interior_fraction : float, optional
        Fraction of each batch drawn from interior (non-boundary) samples.
        Default is 0.70.

    Returns
    -------
    batches : ndarray, shape (n_batches, batch_size)
        Index arrays; each row is one batch.
    n_batches : int
        Number of batches.
    """
    flags = (
        jp.abs(train_data["boundary_alpha"])
        + jp.abs(train_data["boundary_beta"])
        + jp.abs(train_data["boundary_gamma"])
    )
    interior_idx = jp.where(flags == 0)[0]
    boundary_idx = jp.where(flags != 0)[0]

    n_per_batch_interior = int(Training.batch_size * interior_fraction)
    n_per_batch_boundary = Training.batch_size - n_per_batch_interior

    n_batches = min(
        len(interior_idx) // n_per_batch_interior,
        len(boundary_idx) // n_per_batch_boundary,
    )

    rng, k1, k2 = jax.random.split(rng, 3)
    perm_i = interior_idx[
        jax.random.permutation(k1, len(interior_idx))[
            : n_batches * n_per_batch_interior
        ]
    ]
    perm_b = boundary_idx[
        jax.random.permutation(k2, len(boundary_idx))[
            : n_batches * n_per_batch_boundary
        ]
    ]

    perm_i = perm_i.reshape(n_batches, n_per_batch_interior)
    perm_b = perm_b.reshape(n_batches, n_per_batch_boundary)

    return jp.concatenate([perm_i, perm_b], axis=1), n_batches


def train(
    train_data,
    valid_data,
    checkpoint_path=None,
    show_progress=True,
    seed=42,
    model=None,
    resume=True,
    plot_interval=None,
):
    """Train (or resume training of) the gNFW MLP emulator.

    Parameters
    ----------
    train_data : dict
        Training dataset returned by :func:`data.generate_training_data`.
    valid_data : dict
        Validation dataset returned by :func:`data.generate_training_data`.
    checkpoint_path : str or None, optional
        Path for saving/loading dill checkpoints after each epoch. Enables
        resume behaviour. Default is None (no checkpointing).
    show_progress : bool, optional
        Show a tqdm progress bar. Default is True.
    seed : int, optional
        PRNG seed for weight initialisation and batch shuffling. Default is 42.
    model : MLP or None, optional
        Pre-existing model to continue training. If None, a fresh model is
        created. Ignored when resuming from a checkpoint.
    resume : bool, optional
        If True and ``checkpoint_path`` exists, resume from that checkpoint.
        Default is True.
    plot_interval : int or None, optional
        If given, save a training-history PDF every this many epochs (requires
        ``checkpoint_path`` to construct the output filename).

    Returns
    -------
    model : MLP
        Trained emulator.
    history : dict
        Training history with ``train_loss`` and ``valid_loss`` lists.
    """
    has_boundary = "boundary_alpha" in train_data
    n_train = len(train_data["x"])

    if has_boundary:
        _, n_batches = _make_stratified_batches(
            train_data, jax.random.PRNGKey(seed)
        )
    else:
        n_batches = n_train // Training.batch_size

    if model is None:
        model = create_model(seed=seed)

    decay_steps = max(1000, Training.n_epochs * n_batches)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=Training.learning_rate * 0.1,
        peak_value=Training.learning_rate,
        warmup_steps=min(1000, decay_steps // 2),
        decay_steps=decay_steps,
        end_value=Training.learning_rate * 0.001,
    )

    optimizer = nnx.Optimizer(
        model,
        optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule)),
        wrt=nnx.Param,
    )

    history = {"train_loss": [], "valid_loss": []}
    start_epoch = 0

    if not resume and checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = load_checkpoint(checkpoint_path)
        model = ckpt["model"]
        optimizer = ckpt["optimizer"]
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1

    epoch_iterator = range(start_epoch, Training.n_epochs)
    if show_progress:
        epoch_iterator = tqdm(epoch_iterator, desc="Training")

    for epoch in epoch_iterator:
        rng = jax.random.PRNGKey(seed + epoch)

        if has_boundary:
            batches, _ = _make_stratified_batches(train_data, rng)
        else:
            perm = jax.random.permutation(rng, n_train)
            batches = perm[: n_batches * Training.batch_size].reshape(
                n_batches, Training.batch_size
            )

        epoch_loss = jp.float32(0.0)
        for i in range(n_batches):
            batch_idx = batches[i]
            batch = {key: val[batch_idx] for key, val in train_data.items()}
            loss = train_step(model, optimizer, batch)
            epoch_loss = epoch_loss + loss

        train_loss = (epoch_loss / n_batches).item()
        history["train_loss"].append(train_loss)

        valid_loss = mse_loss(model, valid_data).item()
        history["valid_loss"].append(valid_loss)

        if show_progress:
            epoch_iterator.set_postfix(
                {"train_loss": train_loss, "valid_loss": valid_loss}
            )

        if checkpoint_path:
            save_checkpoint(checkpoint_path, model, optimizer, epoch, history)

        if plot_interval and (epoch + 1) % plot_interval == 0:
            import matplotlib.pyplot as plt

            from . import plot as emulator_plot

            stem = os.path.splitext(checkpoint_path or "history")[0]
            emulator_plot.plot_history(history)
            plt.savefig(
                f"{stem}_history_{epoch + 1:04d}.pdf",
                format="pdf",
                dpi=150,
            )
            plt.close()

    return model, history
