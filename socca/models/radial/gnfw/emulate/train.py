"""Training utilities for the gNFW emulator.

This module provides functions for training the neural network
emulator, including checkpoint/resume functionality and learning
rate scheduling.
"""

import os

import dill
import jax
import jax.numpy as jp
import optax
from flax import nnx
from tqdm import tqdm

from .config import Training
from .emulator import create_model


def mse_loss(model, batch):
    """Compute mean squared error loss in log space.

    Parameters
    ----------
    model : MLP
        Neural network model to evaluate.
    batch : dict
        Batch dictionary with keys 'x', 'alpha', 'beta', 'gamma', 'logy'.

    Returns
    -------
    float
        Mean squared error between predictions and targets in log space.
    """
    pred = model(
        batch["x"], batch["alpha"], batch["beta"], batch["gamma"], log=True
    )
    return jp.mean((pred - batch["logy"]) ** 2)


@nnx.jit
def train_step(model, optimizer, batch):
    """Execute a single training step.

    Parameters
    ----------
    model : MLP
        Neural network model to train.
    optimizer : nnx.Optimizer
        Optimizer managing the model parameters.
    batch : dict
        Training batch dictionary.

    Returns
    -------
    float
        Loss value for this batch.
    """
    loss, grads = nnx.value_and_grad(mse_loss)(model, batch)
    optimizer.update(model, grads)
    return loss


def save_checkpoint(
    path, model, optimizer, epoch, history, train_data, valid_data, seed
):
    """Save a training checkpoint.

    Parameters
    ----------
    path : str
        Path to save the checkpoint file.
    model : MLP
        The model being trained.
    optimizer : nnx.Optimizer
        The optimizer state.
    epoch : int
        Current epoch number (0-indexed, completed epochs).
    history : dict
        Training history with loss values.
    train_data : dict
        Training dataset.
    valid_data : dict
        Validation dataset.
    seed : int
        Random seed used for training.
    """
    checkpoint = {
        "model": model,
        "optimizer": optimizer,
        "epoch": epoch,
        "history": history,
        "train_data": train_data,
        "valid_data": valid_data,
        "seed": seed,
    }
    with open(path, "wb") as f:
        dill.dump(checkpoint, f, dill.HIGHEST_PROTOCOL)


def load_checkpoint(path):
    """Load a training checkpoint.

    Parameters
    ----------
    path : str
        Path to the checkpoint file.

    Returns
    -------
    dict
        Checkpoint dictionary containing model, optimizer, epoch, history,
        train_data, valid_data, and seed.
    """
    with open(path, "rb") as f:
        return dill.load(f)


def train(
    train_data,
    valid_data,
    show_progress=True,
    seed=42,
    n_epochs=None,
    checkpoint=None,
    checkpoint_every=50,
    resume=False,
):
    """Train the gNFW emulator model.

    Parameters
    ----------
    train_data : dict
        Training dataset with keys 'x', 'alpha', 'beta', 'gamma', 'logy'.
    valid_data : dict
        Validation dataset with the same structure.
    show_progress : bool, optional
        Whether to show a progress bar. Default is True.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    n_epochs : int, optional
        Total number of epochs to train. If None, uses n_epochs
        from config (default 500). When resuming, set this higher
        than the original to extend training.
    checkpoint : str, optional
        Path to save checkpoints (without extension). If None, no checkpoints
        are saved. Checkpoints are saved as '{checkpoint}.ckpt'.
    checkpoint_every : int, optional
        Save checkpoint every N epochs. Default is 50.
    resume : bool, optional
        If True, attempt to resume from '{checkpoint}.ckpt' if it exists.
        Default is False.

    Returns
    -------
    model : MLP
        Trained model.
    history : dict
        Training history with 'train_loss' and 'valid_loss' lists.

    Notes
    -----
    When resuming from a checkpoint, the train_data and valid_data parameters
    are ignored and the data from the checkpoint is used instead to ensure
    consistency.

    The learning rate follows a warmup-cosine-decay schedule:

    - Warmup from 0.1x to 1x peak learning rate
    - Cosine decay to 0.01x peak learning rate
    """
    if n_epochs is None:
        n_epochs = Training.n_epochs

    start_epoch = 0
    ckpt_file = f"{checkpoint}.ckpt" if checkpoint else None

    if resume and ckpt_file and os.path.exists(ckpt_file):
        ckpt = load_checkpoint(ckpt_file)
        model = ckpt["model"]
        optimizer = ckpt["optimizer"]
        start_epoch = ckpt["epoch"] + 1
        history = ckpt["history"]
        train_data = ckpt["train_data"]
        valid_data = ckpt["valid_data"]
        seed = ckpt["seed"]

        if show_progress:
            print(f"Resuming from epoch {start_epoch}")
    else:
        n_train = len(train_data["x"])
        n_batches = n_train // Training.batch_size

        model = create_model(seed=seed)

        decay_steps = max(1000, n_epochs * n_batches)
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=Training.learning_rate * 0.1,
            peak_value=Training.learning_rate,
            warmup_steps=min(1000, decay_steps // 2),
            decay_steps=decay_steps,
            end_value=Training.learning_rate * 0.01,
        )

        optimizer = nnx.Optimizer(model, optax.adam(schedule), wrt=nnx.Param)
        history = {"train_loss": [], "valid_loss": []}

    n_train = len(train_data["x"])
    n_batches = n_train // Training.batch_size

    epoch_iterator = range(start_epoch, n_epochs)
    if show_progress:
        epoch_iterator = tqdm(
            epoch_iterator,
            desc="Training",
            initial=start_epoch,
            total=n_epochs,
        )

    for epoch in epoch_iterator:
        perm = jax.random.permutation(
            jax.random.PRNGKey(seed + epoch), n_train
        )
        perm_data = {key: val[perm] for key, val in train_data.items()}

        epoch_loss = 0.0
        for i in range(n_batches):
            start_idx = i * Training.batch_size
            end_idx = (i + 1) * Training.batch_size
            batch = {
                key: val[start_idx:end_idx] for key, val in perm_data.items()
            }
            loss = train_step(model, optimizer, batch)
            epoch_loss += loss.item()

        train_loss = epoch_loss / n_batches
        history["train_loss"].append(train_loss)

        valid_loss = mse_loss(model, valid_data).item()
        history["valid_loss"].append(valid_loss)

        if show_progress:
            epoch_iterator.set_postfix(
                {"train_loss": train_loss, "valid_loss": valid_loss}
            )

        if ckpt_file and (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(
                ckpt_file,
                model,
                optimizer,
                epoch,
                history,
                train_data,
                valid_data,
                seed,
            )
            if show_progress:
                tqdm.write(f"Checkpoint saved at epoch {epoch + 1}")

    return model, history
