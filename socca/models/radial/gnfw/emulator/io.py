"""I/O utilities for gNFW emulator checkpoints and training data."""

import dill
import h5py
import numpy as np
import jax.numpy as jp


def save_data(path, data):
    """Save a training/validation dataset to an HDF5 file.

    Parameters
    ----------
    path : str
        Output file path.
    data : dict
        Dictionary mapping field names to arrays.
    """
    with h5py.File(path, "w") as f:
        for key, val in data.items():
            f.create_dataset(key, data=np.array(val))


def load_data(path):
    """Load a training/validation dataset from an HDF5 file.

    Parameters
    ----------
    path : str
        Input file path.

    Returns
    -------
    dict
        Dictionary mapping field names to JAX arrays.
    """
    with h5py.File(path, "r") as f:
        return {key: jp.array(f[key][:]) for key in f.keys()}


def save_checkpoint(path, model, optimizer, epoch, history):
    """Save a training checkpoint to a dill file.

    Parameters
    ----------
    path : str
        Output file path.
    model : MLP
        Current emulator state.
    optimizer : nnx.Optimizer
        Current optimiser state.
    epoch : int
        Last completed epoch index.
    history : dict
        Training history with ``train_loss`` and ``valid_loss`` lists.
    """
    with open(path, "wb") as f:
        dill.dump(
            {
                "model": model,
                "optimizer": optimizer,
                "epoch": epoch,
                "history": history,
            },
            f,
        )


def load_checkpoint(path):
    """Load a training checkpoint from a dill file.

    Parameters
    ----------
    path : str
        Input file path.

    Returns
    -------
    dict
        Checkpoint dictionary with keys ``model``, ``optimizer``, ``epoch``,
        and ``history``.
    """
    with open(path, "rb") as f:
        return dill.load(f)
