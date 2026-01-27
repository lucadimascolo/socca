"""gNFW emulator training module.

This module provides tools for training a neural network emulator
that approximates the gNFW Abel integral. The trained emulator
enables faster evaluation of gNFW profiles compared to numerical
integration.

Main functions
--------------
rerun
    Train or resume training of the emulator.
generate_training_data
    Generate training data by evaluating the numerical integral.
train
    Core training loop with checkpoint support.

Submodules
----------
plotting
    Visualization tools for training history and emulator performance.

Example usage
-------------
Train the default emulator:

>>> from socca.models.radial.gnfw import emulate
>>> model, history = emulate.rerun()

Resume interrupted training:

>>> model, history = emulate.rerun(resume=True)

Plot training results:

>>> from socca.models.radial.gnfw.emulate import plotting
>>> fig, stats = plotting.summary_plot(model, history)
>>> plotting.print_statistics(stats)

Command-line usage:

.. code-block:: bash

    gnfw-emulator --n_train 100000 --n_epochs 500
"""

import os

os.environ["JAX_ENABLE_X64"] = "True"
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)

import dill

from .data import generate_training_data
from .train import train
from . import plotting

# Default model path in the emulate directory
_MODULE_DIR = os.path.dirname(__file__)
DEFAULT_MODEL_PATH = os.path.join(_MODULE_DIR, "gnfw_emulator")


def rerun(
    name=None,
    n_train=100000,
    n_valid=10000,
    n_radial=100,
    n_epochs=None,
    checkpoint_every=50,
    resume=False,
    seed=42,
    plot=False,
):
    """Train or resume training of a gNFW emulator model.

    Parameters
    ----------
    name : str, optional
        Base name for output files (without extension). If None, saves to
        the default location in the emulate directory so that gNFWEmulator
        can find it automatically.
    n_train : int, optional
        Number of training samples to generate. Default is 100,000.
        Ignored when resuming from checkpoint.
    n_valid : int, optional
        Number of validation samples to generate. Default is 10,000.
        Ignored when resuming from checkpoint.
    n_radial : int, optional
        Number of radial points per sample. Default is 100.
        Ignored when resuming from checkpoint.
    n_epochs : int, optional
        Total number of epochs to train. If None, uses the default from
        config (500). When resuming, set this higher than the original
        to extend training beyond the initial run.
    checkpoint_every : int, optional
        Save checkpoint every N epochs. Default is 50.
    resume : bool, optional
        If True, attempt to resume from checkpoint if it exists.
        Default is False.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
        Ignored when resuming from checkpoint.
    plot : bool, optional
        If True, display summary plots and performance statistics
        after training. Default is False.

    Returns
    -------
    model : MLP
        Trained emulator model.
    history : dict
        Training history with 'train_loss' and 'valid_loss'.

    Examples
    --------
    Train and save to default location (recommended):

    >>> from socca.models.radial.gnfw import emulate
    >>> model, history = emulate.rerun()

    Resume interrupted training:

    >>> model, history = emulate.rerun(resume=True)

    Extend training to more epochs:

    >>> model, history = emulate.rerun(n_epochs=1000, resume=True)

    Train with plotting enabled:

    >>> model, history = emulate.rerun(plot=True)

    Save to a custom location:

    >>> model, history = emulate.rerun(name='/path/to/my_model')
    """
    if name is None:
        name = DEFAULT_MODEL_PATH

    if resume and os.path.exists(f"{name}.ckpt"):
        train_data = None
        valid_data = None
    else:
        train_data = generate_training_data(
            n_samples=n_train, n_radial=n_radial, seed=seed + 10
        )
        valid_data = generate_training_data(
            n_samples=n_valid, n_radial=n_radial, seed=seed + 20
        )

    model, history = train(
        train_data,
        valid_data,
        seed=seed,
        n_epochs=n_epochs,
        checkpoint=name,
        checkpoint_every=checkpoint_every,
        resume=resume,
    )

    with open(f"{name}.dill", "wb") as f:
        dill.dump(dict(model=model, history=history), f, dill.HIGHEST_PROTOCOL)

    if plot:
        import matplotlib.pyplot as plt

        fig, stats = plotting.summary_plot(model, history, seed=seed)
        plotting.print_statistics(stats)
        plt.show()

    return model, history
