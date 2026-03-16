"""gNFW emulator subpackage.

Provides tools to train and use a neural-network emulator for the
Abel-projected generalised NFW pressure profile integral, as an
alternative to slow numerical quadrature.

Typical workflow
----------------
1. Generate training/validation data::

    from socca.models.radial.gnfw.emulator import data
    train = data.generate_training_data(5000, 200, n_edge=0.15,
                                        output_path="train.h5")
    valid = data.generate_training_data(500,   25, n_edge=0.15,
                                        output_path="valid.h5")

2. Train the emulator::

    from socca.models.radial.gnfw.emulator import train
    model, history = train(train, valid, checkpoint_path="gnfw.dill")

3. Use the emulator inside a gNFW profile::

    import socca.models as sm
    gnfw = sm.gNFW(emulator="gnfw.dill")

Optional dependencies
---------------------
Training requires ``flax``, ``optax``, and ``tqdm``.
Inference (loading and evaluating a pre-trained model) requires ``flax``.
"""

from . import data
from .. import model
from . import io
from . import plot
from .trainer import train
from .emulator import MLP, create_model, predict

__all__ = [
    "data",
    "model",
    "io",
    "plot",
    "train",
    "MLP",
    "create_model",
    "predict",
]
