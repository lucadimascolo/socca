"""Neural network emulator for the gNFW profile.

This module defines the MLP architecture and utility functions
for the neural network emulator that approximates the gNFW
Abel integral.
"""

import jax.numpy as jp
from flax import nnx

from .config import Emulator, Model


class MLP(nnx.Module):
    """Multi-layer perceptron for gNFW profile emulation.

    A feedforward neural network with residual connections that maps
    (x, alpha, beta, gamma) inputs to the gNFW Abel integral values.

    Parameters
    ----------
    features : list[int]
        Number of neurons in each hidden layer.
    activation : callable
        Activation function to use between layers.
    rngs : nnx.Rngs
        Random number generator state for weight initialization.

    Attributes
    ----------
    features : list[int]
        Number of neurons in each hidden layer.
    activation : callable
        Activation function used between layers.
    input_layer : nnx.Linear
        Input layer mapping 4 inputs to first hidden size.
    hidden_layers : nnx.List[nnx.Linear]
        List of hidden layers with residual connections.
    output_layer : nnx.Linear
        Output layer producing single scalar output.
    """

    def __init__(self, features, activation, rngs):
        """Initialize the MLP.

        Parameters
        ----------
        features : list[int]
            Number of neurons in each hidden layer.
        activation : callable
            Activation function to use between layers.
        rngs : nnx.Rngs
            Random number generator state for weight initialization.
        """
        self.features = features
        self.activation = activation

        self.input_layer = nnx.Linear(4, features[0], rngs=rngs)

        hidden_layers = []
        for i in range(1, len(features)):
            hidden_layers.append(
                nnx.Linear(features[i - 1], features[i], rngs=rngs)
            )
        self.hidden_layers = nnx.List(hidden_layers)

        self.output_layer = nnx.Linear(features[-1], 1, rngs=rngs)

    def __call__(self, x, alpha, beta, gamma, log=False):
        """Evaluate the emulator at given parameters.

        Parameters
        ----------
        x : float or ndarray
            Dimensionless radius r/rc.
        alpha : float
            Intermediate slope parameter.
        beta : float
            Outer slope parameter.
        gamma : float
            Inner slope parameter.
        log : bool, optional
            If True, return log10(y). If False, return y.
            Default is False.

        Returns
        -------
        float or ndarray
            Emulated Abel integral value(s). Returns log10(y) if
            log=True, otherwise returns y.
        """
        x = jp.asarray(x, dtype=jp.float64)
        alpha = jp.broadcast_to(jp.float64(alpha), x.shape)
        beta = jp.broadcast_to(jp.float64(beta), x.shape)
        gamma = jp.broadcast_to(jp.float64(gamma), x.shape)

        x_ = normalize(jp.log10(x), jp.log10(Model.x[0]), jp.log10(Model.x[1]))
        alpha_ = normalize(alpha, Model.alpha[0], Model.alpha[1])
        beta_ = normalize(beta, Model.beta[0], Model.beta[1])
        gamma_ = normalize(gamma, Model.gamma[0], Model.gamma[1])

        z = jp.stack([x_, alpha_, beta_, gamma_], axis=-1)

        z = self.input_layer(z)
        z = self.activation(z)

        for layer in self.hidden_layers:
            z_ = self.activation(layer(z))
            z = z + z_ if z.shape[-1] == z_.shape[-1] else z_

        logy = self.output_layer(z).squeeze(-1)
        return logy if log else jp.power(10.0, logy)


def normalize(p, pmin, pmax):
    """Normalize a parameter to the [0, 1] range.

    Parameters
    ----------
    p : float or ndarray
        Parameter value(s) to normalize.
    pmin : float
        Minimum value of the parameter range.
    pmax : float
        Maximum value of the parameter range.

    Returns
    -------
    float or ndarray
        Normalized value(s) in [0, 1].
    """
    return (p - pmin) / (pmax - pmin)


def create_model(seed=42, **kwargs):
    """Create a new MLP model instance.

    Parameters
    ----------
    seed : int, optional
        Random seed for weight initialization. Default is 42.
    **kwargs : dict, optional
        Override default configuration:
        - features : list[int], hidden layer sizes
        - activation : callable, activation function

    Returns
    -------
    MLP
        Initialized neural network model.
    """
    rngs = nnx.Rngs(seed)
    features = kwargs.get("features", Emulator.features)
    activation = kwargs.get("activation", Emulator.activation)
    return MLP(features=features, activation=activation, rngs=rngs)


@nnx.jit
def predict(model, x, alpha, beta, gamma):
    """Make predictions with a trained model.

    Parameters
    ----------
    model : MLP
        Trained emulator model.
    x : float or ndarray
        Dimensionless radius r/rc.
    alpha : float
        Intermediate slope parameter.
    beta : float
        Outer slope parameter.
    gamma : float
        Inner slope parameter.

    Returns
    -------
    float or ndarray
        Predicted Abel integral values (not log-transformed).
    """
    return model(x, alpha, beta, gamma, log=False)
