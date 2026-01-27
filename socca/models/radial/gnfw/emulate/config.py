"""Configuration classes for the gNFW emulator."""

from dataclasses import asdict, dataclass, field

from flax import nnx


@dataclass
class Config:
    """Base configuration class with serialization support."""

    def to_dict(self):
        """Convert configuration to dictionary.

        Returns
        -------
        dict
            Dictionary representation of the configuration.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        """Create configuration from dictionary.

        Parameters
        ----------
        d : dict
            Dictionary with configuration values.

        Returns
        -------
        Config
            Configuration instance.
        """
        return cls(**d)


@dataclass
class ModelConfig(Config):
    """Configuration for gNFW model parameter ranges.

    Defines the valid ranges for each gNFW profile parameter,
    used both for training data generation and input validation.

    Attributes
    ----------
    alpha : list[float]
        Range [min, max] for the intermediate slope parameter.
        Default is [0.25, 10.0].
    beta : list[float]
        Range [min, max] for the outer slope parameter.
        Default is [0.25, 10.0].
    gamma : list[float]
        Range [min, max] for the inner slope parameter.
        Default is [-5.0, 5.0].
    x : list[float]
        Range [min, max] for the dimensionless radius r/rc.
        Default is [1e-6, 50.0].
    random_type : str
        Distribution type for random sampling. Default is "uniform".
    """

    alpha: list[float] = field(default_factory=lambda: [0.25, 10.00])
    beta: list[float] = field(default_factory=lambda: [0.25, 10.00])
    gamma: list[float] = field(default_factory=lambda: [-5.00, 5.00])
    x: list[float] = field(default_factory=lambda: [1.00e-06, 2.00e01])
    eps: float = 1.00e-08
    random_type: str = "uniform"


Model = ModelConfig()


@dataclass
class EmulatorConfig(Config):
    """Configuration for the neural network emulator architecture.

    Attributes
    ----------
    features : list[int]
        Number of neurons in each hidden layer.
        Default is [128, 128, 128, 128].
    activation_name : str
        Name of the activation function to use.
        Options: "gelu", "relu", "tanh", "silu".
        Default is "gelu".
    """

    features: list[int] = field(default_factory=lambda: [128, 128, 128, 128])
    activation_name: str = "gelu"

    @property
    def activation(self):
        """Get activation function from name.

        Returns
        -------
        callable
            The activation function corresponding to activation_name.

        Raises
        ------
        ValueError
            If activation_name is not recognized.
        """
        activations = {
            "gelu": nnx.gelu,
            "relu": nnx.relu,
            "tanh": nnx.tanh,
            "silu": nnx.silu,
        }
        if self.activation_name not in activations:
            raise ValueError(
                f"Unknown activation: {self.activation_name}. "
                f"Available: {list(activations.keys())}"
            )
        return activations[self.activation_name]


Emulator = EmulatorConfig()


@dataclass
class TrainingConfig(Config):
    """Configuration for training hyperparameters.

    Attributes
    ----------
    n_epochs : int
        Total number of training epochs. Default is 500.
    batch_size : int
        Number of samples per training batch. Default is 1024.
    learning_rate : float
        Peak learning rate for the optimizer. Default is 1e-3.
    """

    n_epochs: int = 500
    batch_size: int = 1024
    learning_rate: float = 1.00e-03


Training = TrainingConfig()
