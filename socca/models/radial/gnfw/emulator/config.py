"""Configuration dataclasses for gNFW emulator training and inference."""

from dataclasses import asdict, dataclass, field


@dataclass
class Config:
    """Base configuration class."""

    def to_dict(self):
        """Serialize to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        """Deserialize from dict."""
        return cls(**d)


@dataclass
class ProfileConfig(Config):
    """Parameter ranges for the gNFW emulator training domain."""

    alpha: list = field(default_factory=lambda: [0.05, 10.00])
    beta: list = field(default_factory=lambda: [3.00, 10.00])
    gamma: list = field(default_factory=lambda: [-5.00, 1.00])
    x: list = field(default_factory=lambda: [1.00e-08, 1.00e02])
    random_type: str = "uniform"


@dataclass
class EmulatorConfig(Config):
    """Architecture configuration for the MLP emulator."""

    features: list = field(default_factory=lambda: [128, 128, 128, 128])
    activation_name: str = "gelu"

    @property
    def activation(self):
        """Return the activation function from its name."""
        try:
            from flax import nnx
        except ImportError as e:
            raise ImportError(
                "The gNFW emulator requires 'flax'. "
                "Install with: pip install flax"
            ) from e

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


@dataclass
class TrainingConfig(Config):
    """Hyperparameters for emulator training."""

    n_epochs: int = 2000
    batch_size: int = 2048
    learning_rate: float = 1.00e-03


Profile = ProfileConfig()
Emulator = EmulatorConfig()
Training = TrainingConfig()
