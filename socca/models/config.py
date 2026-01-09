"""Default configuration parameters for model profiles."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Component:
    """Base configuration for model components."""

    positive: bool = False


@dataclass(frozen=True)
class Profile:
    """Default parameters for profile position and geometry."""

    xc: float = None
    yc: float = None
    theta: float = 0.00
    e: float = 0.00
    cbox: float = 0.00


@dataclass(frozen=True)
class Beta:
    """Default parameters for Beta profile."""

    rc: float = None
    Ic: float = None
    beta: float = 0.70 * 1.50 - 0.50


@dataclass(frozen=True)
class gNFW:
    """Default parameters for generalized NFW profile."""

    rc: float = None
    Ic: float = None
    alpha: float = 1.0510e00
    beta: float = 5.4905e00
    gamma: float = 3.0810e-01


@dataclass(frozen=True)
class Sersic:
    """Default parameters for Sersic profile."""

    re: float = None
    Ie: float = None
    ns: float = 0.50


@dataclass(frozen=True)
class Gaussian:
    """Default parameters for Gaussian profile."""

    rs: float = None
    Is: float = None


@dataclass(frozen=True)
class Exponential:
    """Default parameters for Exponential profile."""

    rs: float = None
    Is: float = None


@dataclass(frozen=True)
class PolyExponential:
    """Default parameters for Polynomial Exponential profile."""

    ck: float = 0.00
    rc: float = 1.00 / 3.60e03


@dataclass(frozen=True)
class PolyExpoRefact:
    """Default parameters for refactored Polynomial Exponential profile."""

    I1: float = 0.00
    I2: float = 0.00
    I3: float = 0.00
    I4: float = 0.00
    rc: float = 1.00 / 3.60e03


@dataclass(frozen=True)
class ModExponential:
    """Default parameters for Modified Exponential profile."""

    rm: float = None
    alpha: float = None


@dataclass(frozen=True)
class Point:
    """Default parameters for Point source profile."""

    xc: float = None
    yc: float = None
    Ic: float = None


@dataclass(frozen=True)
class Background:
    """Default parameters for polynomial Background model."""

    positive: bool = False
    rs: float = 1.00 / 60.00 / 60.00
    a0: float = None
    a1x: float = 0.00
    a1y: float = 0.00
    a2xx: float = 0.00
    a2xy: float = 0.00
    a2yy: float = 0.00
    a3xxx: float = 0.00
    a3xxy: float = 0.00
    a3xyy: float = 0.00
    a3yyy: float = 0.00


@dataclass(frozen=True)
class Height:
    """Default parameters for line-of-sight Height integration."""

    inc: float = 0.00
    losdepth: float = 10.00 / 60.00 / 60.00
    losbins: int = 200


@dataclass(frozen=True)
class HyperSecantHeight:
    """Default parameters for Hyperbolic Secant height profile."""

    zs: float = None
    alpha: float = 2.00


@dataclass(frozen=True)
class ExponentialHeight:
    """Default parameters for Exponential height profile."""

    zs: float = None
