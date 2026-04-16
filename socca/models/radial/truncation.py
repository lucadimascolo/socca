"""Truncation functions for radial surface brightness profiles."""

from abc import abstractmethod

import jax.numpy as jp

from .. import config


class Truncation:
    """Base class for radial truncation functions."""

    def __init__(self, **kwargs):
        self.rt = kwargs.get("rt", config.Truncation.rt)
        self.wt = kwargs.get("wt", config.Truncation.wt)

        self.units = dict(rt="deg", wt="deg")
        self.description = dict(rt="Truncation radius", wt="Truncation width")

    @abstractmethod
    def profile(self, r):
        """Return the truncation weight at radius r."""
        pass


class Exponential(Truncation):
    """Exponential truncation: smoothly cuts off the profile beyond rt."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def profile(r, rt, wt):
        """Exponential truncation profile."""
        num = 1.00 - jp.exp((r - rt) / wt)
        den = 1.00 - jp.exp(-rt / wt)
        return jp.clip(num / den, 0.00, None)


class HyperTangent(Truncation):
    """Hyperbolic tangent truncation: smoothly cuts off the profile beyond rt."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def profile(r, rt, wt):
        """Hyperbolic tangent truncation profile."""
        num = 1.00 - jp.tanh((r - rt) / wt)
        den = 1.00 - jp.tanh(-rt / wt)
        return num / den
