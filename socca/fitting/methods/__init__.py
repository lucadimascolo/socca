"""
Sampling methods for Bayesian inference.

This module provides various backends for posterior sampling:

Nested Sampling
---------------
- :mod:`~socca.fitting.methods.nautilus`: Neural network-accelerated nested
  sampling
- :mod:`~socca.fitting.methods.dynesty`: Dynamic nested sampling

Monte Carlo Sampling
--------------------
- :mod:`~socca.fitting.methods.pocomc`: Preconditioned Monte Carlo sampling
- :mod:`~socca.fitting.methods.emcee`: Affine-invariant ensemble MCMC
- :mod:`~socca.fitting.methods.numpyro`: NUTS sampler (experimental)

Optimization
------------
- :mod:`~socca.fitting.methods.optimizer`: Maximum a posteriori optimization
"""

from .nautilus import run_nautilus
from .dynesty import run_dynesty
from .pocomc import run_pocomc
from .numpyro import run_numpyro
from .emcee import run_emcee
from .optimizer import run_optimizer

__all__ = [
    "run_nautilus",
    "run_dynesty",
    "run_pocomc",
    "run_numpyro",
    "run_emcee",
    "run_optimizer",
]
