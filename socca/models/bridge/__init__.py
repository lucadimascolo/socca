"""Bridge/filament models for intracluster structures.

This module provides models for describing elongated emission structures
such as intracluster bridges and filaments connecting galaxy clusters.

Classes
-------
SimpleBridge
    Bridge model with multiplicative profile combination.
MesaBridge
    Bridge model with harmonic mean (mesa-like) profile combination.
"""

from .base import SimpleBridge, MesaBridge

__all__ = ["SimpleBridge", "MesaBridge"]
