"""Multiprocessing pool for parallel likelihood evaluation."""

import dill
from loky import ProcessPoolExecutor
import loky.process_executor as _loky_pe

# Raise loky's memory-leak threshold to 1 GB.  JAX JIT compilation
# legitimately allocates large buffers in each worker; the default
# 300 MB limit causes loky to kill and restart workers repeatedly.
_loky_pe._MAX_MEMORY_LEAK_SIZE = int(1e9)

_POOL_FUNC = None


def _pool_init(func_bytes):
    """Initialize worker with deserialized likelihood function."""
    global _POOL_FUNC
    _POOL_FUNC = dill.loads(func_bytes)


class PoolFunctionTag:
    """Picklable proxy that invokes the pre-loaded likelihood.

    Analogous to :class:`~socca.pool.mpi.FunctionTag` for MPI.
    Each worker process pre-loads the likelihood via the pool
    initializer; this lightweight tag simply forwards calls to
    that pre-loaded function, avoiding re-serialization of the
    full fitter state at runtime.
    """

    def __call__(self, theta):  # noqa: D102
        return float(_POOL_FUNC(theta))

    def __reduce__(self):  # noqa: D105
        return (PoolFunctionTag, ())


class MultiPool:
    """Process pool for parallel likelihood evaluation.

    Uses loky for robust process management with cloudpickle
    serialization. Workers do not re-import ``__main__``, so no
    ``if __name__ == "__main__":`` guard is needed in user scripts.

    The likelihood is pre-loaded in each worker via the pool
    initializer for fast repeated evaluation. Other functions
    (e.g. nautilus neural network training) are serialized with
    cloudpickle transparently by loky.

    Parameters
    ----------
    processes : int
        Number of worker processes.
    likelihood : callable
        Likelihood function to evaluate in parallel.
    """

    def __init__(self, processes, likelihood):
        self._n_processes = processes
        func_bytes = dill.dumps(likelihood)
        self._executor = ProcessPoolExecutor(
            max_workers=processes,
            initializer=_pool_init,
            initargs=(func_bytes,),
            timeout=None,
        )

    @property
    def size(self):
        """Return the number of worker processes."""
        return self._n_processes

    @property
    def likelihood(self):
        """Return a picklable proxy for the pre-loaded likelihood."""
        return PoolFunctionTag()

    def map(self, func, iterable):
        """Map a function over an iterable in parallel."""
        return list(self._executor.map(func, iterable))

    def close(self):
        """Shut down the worker pool."""
        self._executor.shutdown(wait=True)
