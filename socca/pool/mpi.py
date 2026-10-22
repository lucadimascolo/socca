"""MPI pool for parallel likelihood evaluation."""

from functools import wraps

import dill
import os
import sys
import traceback

KWCAST = ["samples", "weights", "logw", "logz", "logz_prior"]


# Environment variables set by common MPI launchers
# --------------------------------------------------------
_MPI_ENV_VARS = [
    "OMPI_COMM_WORLD_SIZE",  # Open MPI
    "PMI_SIZE",  # MPICH / Intel MPI
    "MPI_LOCALNRANKS",  # MS-MPI
    "SLURM_MPI_TYPE",  # SLURM + MPI
]


# Initialize MPI
# --------------------------------------------------------
try:
    from mpi4py import MPI

    MPI.pickle.__init__(dill.dumps, dill.loads)

    MPI_COMM = MPI.COMM_WORLD
    MPI_RANK = MPI_COMM.Get_rank()
    MPI_SIZE = MPI_COMM.Get_size()

except ImportError:
    if any(var in os.environ for var in _MPI_ENV_VARS):
        raise ImportError(
            "mpi4py is not installed, but the process was "
            "launched with mpirun/mpiexec. "
            "To enable MPI support, install mpi4py (pip install mpi4py) "
            "or re-install socca with MPI support (pip install socca[mpi])."
        )
    MPI_COMM = None
    MPI_RANK = 0
    MPI_SIZE = 1


# MPI-aware stdout wrapper
# --------------------------------------------------------
class _MPIStream:
    r"""Stdout wrapper that emulates ``\r`` overwriting through MPI pipes.

    MPI launchers forward pipe output line-by-line, so progress
    updates that end with ``\r`` (carriage return) are held in the
    pipe buffer until a ``\n`` arrives.  This wrapper converts
    ``\r`` to ``\n`` for immediate flushing, then uses ANSI escape
    codes (cursor-up + clear-line) on the next write to overwrite
    the previous line, preserving the visual effect of in-place
    updates.

    Two progress-output patterns are handled:

    * **Trailing** ``\r`` (nautilus): ``print(text, end='\\r')``
      produces ``write(text)`` then ``write('\\r')``.
    * **Leading** ``\r`` (dynesty): ``stderr.write('\\r' + text)``
      produces a single ``write('\\r...')``.
    """

    _UP_AND_CLEAR = "\033[A\033[2K\r"

    def __init__(self, stream):
        self._stream = stream
        self._cr_pending = False

    def write(self, text):
        r"""Write *text*, emulating ``\r`` via ANSI escapes."""
        if not text:
            return 0

        # Leading \r (dynesty pattern: "\r" + text)
        # Strip it and treat the write as an overwrite line.
        has_leading_cr = len(text) > 1 and text[0] == "\r" and text[1] != "\n"
        if has_leading_cr:
            text = text.lstrip("\r")
            if self._cr_pending:
                self._stream.write(self._UP_AND_CLEAR)
            text = text + "\n"
            self._cr_pending = True
            return self._stream.write(text)

        # Trailing \r (nautilus pattern: write(text) then write('\r'))
        if self._cr_pending:
            self._stream.write(self._UP_AND_CLEAR)
            self._cr_pending = False
        if text.endswith("\r"):
            text = text[:-1] + "\n"
            self._cr_pending = True
        return self._stream.write(text)

    def flush(self):
        """Flush the underlying stream."""
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


if MPI_SIZE > 1:
    sys.stdout.reconfigure(write_through=True)
    sys.stdout = _MPIStream(sys.stdout)

    sys.stderr.reconfigure(write_through=True)
    sys.stderr = _MPIStream(sys.stderr)


# MPI-aware function wrapper
# --------------------------------------------------------
class FunctionTag:
    """Picklable function wrapper for MPI transport.

    Each MPI rank sets the class-level ``_func`` to its own local
    function before entering the worker loop. When pickled,
    ``__reduce__`` reconstructs a new instance on the receiving
    rank that calls that rank's own ``_func``, avoiding
    serialization of the full fitter state.
    """

    _func = None

    def __call__(self, theta):  # noqa: D102
        return float(FunctionTag._func(theta))

    def __reduce__(self):  # noqa: D105
        return (FunctionTag, ())


# Custom MPI pool
# --------------------------------------------------------
class MPIPool:
    """MPI pool where all ranks evaluate tasks."""

    # pool of the run in progress on this rank, until its results are sent
    _active = None

    def __init__(self, comm=None):
        if comm is None:
            comm = MPI.COMM_WORLD

        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.closed = False
        MPIPool._active = self

    def is_master(self):
        """Return True if this is the master rank."""
        return self.rank == 0

    def map(self, func, iterable):
        """Scatter tasks, evaluate on all ranks, gather results."""
        tasks = list(iterable)
        n = len(tasks)

        chunk_size = -(-n // self.size)
        padded = tasks + [None] * (chunk_size * self.size - n)
        # serialize before any message, so a failure here leaves workers idle
        func = dill.dumps(func)
        chunks = [
            dill.dumps(padded[i * chunk_size : (i + 1) * chunk_size])
            for i in range(self.size)
        ]

        self.comm.bcast(func, root=0)
        local_chunk = self.comm.scatter(chunks, root=0)
        try:
            results = _evaluate(func, local_chunk)
        finally:
            all_results = self.comm.gather(None, root=0)

        for chunk in all_results[1:]:
            if isinstance(chunk, _Failure):
                raise RuntimeError(f"An MPI worker failed:\n{chunk}")
            results.extend(dill.loads(chunk))
        return results[:n]

    def wait(self):
        """Block non-root ranks, processing batches until closed."""
        while True:
            func = self.comm.bcast(None, root=0)
            if not func:
                return

            local_chunk = self.comm.scatter(None, root=0)
            try:
                local_results = dill.dumps(_evaluate(func, local_chunk))
            except Exception:
                local_results = _Failure(traceback.format_exc())
            self.comm.gather(local_results, root=0)

    def close(self):
        """Signal workers to exit."""
        if not self.closed:
            self.closed = True
            self.comm.bcast(False, root=0)


class _Failure(str):
    """Error message sent between ranks in place of results."""


def _evaluate(func, chunk):
    """Evaluate a serialized function on a serialized chunk of tasks."""
    func, chunk = dill.loads(func), dill.loads(chunk)
    return [func(t) if t is not None else None for t in chunk]


def send_results(fitter, keys=KWCAST):
    """Send the results of a run from the master to the workers."""
    results = {key: getattr(fitter, key) for key in keys}
    MPIPool._active = None
    MPI_COMM.bcast(results, root=0)


def stop_ranks(error):
    """After a failed run, make waiting workers raise, or end all ranks."""
    pool, MPIPool._active = MPIPool._active, None
    if pool is None:
        return
    if not isinstance(error, Exception):
        print(
            f"\nRun interrupted on MPI rank {MPI_RANK}: ending all ranks.",
            file=sys.stderr,
            flush=True,
        )
        MPI_COMM.Abort(1)
    if pool.is_master():
        pool.close()
        MPI_COMM.bcast(_Failure(f"{type(error).__name__}: {error}"), root=0)


def receive_results(fitter):
    """Set the results of a run on a worker, or raise if the master failed."""
    results = MPI_COMM.bcast(None, root=0)
    MPIPool._active = None
    if isinstance(results, _Failure):
        raise RuntimeError(f"The MPI master process failed: {results}")
    for key, value in results.items():
        setattr(fitter, key, value)


# Root decorator for mpi-safe functions
# --------------------------------------------------------
def root_only(func):
    """Only execute on rank 0. Returns None on other ranks."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if MPI_RANK == 0:
            return func(*args, **kwargs)

    return wrapper
