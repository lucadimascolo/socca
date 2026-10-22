"""Tests for socca.pool parallelization module."""

import io
import os
import shutil
import signal
import subprocess
import sys

import dill
import numpy as np
import pytest

import socca

from socca.pool.mp import (
    MultiPool,
    close_pools,
    PoolFunctionTag,
    _pool_init,
    _POOL_FUNC,
)
from socca.pool.mpi import (
    FunctionTag,
    MPI_RANK,
    MPI_SIZE,
    MPI_COMM,
    KWCAST,
    _MPIStream,
    root_only,
)


# ========================================================
# MultiPool (multi-process parallelization)
# ========================================================
class TestMultiPool:
    """Tests for the loky-based MultiPool."""

    @staticmethod
    def _square(x):
        return float(x) ** 2

    def test_creation(self):
        """Test that MultiPool can be created."""
        pool = MultiPool(2, self._square)
        assert pool.size == 2
        pool.close()

    def test_map_returns_correct_values(self):
        """Test that map evaluates the function correctly."""
        pool = MultiPool(2, self._square)
        inputs = [1.0, 2.0, 3.0, 4.0]
        results = pool.map(pool.likelihood, inputs)
        expected = [1.0, 4.0, 9.0, 16.0]
        np.testing.assert_allclose(results, expected)
        pool.close()

    def test_map_with_many_items(self):
        """Test map with more items than workers."""
        pool = MultiPool(2, self._square)
        inputs = list(range(20))
        results = pool.map(pool.likelihood, inputs)
        expected = [float(x) ** 2 for x in inputs]
        np.testing.assert_allclose(results, expected)
        pool.close()

    def test_close_pools_keeps_older_pools(self):
        """Only pools opened after the given count are killed."""
        older = MultiPool(2, self._square)
        start = len(MultiPool._open)
        newer = MultiPool(2, self._square)
        close_pools(start)
        assert newer not in MultiPool._open
        assert older.map(older.likelihood, [2.0]) == [4.0]
        older.close()

    def test_likelihood_returns_pool_function_tag(self):
        """Test that likelihood property returns a PoolFunctionTag."""
        pool = MultiPool(2, self._square)
        assert isinstance(pool.likelihood, PoolFunctionTag)
        pool.close()

    def test_close_is_idempotent(self):
        """Test that closing the pool does not raise."""
        pool = MultiPool(2, self._square)
        pool.close()
        # Second close should not error.
        pool.close()


class TestPoolFunctionTag:
    """Tests for PoolFunctionTag picklability."""

    def test_picklable_roundtrip(self):
        """Test that PoolFunctionTag survives dill serialization."""
        tag = PoolFunctionTag()
        data = dill.dumps(tag)
        restored = dill.loads(data)
        assert isinstance(restored, PoolFunctionTag)

    def test_calls_global_pool_func(self):
        """Test that __call__ invokes the worker-local function."""
        import socca.pool.mp as mp_mod

        original = mp_mod._POOL_FUNC
        try:
            mp_mod._POOL_FUNC = lambda x: float(x) * 3.0
            tag = PoolFunctionTag()
            assert tag(5.0) == 15.0
        finally:
            mp_mod._POOL_FUNC = original


class TestPoolInit:
    """Tests for the _pool_init worker initializer."""

    def test_sets_global_func(self):
        """Test that _pool_init deserializes into _POOL_FUNC."""
        import socca.pool.mp as mp_mod

        original = mp_mod._POOL_FUNC
        try:
            func = lambda x: float(x) + 1.0  # noqa: E731
            _pool_init(dill.dumps(func))
            assert mp_mod._POOL_FUNC is not None
            assert mp_mod._POOL_FUNC(4.0) == 5.0
        finally:
            mp_mod._POOL_FUNC = original


# ========================================================
# MPI module (non-MPI environment defaults)
# ========================================================
class TestMPIDefaults:
    """Tests for MPI state in a non-MPI environment."""

    def test_rank_is_zero(self):
        """MPI_RANK should be 0 when not under MPI."""
        assert MPI_RANK == 0

    def test_size_is_one(self):
        """MPI_SIZE should be 1 when not under MPI."""
        assert MPI_SIZE == 1

    def test_kwcast_keys(self):
        """KWCAST should contain the expected result attributes."""
        expected = {"samples", "weights", "logw", "logz", "logz_prior"}
        assert set(KWCAST) == expected


class TestFunctionTag:
    """Tests for the MPI FunctionTag wrapper."""

    def test_picklable_roundtrip(self):
        """Test that FunctionTag survives dill serialization."""
        tag = FunctionTag()
        data = dill.dumps(tag)
        restored = dill.loads(data)
        assert isinstance(restored, FunctionTag)

    def test_calls_class_func(self):
        """Test that __call__ uses the class-level _func."""
        original = FunctionTag._func
        try:
            FunctionTag._func = lambda x: float(x) * 2.0
            tag = FunctionTag()
            assert tag(7.0) == 14.0
        finally:
            FunctionTag._func = original


class TestRootOnly:
    """Tests for the root_only decorator."""

    def test_executes_on_rank_zero(self):
        """Function should execute normally when MPI_RANK is 0."""

        @root_only
        def greet():
            return "hello"

        # MPI_RANK is 0 in non-MPI environment.
        assert greet() == "hello"

    def test_passes_arguments(self):
        """Decorated function should receive positional and kw args."""

        @root_only
        def add(a, b, offset=0):
            return a + b + offset

        assert add(2, 3, offset=10) == 15


class TestMPIStream:
    """Tests for the _MPIStream stdout wrapper."""

    def test_plain_text_passthrough(self):
        """Regular text should pass through unchanged."""
        buf = io.StringIO()
        stream = _MPIStream(buf)
        stream.write("hello\n")
        assert buf.getvalue() == "hello\n"

    def test_trailing_cr_converted_to_newline(self):
        r"""Trailing \r should be replaced with \n."""
        buf = io.StringIO()
        stream = _MPIStream(buf)
        stream.write("progress 50%\r")
        assert "\n" in buf.getvalue()
        assert "\r" not in buf.getvalue().rstrip("\r")

    def test_leading_cr_converted(self):
        r"""Leading \r (dynesty pattern) should produce a newline."""
        buf = io.StringIO()
        stream = _MPIStream(buf)
        stream.write("\rupdate line")
        assert "\n" in buf.getvalue()

    def test_flush_delegates(self):
        """flush() should delegate to the underlying stream."""
        buf = io.StringIO()
        stream = _MPIStream(buf)
        stream.write("data")
        stream.flush()
        # No error means flush was forwarded.

    def test_getattr_delegates(self):
        """Attribute access should fall through to underlying stream."""
        buf = io.StringIO()
        stream = _MPIStream(buf)
        # StringIO has a getvalue method; _MPIStream should expose it.
        stream.write("test")
        assert stream.getvalue() == "test"


# ========================================================
# Sampler pool/ncores argument validation
# ========================================================
class TestSamplerPoolValidation:
    """Tests for ncores/pool argument handling in sampler methods."""

    @pytest.fixture
    def simple_fitter(self, simple_hdu, gaussian_psf):
        """Create a minimal fitter for argument validation tests."""
        import socca.data as data
        import socca.fitting as fitting
        import socca.models as models
        import socca.noise as noise
        import socca.priors as priors

        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))

        xc = simple_hdu.header["CRVAL1"]
        yc = simple_hdu.header["CRVAL2"]

        gaussian = models.Gaussian(
            xc=priors.uniform(xc - 0.01, xc + 0.01),
            yc=priors.uniform(yc - 0.01, yc + 0.01),
            rs=0.005,
            Is=10.0,
        )
        mod = models.Model(gaussian)
        return fitting.fitter(img=img, mod=mod)

    def test_nautilus_ncores_and_pool_raises(self, simple_fitter):
        """Nautilus should reject ncores + pool together."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            simple_fitter.run(method="nautilus", ncores=2, pool=2)

    def test_dynesty_ncores_and_pool_raises(self, simple_fitter):
        """Dynesty should reject ncores + pool together."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            simple_fitter.run(method="dynesty", ncores=2, pool=2)

    def test_pocomc_ncores_and_pool_raises(self, simple_fitter):
        """PocoMC should reject ncores + pool together."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            simple_fitter.run(method="pocomc", ncores=2, pool=2)

    def test_optimizer_rejects_pool(self, simple_fitter):
        """Optimizer should reject pool argument."""
        with pytest.raises(ValueError, match="does not support"):
            simple_fitter._run_optimizer(pinits="median", pool=2)

    def test_optimizer_rejects_ncores(self, simple_fitter):
        """Optimizer should reject ncores argument."""
        with pytest.raises(ValueError, match="does not support"):
            simple_fitter._run_optimizer(pinits="median", ncores=2)


# ========================================================
# Integration tests: parallel sampling
# ========================================================
@pytest.mark.slow
class TestParallelSamplingIntegration:
    """Integration tests running samplers with ncores > 1.

    Marked as slow; skip with ``pytest -m "not slow"``.
    """

    @pytest.fixture
    def multi_param_fitter(self, simple_hdu, gaussian_psf):
        """Create a fitter with 2 free parameters."""
        import socca.data as data
        import socca.fitting as fitting
        import socca.models as models
        import socca.noise as noise
        import socca.priors as priors

        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))

        xc = simple_hdu.header["CRVAL1"]
        yc = simple_hdu.header["CRVAL2"]

        gaussian = models.Gaussian(
            xc=priors.uniform(xc - 0.01, xc + 0.01),
            yc=priors.uniform(yc - 0.01, yc + 0.01),
            rs=0.005,
            Is=10.0,
        )
        mod = models.Model(gaussian)
        return fitting.fitter(img=img, mod=mod)

    def test_nautilus_parallel(self, multi_param_fitter, tmp_path):
        """Nautilus should produce valid output with ncores=2."""
        checkpoint = tmp_path / "nautilus_par.hdf5"
        multi_param_fitter.run(
            method="nautilus",
            ncores=2,
            n_live=1000,
            n_like_max=5000,
            checkpoint=str(checkpoint),
            verbose=False,
            resume=False,
        )

        assert multi_param_fitter.samples.shape[0] > 0
        assert multi_param_fitter.samples.shape[1] == 2
        assert np.isfinite(multi_param_fitter.logz)
        assert np.isclose(multi_param_fitter.weights.sum(), 1.0)

    def test_dynesty_parallel(self, multi_param_fitter):
        """Dynesty should produce valid output with ncores=2."""
        multi_param_fitter.run(
            method="dynesty",
            ncores=2,
            nlive=500,
            maxiter=3000,
            resume=False,
        )

        assert multi_param_fitter.samples.shape[0] > 0
        assert multi_param_fitter.samples.shape[1] == 2
        assert np.isfinite(multi_param_fitter.logz)
        assert np.isclose(multi_param_fitter.weights.sum(), 1.0)

    def test_nautilus_pool_int(self, multi_param_fitter, tmp_path):
        """Nautilus should accept pool=<int> as alias for ncores."""
        checkpoint = tmp_path / "nautilus_pool.hdf5"
        multi_param_fitter.run(
            method="nautilus",
            pool=2,
            n_live=1000,
            n_like_max=5000,
            checkpoint=str(checkpoint),
            verbose=False,
            resume=False,
        )

        assert multi_param_fitter.samples.shape[0] > 0
        assert np.isfinite(multi_param_fitter.logz)


# ========================================================
# Runs under mpiexec
# ========================================================
_MPI_SCRIPT = """
import sys
import numpy as np
from astropy.io import fits
from socca import data, fitting, models, noise, priors
from socca.pool.mpi import MPI_RANK

method, checkpoint, mode = sys.argv[1:]
header = fits.Header()
header.update(NAXIS=2, NAXIS1=16, NAXIS2=16, CRPIX1=8.5, CRPIX2=8.5,
              CRVAL1=180.0, CRVAL2=0.0, CDELT1=-1 / 3600, CDELT2=1 / 3600,
              CTYPE1="RA---TAN", CTYPE2="DEC--TAN")
y, x = np.mgrid[:16, :16]
image = np.exp(-((x - 8) ** 2 + (y - 8) ** 2) / 8.0)
scale = 2.0 if mode == "changed" else 1.0
gaussian = models.Gaussian(xc=priors.uniform(179.999, 180.001), yc=0.0,
                           rs=1e-3, Is=priors.uniform(0.1, 2.0))
fit = fitting.fitter(
    img=data.Image(fits.PrimaryHDU(image * scale, header),
                   noise=noise.Normal(sigma=0.1)),
    mod=models.Model(gaussian),
)
if mode == "raise" and MPI_RANK == 1:
    def fail(pp):
        raise ZeroDivisionError("likelihood failed")
    fit._log_likelihood = fail
kwargs = dict(emcee=dict(nsteps=10, nwalkers=4, progress=False),
              nautilus=dict(nlive=20, n_like_max=200, verbose=False))[method]
try:
    fit.run(method=method, checkpoint=checkpoint,
            resume=mode == "changed", **kwargs)
    print(f"RESULT {MPI_RANK} {len(fit.samples)}", flush=True)
except Exception as error:
    print(f"RESULT {MPI_RANK} {type(error).__name__}", flush=True)
"""


_MPI_POOL_SCRIPT = """
from socca.pool.mpi import MPI_RANK, MPIPool


class Unpicklable:
    def __call__(self, x):
        return x

    def __reduce__(self):
        raise TypeError("cannot pickle")


class Unloadable(Unpicklable):
    def __reduce__(self):
        return (Unloadable, (), {})

    def __setstate__(self, state):
        raise TypeError("cannot unpickle")


pool = MPIPool()
if not pool.is_master():
    pool.wait()
    print(f"RESULT {MPI_RANK} closed", flush=True)
else:
    for func in [Unpicklable(), Unloadable()]:
        try:
            pool.map(func, [1, 2, 3])
        except TypeError as error:
            print(f"RESULT {MPI_RANK} {error}".replace("cannot ", "cannot_"))
    print(f"RESULT {MPI_RANK} {sum(pool.map(abs, [-1, -2, -3]))}", flush=True)
    pool.close()
"""


@pytest.mark.skipif(MPI_COMM is None, reason="mpi4py is not installed")
class TestMPIRun:
    """Runs under mpiexec finish, or fail on every rank, without hanging."""

    @staticmethod
    def _mpiexec(tmp_path, *args, script=_MPI_SCRIPT):
        mpiexec = shutil.which("mpiexec") or shutil.which("mpirun")
        if mpiexec is None:
            pytest.skip("mpiexec is not available")

        path = tmp_path / "run.py"
        path.write_text(script)
        root = os.path.dirname(os.path.dirname(socca.__file__))
        env = dict(os.environ, PYTHONPATH=root)
        process = subprocess.Popen(
            [mpiexec, "-n", "2", sys.executable, str(path), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        try:
            out, _ = process.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            pytest.fail("the MPI run hung")
        lines = [line.split() for line in out.splitlines()]
        return [
            (int(line[1]), line[2]) for line in lines if line[:1] == ["RESULT"]
        ]

    def test_emcee_checkpoint(self, tmp_path):
        """An emcee run with a checkpoint finishes under MPI."""
        results = dict(
            self._mpiexec(tmp_path, "emcee", str(tmp_path / "e.h5"), "fresh")
        )
        assert results.keys() == {0, 1}
        assert len(set(results.values())) == 1
        assert results[0].isdigit()

    def test_refused_resume(self, tmp_path):
        """A refused resume stops the workers too."""
        checkpoint = str(tmp_path / "n.h5")
        self._mpiexec(tmp_path, "nautilus", checkpoint, "fresh")
        results = dict(
            self._mpiexec(tmp_path, "nautilus", checkpoint, "changed")
        )
        assert results == {0: "ValueError", 1: "RuntimeError"}

    def test_worker_error(self, tmp_path):
        """An error on a worker stops every rank."""
        results = dict(
            self._mpiexec(
                tmp_path, "nautilus", str(tmp_path / "n.h5"), "raise"
            )
        )
        assert results == {0: "RuntimeError", 1: "RuntimeError"}

    def test_emcee_bad_checkpoint(self, tmp_path):
        """A checkpoint that cannot be created stops the workers too."""
        results = dict(
            self._mpiexec(
                tmp_path, "emcee", str(tmp_path / "missing" / "e.h5"), "fresh"
            )
        )
        assert results == {0: "FileNotFoundError", 1: "RuntimeError"}

    def test_unpicklable_tasks(self, tmp_path):
        """Tasks that fail to (un)pickle raise and keep the ranks in step."""
        results = self._mpiexec(tmp_path, script=_MPI_POOL_SCRIPT)
        assert sorted(results) == [
            (0, "6"),
            (0, "cannot_pickle"),
            (0, "cannot_unpickle"),
            (1, "closed"),
        ]
