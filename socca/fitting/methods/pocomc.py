"""pocoMC preconditioned Monte Carlo sampling backend."""

import jax
import pocomc
import numpyro
import numpy as np

import dill
import inspect
import glob
import os

import time

import tqdm.utils


# Patch tqdm ObjectWrapper to prevent RecursionError during
# dill unpickling of pocomc checkpoint files. When dill
# deserializes a tqdm progress bar, _wrapped may not yet be
# set, causing __getattr__ to recurse infinitely.
def _safe_ow_getattr(self, name):
    if name == "_wrapped":
        raise AttributeError(name)
    try:
        wrapped = object.__getattribute__(self, "_wrapped")
    except AttributeError:
        raise AttributeError(name)
    return getattr(wrapped, name)


tqdm.utils.ObjectWrapper.__getattr__ = _safe_ow_getattr

from .utils import check_resume, get_imp_weights, warn_unused

from ...priors import pocomcPrior

from ...pool.mpi import FunctionTag, MPIPool
from ...pool.mpi import MPI_COMM, MPI_RANK, MPI_SIZE
from ...pool.mpi import KWCAST

from ...pool.mp import MultiPool


class _Sampler(pocomc.Sampler):
    """pocomc sampler that keeps its functions and pool when resuming."""

    _keep = [
        "prior",
        "log_prior",
        "sample_prior",
        "bounds",
        "log_likelihood",
        "vectorize",
        "pool",
        "distribute",
    ]

    def load_state(self, path):
        """Load a saved state, but not its likelihood, prior or pool."""
        keep = {key: self.__dict__[key] for key in self._keep}
        super().load_state(path)
        self.__dict__.update(keep)
        # older states hold read-only JAX arrays
        if isinstance(self.prior_samples, jax.Array):
            self.prior_samples = np.array(self.prior_samples)
        for key, value in (self.current_particles or {}).items():
            if isinstance(value, jax.Array):
                self.current_particles[key] = np.array(value)


def _latest_state(pocodir):
    """Most recent pocomc state file in pocodir, by iteration, or None."""

    def order(path):
        tag = os.path.basename(path)[: -len(".state")].rsplit("_", 1)[-1]
        return (tag == "final", int(tag) if tag.isdigit() else -1)

    states = glob.glob(os.path.join(pocodir, "*.state"))
    return max(states, key=order) if states else None


#   Fitting method - PocoMC sampler
#   --------------------------------------------------------
def run_pocomc(
    self,
    log_likelihood,
    log_prior,
    checkpoint,
    resume,
    getzprior,
    **kwargs,
):
    """
    Run nested sampling using the pocoMC sampler.

    Performs Bayesian parameter estimation using preconditioned
    Monte Carlo nested sampling via the pocoMC package. Supports
    checkpointing, resuming, and optional prior evidence computation.

    Parameters
    ----------
    log_likelihood : callable
        Function that computes the log-likelihood given parameters.
    log_prior : callable
        Function that computes the log-prior given parameters.
    checkpoint : str or None
        Path prefix for checkpoint files, saved in a directory named
        "{checkpoint}_pocomc_dump". If None, no checkpointing is
        performed.
    resume : bool
        If True and a checkpoint is given, resume from the latest saved
        state in the checkpoint directory.
    getzprior : bool
        If True, run a second nested sampling to estimate the prior
        evidence for Bayesian model comparison with prior deboosting.
    **kwargs : dict
        Additional keyword arguments passed to pocomc.Sampler and
        its run method. Common options include:

        - nlive/n_live/n_effective : int, effective sample size
            (default: 1000)
        - n_active : int, number of active particles
            (default: nlive // 2)
        - save_every : int, save state every N iterations
            (default: 10)
        - seed : int, random seed (default: 0)
        - vectorize : bool, if True evaluate the likelihood on the
            full batch of active particles at once using JAX vmap,
            rather than calling it once per particle. Incompatible
            with pool/ncores and MPI (raises ValueError if combined).
            (default: False)

    Attributes Set
    --------------
    sampler : pocomc.Sampler
        The main pocoMC sampler object.
    samples : ndarray
        Posterior samples from nested sampling.
    logw : ndarray
        Log-weights for each sample.
    weights : ndarray
        Normalized importance weights for each sample.
    logz : float
        Log-evidence (marginal likelihood) estimate.
    sampler_prior : pocomc.Sampler or None
        Prior sampler object if getzprior=True, else None.
    logz_prior : float or None
        Prior evidence if getzprior=True, else None.

    Notes
    -----
    pocoMC uses normalizing flows for preconditioned sampling,
    making it efficient for complex, multimodal posteriors.
    Prior distributions must be NumPyro distributions.

    References
    ----------
    Karamanis, M., Beutler, F., Peacock, J. A., Nabergoj, D., Seljak, U., MNRAS, 516, 1644 (2022)
    Karamanis, M., Nabergoj, D., Beutler, F., Peacock, J. A., Seljak, U., arXiv:2207.05660 (2022)
    pocoMC documentation: https://pocomc.readthedocs.io/en/latest/
    """
    if checkpoint is None:
        resume = False
    loglike = log_likelihood

    save_every = (
        kwargs.pop("save_every", 10) if checkpoint is not None else None
    )

    pocodir = (
        "{0}_pocomc_dump".format(
            checkpoint.replace(".hdf5", "").replace(".h5", "")
        )
        if checkpoint is not None
        else None
    )

    nlive = 1000
    for key in ["nlive", "n_live", "n_effective"]:
        nlive = kwargs.pop(key, nlive)

    n_active = kwargs.pop("n_active", nlive // 2)

    progress = kwargs.pop("progress", True)

    if "ncores" in kwargs and "pool" in kwargs:
        raise ValueError("Cannot specify both 'ncores' and 'pool' arguments.")

    vectorize = kwargs.pop("vectorize", False)

    pool = None
    for key in ["ncores", "pool"]:
        pool = kwargs.pop(key, pool)

    ncores = None
    if isinstance(pool, int):
        ncores = pool
        pool = None

    if vectorize and (ncores is not None or pool is not None):
        raise ValueError(
            "pocomc does not support combining vectorize=True with a pool. "
            "Use either JAX vectorization (vectorize=True) or "
            "multiprocessing (pool/ncores), not both."
        )

    if vectorize and MPI_SIZE > 1:
        raise ValueError(
            "pocomc does not support combining vectorize=True with MPI. "
            "Use either JAX vectorization (vectorize=True, single node) or "
            "MPI parallelization (vectorize=False), not both."
        )

    sampler_kwargs = {}
    for key in inspect.signature(pocomc.Sampler).parameters.keys():
        if key not in [
            "likelihood",
            "prior",
            "n_effective",
            "n_active",
            "output_dir",
            "pool",
            "vectorize",
        ]:
            if key in kwargs:
                sampler_kwargs[key] = kwargs.pop(key)

    run_kwargs = {}
    for key in inspect.signature(pocomc.Sampler.run).parameters.keys():
        if key not in ["save_every", "resume_state_path", "progress"]:
            if key in kwargs:
                run_kwargs[key] = kwargs.pop(key)

    prior = []
    for key in self.mod.params:
        if isinstance(
            self.mod.priors[key], numpyro.distributions.Distribution
        ):
            prior.append(self.mod.priors[key])
    prior = pocomcPrior(prior, seed=kwargs.pop("seed", 0))
    warn_unused(kwargs, "pocomc")

    if MPI_SIZE > 1:
        if MPI_RANK == 0:
            print(
                "\npocomc might not benefit signicantly "
                " from MPI parallelization.\nConsider using "
                "method='nautilus' for MPI runs."
            )

        FunctionTag._func = log_likelihood

        pool = MPIPool()

        if not pool.is_master():
            pool.wait()

            results = MPI_COMM.bcast(None, root=0)
            for key in KWCAST:
                setattr(self, key, results[key])
            return

    if ncores is not None and pool is None:
        pool = MultiPool(ncores, log_likelihood)

    print("\n* Running the main sampling step")
    if isinstance(pool, MPIPool):
        log_likelihood = FunctionTag()
    elif isinstance(pool, MultiPool):
        log_likelihood = pool.likelihood

    if vectorize:
        _ll_vmap = jax.jit(jax.vmap(log_likelihood))

        def log_likelihood(theta_batch):
            return np.array(_ll_vmap(theta_batch))

    state = _latest_state(pocodir) if resume else None
    if state is not None:
        with open(state, "rb") as f:
            particles = dill.load(f)["particles"]
        x = particles.get("x", -1)
        check_resume(state, loglike, x, particles.get("logl", -1))
        check_resume(
            state,
            lambda xi: prior.logpdf(xi[None])[0],
            x,
            particles.get("logp", -1),
        )

    self.sampler = _Sampler(
        likelihood=log_likelihood,
        prior=prior,
        n_effective=nlive,
        n_active=n_active,
        vectorize=vectorize,
        output_dir=pocodir if checkpoint is not None else None,
        pool=pool,
        **sampler_kwargs,
    )

    if MPI_SIZE > 1 and MPI_RANK == 0 or isinstance(pool, int):
        toc = time.time()

    self.sampler.run(
        save_every=save_every,
        resume_state_path=state,
        progress=progress,
        **run_kwargs,
    )

    if MPI_SIZE > 1 and MPI_RANK == 0 or isinstance(pool, int):
        tic = time.time()

        dt = tic - toc
        dt = (
            "{0:.2f} s".format(dt)
            if dt < 60.00
            else "{0:.2f} m".format(dt / 60.00)
        )
        print(f"Elapsed time: {dt}")

    self.samples, self.logw, _, _ = self.sampler.posterior(return_logw=True)
    self.logz, _ = self.sampler.evidence()

    self.weights = get_imp_weights(self.logw, self.logz)

    if getzprior:
        print("\n* Sampling the prior probability")
        self.sampler_prior = pocomc.Sampler(
            likelihood=log_prior,
            prior=prior,
            n_effective=nlive,
            n_active=n_active,
            vectorize=False,
            **sampler_kwargs,
        )
        self.sampler_prior.run(progress=progress, **run_kwargs)
        self.logz_prior, _ = self.sampler_prior.evidence()
    else:
        self.sampler_prior = None
        self.logz_prior = None

    if pool is not None:
        pool.close()

    if MPI_SIZE > 1:
        MPI_COMM.bcast({key: getattr(self, key) for key in KWCAST}, root=0)
