"""emcee affine-invariant ensemble MCMC backend."""

import emcee

import jax

import numpy as np

import inspect

from ...pool.mpi import FunctionTag, MPIPool
from ...pool.mpi import MPI_COMM, MPI_SIZE
from ...pool.mpi import KWCAST

from ...pool.mp import MultiPool


#   Fitting method - emcee sampler
#   --------------------------------------------------------
def _run_emcee(
    self,
    log_likelihood,
    log_prior,
    checkpoint,
    resume,
    **kwargs,
):
    """
    Run ensemble MCMC sampling using the emcee package.

    Performs Bayesian parameter estimation using the affine-invariant
    ensemble sampler from the emcee package. Supports checkpointing
    via HDF5 backends and parallelization via MPI or multiprocessing.

    Parameters
    ----------
    log_likelihood : callable
        Function that computes the log-likelihood given parameters.
    log_prior : callable
        Function that computes the log-prior given parameters.
    checkpoint : str or None
        Path to the HDF5 backend file for checkpointing. If None
        and resume=True, defaults to "run.hdf5".
    resume : bool
        If True, resume from the checkpoint file.
    **kwargs : dict
        Additional keyword arguments passed to emcee.EnsembleSampler
        and its run_mcmc method. Common options include:

        - nwalkers : int, number of walkers (default: 2 * ndim)
        - nsteps/n_steps/num_steps : int, number of MCMC steps
            (default: 5000)
        - discard/nburn/n_burn : int, number of burn-in steps to
            discard (default: 0)
        - thin : int, thinning factor (default: 1)
        - seed : int, random seed (default: None)

    Attributes Set
    --------------
    sampler : emcee.EnsembleSampler
        The emcee sampler object.
    samples : ndarray
        Posterior samples, shape (n_samples, n_params).
    weights : ndarray
        Uniform weights (all ones) since MCMC samples are unweighted.

    Notes
    -----
    emcee uses an affine-invariant ensemble of walkers to explore the
    posterior. It does not compute evidence, so it cannot be used for
    Bayesian model comparison. Initial walker positions are drawn from
    the prior distributions.
    """
    if checkpoint is None and resume:
        checkpoint = "run"

    ndim = len(self.mod.paridx)

    nwalkers = kwargs.pop("nwalkers", 2 * ndim)
    if nwalkers < 2 * ndim:
        nwalkers = 2 * ndim

    nsteps = 5000
    for key in ["nsteps", "n_steps", "num_steps"]:
        nsteps = kwargs.pop(key, nsteps)

    discard = 0
    for key in ["discard", "nburn", "n_burn"]:
        discard = kwargs.pop(key, discard)

    thin = kwargs.pop("thin", 1)

    progress = kwargs.pop("progress", True)

    seed = kwargs.pop("seed", None)

    if "ncores" in kwargs and "pool" in kwargs:
        raise ValueError("Cannot specify both 'ncores' and 'pool' arguments.")

    pool = None
    for key in ["ncores", "pool"]:
        pool = kwargs.pop(key, pool)

    ncores = None
    if isinstance(pool, int):
        ncores = pool
        pool = None

    def log_posterior(theta):
        lp = log_prior(theta)
        if not np.isfinite(float(lp)):
            return -np.inf
        ll = log_likelihood(theta)
        return float(ll) + float(lp)

    backend = None
    if checkpoint is not None:
        backend = emcee.backends.HDFBackend(checkpoint)
        if not resume:
            backend.reset(nwalkers, ndim)

    sampler_kwargs = {}
    for key in inspect.signature(emcee.EnsembleSampler).parameters.keys():
        if key not in [
            "nwalkers",
            "ndim",
            "log_prob_fn",
            "pool",
            "backend",
        ]:
            if key in kwargs:
                sampler_kwargs[key] = kwargs.pop(key)

    run_kwargs = {}
    for key in inspect.signature(
        emcee.EnsembleSampler.run_mcmc
    ).parameters.keys():
        if key not in [
            "initial_state",
            "nsteps",
            "progress",
        ]:
            if key in kwargs:
                run_kwargs[key] = kwargs.pop(key)

    if MPI_SIZE > 1:
        FunctionTag._func = log_posterior

        pool = MPIPool()

        if not pool.is_master():
            pool.wait()

            results = MPI_COMM.bcast(None, root=0)
            for key in KWCAST:
                setattr(self, key, results[key])
            return

    if ncores is not None and pool is None:
        pool = MultiPool(ncores, log_posterior)

    print("\n* Running emcee ensemble sampler")
    if isinstance(pool, MPIPool):
        log_posterior_fn = FunctionTag()
    elif isinstance(pool, MultiPool):
        log_posterior_fn = pool.likelihood
    else:
        log_posterior_fn = log_posterior

    self.sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_posterior_fn,
        pool=pool,
        backend=backend,
        **sampler_kwargs,
    )

    if resume and backend is not None and backend.iteration > 0:
        initial_state = None
        nsteps = max(0, nsteps - backend.iteration)
        print(f"  Resuming from iteration {backend.iteration}")
    else:
        rng = np.random.default_rng(seed)
        initial_state = np.empty((nwalkers, ndim))
        for wi in range(nwalkers):
            rkey = jax.random.PRNGKey(rng.integers(0, 2**31))
            keys = jax.random.split(rkey, ndim)
            for pi in range(ndim):
                prior_key = self.mod.params[self.mod.paridx[pi]]
                initial_state[wi, pi] = float(
                    self.mod.priors[prior_key].sample(keys[pi])
                )

    if nsteps > 0:
        self.sampler.run_mcmc(
            initial_state,
            nsteps,
            progress=progress,
            **run_kwargs,
        )

    self.samples = self.sampler.get_chain(
        discard=discard, thin=thin, flat=True
    )
    self.weights = np.ones(self.samples.shape[0])

    if pool is not None:
        pool.close()

    if MPI_SIZE > 1:
        MPI_COMM.bcast({key: getattr(self, key) for key in KWCAST}, root=0)
