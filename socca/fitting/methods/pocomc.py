"""pocoMC preconditioned Monte Carlo sampling backend."""

import pocomc
import numpyro

import inspect
import glob

import time

from .utils import get_imp_weights

from ...priors import pocomcPrior

from ...pool.mpi import FunctionTag, MPIPool
from ...pool.mpi import MPI_COMM, MPI_RANK, MPI_SIZE
from ...pool.mpi import KWCAST

from ...pool.mp import MultiPool


#   Fitting method - PocoMC sampler
#   --------------------------------------------------------
def _run_pocomc(
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
        Path prefix for checkpoint files. If None and resume=True,
        defaults to "run". Checkpoint files are saved in a directory
        named "{checkpoint}_pocomc_dump".
    resume : bool
        If True, resume from the latest saved state in the checkpoint
        directory.
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
    """
    if checkpoint is None and resume:
        checkpoint = "run"

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

    n_active = kwargs.get("n_active", nlive // 2)

    progress = kwargs.pop("progress", True)

    if "ncores" in kwargs and "pool" in kwargs:
        raise ValueError("Cannot specify both 'ncores' and 'pool' arguments.")

    pool = None
    for key in ["ncores", "pool"]:
        pool = kwargs.pop(key, pool)

    ncores = None
    if isinstance(pool, int):
        ncores = pool
        pool = None

    sampler_kwargs = {}
    for key in inspect.signature(pocomc.Sampler).parameters.keys():
        if key not in [
            "likelihood",
            "prior",
            "n_effective",
            "n_active",
            "output_dir",
            "pool",
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

    if MPI_SIZE > 1:
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

    self.sampler = pocomc.Sampler(
        likelihood=log_likelihood,
        prior=prior,
        n_effective=nlive,
        n_active=n_active,
        vectorize=False,
        output_dir=pocodir if resume else None,
        pool=pool,
        **sampler_kwargs,
    )

    if MPI_SIZE > 1 and MPI_RANK == 0 or isinstance(pool, int):
        toc = time.time()

    states_ = sorted(glob.glob(f"{pocodir}/*.state")) if resume else []
    self.sampler.run(
        save_every=save_every,
        resume_state_path=states_[-1] if len(states_) else None,
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

    self.samples, self.logw, _, _ = self.sampler.posterior()
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
