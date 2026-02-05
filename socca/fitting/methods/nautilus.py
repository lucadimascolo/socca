"""Nautilus neural network-accelerated nested sampling backend."""

import nautilus
import inspect

import time

from .utils import get_imp_weights

from ...pool.mpi import FunctionTag, MPIPool
from ...pool.mpi import MPI_COMM, MPI_RANK, MPI_SIZE
from ...pool.mpi import KWCAST

from ...pool.mp import MultiPool


#   Fitting method - Nautilus sampler
#   --------------------------------------------------------
def run_nautilus(
    self,
    log_likelihood,
    log_prior,
    prior_transform,
    checkpoint,
    resume,
    getzprior,
    **kwargs,
):
    """
    Run nested sampling using the Nautilus sampler.

    Performs Bayesian parameter estimation using neural network-
    accelerated nested sampling via the Nautilus package. Supports
    checkpointing, resuming, and optional prior evidence computation.

    Parameters
    ----------
    log_likelihood : callable
        Function that computes the log-likelihood given parameters.
    log_prior : callable
        Function that computes the log-prior given parameters.
    prior_transform : callable
        Function that transforms unit hypercube to parameter space.
    checkpoint : str or None
        Path to checkpoint file for saving/resuming the sampler state.
    resume : bool
        If True and checkpoint file exists, resume from saved state.
    getzprior : bool
        If True, run a second nested sampling to estimate the prior
        evidence for Bayesian model comparison with prior deboosting.
    **kwargs : dict
        Additional keyword arguments passed to nautilus.Sampler and
        its run method. Common options include:

        - nlive/n_live : int, number of live points (default: 1000)
        - flive/f_live : float, stopping criterion (default: 0.01)
        - discard_exploration : bool, discard exploration phase
            samples (default: True)

    Attributes Set
    --------------
    sampler : nautilus.Sampler
        The main Nautilus sampler object.
    samples : ndarray
        Posterior samples from nested sampling.
    logw : ndarray
        Log-weights for each sample.
    weights : ndarray
        Normalized importance weights for each sample.
    logz : float
        Log-evidence (marginal likelihood) estimate.
    sampler_prior : nautilus.Sampler or None
        Prior sampler object if getzprior=True, else None.
    logz_prior : float or None
        Prior evidence if getzprior=True, else None.

    Notes
    -----
    Nautilus uses neural networks to learn the iso-likelihood
    contours, making it efficient for high-dimensional problems.
    The method prints elapsed time after completion.

    References
    ----------
    Lange, J. U., MNRAS, 525, 3181 (2023)
    Nautilus documentation: https://nautilus-sampler.readthedocs.io/en/latest/
    """
    ndims = len(self.mod.paridx)

    nlive = 1000
    for key in ["nlive", "n_live"]:
        nlive = kwargs.pop(key, nlive)

    flive = 0.01
    for key in ["flive", "f_live"]:
        flive = kwargs.pop(key, flive)

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
    for key in inspect.signature(nautilus.Sampler).parameters.keys():
        if key not in [
            "loglikelihood",
            "prior_transform",
            "n_dim",
            "n_live",
            "filepath",
            "resume",
            "pool",
        ]:
            if key in kwargs:
                sampler_kwargs[key] = kwargs.pop(key)

    run_kwargs = {}
    for key in inspect.signature(nautilus.Sampler.run).parameters.keys():
        if key not in ["f_live", "verbose", "discard_exploration"]:
            if key in kwargs:
                run_kwargs[key] = kwargs.pop(key)

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

    self.sampler = nautilus.Sampler(
        prior=prior_transform,
        likelihood=log_likelihood,
        n_dim=ndims,
        n_live=nlive,
        filepath=checkpoint,
        resume=resume,
        pool=pool,
        **sampler_kwargs,
    )

    discard_exploration = kwargs.pop("discard_exploration", True)

    if MPI_RANK == 0:
        toc = time.time()

    self.sampler.run(
        f_live=flive,
        verbose=True,
        discard_exploration=discard_exploration,
        **run_kwargs,
    )

    if MPI_RANK == 0:
        tic = time.time()

        dt = tic - toc
        dt = (
            "{0:.2f} s".format(dt)
            if dt < 60.00
            else "{0:.2f} m".format(dt / 60.00)
        )
        print(f"Elapsed time: {dt}")

    self.samples, self.logw, _ = self.sampler.posterior()
    self.logz = self.sampler.log_z

    self.weights = get_imp_weights(self.logw, self.logz)

    if getzprior:
        print("\n* Sampling the prior probability")
        self.sampler_prior = nautilus.Sampler(
            prior_transform,
            log_prior,
            n_live=nlive,
            n_dim=ndims,
            **sampler_kwargs,
        )
        self.sampler_prior.run(
            f_live=flive,
            verbose=True,
            discard_exploration=discard_exploration,
            **run_kwargs,
        )
        self.logz_prior = self.sampler_prior.log_z
    else:
        self.sampler_prior = None
        self.logz_prior = None

    if pool is not None:
        pool.close()

    if MPI_SIZE > 1:
        MPI_COMM.bcast({key: getattr(self, key) for key in KWCAST}, root=0)
