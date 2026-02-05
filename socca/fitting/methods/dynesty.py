"""Dynesty nested sampling backend."""

import dill

import dynesty
import dynesty.utils

dynesty.utils.pickle_module = dill

import inspect
import os

from ...pool.mpi import FunctionTag, MPIPool
from ...pool.mpi import MPI_COMM, MPI_RANK, MPI_SIZE
from ...pool.mpi import KWCAST

from ...pool.mp import MultiPool


#   Fitting method - Dynesty sampler
#   --------------------------------------------------------
def run_dynesty(
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
    Run nested sampling using the Dynesty sampler.

    Performs Bayesian parameter estimation using nested sampling
    via the Dynesty package. Supports checkpointing, resuming,
    and optional prior evidence computation.

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
        If None, no checkpointing is performed.
    resume : bool
        If True and checkpoint file exists, resume from saved state.
    getzprior : bool
        If True, run a second nested sampling to estimate the prior
        evidence for Bayesian model comparison with prior deboosting.
    **kwargs : dict
        Additional keyword arguments passed to dynesty.NestedSampler
        and its run_nested method. Common options include:

        - nlive : int, number of live points (default: 1000)
        - dlogz : float, stopping criterion (default: 0.01)

    Attributes Set
    --------------
    sampler : dynesty.NestedSampler
        The main Dynesty sampler object.
    samples : ndarray
        Posterior samples from nested sampling.
    weights : ndarray
        Importance weights for each sample.
    logz : float
        Log-evidence (marginal likelihood) estimate.
    sampler_prior : dynesty.NestedSampler or None
        Prior sampler object if getzprior=True, else None.
    logz_prior : float or None
        Prior evidence if getzprior=True, else None.

    Notes
    -----
    The method automatically extracts valid kwargs for NestedSampler
    and run_nested based on their function signatures.

    References
    ----------
    Speagle, J. S., MNRAS, 493, 3132 (2020)
    Dynesty documentation: https://dynesty.readthedocs.io/en/v3.0.0/
    """
    if checkpoint is None:
        resume = False

    ndims = len(self.mod.paridx)

    nlive = kwargs.pop("nlive", 1000)
    dlogz = kwargs.pop("dlogz", 0.01)

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
    for key in inspect.signature(dynesty.NestedSampler).parameters.keys():
        if key not in [
            "loglikelihood",
            "prior_transform",
            "ndim",
            "nlive",
            "pool",
        ]:
            if key in kwargs:
                sampler_kwargs[key] = kwargs.pop(key)

    run_kwargs = {}
    for key in inspect.signature(
        dynesty.NestedSampler.run_nested
    ).parameters.keys():
        if key not in ["dlogz", "checkpoint_file", "resume"]:
            if key in kwargs:
                run_kwargs[key] = kwargs.pop(key)

    if MPI_SIZE > 1:
        if MPI_RANK == 0:
            print(
                "\nDynesty might not benefit from MPI "
                "parallelization due to its sequential "
                "point proposal.\nConsider using "
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

    if not resume or not os.path.exists(checkpoint):
        sampler = dynesty.NestedSampler(
            loglikelihood=log_likelihood,
            prior_transform=prior_transform,
            ndim=ndims,
            nlive=nlive,
            pool=pool,
            **sampler_kwargs,
        )
        sampler.run_nested(
            dlogz=dlogz, checkpoint_file=checkpoint, **run_kwargs
        )
    else:
        sampler = dynesty.NestedSampler.restore(checkpoint, pool=pool)
        sampler.run_nested(resume=True)

    results = sampler.results

    self.samples = results["samples"].copy()
    self.weights = results.importance_weights().copy()
    self.logw = results["logwt"].copy()
    self.logz = results["logz"][-1]

    if getzprior:
        print("\n* Sampling the prior probability")
        sampler_prior = dynesty.NestedSampler(
            log_prior,
            prior_transform,
            ndim=ndims,
            nlive=nlive,
            **sampler_kwargs,
        )
        sampler_prior.run_nested(dlogz=dlogz)
        self.logz_prior = sampler_prior.results["logz"][-1]
    else:
        self.logz_prior = None

    if pool is not None:
        pool.close()

    if MPI_SIZE > 1:
        MPI_COMM.bcast({key: getattr(self, key) for key in KWCAST}, root=0)
