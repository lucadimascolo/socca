"""emcee affine-invariant ensemble MCMC backend."""

import numpy as np
import jax

import inspect
import warnings

import emcee

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
            (default: 5000). Acts as maximum when converge=True.
        - discard/nburn/n_burn : int, number of burn-in steps to
            discard (default: 0). When converge=True and not set,
            auto-set to 2 * max(tau).
        - thin : int, thinning factor (default: 1)
        - seed : int, random seed (default: None)
        - converge : bool, if True run until convergence based on
            autocorrelation time estimates (default: False)
        - check_every : int, check convergence every N steps
            (default: 100)
        - tau_factor : float, require chain length > tau_factor * tau
            for convergence (default: 50)
        - tau_rtol : float, relative tolerance for tau stability
            (default: 0.01)
        - thin_factor : float, thinning factor applied to tau when checking convergence
            (default: 0.50)
        - discard_factor : float, factor multiplied by tau to determine burn-in discard when converge=True
            (default: 2.0)

    Attributes Set
    --------------
    sampler : emcee.EnsembleSampler
        The emcee sampler object.
    samples : ndarray
        Posterior samples, shape (n_samples, n_params).
    weights : ndarray
        Uniform weights (all ones) since MCMC samples are unweighted.
    tau : ndarray or None
        Integrated autocorrelation time per parameter, if computed.

    Notes
    -----
    emcee uses an affine-invariant ensemble of walkers to explore the
    posterior. It does not compute evidence, so it cannot be used for
    Bayesian model comparison. Initial walker positions are drawn from
    the prior distributions.

    When converge=True, the sampler runs in batches and checks the
    integrated autocorrelation time after each batch. Convergence
    is declared when (1) the chain is longer than tau_factor * tau
    for all parameters, and (2) the tau estimate has stabilized
    (relative change < tau_rtol). The burn-in (discard) is then
    auto-set to 2 * max(tau) unless explicitly provided.
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

    _discard_sentinel = object()
    discard = _discard_sentinel
    for key in ["discard", "nburn", "n_burn"]:
        discard = kwargs.pop(key, discard)
    user_set_discard = discard is not _discard_sentinel
    if not user_set_discard:
        discard = 0

    _thin_sentinel = object()
    thin = _thin_sentinel
    for key in ["thin"]:
        thin = kwargs.pop(key, thin)
    user_set_thin = thin is not _thin_sentinel
    if not user_set_thin:
        thin = 1

    converge = kwargs.pop("converge", False)
    check_every = kwargs.pop("check_every", 100)
    tau_factor = kwargs.pop("tau_factor", 50)
    tau_rtol = kwargs.pop("tau_rtol", 0.01)
    thin_factor = kwargs.pop("thin_factor", 0.50)
    discard_factor = kwargs.pop("discard_factor", 2.00)

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

    istep = 0
    nsteps_total = nsteps
    if resume and backend is not None and backend.iteration > 0:
        initial_state = None
        istep = backend.iteration
        nsteps = max(0, nsteps - istep)
        print(f"Resuming from iteration {istep}")
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

    self.tau = None
    self.tau_history = []
    if nsteps > 0:
        if converge:
            old_tau = np.inf
            state = initial_state
            for batch_start in range(
                check_every * (istep // check_every), nsteps_total, check_every
            ):
                batch = min(check_every, nsteps_total - batch_start)
                batch = batch - istep % check_every

                state = self.sampler.run_mcmc(
                    state, batch, progress=progress, **run_kwargs
                )

                istep = self.sampler.iteration

                try:
                    tau = self.sampler.get_autocorr_time(tol=0)
                except emcee.autocorr.AutocorrError:
                    if progress:
                        print(
                            f"Step {istep}/{nsteps_total}: tau not yet reliable\n"
                        )
                    continue

                if not np.all(np.isfinite(tau)):
                    if progress:
                        nfinite = np.sum(np.isfinite(tau))
                        print(
                            f"Step {istep}"
                            f"/{nsteps_total}: "
                            f"tau reliable for {nfinite}"
                            f"/{ndim} params\n"
                        )
                    continue

                self.tau_history.append((istep, tau.copy()))

                is_converged = np.all(tau * tau_factor < istep)
                is_converged &= np.all(np.abs(old_tau - tau) / tau < tau_rtol)

                if progress:
                    print(
                        f"Step {istep}"
                        f"/{nsteps_total}: "
                        f"max(tau) = {np.max(tau):.1f}\n"
                    )

                if is_converged:
                    print(f"Converged at step {istep}")
                    break

                old_tau = tau
            else:
                warnings.warn(
                    f"Chain reached maximum steps ({nsteps_total})"
                    f" without converging. Consider increasing"
                    f" nsteps.\n"
                )

            try:
                tau_final = self.sampler.get_autocorr_time(tol=0)
                if np.all(np.isfinite(tau_final)):
                    self.tau = tau_final
            except emcee.autocorr.AutocorrError:
                pass
        else:
            self.sampler.run_mcmc(
                initial_state,
                nsteps,
                progress=progress,
                **run_kwargs,
            )
    elif nsteps == 0 and converge:
        if len(self.tau_history) == 0:
            samples_ = self.sampler.get_chain(discard=0, thin=1)

            for batch in range(0, samples_.shape[0], check_every):
                batch_samples = samples_[0 : batch + check_every]
                tau = emcee.autocorr.integrated_time(batch_samples, tol=0)

                if np.all(np.isfinite(tau)):
                    self.tau_history.append((batch, tau.copy()))
            del samples_

        if self.tau is None:
            try:
                tau_final = self.sampler.get_autocorr_time(tol=0)
                if np.all(np.isfinite(tau_final)):
                    self.tau = tau_final
            except emcee.autocorr.AutocorrError:
                pass

    if self.tau is not None and converge:
        if not user_set_discard:
            discard = int(discard_factor * np.max(self.tau))
            print(f"Discarding {discard} steps based on tau estimates.")

        if not user_set_thin:
            thin = max(1, int(thin_factor * np.min(self.tau)))
            print(f"Auto-setting thin={thin} based on tau estimates.")

    self.samples = self.sampler.get_chain(
        discard=discard, thin=thin, flat=True
    )
    self.weights = np.ones(self.samples.shape[0])

    if pool is not None:
        pool.close()

    if MPI_SIZE > 1:
        MPI_COMM.bcast({key: getattr(self, key) for key in KWCAST}, root=0)
