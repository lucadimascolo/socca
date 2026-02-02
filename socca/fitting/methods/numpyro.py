"""NumPyro NUTS Hamiltonian Monte Carlo backend."""

import numpyro

import jax
import numpy as np

import inspect

from ...pool.mpi import MPI_COMM, MPI_RANK, MPI_SIZE


#   Fitting method - Numpyro NUTS
#   --------------------------------------------------------
def _run_numpyro(self, log_likelihood, **kwargs):
    """
    Run Hamiltonian Monte Carlo sampling using NumPyro's NUTS.

    Performs Bayesian parameter estimation using the No-U-Turn
    Sampler (NUTS), a variant of Hamiltonian Monte Carlo (HMC),
    via the NumPyro package.

    Parameters
    ----------
    log_likelihood : callable
        Function that computes the log-likelihood given parameters.
    **kwargs : dict
        Additional keyword arguments passed to numpyro.infer.NUTS
        and numpyro.infer.MCMC. Common options include:

        - n_warmup/nwarmup/num_warmup : int, number of warmup
            iterations (default: 1000)
        - n_samples/nsamples/num_samples : int, number of posterior
            samples (default: 2000)
        - seed : int, random seed (default: 0)

    Attributes Set
    --------------
    samples : ndarray
        Posterior samples from MCMC, shape (n_samples, n_params).
    weights : ndarray
        Uniform weights (all ones) since MCMC samples are unweighted.

    Notes
    -----
    NUTS automatically tunes step size and number of leapfrog steps
    during the warmup phase. This method does not compute evidence,
    so it cannot be used for Bayesian model comparison. The method
    requires parameter priors to be NumPyro distributions.
    """
    if MPI_SIZE > 1:
        MPI_COMM.bcast(None, root=0)
        if MPI_RANK != 0:
            raise SystemExit(0)
        raise ValueError(
            "NumPyro NUTS does not support MPI parallelization.\n "
        )

    num_chains = 1
    for key in ["pool", "ncores", "n_cores", "num_chains"]:
        num_chains = kwargs.pop(key, num_chains)

    nwarmup = 1000
    for key in ["n_warmup", "nwarmup", "num_warmup"]:
        nwarmup = kwargs.pop(key, nwarmup)

    nsamples = 2000
    for key in ["n_samples", "nsamples", "num_samples"]:
        nsamples = kwargs.pop(key, nsamples)

    def model():
        pp = []
        for pi, p in enumerate(self.mod.paridx):
            key = self.mod.params[self.mod.paridx[pi]]
            pp.append(numpyro.sample(key, self.mod.priors[key]))

        numpyro.factor("post", log_likelihood(pp))

    rkey = jax.random.PRNGKey(kwargs.pop("seed", 0))
    rkey, seed = jax.random.split(rkey)

    nuts_kwargs = {}
    for key in inspect.signature(numpyro.infer.NUTS).parameters.keys():
        if key not in ["model"]:
            if key in kwargs:
                nuts_kwargs[key] = kwargs.pop(key)

    mcmc_kwargs = {}
    for key in inspect.signature(numpyro.infer.MCMC).parameters.keys():
        if key not in ["nuts", "num_warmup", "num_samples", "num_chains"]:
            if key in kwargs:
                mcmc_kwargs[key] = kwargs.pop(key)

    nuts = numpyro.infer.NUTS(model, **nuts_kwargs)
    mcmc = numpyro.infer.MCMC(
        nuts,
        num_warmup=nwarmup,
        num_samples=nsamples,
        num_chains=num_chains,
        **mcmc_kwargs,
    )
    mcmc.run(seed, **kwargs)

    samp = mcmc.get_samples()

    self.samples = np.array(
        [samp[self.mod.params[p]] for p in self.mod.paridx]
    ).T
    self.weights = np.ones(self.samples.shape[0])
