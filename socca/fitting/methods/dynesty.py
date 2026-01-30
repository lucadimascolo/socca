"""Dynesty nested sampling backend."""

import dynesty
import inspect
import os


#   Fitting method - Dynesty sampler
#   --------------------------------------------------------
def _run_dynesty(
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
    """
    if checkpoint is None:
        resume = False

    ndims = len(self.mod.paridx)

    nlive = kwargs.pop("nlive", 1000)
    dlogz = kwargs.pop("dlogz", 0.01)

    sampler_kwargs = {}
    for key in inspect.signature(dynesty.NestedSampler).parameters.keys():
        if key not in [
            "loglikelihood",
            "prior_transform",
            "ndim",
            "nlive",
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

    print("\n* Running the main sampling step")
    if ~resume or not os.path.exists(checkpoint):
        sampler = dynesty.NestedSampler(
            loglikelihood=log_likelihood,
            prior_transform=prior_transform,
            ndim=ndims,
            nlive=nlive,
            **sampler_kwargs,
        )
        sampler.run_nested(
            dlogz=dlogz, checkpoint_file=checkpoint, **run_kwargs
        )
    else:
        sampler = dynesty.NestedSampler.restore(checkpoint)
        sampler.run_nested(resume=True)

    self.sampler = sampler

    results = sampler.results

    self.samples = results["samples"].copy()
    self.weights = results.importance_weights().copy()
    self.logz = results["logz"][-1]

    if getzprior:
        print("\n* Sampling the prior probability")
        self.sampler_prior = dynesty.NestedSampler(
            log_prior,
            prior_transform,
            ndim=ndims,
            nlive=nlive,
            **sampler_kwargs,
        )
        self.sampler_prior.run_nested(dlogz=dlogz)
        self.logz_prior = self.sampler_prior.results["logz"][-1]
    else:
        self.sampler_prior = None
        self.logz_prior = None
