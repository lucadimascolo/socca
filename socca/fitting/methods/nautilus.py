"""Nautilus neural network-accelerated nested sampling backend."""

import nautilus
import inspect

import time

from .utils import get_imp_weights


#   Fitting method - Nautilus sampler
#   --------------------------------------------------------
def _run_nautilus(
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
    """
    ndims = len(self.mod.paridx)

    nlive = 1000
    for key in ["nlive", "n_live"]:
        nlive = kwargs.pop(key, nlive)

    flive = 0.01
    for key in ["flive", "f_live"]:
        flive = kwargs.pop(key, flive)

    sampler_kwargs = {}
    for key in inspect.signature(nautilus.Sampler).parameters.keys():
        if key not in [
            "loglikelihood",
            "prior_transform",
            "n_dim",
            "n_live",
            "filepath",
            "resume",
        ]:
            if key in kwargs:
                sampler_kwargs[key] = kwargs.pop(key)

    run_kwargs = {}
    for key in inspect.signature(nautilus.Sampler.run).parameters.keys():
        if key not in ["f_live", "verbose", "discard_exploration"]:
            if key in kwargs:
                run_kwargs[key] = kwargs.pop(key)

    print("\n* Running the main sampling step")
    self.sampler = nautilus.Sampler(
        prior=prior_transform,
        likelihood=log_likelihood,
        n_dim=ndims,
        n_live=nlive,
        filepath=checkpoint,
        resume=resume,
        **sampler_kwargs,
    )

    discard_exploration = kwargs.pop("discard_exploration", True)

    toc = time.time()
    self.sampler.run(
        f_live=flive,
        verbose=True,
        discard_exploration=discard_exploration,
        **run_kwargs,
    )
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
