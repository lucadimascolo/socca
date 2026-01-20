"""Bayesian inference engine for astronomical image modeling."""

from functools import partial

import jax
import jax.numpy as jp
import numpy as np

import numpyro
import numpyro.distributions

import dynesty
import nautilus
import pocomc
import scipy.special
import scipy.optimize

from .priors import pocomcPrior
from .plotting import Plotter

import inspect
import dill
import glob
import os
import time
import warnings
from pathlib import Path

from astropy.io import fits


# Support functions
# ========================================================
# Compute importance weights for nested sampling
# --------------------------------------------------------
def get_imp_weights(logw, logz=None):
    r"""
    Compute importance weights from log-weights and log-evidence.

    Converts log-weights to normalized importance weights using the
    log-evidence for numerical stability. The weights are normalized
    such that they sum to 1.0.

    Parameters
    ----------
    logw : array_like
        Log-weights from nested sampling.
    logz : float or array_like, optional
        Log-evidence value(s). If None, uses the maximum log-weight.
        If not None and not iterable, converts to a single-element list.
        Default is None.

    Returns
    -------
    weights : ndarray
        Normalized importance weights in linear space.

    Notes
    -----
    The importance weights are computed as:

    .. math::
        w_i = \exp(\log w_i - \log Z - \log\sum_j \exp(\log w_j - \log Z))

    where :math:`\log Z` is the log-evidence (logz[-1]).
    """
    if logz is None:
        logz = [logw.max()]
    if not hasattr(logz, "__len__"):
        logz = [logz]

    wt = logw - logz[-1]
    wt = wt - scipy.special.logsumexp(wt)
    return np.exp(wt)


# Fitter constructor
# ========================================================
# Initialize fitter structure
# --------------------------------------------------------
class fitter:
    """
    Main inference engine for fitting astronomical models to image data.

    The fitter class orchestrates Bayesian inference using nested sampling
    (dynesty, nautilus) or MCMC (numpyro) methods. It manages likelihood
    computation, prior transformations, and sampler execution.
    """

    def __init__(self, img, mod):
        """
        Initialize the fitter with an image and model.

        Sets up the fitting infrastructure by extracting noise properties,
        parameter labels, and initializing the plotting interface.

        Parameters
        ----------
        img : Image
            Image object containing the data, noise model, response function,
            and exposure map.
        mod : Model
            Model object defining the forward model, parameters, priors, and
            parameter transformations.

        Attributes
        ----------
        img : Image
            Reference to the input image object.
        mod : Model
            Reference to the input model object.
        mask : ndarray
            Boolean mask from the noise model indicating valid pixels.
        pdfnoise : callable
            Log-probability density function from the noise model.
        pdfkwarg : list of str
            Parameter names expected by the noise PDF function.
        labels : list of str
            Parameter names for the fitted parameters.
        units : list of str
            Physical units for each fitted parameter.
        plot : Plotter
            Plotting interface for visualization of results.
        """
        self.img = img
        self.mod = mod

        self.mask = self.img.noise.mask
        self.pdfnoise = self.img.noise.logpdf
        self.pdfkwarg = [
            key for key in inspect.signature(self.pdfnoise).parameters.keys()
        ]

        if not hasattr(self.img, "shape"):
            setattr(self.img, "shape", self.img.data.shape)
        else:
            self.img.shape = self.img.data.shape

        self.labels = [self.mod.params[idx] for idx in self.mod.paridx]
        self.units = [
            self.mod.units[self.mod.params[idx]] for idx in self.mod.paridx
        ]

        self.plot = Plotter(self)

    #   Compute total model
    #   --------------------------------------------------------
    def _get_model(self, pp):
        """
        Compute the total model with response and exposure corrections.

        Automatically determines whether to apply the instrument response
        function and exposure map based on whether they deviate from unity.

        Parameters
        ----------
        pp : array_like
            Model parameters in the parameter space.

        Returns
        -------
        model_raw : ndarray
            Raw model before convolution and response application.
        model_smooth : ndarray
            Model after convolution and response/exposure corrections.
        model_background : ndarray
            Background component of the model.
        negative_flag : ndarray
            Boolean array indicating pixels with negative values.

        Notes
        -----
        Response is applied if any element of img.resp differs from 1.0.
        Exposure is applied if any element of img.exp differs from 1.0.
        """
        doresp = ~np.all(np.array(self.img.resp) == 1.00)  # True
        doexp = ~np.all(np.array(self.img.exp) == 1.00)  # True
        return self.mod.getmodel(
            self.img, pp, doresp=doresp, doexp=doexp, component=None
        )

    #   Compute log-likelihood
    #   --------------------------------------------------------
    @partial(jax.jit, static_argnames=["self"])
    def _log_likelihood(self, pp):
        """
        Compute the log-likelihood for given parameters.

        Evaluates the noise model's log-probability density function on
        the masked pixels. Returns negative infinity if any masked pixel
        has a negative model value.

        Parameters
        ----------
        pp : array_like
            Model parameters in the parameter space.

        Returns
        -------
        log_likelihood : float
            Log-likelihood value. Returns -inf if negative model values
            are detected in the masked region.

        Notes
        -----
        This method is JIT-compiled with JAX for performance. The mask
        is applied to select valid pixels before computing the likelihood.
        The noise PDF is evaluated using the parameters xs (model data)
        and xr (raw model data) extracted from pdfkwarg.
        """
        xr, xs, _, neg = self._get_model(pp)

        xs = xs.at[self.mask].get()
        xr = xr.at[self.mask].get()
        local_vars = {"xs": xs, "xr": xr}
        pdf = self.pdfnoise(**{key: local_vars[key] for key in self.pdfkwarg})
        return jp.where(jp.any(neg.at[self.mask].get() == 1), -jp.inf, pdf)

    #   Prior probability distribution
    #   --------------------------------------------------------
    @partial(jax.jit, static_argnames=["self"])
    def _log_prior(self, theta):
        """
        Compute the log-prior probability for given parameters.

        Evaluates the log-prior by summing the log-probabilities from
        each parameter's individual prior distribution.

        Parameters
        ----------
        theta : dict or array_like
            Parameter values. Can be a dictionary with parameter names
            as keys, or an array-like object indexed by parameter names.

        Returns
        -------
        log_prior : float
            Total log-prior probability computed as the sum of individual
            parameter log-priors.

        Notes
        -----
        This method is JIT-compiled with JAX for performance. The prior
        is computed by summing log_prob values from each parameter's
        prior distribution defined in self.mod.priors.
        """
        prob = 0.00
        for idx in self.mod.paridx:
            key = self.mod.params[idx]
            prob += self.mod.priors[key].log_prob(theta[key])
        return prob

    #   Transform prior hypercube
    #   --------------------------------------------------------
    @partial(jax.jit, static_argnames=["self"])
    def _prior_transform(self, pp):
        """
        Transform unit hypercube to parameter space for nested sampling.

        Applies the inverse cumulative distribution function (quantile
        function) of each parameter's prior to transform uniform [0, 1]
        samples to the prior distribution.

        Parameters
        ----------
        pp : array_like
            Parameter values in the unit hypercube, with each element
            in the range [0, 1].

        Returns
        -------
        parameters : jax.numpy.ndarray
            Transformed parameters in the physical parameter space.

        Notes
        -----
        This method is JIT-compiled with JAX for performance. The
        transformation is used by nested sampling algorithms that
        sample from a unit hypercube and need to map to the prior.
        Each parameter's prior must implement an icdf (inverse CDF)
        method.
        """
        prior = []
        for pi, p in enumerate(pp):
            key = self.mod.params[self.mod.paridx[pi]]
            prior.append(self.mod.priors[key].icdf(p))
        return jp.array(prior)

    #   Main sampler function
    #   --------------------------------------------------------
    def run(
        self,
        method="nautilus",
        checkpoint=None,
        resume=True,
        getzprior=False,
        **kwargs,
    ):
        """
        Execute Bayesian inference using the specified sampling method.

        Parameters
        ----------
        method : str, optional
            Sampling method: 'nautilus', 'dynesty', or 'numpyro'.
            Default is 'nautilus'.
        checkpoint : str, optional
            Path to checkpoint file for saving/loading sampler state.
        resume : bool, optional
            Whether to resume from checkpoint if available. Default is True.
        getzprior : bool, optional
            Whether to compute log-evidence prior normalization.
            Default is False.
        **kwargs : dict
            Additional keyword arguments passed to the sampler.
        """
        self.method = method

        def log_likelihood(theta):
            return self._log_likelihood(theta)

        def log_prior(theta):
            return self._log_prior(theta)

        def prior_transform(utheta):
            return self._prior_transform(utheta)

        sampler_methods = {
            "dynesty": self._run_dynesty,
            "nautilus": self._run_nautilus,
            "pocomc": self._run_pocomc,
            "optimizer": self._run_optimizer,
            "numpyro": self._run_numpyro,
        }

        self.logz_prior = None

        if isinstance(checkpoint, str) and self.method != "pocomc":
            checkpoint_glob = glob.glob(f"{checkpoint}*.h*5")
            print(checkpoint_glob)
            if len(checkpoint_glob) > 0:
                checkpoint = sorted(checkpoint_glob)[-1]
                warnings.warn(
                    f"Found existing checkpoint file '{checkpoint}'. "
                    "Resuming from this file. If this is not intended, "
                    "please delete or rename the checkpoint "
                    "file before running."
                )

            if not checkpoint.endswith((".hdf5", ".h5")):
                checkpoint = f"{checkpoint}.hdf5"

        if self.method in sampler_methods:
            local_vars = {
                "log_likelihood": log_likelihood,
                "log_prior": log_prior,
                "prior_transform": prior_transform,
                "checkpoint": checkpoint,
                "resume": resume,
                "getzprior": getzprior,
            }
            sampler_params = list(
                inspect.signature(
                    sampler_methods[self.method]
                ).parameters.keys()
            )
            sampler_kwargs = {
                key: local_vars[key]
                for key in sampler_params
                if key != "kwargs"
            }
            sampler_methods[self.method](**sampler_kwargs, **kwargs)
        else:
            raise ValueError(f"Unknown sampling method: {self.method}")

        self.logz_data = self.img.data.at[self.mask].get()
        self.logz_data = self.pdfnoise(
            **{key: jp.zeros(self.logz_data.shape) for key in self.pdfkwarg}
        )
        self.logz_data = self.logz_data.sum()

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

        sampler_kwargs = {}
        for key in inspect.signature(pocomc.Sampler).parameters.keys():
            if key not in [
                "likelihood",
                "prior",
                "n_effective",
                "n_active",
                "output_dir",
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

        print("\n* Running the main sampling step")
        self.sampler = pocomc.Sampler(
            likelihood=log_likelihood,
            prior=prior,
            n_effective=nlive,
            n_active=n_active,
            vectorize=False,
            output_dir=pocodir if resume else None,
            **sampler_kwargs,
        )

        states_ = sorted(glob.glob(f"{pocodir}/*.state")) if resume else []
        self.sampler.run(
            save_every=save_every,
            resume_state_path=states_[-1] if len(states_) else None,
            progress=True,
            **run_kwargs,
        )

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
            self.sampler_prior.run(progress=True, **run_kwargs)
            self.logz_prior, _ = self.sampler_prior.evidence()
        else:
            self.sampler_prior = None
            self.logz_prior = None

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
            if key not in ["nuts", "num_warmup", "num_samples"]:
                if key in kwargs:
                    mcmc_kwargs[key] = kwargs.pop(key)

        nuts = numpyro.infer.NUTS(model, **nuts_kwargs)
        mcmc = numpyro.infer.MCMC(
            nuts, num_warmup=nwarmup, num_samples=nsamples, **mcmc_kwargs
        )
        mcmc.run(seed, **kwargs)

        samp = mcmc.get_samples()

        self.samples = np.array(
            [samp[self.mod.params[p]] for p in self.mod.paridx]
        ).T
        self.weights = np.ones(self.samples.shape[0])

    #   Fitting method - optimizer
    #   --------------------------------------------------------
    def _run_optimizer(self, pinits, **kwargs):
        """
        Run maximum likelihood optimization using scipy.optimize.

        Finds the maximum likelihood estimate (MLE) or maximum a
        posteriori (MAP) estimate using L-BFGS-B optimization with
        automatic differentiation via JAX.

        Parameters
        ----------
        pinits : array_like, str
            Initial parameter values. Can be:

            - array_like : specific initial values in parameter space
              (will be transformed to unit hypercube internally)
            - "median" : start from the median of each prior (0.5 in
              unit hypercube)
            - "random" : start from random values in unit hypercube
        **kwargs : dict
            Additional keyword arguments passed to scipy.optimize.minimize.
            Common options include:

            - tol : float, tolerance for termination
            - options : dict, solver-specific options

        Attributes Set
        --------------
        results : scipy.optimize.OptimizeResult
            Optimization result object containing:

            - x : optimal parameters in unit hypercube
            - fun : negative log-likelihood at optimum
            - success : whether optimization succeeded
            - message : description of termination cause

        Raises
        ------
        ValueError
            If pinits is a string other than "median" or "random".

        Notes
        -----
        The optimization is performed in the unit hypercube space
        with bounds [0, 1] for each parameter. The objective function
        is the negative log-likelihood, and gradients are computed
        automatically using JAX. The L-BFGS-B method is used for
        box-constrained optimization.
        """
        opt_kwargs = {}
        for key in inspect.signature(
            scipy.optimize.minimize
        ).parameters.keys():
            if key not in ["fun", "x0", "jac", "bounds", "method"]:
                if key in kwargs:
                    opt_kwargs[key] = kwargs.pop(key)

        def _opt_prior(pp):
            return jp.array(self._prior_transform(pp))

        def _opt_func(pp):
            pars = _opt_prior(pp)
            return -self._log_likelihood(pars)

        opt_func_jac = jax.jit(jax.value_and_grad(_opt_func))

        if isinstance(pinits, (list, tuple, np.ndarray, jp.ndarray)):
            pinits = jp.asarray(pinits)
            for pi, p in enumerate(pinits):
                key = self.mod.params[self.mod.paridx[pi]]
                pinits = pinits.at[pi].set(self.mod.priors[key].cdf(p))
        else:
            if pinits == "median":
                pinits = jp.array([0.50 for _ in self.mod.paridx])
            elif pinits == "random":
                pinits = jp.array([np.random.rand() for _ in self.mod.paridx])
            else:
                raise ValueError(
                    "Unknown pinits option. Use 'median', 'random', "
                    "or provide an array-like object of initial values."
                )

        bounds = [(0.00, 1.00) for _ in self.mod.paridx]

        self.results = scipy.optimize.minimize(
            fun=opt_func_jac,
            x0=pinits,
            jac=True,
            bounds=bounds,
            method="L-BFGS-B",
            **opt_kwargs,
        )

    #   Compute standard Bayesian Model Selection estimators
    #   --------------------------------------------------------
    def bmc(self, verbose=True):
        """
        Compute Bayesian model comparison estimators.

        Calculates the Bayes factor and effective detection significance
        for model comparison against the null model (data only). Optionally
        computes prior-deboosted values if prior evidence is available.

        Parameters
        ----------
        verbose : bool, optional
            If True, print the computed statistics. Default is True.

        Returns
        -------
        lnBF_raw : float
            Natural logarithm of the raw Bayes factor (model vs. null).
        seff_raw : float
            Effective Gaussian detection significance for raw Bayes factor,
            computed as ``sign(ln BF) * sqrt(2 * |ln BF|)``.
        lnBF_cor : float or None
            Natural logarithm of the prior-deboosted Bayes factor.
            None if prior evidence was not computed.
        seff_cor : float or None
            Effective significance for prior-deboosted Bayes factor.
            None if prior evidence was not computed.

        Warnings
        --------
        UserWarning
            If prior evidence (logz_prior) is None, warns that prior
            deboosting cannot be applied.

        Notes
        -----
        The raw Bayes factor compares the model evidence to the null
        model evidence (data-only). The prior-deboosted Bayes factor
        additionally accounts for the prior volume to avoid Occam's
        razor penalty when the prior is uninformative.

        The effective significance approximates the detection significance
        in terms of Gaussian standard deviations using the Wilks' theorem
        approximation: ``sigma_eff = sign(ln BF) * sqrt(2 * |ln BF|)``.
        """
        lnBF_raw = self.logz - self.logz_data
        seff_raw = np.sign(lnBF_raw) * np.sqrt(2.00 * np.abs(lnBF_raw))

        if verbose:
            print("\nnull-model comparison")
            print("=" * 21)
            print(f"ln(Bayes factor) : {lnBF_raw:10.3E}")
            print(f"effective sigma  : {seff_raw:10.3E}")

        if self.logz_prior is None:
            lnBF_cor = None
            seff_cor = None
            warnings.warn(
                "Prior evidence not computed. Cannot apply prior deboosting."
            )
        else:
            lnBF_cor = lnBF_raw - self.logz_prior
            seff_cor = np.sign(lnBF_cor) * np.sqrt(2.00 * np.abs(lnBF_cor))

            if verbose:
                print("\nprior deboosted")
                print("=" * 21)
                print(f"ln(Bayes factor) : {lnBF_cor:10.3E}")
                print(f"effective sigma  : {seff_cor:10.3E}\n")

        return lnBF_raw, seff_raw, lnBF_cor, seff_cor

    #   Dump results
    #   --------------------------------------------------------
    def dump(self, filename):
        """
        Save the fitter object to a pickle file.

        Serializes the entire fitter object state including samples,
        weights, sampler objects, and all attributes to a file using
        dill for enhanced pickling support.

        Parameters
        ----------
        filename : str or Path
            Output file path. If the filename does not have a pickle
            extension (.pickle, .pkl, .pck), ".pickle" is appended
            automatically.

        Notes
        -----
        Uses dill instead of pickle to handle complex objects like
        JAX-compiled functions and lambda functions. The file is
        written with the highest protocol for optimal compression.

        See Also
        --------
        load : Load a fitter object from a pickle file.
        """
        odict = {key: self.__dict__[key] for key in self.__dict__.keys()}
        # ensure filename has a pickle-like suffix
        p = Path(filename)
        if p.suffix.lower() not in [".pickle", ".pkl", ".pck"]:
            filename = str(p) + ".pickle"

        with open(filename, "wb") as f:
            dill.dump(odict, f, dill.HIGHEST_PROTOCOL)

    #   Generate best-fit/median model
    #   --------------------------------------------------------
    def getmodel(
        self,
        what="all",
        component=None,
        usebest=True,
        img=None,
        doresp=False,
        doexp=False,
    ):
        """
        Generate best-fit or median model from sampling results.

        Computes model realizations using either the weighted median
        parameters or by marginalizing over all posterior samples.

        Parameters
        ----------
        what : str or list of str, optional
            Which model component(s) to return. Options include:

            - "all" : return all components (raw, smooth, background)
            - "raw" : raw model before convolution
            - "smo"/"smooth"/"smoothed"/"conv"/"convolved" : model after
              PSF convolution
            - "bkg"/"background" : background component

            Default is "all".
        component : None, str, int, list, or Profile, optional
            Model component(s) to include in the computation. Can be:

            - None: Include all model components (default)
            - str: Single component name (e.g., 'comp_00')
            - int: Component index (e.g., 0 for the first component)
            - list: Multiple components as names, indices, or Profile objects
            - Profile: Object with `id` attribute specifying the component

            This is useful for generating images of individual model
            components. Default is None (all components).
        usebest : bool, optional
            If True, compute model at weighted median parameters.
            If False, compute median model by marginalizing over all
            samples. Default is True.
        img : Image, optional
            Image object to use for model computation. If None, uses
            self.img. Default is None.
        doresp : bool, optional
            Whether to apply instrument response. Default is False.
        doexp : bool, optional
            Whether to apply exposure map. Default is False.

        Returns
        -------
        model_raw : ndarray
            Raw model before convolution. Returned if "all" or "raw"
            is requested.
        model_smooth : ndarray
            Model after convolution and background subtraction. Returned
            if "all" or a smoothed variant is requested.
        model_background : ndarray
            Background component. Returned if "all" or "bkg" is requested.

        Raises
        ------
        ValueError
            If an unknown model component name is provided in `what`.

        Notes
        -----
        For optimizer results, only usebest=True mode is supported.
        The weighted median uses importance weights for nested sampling
        results. When usebest=False, the method marginalizes over all
        posterior samples to compute the median model, which can be
        computationally expensive for large sample sets.

        Examples
        --------
        >>> # Get full model with all components
        >>> mraw, msmo, mbkg = fit.getmodel()
        >>> # Get only the first component
        >>> mraw, msmo, mbkg = fit.getmodel(component=0)
        >>> # Get specific components by name
        >>> mraw, msmo, mbkg = fit.getmodel(component=['comp_00', 'comp_02'])
        """
        name_map = {
            "raw": "raw",
            "smo": "smoothed",
            "smooth": "smoothed",
            "smoothed": "smoothed",
            "conv": "convolved",
            "convolved": "convolved",
            "bkg": "background",
            "background": "background",
            "all": "all",
        }

        if isinstance(what, str):
            label = name_map.get(what.lower(), what)
            print(f"Generating {label} model")
        else:
            labels = [name_map.get(w.lower(), w) for w in what]
            if len(labels) == 2:
                print(f"Generating {' and '.join(labels)} models")
            else:
                print(
                    f"Generating {', '.join(labels[:-1])} and {labels[-1]} models"
                )

        def gm(pp):
            return self.mod.getmodel(
                self.img if img is None else img, pp, doresp, doexp, component
            )

        if self.method == "optimizer":
            p = self._prior_transform(self.results.x)
            mraw, msmo, mbkg, _ = gm(p)
            msmo = msmo - mbkg
        else:
            if usebest:
                p = np.array(
                    [
                        np.quantile(
                            samp,
                            0.50,
                            method="inverted_cdf",
                            weights=self.weights,
                        )
                        for samp in self.samples.T
                    ]
                )
                mraw, msmo, mbkg, _ = gm(p)
                msmo = msmo - mbkg
            else:
                mraw, msmo = [], []
                for sample in self.samples:
                    mraw_, msmo_, mbkg_, _ = gm(sample)
                    msmo_ = msmo_ - mbkg_
                    mraw.append(mraw_)
                    del mraw_
                    msmo.append(msmo_)
                    del msmo_
                    mbkg.append(mbkg_)
                    del mbkg_

                mraw = np.quantile(
                    mraw,
                    0.50,
                    axis=0,
                    method="inverted_cdf",
                    weights=self.weights,
                )
                msmo = np.quantile(
                    msmo,
                    0.50,
                    axis=0,
                    method="inverted_cdf",
                    weights=self.weights,
                )
                mbkg = np.quantile(
                    mbkg,
                    0.50,
                    axis=0,
                    method="inverted_cdf",
                    weights=self.weights,
                )

        if isinstance(what, str):
            if what.lower() == "all":
                return mraw, msmo, mbkg
            else:
                what = [what]

        mout = []
        for w in what:
            if w.lower() in ["raw"]:
                mout.append(mraw)
            elif w.lower() in [
                "smo",
                "smooth",
                "smoothed",
                "conv",
                "convolved",
            ]:
                mout.append(msmo)
            elif w.lower() in ["bkg", "background"]:
                mout.append(mbkg)
            else:
                raise ValueError(f"Unknown model component: {w}")

        return mout if len(mout) > 1 else mout[0]

    #   Save best-fit/median model to file
    #   --------------------------------------------------------
    def savemodel(self, name, component=None, **kwargs):
        """
        Save best-fit or median model to a FITS file.

        Generates a model image using `getmodel()` and writes it to a
        FITS file with the WCS header from the input image preserved.

        Parameters
        ----------
        name : str or Path
            Output FITS filename. The '.fits' extension is added
            automatically if not present.
        component : None, str, int, list, or Profile, optional
            Model component(s) to include in the saved image. Can be:

            - None: Include all model components (default)
            - str: Single component name (e.g., 'comp_00')
            - int: Component index (e.g., 0 for the first component)
            - list: Multiple components as names, indices, or Profile objects
            - Profile: Object with `id` attribute specifying the component

            This is useful for saving images of individual model
            components. Default is None (all components).
        **kwargs : dict
            Additional keyword arguments passed to `getmodel()`.
            Common options include:

            - what : str, optional
                Model component to save. Default is 'convolved'.
                Options: 'raw', 'convolved'/'smoothed', 'background'.
            - usebest : bool, optional
                If True (default), use weighted median parameters.
                If False, compute median of model realizations.
            - doresp : bool, optional
                Apply response correction. Default is False.
            - doexp : bool, optional
                Apply exposure map. Default is False.

        See Also
        --------
        getmodel : Generate model images from posterior samples.
        dump : Save entire fitter object to pickle file.

        Examples
        --------
        >>> # Save the PSF-convolved model
        >>> fit.savemodel('best_fit_model.fits')
        >>>
        >>> # Save the raw (unconvolved) model
        >>> fit.savemodel('raw_model.fits', what='raw')
        >>>
        >>> # Save model computed by marginalizing over samples
        >>> fit.savemodel('median_model.fits', usebest=False)
        >>>
        >>> # Save multiple components as a multi-slice FITS
        >>> fit.savemodel('all_components.fits', what=['raw', 'convolved'])
        >>>
        >>> # Save only the first component
        >>> fit.savemodel('component_0.fits', component=0)
        """
        what = kwargs.pop("what", "convolved")
        mod = self.getmodel(what=what, component=component, **kwargs)

        # Normalize component names for header
        name_map = {
            "raw": "raw",
            "smo": "smoothed",
            "smooth": "smoothed",
            "smoothed": "smoothed",
            "conv": "convolved",
            "convolved": "convolved",
            "bkg": "background",
            "background": "background",
            "all": "all",
        }

        header = self.img.wcs.to_header()

        if isinstance(what, list):
            data = np.array([np.array(m) for m in mod])
            header["NSLICES"] = (len(what), "Number of model slices")
            for i, w in enumerate(what):
                label = name_map.get(w.lower(), w)
                header[f"SLICE{i + 1}"] = (
                    label,
                    f"Model component in slice {i}",
                )
        else:
            data = np.array(mod)
            label = name_map.get(what.lower(), what)
            header["MODEL"] = (label, "Model component")

        hdu = fits.PrimaryHDU(data=data, header=header)
        hdu.writeto(name, overwrite=True)
        print(f"Saved to {name}")


#   Load results
#   --------------------------------------------------------
def load(filename):
    """
    Load a fitter object from a pickle file.

    Deserializes a previously saved fitter object, restoring all
    samples, weights, sampler objects, and attributes.

    Parameters
    ----------
    filename : str or Path
        Path to the pickle file created by fitter.dump().

    Returns
    -------
    fit : fitter
        Restored fitter object with all attributes and state.

    Notes
    -----
    Uses dill for deserialization to handle complex objects like
    JAX-compiled functions. The loaded fitter object is fully
    functional and can be used for plotting, model generation,
    and further analysis.

    See Also
    --------
    fitter.dump : Save a fitter object to a pickle file.

    Examples
    --------
    >>> fit = load('results.pickle')
    >>> mraw, msmo, mbkg = fit.getmodel()
    """
    with open(filename, "rb") as f:
        odict = dill.load(f)
    fit = fitter(img=odict["img"], mod=odict["mod"])
    for key in odict.keys():
        fit.__dict__[key] = odict[key]
    return fit
