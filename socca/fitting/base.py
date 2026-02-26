"""Bayesian inference engine for astronomical image modeling."""

from functools import partial

import jax
import jax.numpy as jp
import numpy as np

from ..plotting import Plotter

import inspect
import dill
import os
import warnings
from pathlib import Path

from astropy.io import fits

from .methods import (
    run_nautilus,
    run_dynesty,
    run_pocomc,
    run_numpyro,
    run_emcee,
    run_optimizer,
)

from ..pool.mpi import MPI_RANK
from ..pool.mpi import root_only


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

    _run_nautilus = run_nautilus
    _run_dynesty = run_dynesty
    _run_pocomc = run_pocomc
    _run_numpyro = run_numpyro
    _run_emcee = run_emcee
    _run_optimizer = run_optimizer

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

        self.img._init_noise()

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
    def _getmodel(self, pp):
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
        Response is applied if any element of img.response differs from 1.0.
        Exposure is applied if any element of img.exposure differs from 1.0.
        """
        doresp = ~np.all(np.array(self.img.response) == 1.00)  # True
        doexp = ~np.all(np.array(self.img.exposure) == 1.00)  # True
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
        xr, xs, _, neg = self._getmodel(pp)

        xs = xs.at[self.mask].get()
        xr = xr.at[self.mask].get()
        local_vars = {"xs": xs, "xr": xr}
        pdf = self.pdfnoise(**{key: local_vars[key] for key in self.pdfkwarg})
        return jp.where(jp.any(neg == 1), -jp.inf, pdf)

    #   Prior probability distribution
    #   --------------------------------------------------------
    @partial(jax.jit, static_argnames=["self"])
    def _log_prior(self, pp):
        """
        Compute the log-prior probability for given parameters.

        Evaluates the log-prior by summing the log-probabilities from
        each parameter's individual prior distribution.

        Parameters
        ----------
        pp : dict or array_like
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
        for pi, p in enumerate(pp):
            key = self.mod.params[self.mod.paridx[pi]]
            prob_ = self.mod.priors[key].log_prob(p)
            prob_ = jp.where(
                self.mod.priors[key].support.check(p), prob_, -jp.inf
            )
            prob += jp.where(jp.isfinite(prob_), prob_, -jp.inf)
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
            "emcee": self._run_emcee,
        }

        if self.method not in sampler_methods:
            raise ValueError(f"Unknown sampling method: {self.method}")

        self.logz_prior = None

        if isinstance(checkpoint, str) and self.method != "pocomc":
            for ext in [".hdf5", ".h5"]:
                if os.path.exists(f"{checkpoint}{ext}"):
                    checkpoint = f"{checkpoint}{ext}"
                    print(f"Resuming from checkpoint: {checkpoint}")
                    break

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

    #   Compute standard Bayesian Model Selection estimators
    #   --------------------------------------------------------
    @root_only
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
        if self.method in ["emcee", "numpyro", "optimizer"]:
            raise ValueError(
                f"Bayesian model comparison is not applicable for {self.method} results."
            )

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
    @root_only
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

        if MPI_RANK == 0:
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
    @root_only
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

    #   Print best-fit parameters
    #   --------------------------------------------------------
    @root_only
    def parameters(self):
        """
        Print best-fit parameters with uncertainties.

        Computes and prints the weighted median (50th percentile) and
        asymmetric uncertainties (16th and 84th percentiles) for each
        fitted parameter, grouped by model component.

        The output format is::

            comp_00
            -------
            param : best-fit [+upper/-lower]

        where upper = 84th percentile - median and lower = median - 16th
        percentile.

        Notes
        -----
        Requires that the sampler has been run and samples are available.
        Uses importance weights for nested sampling results.
        """
        print("\nBest-fit parameters")
        print("=" * 40)

        # Group parameters by component
        components = {}
        for pi, label in enumerate(self.labels):
            parts = label.rsplit("_", 1)
            if len(parts) == 2:
                comp, param = "_".join(label.split("_")[:-1]), parts[-1]
            else:
                comp, param = "model", label
            if comp not in components:
                components[comp] = []
            components[comp].append((pi, param))

        # Print grouped by component
        for comp, params in components.items():
            print(f"\n{comp}")
            print("-" * len(comp))

            # Find max param name length for alignment
            max_len = max(len(p[1]) for p in params)

            for pi, param in params:
                samp = self.samples[:, pi]

                p16, p50, p84 = np.quantile(
                    samp,
                    q=[0.16, 0.50, 0.84],
                    method="inverted_cdf",
                    weights=self.weights,
                )

                upper = p84 - p50
                lower = p50 - p16

                print(
                    f"{param:<{max_len}} : {p50:11.4E} [+{upper:10.4E}/-{lower:10.4E}]"
                )


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
