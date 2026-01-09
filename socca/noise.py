"""Noise models and likelihood functions for astronomical images."""

from .utils import _img_loader

import jax
import jax.numpy as jp
import numpy as np

import warnings

from astropy.convolution import convolve, CustomKernel
from scipy.stats import median_abs_deviation


# Multi-variate normal noise (no correlation)
# ========================================================
class Normal:
    """
    Multi-variate normal noise model with no correlation between pixels.

    Attributes
    ----------
    options : dict
        Dictionary of accepted noise model identifiers and their aliases.
    select : str or None
        Selected noise model identifier from the provided keyword arguments.
    kwargs : dict
        Keyword arguments provided for specifying the noise model.
    data : jax.numpy.ndarray
        Image data array. This is inherited from the parent Data class.
    mask : jax.numpy.ndarray
        Image mask array. This is inherited from the parent Data class.
    """

    def __init__(self, **kwargs):
        """
        Initialize normal noise model.

        Parameters
        ----------
        kwargs : dict
        Keyword arguments for specifying the noise model.
        Accepted keywordss (with aliases):
            - sigma: float or str, optional
                Standard deviation of the noise. Default is None, in which
                case the noise level is estimated using the median absolute
                deviation. Accepted aliases: ``sig``, ``std``, ``rms``,
                ``stddev``
            - var: float or str, optional
                Variance of the noise
                Accepted aliases: ``var``, ``variance``
            - wht: float or str, optional
                Weight (inverse variance) of the noise
                Accepted aliases: ``wht``, ``wgt``, ``weight``,
                ``weights``, ``invvar``
            - idx: int, optional
                HDU index to use when loading noise maps from FITS files.
                Default is 0.

        If a string is provided for any of these, it is treated as a path
        to a FITS file containing the corresponding map.
        """
        self.options = {
            "sig": ["sigma", "sig", "std", "rms", "stddev"],
            "var": ["var", "variance"],
            "wht": ["wht", "wgt", "weight", "weights", "invvar"],
        }

        options = (
            self.options["sig"] + self.options["var"] + self.options["wht"]
        )

        self.select = np.array([key for key in options if key in kwargs])
        if self.select.shape[0] > 1:
            raise ValueError(
                "Multiple noise identifiers found in kwargs. "
                "Please use only one of sigma, variance, or weight."
            )
        self.select = self.select[0] if self.select.shape[0] == 1 else None

        self.kwargs = {key: kwargs[key] for key in options if key in kwargs}
        self.kwargs.update({"idx": kwargs.get("idx", 0)})

    #   Estimate/build noise statistics
    #   --------------------------------------------------------
    def getsigma(self):
        """
        Estimate or build the noise standard deviation map.

        Returns
        -------
        Array
            The per-pixel standard deviation map as a JAX array
            (same shape as ``self.data``).
        """
        if self.select is None:
            print("Using MAD for estimating noise level")
            sigma = median_abs_deviation(
                x=self.data.at[self.mask].get(),
                scale="normal",
                nan_policy="omit",
                axis=None,
            )
            sigma = float(sigma)
            print(f"- noise level: {sigma:.2E} [image units]")
        elif isinstance(self.select, str):
            if isinstance(self.kwargs[self.select], (float, int)):
                sigma = self.kwargs[self.select]
            else:
                sigma = _img_loader(
                    self.kwargs[self.select], self.kwargs.get("idx", 0)
                )
                sigma = sigma.data.copy()

            if self.select in self.options["var"]:
                sigma = np.sqrt(sigma)
            elif self.select in self.options["wht"]:
                self.mask.at[sigma == 0.00].set(0)
                if isinstance(sigma, (float, int)) and sigma != 0.00:
                    sigma = 1.00 / np.sqrt(sigma)
                else:
                    sigma[sigma != 0.00] = 1.00 / np.sqrt(sigma[sigma != 0.00])
            elif self.select not in self.options["sig"]:
                raise ValueError("Unrecognized noise identifier")

        if isinstance(sigma, (float, int)):
            sigma = np.full(self.data.shape, sigma).astype(float)

        sigma[np.isinf(sigma)] = 0.00
        sigma[np.isnan(sigma)] = 0.00
        self.mask.at[sigma == 0.00].set(0)

        return jp.array(sigma)

    #   Set up noise model
    #   --------------------------------------------------------
    def __call__(self, data, mask):
        """
        Set up the noise model with data and mask.

        Processes the input data and mask, computes or loads the noise sigma,
        and creates the log probability density function for likelihood
        evaluation.

        Parameters
        ----------
        data : array_like
            Image data array to be modeled.
        mask : array_like
            Binary mask array (1=valid, 0=masked). Pixels where mask==0
            are excluded from likelihood calculations.

        Notes
        -----
        - Computes noise standard deviation via getsigma()
        - Applies mask to extract valid pixels
        - Creates a lambda function for log-likelihood evaluation
        """
        self.mask = mask.astype(int).copy()
        self.data = data.copy()

        self.sigma = self.getsigma()

        self.mask = self.mask == 1.00
        self.data = self.data.at[self.mask].get()
        self.sigma = self.sigma.at[self.mask].get()

        self.logpdf = lambda xs: self._logpdf(xs, self.data, self.sigma)

    #   Noise log-pdf/likelihood function
    #   --------------------------------------------------------
    @staticmethod
    @jax.jit
    def _logpdf(x, data, sigma):
        """
        Compute log probability density for independent normal noise.

        JAX JIT-compiled static method that evaluates the sum of log
        probabilities for independent normal distributions at each pixel.

        Parameters
        ----------
        x : jax.numpy.ndarray
            Model values at each pixel (flattened, masked pixels excluded).
        data : jax.numpy.ndarray
            Observed data values (flattened, masked pixels excluded).
        sigma : jax.numpy.ndarray
            Noise standard deviation at each pixel (flattened, masked excluded).

        Returns
        -------
        float
            Total log probability (sum over all valid pixels).
        """
        return jax.scipy.stats.norm.logpdf(x, loc=data, scale=sigma).sum()


# Correlated multi-variate normal noise
# ========================================================
class NormalCorrelated:
    """
    Noise model with correlated multi-variate normal distribution.

    Attributes
    ----------
    cov : jax.numpy.ndarray
        Covariance matrix of the noise.
    icov : jax.numpy.ndarray
        Inverse of the covariance matrix.
    norm : float
        Normalization factor for the log-likelihood.
    data : jax.numpy.ndarray
        Image data array.
    mask : jax.numpy.ndarray
        Boolean mask array.
    """

    def __init__(self, cov=None, icov=None, cube=None, **kwargs):
        """
        Initialize correlated normal noise model.

        Parameters
        ----------
        cov : array_like, optional
            Covariance matrix. If not provided, computed from cube.
        icov : array_like, optional
            Inverse covariance matrix. If provided, used directly.
        cube : array_like, optional
            3D array of noise realizations. First dimension is the number
            of realizations. Used to compute covariance if cov is None.
        **kwargs : dict
            Additional keyword arguments:

            - smooth : int, optional
                Number of smoothing iterations for covariance matrix.
                Default is 3.
            - kernel : array_like, optional
                Custom smoothing kernel. If None, uses a 5-point stencil.
        """
        if icov is not None:
            self.cov = None
            self.icov = jp.asarray(icov.astype(float))

            if cov is not None:
                warnings.warn(
                    "Both covariance and inverse covariance matrix provided. \
                            Using inverse covariance matrix."
                )

            if cube is not None:
                warnings.warn(
                    "Both inverse covariance matrix and noise cube provided. \
                            Using inverse covariance matrix."
                )
        else:
            if cube is not None and cov is None:
                cov = np.cov(cube.reshape(cube.shape[0], -1), rowvar=True)

                smooth = kwargs.get("smooth", 3)
                if smooth > 0:
                    kernel = kwargs.get("kernel", None)
                    if kernel is None:
                        kernel = (
                            np.array(
                                [
                                    [0.00, 1.00, 0.00],
                                    [1.00, 1.00, 1.00],
                                    [0.00, 1.00, 0.00],
                                ]
                            )
                            / 5.00
                        )
                        kernel = CustomKernel(kernel)

                    for _ in range(smooth):
                        cov = jp.array(convolve(cov, kernel, boundary="wrap"))
            elif cov is not None and cube is not None:
                warnings.warn(
                    "Both covariance matrix and noise cube provided. \
                            Using covariance matrix."
                )
            elif cov is None and cube is None:
                raise ValueError(
                    "Either covariance matrix or noise realization cube "
                    "must be provided."
                )

            self.cov = jp.asarray(cov.astype(float))
            self.icov = jp.linalg.inv(self.cov)

    #   Set up noise model
    #   --------------------------------------------------------
    def __call__(self, data, mask):
        """
        Set up the correlated noise model with data and mask.

        Processes the input data and mask, computes normalization for the
        multivariate normal distribution, and creates the log probability
        density function using the covariance/inverse covariance matrix.

        Parameters
        ----------
        data : array_like
            Image data array to be modeled.
        mask : array_like
            Binary mask array (1=valid, 0=masked). Pixels where mask==0
            are excluded from likelihood calculations.

        Notes
        -----
        - Computes normalization constant from inverse covariance determinant
        - Creates lambda function that accounts for pixel correlations
        - Uses quadratic form with inverse covariance for log-likelihood
        """
        self.mask = mask.copy()
        self.mask = self.mask == 1.00
        self.data = data.at[self.mask].get()

        self.norm = -float(jp.linalg.slogdet(self.icov / 2.00 / jp.pi)[1])
        self.logpdf = (
            lambda xs: self._logpdf(xs, self.data, self.icov)
            - 0.50 * self.norm
        )

    #   Noise log-pdf/likelihood function
    #   --------------------------------------------------------
    @staticmethod
    @jax.jit
    def _logpdf(x, data, icov):
        """
        Compute log probability for multivariate normal with covariance.

        JAX JIT-compiled static method that evaluates the quadratic form
        for a multivariate normal distribution using the inverse covariance
        matrix.

        Parameters
        ----------
        x : jax.numpy.ndarray
            Model values (flattened vector, masked pixels excluded).
        data : jax.numpy.ndarray
            Observed data values (flattened vector, masked pixels excluded).
        icov : jax.numpy.ndarray
            Inverse covariance matrix (2D array).

        Returns
        -------
        float
            Log probability: -0.5 * (x-data)^T @ icov @ (x-data)

        Notes
        -----
        The normalization constant is added separately in __call__().
        """
        res = x - data
        return -0.50 * jp.einsum("i,ij,j->", res, icov, res)


# Fourier-space independente noise
# ========================================================
class NormalFourier:
    """
    Noise model with independent noise in Fourier space.

    The noise covariance is defined in Fourier space.

    Attributes
    ----------
    cmask : jax.numpy.ndarray
        Boolean mask indicating which Fourier modes have non-zero noise.
    ftype : str
        Type of Fourier transform to use. Options are:
        - 'real' or 'rfft': real-to-complex FFT (for real input data)
        - 'full' or 'fft': complex-to-complex FFT (for complex input data)
    apod : jax.numpy.ndarray
        Apodization map applied to the data before Fourier transforming.
    cov : jax.numpy.ndarray
        Noise covariance in Fourier space.
    icov : jax.numpy.ndarray
        Inverse noise covariance in Fourier space.
    norm : float
        Normalization constant for the log-pdf.
    data : jax.numpy.ndarray
        Image data array. This is set when the model is called.
    mask : jax.numpy.ndarray
        Image mask array. This is set when the model is called.
    """

    def __init__(self, cov=None, icov=None, cube=None, ftype="real", **kwargs):
        """
        Initialize Fourier-space noise model.

        Parameters
        ----------
        cov : array_like, optional
            Noise covariance in Fourier space. If not provided,
            computed from cube.
        icov : array_like, optional
            Inverse noise covariance in Fourier space. If provided,
            used directly.
        cube : array_like, optional
            3D array of noise realizations. First dimension is the number
            of realizations. Used to compute covariance if cov is None.
        ftype : str, optional, default 'real'
            Type of Fourier transform to use. Options are:
            - 'real' or 'rfft': real-to-complex FFT (for real input data)
            - 'full' or 'fft': complex-to-complex FFT (for complex input data)
        **kwargs : dict
            Additional keyword arguments:

            - apod : array_like, optional
                Apodization map applied to the data before Fourier
                transforming. Default is no apodization.
            - smooth : int, optional
                Number of smoothing iterations for covariance matrix.
                Default is 3.
            - kernel : array_like, optional
                Custom smoothing kernel. If None, uses a 5-point stencil.
        """
        if ftype not in ["real", "rfft", "full", "fft"]:
            raise ValueError(
                "ftype must be either 'real'/'rfft' or 'full'/'fft'."
            )
        self.ftype = ftype

        if icov is not None:
            self.cov = None
            self.icov = jp.asarray(icov.astype(float))

            if cov is not None:
                warnings.warn(
                    "Both covariance and inverse covariance matrix provided. \
                            Using inverse covariance matrix."
                )

            if cube is not None:
                warnings.warn(
                    "Both inverse covariance matrix and noise cube provided. \
                            Using inverse covariance matrix."
                )
        else:
            if cube is not None and cov is None:
                cube = jp.array(cube)

                self.apod = kwargs.get(
                    "apod", jp.ones((cube.shape[-2], cube.shape[-1]))
                )

                fft = (
                    jp.fft.rfft2 if ftype in ["real", "rfft"] else jp.fft.fft2
                )
                cov = fft(cube * self.apod[None, ...], axes=(-2, -1))
                cov = jp.mean(jp.abs(cov) ** 2, axis=0) / jp.mean(self.apod**2)

                smooth = kwargs.get("smooth", 3)
                if smooth > 0:
                    kernel = kwargs.get("kernel", None)
                    if kernel is None:
                        kernel = (
                            np.array(
                                [
                                    [0.00, 1.00, 0.00],
                                    [1.00, 1.00, 1.00],
                                    [0.00, 1.00, 0.00],
                                ]
                            )
                            / 5.00
                        )
                        kernel = CustomKernel(kernel)

                    for _ in range(smooth):
                        cov = jp.array(convolve(cov, kernel, boundary="wrap"))
            elif cov is not None and cube is not None:
                warnings.warn(
                    "Both covariance matrix and noise cube provided. \
                            Using covariance matrix."
                )
            elif cov is None and cube is None:
                raise ValueError(
                    "Either covariance matrix or noise realization cube "
                    "must be provided."
                )

            self.cov = jp.asarray(cov.astype(float))
            self.icov = 1.00 / self.cov

        mask_icov = jp.logical_or(self.icov == jp.inf, jp.isnan(self.icov))
        self.icov = self.icov.at[mask_icov].set(0.00)
        del mask_icov

    #   Set up noise model
    #   --------------------------------------------------------
    def __call__(self, data, mask):
        """
        Set up the Fourier-space noise model with data and mask.

        Processes the input data and mask, computes normalization for
        Fourier-space likelihood, and creates the JIT-compiled log
        probability density function.

        Parameters
        ----------
        data : array_like
            Image data array to be modeled.
        mask : array_like
            Binary mask array (1=valid, 0=masked). All pixels must be
            unmasked for this noise model.

        Raises
        ------
        ValueError
            If any pixels are masked (NormalFourier requires full images).

        Notes
        -----
        - Requires all pixels to be valid (no masking)
        - Computes covariance mask (cmask) for non-zero modes
        - Creates JIT-compiled log-likelihood with apodization
        """
        self.mask = mask.copy()
        self.mask = self.mask == 1.00
        self.data = data.copy()

        if not np.all(self.mask):
            raise ValueError(
                "NormalFourier noise model requires full image "
                "(no masked pixels)."
            )

        self.cmask = self.icov != 0.00

        self.norm = jp.log(2.00 * jp.pi / self.icov.at[self.cmask].get()).sum()

        def _logpdf(xs):
            factor = self._logpdf(
                xs,
                self.data,
                self.icov,
                self.mask,
                self.cmask,
                self.apod,
                self.ftype,
            )
            return factor - 0.50 * self.norm

        self.logpdf = jax.jit(_logpdf)

    #   Noise log-pdf/likelihood function
    #   --------------------------------------------------------
    @staticmethod
    def _logpdf(x, data, icov, dmask, cmask, apod, ftype="real"):
        """
        Compute log probability for independent noise in Fourier space.

        Static method that evaluates chi-squared in Fourier space using the
        inverse covariance. Residuals are apodized before transformation.

        Parameters
        ----------
        x : jax.numpy.ndarray
            Model values (flattened vector).
        data : jax.numpy.ndarray
            Observed data array (2D image).
        icov : jax.numpy.ndarray
            Inverse covariance in Fourier space (2D).
        dmask : jax.numpy.ndarray
            Boolean mask for valid pixels in data space.
        cmask : jax.numpy.ndarray
            Boolean mask for non-zero modes in Fourier space.
        apod : jax.numpy.ndarray
            Apodization array applied before FFT.
        ftype : str, optional
            Fourier transform type ('real'/'rfft' or 'full'/'fft').
            Default is 'real'.

        Returns
        -------
        float
            Log probability: -0.5 * sum(icov * |FFT(residual * apod)|^2)

        Notes
        -----
        The normalization constant is added separately in __call__().
        """
        fft = jp.fft.rfft2 if ftype in ["real", "rfft"] else jp.fft.fft2

        xmap = jp.zeros(dmask.shape)
        xmap = xmap.at[dmask].set(x)

        chisq = fft((xmap - data) * apod, axes=(-2, -1))
        chisq = icov * jp.abs(chisq) ** 2
        return -0.50 * jp.sum(chisq.at[cmask].get())


# Correlated noise for radio-interferometric data
# ========================================================
class NormalRI:
    """Radio-interferometric noise model (placeholder)."""

    def __init__(self):
        """
        Initialize radio-interferometric noise model (not yet implemented).

        Raises
        ------
        NotImplementedError
            This noise model is a placeholder for future implementation of
            radio-interferometric data handling with visibility-space noise.
        """
        raise NotImplementedError(
            "NormalRI noise model is not yet implemented."
        )
