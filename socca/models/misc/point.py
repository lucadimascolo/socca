"""Point source model component."""

import jax.numpy as jp
import numpyro.distributions

from .. import config
from ..base import Component

import warnings


# Point source
# --------------------------------------------------------
class Point(Component):
    """
    Point source model for unresolved sources.

    Models point sources (stars, quasars, unresolved AGN) that are handled
    efficiently in Fourier space. The source is convolved with the PSF to
    produce the observed image.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments including:

        xc : float, optional
            Right ascension in degrees.
        yc : float, optional
            Declination in degrees.
        Ic : float, optional
            Integrated flux or peak brightness (same units as image).
        positive : bool, optional
            Whether to enforce positivity constraint.

    Attributes
    ----------
    xc : float
        Source right ascension (deg).
    yc : float
        Source declination (deg).
    Ic : float
        Source intensity.

    Notes
    -----
    Point sources are special-cased in the model evaluation:

    - Handled in Fourier space using phase shifts
    - Always PSF-convolved (not meaningful without PSF)
    - More efficient than evaluating a very narrow Gaussian
    - Can account for instrumental response at the source position
    """

    def __init__(self, **kwargs):
        """
        Initialize a point source component.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments including xc, yc, Ic, positive.
        """
        super().__init__(**kwargs)
        self.xc = kwargs.get("xc", config.Point.xc)
        self.yc = kwargs.get("yc", config.Point.yc)
        self.Ic = kwargs.get("Ic", config.Point.Ic)

        self.units.update(dict(xc="deg", yc="deg", Ic="image"))

        self.description.update(
            dict(
                xc="Right ascension",
                yc="Declination",
                Ic="Peak surface brightness",
            )
        )
        self._initialized = True

    @staticmethod
    def profile(xc, yc, Ic):
        """
        Point source profile placeholder (not used).

        Point sources are handled specially in Fourier space and do not use
        a standard radial profile function.

        Parameters
        ----------
        xc : float
            Right ascension (deg).
        yc : float
            Declination (deg).
        Ic : float
            Source intensity.

        Notes
        -----
        This method is a placeholder to maintain API consistency. Point sources
        are actually evaluated in getmap() using Fourier space phase shifts.
        """
        pass

    @staticmethod
    def getgrid():
        """
        Point source grid placeholder (not used).

        Point sources do not use a spatial grid; they are handled in Fourier space.

        Notes
        -----
        This method is a placeholder to maintain API consistency with other
        profile types that use getgrid() for coordinate transformations.
        """
        pass

    def _build_kwargs(self, pars, comp_prefix):
        """
        Build keyword arguments for _evaluate from the full parameters dict.

        Parameters
        ----------
        pars : dict
            Full parameters dictionary with prefixed keys.
        comp_prefix : str
            Component prefix (e.g., 'comp_00').

        Returns
        -------
        dict
            Keyword arguments for _evaluate.
        """
        return {
            key.replace(f"{comp_prefix}_", ""): pars[key]
            for key in pars
            if key.startswith(f"{comp_prefix}_")
        }

    def _evaluate(self, img, **kwarg):
        """
        Evaluate point source in Fourier space with explicit parameters.

        This internal method computes the Fourier-space representation of a
        point source using phase shifts. It is used by both getmap() and
        Model.getmodel() to avoid code duplication.

        Parameters
        ----------
        img : Image
            Image object containing FFT information.
        **kwarg : dict
            Parameters including xc, yc, Ic.

        Returns
        -------
        ndarray
            Fourier-space representation of the point source (complex array).
        """
        xc = kwarg["xc"]
        yc = kwarg["yc"]
        Ic = kwarg["Ic"]
        phase = img.fft.shift(xc, yc)
        return Ic * img.fft.pulse * phase  # jp.exp(-(uphase + vphase))

    def getmap(self, img, convolve=False):
        """
        Generate point source image via Fourier space phase shifts.

        Creates a point source image by computing the appropriate phase shift
        in Fourier space and optionally convolving with the PSF.

        Parameters
        ----------
        img : Image
            Image object containing FFT information and PSF.
        convolve : bool, optional
            If True, convolve with PSF. If False, return unconvolved point source
            (delta function on pixel grid). Default is False.

        Returns
        -------
        ndarray
            Point source image on the image grid.

        Raises
        ------
        ValueError
            If any parameter is a prior distribution or set to None.

        Warns
        -----
        UserWarning
            If convolve=True but no PSF is defined.

        Notes
        -----
        The point source is created using the Fourier shift theorem:

        - The source is placed at (xc, yc) via phase shifts
        - Multiplication by PSF in Fourier space performs convolution
        - More efficient than spatial convolution for point sources
        - The 'pulse' factor accounts for Fourier normalization

        For point sources, PSF convolution is typically essential since an
        unconvolved point source is a delta function (single bright pixel).

        Examples
        --------
        >>> from socca.models import Point
        >>> from socca.data import Image
        >>> point = Point(xc=180.5, yc=45.2, Ic=1000.0)
        >>> img = Image('observation.fits')
        >>> psf_convolved = point.getmap(img, convolve=True)
        """
        kwarg = {key: getattr(self, key) for key in ["xc", "yc", "Ic"]}

        for key in kwarg.keys():
            if isinstance(kwarg[key], numpyro.distributions.Distribution):
                raise ValueError(
                    "Priors must be fixed values, not distributions."
                )
            if kwarg[key] is None:
                raise ValueError(
                    f"keyword {key} is set to None. "
                    f"Please provide a valid value."
                )

        mgrid = self._evaluate(img, **kwarg)

        kernel = img.fft.center

        if convolve:
            if img.psf is None:
                warnings.warn(
                    "No PSF defined, so no convolution will be performed."
                )
            else:
                kernel = img.convolve.psf_fft

        return img.fft.ifft(mgrid * kernel)
