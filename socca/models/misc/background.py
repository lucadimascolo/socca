"""Background model component."""

from functools import partial

import jax
import jax.numpy as jp
import numpyro.distributions

from .. import config
from ..base import Component

import warnings


class Background(Component):
    """
    Polynomial background model for large-scale gradients.

    Models smooth background variations using a 2D polynomial up to 3rd order.
    Useful for fitting sky background, instrumental gradients, or scattered light.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments including:

        rs : float, optional
            Reference radius for normalizing polynomial terms (deg).
        a0 : float, optional
            Constant (0th order) term.
        a1x, a1y : float, optional
            Linear (1st order) terms in x and y.
        a2xx, a2xy, a2yy : float, optional
            Quadratic (2nd order) terms.
        a3xxx, a3xxy, a3xyy, a3yyy : float, optional
            Cubic (3rd order) terms.
        positive : bool, optional
            Whether to enforce positivity constraint.

    Attributes
    ----------
    rs : float
        Reference radius for polynomial normalization (deg).
    a0, a1x, a1y, a2xx, a2xy, a2yy, a3xxx, a3xxy, a3xyy, a3yyy : float
        Polynomial coefficients up to 3rd order.

    Notes
    -----
    The background is defined as:

    B(x,y) = a0 + a1x·x' + a1y·y'
             + a2xx·x'^2 + a2xy·x'·y' + a2yy·y'^2
             + a3xxx·x'^3 + a3xxy·x'^2·y' + a3xyy·x'·y'^2 + a3yyy·y'^3

    where x' = x/rs and y' = y/rs are normalized coordinates.

    - Coordinates are relative to the field center (CRVAL1, CRVAL2)
    - x coordinate includes cos(dec) correction for spherical geometry
    - Background is not PSF-convolved (assumed to vary on large scales)
    - Typically use low-order terms (0th and 1st) to avoid overfitting
    """

    def __init__(self, **kwargs):
        """
        Initialize a polynomial background component.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments including rs and polynomial coefficients a0, a1x, etc.
        """
        super().__init__(**kwargs)
        self.rs = kwargs.get("rs", config.Background.rs)
        self.a0 = kwargs.get("a0", config.Background.a0)
        self.a1x = kwargs.get("a1x", config.Background.a1x)
        self.a1y = kwargs.get("a1y", config.Background.a1y)
        self.a2xx = kwargs.get("a2xx", config.Background.a2xx)
        self.a2xy = kwargs.get("a2xy", config.Background.a2xy)
        self.a2yy = kwargs.get("a2yy", config.Background.a2yy)
        self.a3xxx = kwargs.get("a3xxx", config.Background.a3xxx)
        self.a3xxy = kwargs.get("a3xxy", config.Background.a3xxy)
        self.a3xyy = kwargs.get("a3xyy", config.Background.a3xyy)
        self.a3yyy = kwargs.get("a3yyy", config.Background.a3yyy)

        self.units = dict(rs="deg")
        self.units.update(
            {
                key: ""
                for key in [
                    "a0",
                    "a1x",
                    "a1y",
                    "a2xx",
                    "a2xy",
                    "a2yy",
                    "a3xxx",
                    "a3xxy",
                    "a3xyy",
                    "a3yyy",
                ]
            }
        )

        self.description.update(
            dict(
                rs="Reference radius for polynomial terms",
                a0="Polynomial coefficient 0",
                a1x="Polynomial coefficient 1 in x",
                a1y="Polynomial coefficient 1 in y",
                a2xx="Polynomial coefficient 2 in x·x",
                a2xy="Polynomial coefficient 2 in x·y",
                a2yy="Polynomial coefficient 2 in y·y",
                a3xxx="Polynomial coefficient 3 in x·x·x",
                a3xxy="Polynomial coefficient 3 in x·x·y",
                a3xyy="Polynomial coefficient 3 in x·y·y",
                a3yyy="Polynomial coefficient 3 in y·y·y",
            )
        )
        self._initialized = True

    @staticmethod
    @jax.jit
    def profile(
        x, y, a0, a1x, a1y, a2xx, a2xy, a2yy, a3xxx, a3xxy, a3xyy, a3yyy, rs
    ):
        """
        Evaluate 2D polynomial background on coordinate grids.

        Parameters
        ----------
        x, y : ndarray
            Coordinate grids in degrees (relative to field center).
        a0 : float
            Constant term.
        a1x, a1y : float
            Linear coefficients.
        a2xx, a2xy, a2yy : float
            Quadratic coefficients.
        a3xxx, a3xxy, a3xyy, a3yyy : float
            Cubic coefficients.
        rs : float
            Reference radius for normalization (deg).

        Returns
        -------
        ndarray
            Background values on the coordinate grid.

        Notes
        -----
        Evaluates the polynomial:

        B = a0 + a1x·x' + a1y·y' + a2xx·x'^2 + a2xy·x'·y' + a2yy·y'^2
            + a3xxx·x'^3 + a3xxy·x'^2·y' + a3xyy·x'·y'^2 + a3yyy·y'^3

        where x' = x/rs and y' = y/rs.

        The normalization by rs keeps coefficients of different orders at
        comparable scales and improves numerical conditioning.
        """
        xc, yc = x / rs, y / rs
        factor = a0
        factor += a1x * xc + a1y * yc
        factor += a2xy * xc * yc + a2xx * xc**2 + a2yy * yc**2
        factor += (
            a3xxx * xc**3
            + a3yyy * yc**3
            + a3xxy * xc**2 * yc
            + a3xyy * xc * yc**2
        )
        return factor

    @staticmethod
    @partial(jax.jit, static_argnames=["grid"])
    def getgrid(grid, xc, yc):
        """
        Compute relative coordinates for background evaluation.

        Converts absolute celestial coordinates to coordinates relative to the
        field center, with spherical geometry correction.

        Parameters
        ----------
        grid : Grid
            Grid object with .x and .y coordinate arrays (deg).
        xc : float
            Reference RA (field center) in degrees.
        yc : float
            Reference Dec (field center) in degrees.

        Returns
        -------
        xgrid : ndarray
            RA offsets in degrees (with cos(dec) correction).
        ygrid : ndarray
            Dec offsets in degrees.

        Notes
        -----
        The RA offset includes a cos(dec) factor to account for spherical
        geometry, ensuring that distances are approximately correct on the sky.
        """
        return (grid.x - xc) * jp.cos(jp.deg2rad(yc)), grid.y - yc

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
        Evaluate background on the image grid with explicit parameters.

        This internal method computes the polynomial background using the
        provided coefficients. It is used by both getmap() and Model.getmodel()
        to avoid code duplication.

        Parameters
        ----------
        img : Image
            Image object containing grid and WCS information.
        **kwarg : dict
            Polynomial coefficients (a0, a1x, a1y, ..., rs).

        Returns
        -------
        ndarray
            Background map on the image grid.
        """
        yr = jp.mean(img.grid.y, axis=0) - img.hdu.header["CRVAL2"]
        xr = jp.mean(img.grid.x, axis=0) - img.hdu.header["CRVAL1"]
        xr = xr * jp.cos(jp.deg2rad(img.hdu.header["CRVAL2"]))
        return self.profile(xr, yr, **kwarg)

    def getmap(self, img, **kwargs):
        """
        Generate background map on the image grid.

        Evaluates the polynomial background across the entire image field.

        Parameters
        ----------
        img : Image
            Image object containing grid and WCS information.
        **kwargs : dict
            Ignored keyword arguments (for API compatibility).

        Returns
        -------
        ndarray
            Background map on the image grid.

        Raises
        ------
        ValueError
            If any parameter is a prior distribution or set to None.

        Warns
        -----
        UserWarning
            If 'convolve' argument is provided (background is never convolved).

        Notes
        -----
        - Background is evaluated relative to field center (CRVAL1, CRVAL2)
        - Background is not PSF-convolved (assumed to vary smoothly)
        - All polynomial coefficients must have fixed values

        Examples
        --------
        >>> from socca.models import Background
        >>> from socca.data import Image
        >>> bkg = Background(a0=10.0, a1x=0.1, a1y=-0.05, rs=1.0)
        >>> img = Image('observation.fits')
        >>> bkg_map = bkg.getmap(img)
        """
        if "convolve" in kwargs:
            warnings.warn(
                "Background component does not support convolution. "
                "Ignoring `convolve` argument."
            )

        klist = list(self.units.keys())
        kwarg = {key: getattr(self, key) for key in klist}

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

        return self._evaluate(img, **kwarg)
