import inspect
import types
import warnings

import jax
import jax.numpy as jp
import numpy as np

import numpyro.distributions

from . import config
from .base import Component
from .misc import Point, Background
from .disk import Disk
from .bridge import SimpleBridge, MesaBridge
from .radial import (
    Profile,
    CustomProfile,
    Sersic,
    Gaussian,
    Beta,
    gNFW,
    Power,
    TopHat,
    Exponential,
    PolyExponential,
    PolyExpoRefact,
    ModExponential,
)


# Support utilities
# ========================================================
# Print available models
# --------------------------------------------------------
def zoo():
    """
    Print available model profiles in the socca library.

    This function displays a list of all available profile models that can be
    used for fitting astronomical images, including various analytical profiles
    for galaxies, point sources, and background components.

    Notes
    -----
    Available models include:

    - Beta: Beta profile (power-law density)
    - gNFW: Generalized Navarro-Frenk-White profile
    - Power: Power-law profile
    - TopHat: Uniform top-hat profile
    - Sersic: Sersic profile for elliptical galaxies
    - Gaussian: Gaussian profile
    - Exponential: Exponential disk profile
    - PolyExponential: Polynomial-exponential profile
    - PolyExpoRefact: Refactored polynomial-exponential profile
    - ModExponential: Modified exponential profile
    - Point: Point source model
    - Background: Polynomial background model
    - Disk: 3D disk model with finite thickness

    Examples
    --------
    >>> import socca.models as models
    >>> models.zoo()
    Beta
    gNFW
    Power
    TopHat
    Sersic
    Gaussian
    Exponential
    PolyExponential
    PolyExpoRefact
    ModExponential
    Point
    Background
    Disk
    """
    print("\nRadial models")
    print("=============")

    models = [
        "Beta",
        "gNFW",
        "Power",
        "TopHat",
        "Sersic",
        "Gaussian",
        "Exponential",
        "PolyExponential",
        "PolyExpoRefact",
        "ModExponential",
    ]

    for mi, m in enumerate(models):
        print(m)

    print("\nFilaments")
    print("============")
    models = [
        "SharpBridge",
        "MesaBridge",
    ]

    for mi, m in enumerate(models):
        print(m)

    print("\nDisk-like model")
    print("===============")
    print("Disk")

    print("\nOther models")
    print("============")
    models = [
        "Point",
        "Background",
    ]

    for mi, m in enumerate(models):
        print(m)


# Models
# ========================================================
# General model structure
# --------------------------------------------------------
class Model:
    """
    Composite model container for combining multiple profile components.

    The Model class provides a flexible framework for constructing complex
    astronomical models by combining individual profile components (Beta, Sersic,
    Point, etc.). It manages parameter priors, positivity constraints, and model
    evaluation.
    """

    def __init__(self, prof=None, positive=None):
        """
        Initialize a composite model with optional initial profile component.

        Parameters
        ----------
        prof : Profile or Component, optional
            Initial profile component to add to the model. Can be any profile
            instance (Beta, Sersic, Point, etc.). If None, creates an empty
            model to which components can be added later.
        positive : bool, optional
            Whether to enforce positivity constraint on the profile. If None,
            uses the default positivity setting from the profile itself.

        Attributes
        ----------
        ncomp : int
            Number of components in the model.
        priors : dict
            Dictionary mapping parameter names to their values or prior distributions.
        params : list of str
            List of all parameter names in the model.
        paridx : list of int
            Indices of parameters with prior distributions (free parameters).
        positive : list of bool
            Positivity constraints for each component.
        profile : list of callable
            Profile functions for each component.
        gridder : list of callable
            Grid generation functions for each component.
        tied : list of bool
            Indicates which parameters are tied to other parameters.
        type : list of str
            Class names of each component in the model.
        units : dict
            Dictionary mapping parameter names to their physical units.

        Examples
        --------
        >>> from socca.models import Model, Beta
        >>> # Create empty model and add component later
        >>> model = Model()
        >>> model.addcomponent(Beta())
        >>> # Or initialize with a profile directly
        >>> model = Model(Beta())
        """
        self.ncomp = 0
        self.priors = {}
        self.params = []
        self.paridx = []
        self.positive = []
        self.profile = []
        self.gridder = []
        self.tied = []
        self.type = []
        self.units = {}

        if prof is not None:
            self.addcomponent(prof, positive)

    # Add a new component to the model
    # --------------------------------------------------------
    def addcomponent(self, prof, positive=None):
        """
        Add a profile component to the composite model.

        This method incorporates a new profile component into the model, handling
        parameter registration, prior assignments, and constraint management.
        Components are indexed sequentially and their parameters are prefixed
        with the component index.

        Parameters
        ----------
        prof : Profile or Component
            Profile component to add (e.g., Beta, Sersic, Point, Background, Disk).
        positive : bool, optional
            Override the default positivity constraint for this component.
            If None, uses the component's default positivity setting.

        Raises
        ------
        ValueError
            If any parameter in the profile is set to None without a valid
            value or prior distribution.

        Notes
        -----
        - Parameters are stored with names like 'comp_00_xc', 'comp_01_rc', etc.
        - Parameters can be fixed values, prior distributions, or tied (functions).
        - Only parameters with prior distributions are considered free parameters
          during fitting.
        - Tied parameters are evaluated as functions of other parameters.

        Examples
        --------
        >>> from socca.models import Model, Beta, Point
        >>> model = Model()
        >>> # Add a Beta profile
        >>> model.addcomponent(Beta(xc=180.5, yc=45.2, rc=0.01))
        >>> # Add a point source
        >>> model.addcomponent(Point(xc=180.6, yc=45.3), positive=True)
        >>> print(model.ncomp)
        2
        """
        self.type.append(prof.__class__.__name__)
        self.positive.append(prof.positive if positive is None else positive)
        for p in prof.parlist():
            par = eval(f"prof.{p}")

            if par is None:
                raise ValueError(
                    f"Parameter {p} in component {self.ncomp:02d} is set "
                    f"to None. Please provide a valid value or prior."
                )

            self.params.append(f"comp_{self.ncomp:02d}_{p}")
            self.priors.update({f"comp_{self.ncomp:02d}_{p}": par})
            if isinstance(par, numpyro.distributions.Distribution):
                self.paridx.append(len(self.params) - 1)

            if isinstance(par, (types.LambdaType, types.FunctionType)):
                self.tied.append(True)
            else:
                self.tied.append(False)

            self.units.update({f"comp_{self.ncomp:02d}_{p}": prof.units[p]})

        self.profile.append(prof.profile)
        self.gridder.append(prof.getgrid)
        self.ncomp += 1

    def parameters(self, freeonly=False):
        """
        Print a formatted table of model parameters with their values and units.

        This method displays all model parameters, showing fixed values, prior
        distributions, or tied parameter relationships in a human-readable format.

        Parameters
        ----------
        freeonly : bool, optional
            If True, only display parameters with prior distributions (free
            parameters to be fitted). If False (default), display all parameters
            including fixed values and tied parameters.

        Notes
        -----
        Parameter values are displayed as:

        - Fixed values: shown in scientific notation (e.g., 1.5000E-02)
        - Prior distributions: shown as "Distribution: DistributionName"
        - Tied parameters: shown as "Tied parameter"

        The output format is:
        parameter_name [units] : value

        Examples
        --------
        >>> from socca.models import Model, Beta
        >>> import numpyro.distributions as dist
        >>> model = Model(Beta(xc=dist.Uniform(180, 181), yc=45.2, rc=0.01))
        >>> model.parameters()
        Model parameters
        ================
        comp_00_xc   [deg]   : Distribution: Uniform
        comp_00_yc   [deg]   : 4.5200E+01
        comp_00_rc   [deg]   : 1.0000E-02
        ...
        >>> model.parameters(freeonly=True)
        Model parameters
        ================
        comp_00_xc   [deg]   : Distribution: Uniform
        """
        keyout = []
        for key in self.params:
            if freeonly:
                if isinstance(
                    self.priors[key], numpyro.distributions.Distribution
                ):
                    keyout.append(key)
            else:
                keyout.append(key)

        if len(keyout) > 0:
            maxlen = np.max(
                np.array([len(f"{key} [{self.units[key]}]") for key in keyout])
            )

            print("\nModel parameters")
            print("=" * 16)
            for key in keyout:
                keylen = maxlen - len(f" [{self.units[key]}]")
                par = self.priors[key]
                if par is None:
                    keyval = None
                elif isinstance(par, numpyro.distributions.Distribution):
                    keyval = f"Distribution: {par.__class__.__name__}"
                elif isinstance(par, (types.LambdaType, types.FunctionType)):
                    keyval = "Tied parameter"
                else:
                    keyval = f"{par:.4E}"
                print(
                    f"{key:<{keylen}} [{self.units[key]}] : "
                    + f"{keyval}".ljust(10)
                )

        else:
            print("No parameters defined.")

    # Get the model map with fixed parameters
    # --------------------------------------------------------
    def getmap(self, img, convolve=None, addbackground=False):
        """
        Generate a model image map using fixed parameter values only.

        This method evaluates the model on the provided image grid using only
        fixed parameter values. All parameters must be fixed; parameters with
        prior distributions will raise an error.

        Parameters
        ----------
        img : Image
            Image object containing the grid, PSF, and WCS information.
        convolve : bool, optional
            If True, return the PSF-convolved model. If False (default), return
            the unconvolved model. If None, defaults to False.
        addbackground : bool, optional
            If True and convolve is True, include the background component in
            the output. If False (default), exclude background. Note that
            background is only added to convolved maps.

        Returns
        -------
        ndarray
            Model image map on the same grid as img.data. Shape matches img.data.shape.

        Raises
        ------
        ValueError
            If any parameter has a prior distribution instead of a fixed value.

        Warns
        -----
        UserWarning
            If addbackground=True but convolve=False, warns that background
            is only added to convolved maps.

        Notes
        -----
        - All model parameters must be fixed values (float or int).
        - Tied parameters are automatically evaluated from their dependencies.
        - For use during fitting with sampled parameters, use getmodel() instead.

        Examples
        --------
        >>> from socca.models import Model, Beta
        >>> from socca.data import Image
        >>> model = Model(Beta(xc=180.5, yc=45.2, rc=0.01, Ic=100, beta=0.5))
        >>> img = Image('observation.fits')
        >>> # Get unconvolved model
        >>> model_map = model.getmap(img)
        >>> # Get PSF-convolved model
        >>> convolved_map = model.getmap(img, convolve=True)
        """
        pars = {}
        for ki, key in enumerate(self.params):
            if isinstance(self.priors[key], (float, int)):
                pars[key] = self.priors[key]
            elif isinstance(
                self.priors[key], numpyro.distributions.Distribution
            ):
                raise ValueError(
                    f"Parameter {key} is a distribution. "
                    f"Use getmap() with sampled values instead."
                )

        for ki, key in enumerate(self.params):
            if self.tied[ki]:
                kwarg = list(
                    inspect.signature(self.priors[key]).parameters.keys()
                )
                kwarg = {k: pars[k] for k in kwarg}
                pars[key] = self.priors[key](**kwarg)
                del kwarg

        mraw, msmo, mbkg, _ = self.getmodel(
            img, pars, doresp=False, doexp=False
        )

        if not addbackground:
            msmo = msmo - mbkg

        if convolve:
            return msmo
        else:
            if addbackground:
                warnings.warn(
                    "The background component is added only to the "
                    "convolved map. Returning the raw map without background."
                )
            return mraw

    def _comp_filter(self, component=None):
        """
        Parse component identifiers into a list of integer indices.

        This internal method normalizes various component identifier formats
        into a consistent list of integer indices for use in model
        computation and filtering.

        Parameters
        ----------
        component : None, str, int, list, or Profile, optional
            Component identifier(s) to parse. Can be:

            - None: returns all component indices [0, 1, 2, ...]
            - str: parsed to integer (e.g., 'comp_00' -> [0])
            - int: returned as single-element list (e.g., 0 -> [0])
            - list: each element converted individually (can mix types)
            - Profile object with `id` attribute: converted using the id

        Returns
        -------
        list of int
            List of component indices.

        Examples
        --------
        >>> model._comp_filter(None)  # All components
        [0, 1, 2]
        >>> model._comp_filter(0)  # Single index
        [0]
        >>> model._comp_filter([0, 2])  # Multiple indices
        [0, 2]
        >>> model._comp_filter('comp_01')  # String format
        [1]
        """
        if component is None:
            component = [int(ci) for ci in range(self.ncomp)]
        elif isinstance(component, int):
            component = [component]
        elif isinstance(component, str):
            component = component.replace("comp", "").replace("_", "")
            component = [int(component)]
        elif isinstance(component, (list, tuple)):
            component_ = []
            for c in component:
                if isinstance(c, int):
                    component_.append(c)
                elif isinstance(c, str):
                    c_ = c.replace("comp", "").replace("_", "")
                    component_.append(int(c_))
                elif hasattr(c, "id"):
                    c_ = c.id.replace("comp", "").replace("_", "")
                    component_.append(int(c_))
                    del c_
            component = component_
            del component_
        return component

    # Get the model map
    # --------------------------------------------------------
    def getmodel(self, img, pp, doresp=False, doexp=False, component=None):
        """
        Compute the full model image with PSF convolution, response, and exposure.

        This is the main forward modeling function used during fitting. It evaluates
        all model components, applies PSF convolution, instrumental response, and
        exposure corrections as specified.

        Parameters
        ----------
        img : Image
            Image object containing data, grid, PSF, response map, and exposure map.
        pp : array_like or dict
            Parameter values. If array_like, contains values for parameters with
            prior distributions, in the order they appear in paridx. If dict,
            contains all parameter values with keys matching self.params.
        doresp : bool, optional
            If True, multiply by the instrumental response map (img.resp).
            Default is False.
        doexp : bool, optional
            If True, multiply by the exposure map (img.exp). Default is False.
        component : None, str, int, list, or Profile, optional
            Component(s) to include in the model computation. Can be:

            - None: Include all components (default)
            - str: Single component name (e.g., 'comp_00')
            - int: Component index (e.g., 0 for first component)
            - list: Multiple components as names, indices, or Profile objects
            - Profile: Object with `id` attribute specifying the component

            Default is None (all components).

        Returns
        -------
        mraw : ndarray
            Unconvolved model image (raw surface brightness distribution).
        msmo : ndarray
            Final model image after PSF convolution, response, and exposure
            corrections, including background.
        mbkg : ndarray
            Background component only, with exposure applied if doexp=True.
        mneg : ndarray
            Mask indicating pixels where model would be negative (if positivity
            constraints are violated). 1.0 where negative, 0.0 otherwise.

        Notes
        -----
        - Fixed parameters are taken from self.priors
        - Free parameters are extracted from pp array
        - Tied parameters are computed from their functional dependencies
        - Point sources are handled in Fourier space for efficiency
        - Background components are not PSF-convolved
        - Disk components use 3D line-of-sight integration
        - The positivity mask (mneg) can be used to penalize negative values

        Examples
        --------
        >>> from socca.models import Model, Beta
        >>> import numpyro.distributions as dist
        >>> model = Model(Beta(xc=dist.Uniform(180, 181), yc=45.2))
        >>> img = Image('observation.fits')
        >>> # During fitting, pp contains sampled parameter values
        >>> pp = [180.5]  # value for xc
        >>> mraw, msmo, mbkg, mneg = model.getmodel(img, pp, doresp=True, doexp=True)
        """
        pars = {}
        for ki, key in enumerate(self.params):
            if isinstance(self.priors[key], (float, int)):
                pars[key] = self.priors[key]
            elif isinstance(
                self.priors[key], numpyro.distributions.Distribution
            ):
                pars[key], pp = pp[0], pp[1:]

        for ki, key in enumerate(self.params):
            if self.tied[ki]:
                kwarg = list(
                    inspect.signature(self.priors[key]).parameters.keys()
                )
                kwarg = {k: pars[k] for k in kwarg}
                pars[key] = self.priors[key](**kwarg)
                del kwarg

        mbkg = jp.zeros(img.data.shape)
        mraw = jp.zeros(img.data.shape)
        mpts = jp.fft.rfft2(mraw, s=img.data.shape)

        mneg = jp.zeros(img.data.shape)

        comp_ = self._comp_filter(component)

        for nc in comp_:
            if self.type[nc] == "Disk":
                kwarg = {
                    key.replace(f"comp_{nc:02d}_radial.", "r_"): pars[key]
                    for key in self.params
                    if key.startswith(f"comp_{nc:02d}")
                    and key.replace(f"comp_{nc:02d}_radial.", "r_")
                    in list(
                        inspect.signature(self.profile[nc]).parameters.keys()
                    )
                }
                kwarg.update(
                    {
                        key.replace(f"comp_{nc:02d}_vertical.", "z_"): pars[
                            key
                        ]
                        for key in self.params
                        if key.startswith(f"comp_{nc:02d}")
                        and key.replace(f"comp_{nc:02d}_vertical.", "z_")
                        in list(
                            inspect.signature(
                                self.profile[nc]
                            ).parameters.keys()
                        )
                    }
                )

                rcube, zcube = self.gridder[nc](
                    img.grid,
                    pars[f"comp_{nc:02d}_radial.xc"],
                    pars[f"comp_{nc:02d}_radial.yc"],
                    pars[f"comp_{nc:02d}_vertical.losdepth"],
                    pars[f"comp_{nc:02d}_vertical.losbins"],
                    pars[f"comp_{nc:02d}_radial.theta"],
                    pars[f"comp_{nc:02d}_vertical.inc"],
                )

                dx = (
                    2.00
                    * pars[f"comp_{nc:02d}_vertical.losdepth"]
                    / (pars[f"comp_{nc:02d}_vertical.losbins"] - 1)
                )
                mone = self.profile[nc](rcube, zcube, **kwarg)
                mone = jp.trapezoid(mone, dx=dx, axis=1)
                mone = jp.mean(mone, axis=0)
                mraw += mone.copy()
                del mone
            else:
                kwarg = {
                    key.replace(f"comp_{nc:02d}_", ""): pars[key]
                    for key in self.params
                    if key.startswith(f"comp_{nc:02d}")
                    and key.replace(f"comp_{nc:02d}_", "")
                    in list(
                        inspect.signature(self.profile[nc]).parameters.keys()
                    )
                }

                if self.type[nc] == "Point":
                    uphase, vphase = img.fft.shift(kwarg["xc"], kwarg["yc"])

                    mone = (
                        kwarg["Ic"]
                        * img.fft.pulse
                        * jp.exp(-(uphase + vphase))
                    )
                    if self.positive[nc]:
                        mneg = jp.where(mone < 0.00, 1.00, mneg)

                    if doresp:
                        xpts = (
                            kwarg["xc"] - img.hdu.header["CRVAL1"]
                        ) / jp.abs(img.hdu.header["CDELT1"])
                        ypts = (
                            kwarg["yc"] - img.hdu.header["CRVAL2"]
                        ) / jp.abs(img.hdu.header["CDELT2"])

                        xpts = (
                            img.hdu.header["CRPIX1"]
                            - 1
                            + xpts
                            * jp.cos(jp.deg2rad(img.hdu.header["CRVAL2"]))
                        )
                        ypts = img.hdu.header["CRPIX2"] - 1 + ypts
                        mone *= jax.scipy.ndimage.map_coordinates(
                            img.resp,
                            [jp.array([ypts]), jp.array([xpts])],
                            order=1,
                            mode="nearest",
                        )[0]

                    mpts += mone.copy()
                    del mone

                elif self.type[nc] == "Background":
                    yr = jp.mean(img.grid.y, axis=0) - img.hdu.header["CRVAL2"]
                    xr = jp.mean(img.grid.x, axis=0) - img.hdu.header["CRVAL1"]
                    xr = xr * jp.cos(jp.deg2rad(img.hdu.header["CRVAL2"]))

                    mone = self.profile[nc](xr, yr, **kwarg)
                    if self.positive[nc]:
                        mneg = jp.where(mone < 0.00, 1.00, mneg)

                    mbkg += mone.copy()
                    del mone

                else:
                    rgrid = self.gridder[nc](
                        img.grid,
                        pars[f"comp_{nc:02d}_xc"],
                        pars[f"comp_{nc:02d}_yc"],
                        pars[f"comp_{nc:02d}_theta"],
                        pars[f"comp_{nc:02d}_e"],
                        pars[f"comp_{nc:02d}_cbox"],
                    )

                    mone = self.profile[nc](rgrid, **kwarg)
                    mone = jp.mean(mone, axis=0)
                    if self.positive[nc]:
                        mneg = jp.where(mone < 0.00, 1.00, mneg)

                    mraw += mone.copy()
                    del mone

        msmo = mraw.copy()
        if doresp:
            msmo *= img.resp

        if img.psf is not None:
            msmo = (
                mpts + jp.fft.rfft2(jp.fft.fftshift(msmo), s=img.data.shape)
            ) * img.psf_fft
            msmo = jp.fft.ifftshift(jp.fft.irfft2(msmo, s=img.data.shape)).real

        mpts = jp.fft.ifftshift(jp.fft.irfft2(mpts, s=img.data.shape)).real

        if img.psf is None:
            msmo = msmo + mpts

        if doexp:
            msmo *= img.exp
            mbkg *= img.exp

        msmo = msmo + mbkg

        return mraw + mpts, msmo, mbkg, mneg


# Reset component counter after module initialization
# (default arguments in Disk.__init__ create instances at import time)
Component.idcls = 0
