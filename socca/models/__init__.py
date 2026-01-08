from abc import abstractmethod
from functools import partial

import inspect
import types
import warnings

import jax
import jax.numpy as jp
import numpy as np

import numpyro.distributions

from . import config

from quadax import quadgk
from scipy.special import gammaincinv


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
    - Sersic: Sersic profile for elliptical galaxies
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
    Sersic
    Exponential
    PolyExponential
    PolyExpoRefact
    ModExponential
    Point
    Background
    Disk
    """
    models = [
        "Beta",
        "gNFW",
        "Sersic",
        "Exponential",
        "PolyExponential",
        "PolyExpoRefact",
        "ModExponential",
        "Point",
        "Background",
        "Disk",
    ]
    for mi, m in enumerate(models):
        print(m)


# Models
# ========================================================
# General model structure
# --------------------------------------------------------
class Model:
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
                if isinstance(par, numpyro.distributions.Distribution):
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

    # Get the model map
    # --------------------------------------------------------
    def getmodel(self, img, pp, doresp=False, doexp=False):
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

        for nc in range(self.ncomp):
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


# General composable term
# --------------------------------------------------------
class Component:
    """
    Base class for all model components.

    This abstract base class provides the infrastructure for model components,
    including parameter management, unique identification, and formatted output.
    All specific profile types (Beta, Sersic, Point, etc.) inherit from this class.

    Attributes
    ----------
    idcls : int (class variable)
        Class-level counter for assigning unique IDs to component instances.
    id : str
        Unique identifier for this component instance (e.g., 'comp_00', 'comp_01').
    positive : bool
        Whether to enforce positivity constraint on this component.
    hyper : list of str
        Names of hyperparameters (not fitted, control behavior).
    units : dict
        Mapping from parameter names to their physical units.
    description : dict
        Mapping from parameter names to their descriptions.
    okeys : list of str
        Reserved keys that should not be treated as model parameters.
    """

    idcls = 0

    def __init__(self, **kwargs):
        """
        Initialize a component with unique ID and parameter tracking structures.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for component initialization.

            positive : bool, optional
                Override default positivity constraint.

        Notes
        -----
        The component ID is automatically assigned and incremented for each
        new instance. This ensures unique identification when multiple components
        are combined in a model.
        """
        self.id = f"comp_{type(self).idcls:02d}"
        type(self).idcls += 1

        self.positive = kwargs.get("positive", config.Component.positive)
        self.okeys = ["id", "positive", "hyper", "units", "description"]
        self.hyper = []
        self.units = {}
        self.description = {}

    #   Print model parameters
    #   --------------------------------------------------------
    def parameters(self):
        """
        Print a formatted table of component parameters and hyperparameters.

        Displays all parameters (excluding reserved keys) with their values,
        units, and descriptions in a human-readable format. Separates regular
        parameters from hyperparameters.

        Notes
        -----
        Output format:

        Model parameters
        ================
        parameter_name [units] : value | description

        Hyperparameters
        ===============
        hyperparam_name [units] : value | description

        - Parameters are model quantities to be fitted or fixed
        - Hyperparameters control computation but are not fitted
        - Values are shown in scientific notation (e.g., 1.5000E-02)
        - None values are displayed as "None"

        Examples
        --------
        >>> from socca.models import Beta
        >>> beta = Beta(xc=180.5, yc=45.2, rc=0.01, Ic=100, beta=0.5)
        >>> beta.parameters()
        Model parameters
        ================
        xc    [deg]    : 1.8050E+02 | Right ascension of centroid
        yc    [deg]    : 4.5200E+01 | Declination of centroid
        rc    [deg]    : 1.0000E-02 | Core radius
        Ic    [image]  : 1.0000E+02 | Central surface brightness
        beta  []       : 5.0000E-01 | Slope parameter
        theta [rad]    : 0.0000E+00 | Position angle (east from north)
        e     []       : 0.0000E+00 | Projected axis ratio
        cbox  []       : 0.0000E+00 | Projected boxiness
        """
        keyout = []
        for key in self.__dict__.keys():
            if (
                key not in self.okeys
                and key not in self.hyper
                and key != "okeys"
            ):
                keyout.append(key)

        if len(keyout) > 0:
            maxlen = np.max(
                np.array(
                    [
                        len(f"{key} [{self.units[key]}]")
                        for key in keyout + self.hyper
                    ]
                )
            )

            print("\nModel parameters")
            print("=" * 16)
            for key in keyout:
                keylen = maxlen - len(f" [{self.units[key]}]")
                keyval = (
                    None
                    if self.__dict__[key] is None
                    else f"{self.__dict__[key]:.4E}"
                )
                print(
                    f"{key:<{keylen}} [{self.units[key]}] : "
                    + f"{keyval}".ljust(10)
                    + f" | {self.description[key]}"
                )

            if len(self.hyper) > 0:
                print("\nHyperparameters")
                print("=" * 15)
                for key in self.hyper:
                    keylen = maxlen - len(f" [{self.units[key]}]")
                    kvalue = self.__dict__[key]
                    kvalue = None if kvalue is None else f"{kvalue:.4E}"
                    print(
                        f"{key:<{keylen}} [{self.units[key]}] : "
                        + f"{kvalue}".ljust(10)
                        + f" | {self.description[key]}"
                    )

        else:
            print("No parameters defined.")

    def parlist(self):
        """
        Return a list of parameter names for this component.

        Returns
        -------
        list of str
            Names of all parameters (excluding reserved keys like 'id', 'positive',
            'hyper', 'units', 'description', and 'okeys').

        Notes
        -----
        This method is used internally by Model.addcomponent() to register
        parameters when adding a component to a composite model.

        Examples
        --------
        >>> from socca.models import Beta
        >>> beta = Beta(xc=180.5, yc=45.2)
        >>> beta.parlist()
        ['xc', 'yc', 'theta', 'e', 'cbox', 'rc', 'Ic', 'beta']
        """
        return [
            key
            for key in self.__dict__.keys()
            if key not in self.okeys and key != "okeys"
        ]

    def addpar(self, name, value=None, units="", description=""):
        """
        Add a new parameter to the component with metadata.

        Parameters
        ----------
        name : str
            Name of the parameter to add.
        value : float, int, Distribution, or callable, optional
            Parameter value. Can be a fixed value, a numpyro prior distribution,
            or a function for tied parameters. Default is None.
        units : str, optional
            Physical units of the parameter (e.g., 'deg', 'rad', 'image').
            Default is empty string.
        description : str, optional
            Human-readable description of the parameter. Default is empty string.

        Notes
        -----
        This method dynamically adds a parameter attribute to the component
        instance and registers its metadata (units and description).

        Examples
        --------
        >>> from socca.models import Component
        >>> comp = Component()
        >>> comp.addpar('amplitude', value=1.0, units='Jy', description='Source flux')
        >>> comp.amplitude
        1.0
        >>> comp.units['amplitude']
        'Jy'
        """
        setattr(self, name, value)
        self.units.update({name: units})
        self.description.update({name: description})


# General profile class
# --------------------------------------------------------
class Profile(Component):
    """
    Base class for 2D surface brightness profiles.

    This class extends Component to provide common coordinate transformation
    and projection parameters (position, orientation, ellipticity, boxiness)
    for all 2D surface brightness profiles.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments for profile initialization including:

        xc : float, optional
            Right ascension of profile centroid (deg).
        yc : float, optional
            Declination of profile centroid (deg).
        theta : float, optional
            Position angle measured east from north (rad).
        e : float, optional
            Ellipticity, defined as 1 - b/a where b/a is axis ratio (0 <= e < 1).
        cbox : float, optional
            Boxiness parameter (0 = elliptical, >0 = boxy, <0 = disky).
        positive : bool, optional
            Whether to enforce positivity constraint.

    Attributes
    ----------
    xc, yc : float
        Centroid coordinates in celestial degrees.
    theta : float
        Position angle in radians.
    e : float
        Ellipticity parameter.
    cbox : float
        Boxiness/diskyness parameter.

    Notes
    -----
    All Profile subclasses must implement the abstract profile() method that
    defines the radial surface brightness distribution.
    """

    def __init__(self, **kwargs):
        """
        Initialize a profile with standard coordinate and shape parameters.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments including xc, yc, theta, e, cbox, positive.
            See class docstring for parameter descriptions.
        """
        super().__init__(**kwargs)
        self.xc = kwargs.get("xc", config.Profile.xc)
        self.yc = kwargs.get("yc", config.Profile.yc)

        self.theta = kwargs.get("theta", config.Profile.theta)
        self.e = kwargs.get("e", config.Profile.e)
        self.cbox = kwargs.get("cbox", config.Profile.cbox)

        self.units.update(dict(xc="deg", yc="deg", theta="rad", e="", cbox=""))

        self.description.update(
            dict(
                xc="Right ascension of centroid",
                yc="Declination of centroid",
                theta="Position angle (east from north)",
                e="Projected axis ratio",
                cbox="Projected boxiness",
            )
        )

    @abstractmethod
    def profile(self, r):
        pass

    def getmap(self, img, convolve=False):
        """
        Generate an image map from the profile on the given grid.

        Evaluates the profile on the image grid and optionally convolves with
        the PSF. All parameters must have fixed values.

        Parameters
        ----------
        img : Image
            Image object containing grid, PSF, and WCS information.
        convolve : bool, optional
            If True, convolve the model with the PSF. Default is False.

        Returns
        -------
        ndarray
            2D array of surface brightness values on the image grid.
            Shape matches img.data.shape.

        Raises
        ------
        ValueError
            If any parameter is a prior distribution or set to None.

        Warns
        -----
        UserWarning
            If convolve=True but no PSF is defined in img.

        Notes
        -----
        - All profile parameters must be fixed values (not distributions).
        - The elliptical grid is computed using position, ellipticity, and PA.
        - Convolution is performed in Fourier space for efficiency.

        Examples
        --------
        >>> from socca.models import Beta
        >>> from socca.data import Image
        >>> beta = Beta(xc=180.5, yc=45.2, rc=0.01, Ic=100, beta=0.5)
        >>> img = Image('observation.fits')
        >>> model_map = beta.getmap(img, convolve=True)
        """
        kwarg = {
            key: getattr(self, key)
            for key in list(inspect.signature(self.profile).parameters.keys())
            if key != "r"
        }

        rgrid = self.getgrid(img.grid, self.xc, self.yc, self.theta, self.e)

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

        mgrid = self.profile(rgrid, **kwarg)
        mgrid = jp.mean(mgrid, axis=0)

        if convolve:
            if img.psf is None:
                warnings.warn(
                    "No PSF defined, so no convolution will be performed."
                )
            else:
                mgrid = (
                    jp.fft.rfft2(jp.fft.fftshift(mgrid), s=img.data.shape)
                    * img.psf_fft
                )
                mgrid = jp.fft.ifftshift(
                    jp.fft.irfft2(mgrid, s=img.data.shape)
                ).real
        return mgrid

    @staticmethod
    @partial(jax.jit, static_argnames=["grid"])
    def getgrid(grid, xc, yc, theta=0.00, e=0.00, cbox=0.00):
        """
        Compute elliptical radius grid with rotation and projection.

        This static JIT-compiled method transforms celestial coordinates to
        elliptical radius values accounting for position angle, ellipticity,
        and boxiness. Used internally by profile evaluation.

        Parameters
        ----------
        grid : Grid
            Grid object with .x and .y celestial coordinate arrays (deg).
        xc : float
            Right ascension of centroid (deg).
        yc : float
            Declination of centroid (deg).
        theta : float, optional
            Position angle east from north (rad). Default is 0.
        e : float, optional
            Ellipticity (1 - b/a). Default is 0 (circular).
        cbox : float, optional
            Boxiness parameter. Default is 0 (elliptical).

        Returns
        -------
        ndarray
            Elliptical radius grid in degrees. Same shape as grid.x and grid.y.

        Notes
        -----
        The transformation accounts for:
        - Spherical geometry (cos(dec) correction)
        - Position angle rotation
        - Ellipticity via axis ratio correction
        - Generalized elliptical isophotes with boxiness

        The elliptical radius is computed as:
        r = [(|x'|^(2+c) + |y'/(1-e)|^(2+c)]^(1/(2+c))
        where c is the boxiness parameter.

        This function is JIT-compiled for performance during model evaluation.
        """
        sint = jp.sin(theta)
        cost = jp.cos(theta)

        xgrid = (
            -(grid.x - xc) * jp.cos(jp.deg2rad(yc)) * sint
            - (grid.y - yc) * cost
        )
        ygrid = (grid.x - xc) * jp.cos(jp.deg2rad(yc)) * cost - (
            grid.y - yc
        ) * sint

        xgrid = jp.abs(xgrid) ** (cbox + 2.00)
        ygrid = jp.abs(ygrid / (1.00 - e)) ** (cbox + 2.00)
        return jp.power(xgrid + ygrid, 1.00 / (cbox + 2.00))

    def refactor(self):
        """
        Return a refactored version of the profile with equivalent parameterization.

        For most profiles, this returns a copy of the profile with the same
        parameters. Some profiles (e.g., PolyExpoRefact) override this to
        convert to an alternative parameterization.

        Returns
        -------
        Profile
            A new profile instance with refactored parameterization.

        Warns
        -----
        UserWarning
            Warns that this profile has no alternative parameterization.

        Examples
        --------
        >>> from socca.models import Beta
        >>> beta = Beta(xc=180.5, yc=45.2)
        >>> beta_refactored = beta.refactor()
        """
        warnings.warn("Nothing to refactor here.")
        return self.__class__(**self.__dict__)


# Custom profile
# --------------------------------------------------------
class CustomProfile(Profile):
    """
    User-defined custom surface brightness profile.

    Allows users to define arbitrary profile functions with custom parameters,
    enabling modeling of non-standard surface brightness distributions.

    Parameters
    ----------
    parameters : list of dict
        List of parameter specifications. Each dict should contain:
        - 'name': str, parameter name
        - 'unit': str, optional, physical unit (default: 'not specified')
        - 'description': str, optional, parameter description
    profile : callable
        Function defining the profile. Should have signature profile(r, **params)
        where r is the elliptical radius and **params are the custom parameters.
    **kwargs : dict
        Standard profile parameters (xc, yc, theta, e, cbox, positive).

    Notes
    -----
    - The profile function is automatically JIT-compiled for performance.
    - All parameters in the parameters list are initialized to None and must
      be set before use.
    - The profile function should be compatible with JAX (use jax.numpy operations).

    Examples
    --------
    >>> from socca.models import CustomProfile
    >>> import jax.numpy as jp
    >>> # Define a custom Gaussian profile
    >>> def gaussian_profile(r, amplitude, sigma):
    ...     return amplitude * jp.exp(-0.5 * (r / sigma)**2)
    >>> params = [
    ...     {'name': 'amplitude', 'unit': 'image', 'description': 'Peak value'},
    ...     {'name': 'sigma', 'unit': 'deg', 'description': 'Gaussian width'}
    ... ]
    >>> profile = CustomProfile(params, gaussian_profile, xc=180.5, yc=45.2)
    >>> profile.amplitude = 100.0
    >>> profile.sigma = 0.01
    """

    def __init__(self, parameters, profile, **kwargs):
        """
        Initialize a custom profile with user-defined parameters and function.

        Parameters
        ----------
        parameters : list of dict
            Parameter specifications with 'name', 'unit', and 'description'.
        profile : callable
            Profile function with signature profile(r, **params).
        **kwargs : dict
            Standard profile parameters (xc, yc, theta, e, cbox).
        """
        super().__init__(**kwargs)
        self.okeys.append("profile")

        for p in parameters:
            setattr(self, p["name"], None)
            self.units.update({p["name"]: p.get("unit", "not specified")})
            self.description.update(
                {p["name"]: p.get("description", "No description provided")}
            )

        self.profile = jax.jit(profile)


# Beta profile
# --------------------------------------------------------
class Beta(Profile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rc = kwargs.get("rc", config.Beta.rc)
        self.Ic = kwargs.get("Ic", config.Beta.Ic)
        self.beta = kwargs.get("beta", config.Beta.beta)

        self.units.update(dict(rc="deg", beta="", Ic="image"))

        self.description.update(
            dict(
                rc="Core radius",
                Ic="Central surface brightness",
                beta="Slope parameter",
            )
        )

    @staticmethod
    @jax.jit
    def profile(r, Ic, rc, beta):
        """
        Beta profile surface brightness distribution.

        The Beta profile describes a power-law density distribution commonly
        used for modeling galaxy clusters and elliptical galaxies.

        Parameters
        ----------
        r : ndarray
            Elliptical radius in degrees.
        Ic : float
            Central surface brightness (same units as image).
        rc : float
            Core radius in degrees.
        beta : float
            Slope parameter (typically 0.4-1.0 for galaxy clusters).

        Returns
        -------
        ndarray
            Surface brightness at radius r.

        Notes
        -----
        The Beta profile is defined as:
        I(r) = Ic * [1 + (r/rc)^2]^(-beta)

        This profile is widely used in X-ray astronomy for modeling hot gas
        in galaxy clusters. The parameter beta controls the slope of the
        outer profile.

        References
        ----------
        Cavaliere, A., & Fusco-Femiano, R. 1976, A&A, 49, 137

        Examples
        --------
        >>> import jax.numpy as jp
        >>> from socca.models import Beta
        >>> r = jp.linspace(0, 0.1, 100)
        >>> I = Beta.profile(r, Ic=100.0, rc=0.01, beta=0.5)
        """
        return Ic * jp.power(1.00 + (r / rc) ** 2, -beta)


# gNFW profile
# --------------------------------------------------------
class gNFW(Profile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rc = kwargs.get("rc", config.gNFW.rc)
        self.Ic = kwargs.get("Ic", config.gNFW.Ic)
        self.alpha = kwargs.get("alpha", config.gNFW.alpha)
        self.beta = kwargs.get("beta", config.gNFW.beta)
        self.gamma = kwargs.get("gamma", config.gNFW.gamma)

        self.rz = kwargs.get("rz", jp.logspace(-7, 2, 1000))
        self.eps = kwargs.get("eps", 1.00e-08)

        self.okeys.append("rz")
        self.okeys.append("eps")
        self.okeys.append("profile")

        self.units.update(
            dict(rc="deg", alpha="", beta="", gamma="", Ic="image")
        )

        self.description.update(
            dict(
                rc="Scale radius",
                Ic="Characteristic surface brightness",
                alpha="Intermediate slope",
                beta="Outer slope",
                gamma="Inner slope",
            )
        )

        def _profile(r, Ic, rc, alpha, beta, gamma):
            return gNFW._profile(
                r, Ic, rc, alpha, beta, gamma, self.rz, self.eps
            )

        self.profile = jax.jit(_profile)

    @staticmethod
    def _profile(r, Ic, rc, alpha, beta, gamma, rz, eps=1.00e-08):
        """
        Generalized Navarro-Frenk-White (gNFW) profile via Abel deprojection.

        Computes the projected surface brightness profile by Abel transformation
        of a 3D density distribution. Used internally by the gNFW class.

        Parameters
        ----------
        r : ndarray
            Projected elliptical radius in degrees.
        Ic : float
            Characteristic surface brightness (same units as image).
        rc : float
            Scale radius in degrees.
        alpha : float
            Intermediate slope parameter (sharpness of transition).
        beta : float
            Outer slope parameter.
        gamma : float
            Inner slope parameter (central cusp).
        rz : ndarray
            Radial points for numerical integration (in units of rc).
        eps : float, optional
            Absolute and relative error tolerance for integration. Default is 1e-8.

        Returns
        -------
        ndarray
            Projected surface brightness at radius r.

        Notes
        -----
        The 3D density profile is:
        rho(r) = rho0 / [(r/rc)^gamma * (1 + (r/rc)^alpha)^((beta-gamma)/alpha)]

        The surface brightness is obtained by Abel projection (line-of-sight
        integration). This is computed numerically using adaptive quadrature.

        The gNFW profile generalizes several important profiles:
        - NFW (alpha=1, beta=3, gamma=1)
        - Hernquist (alpha=1, beta=4, gamma=1)
        - Einasto-like with varying slopes

        This is a computationally expensive profile due to the numerical
        integration required for each evaluation.

        References
        ----------
        Navarro, J. F., Frenk, C. S., & White, S. D. M. 1996, ApJ, 462, 563
        Zhao, H. 1996, MNRAS, 278, 488
        """

        def radial(u, alpha, beta, gamma):
            factor = 1.00 + u**alpha
            factor = factor ** ((gamma - beta) / alpha)
            return factor / u**gamma

        def integrand(u, uz):
            factor = radial(u, alpha, beta, gamma)
            return 2.00 * factor * u / jp.sqrt(u**2 - uz**2)

        def integrate(rzj):
            return quadgk(
                integrand, [rzj, jp.inf], args=(rzj,), epsabs=eps, epsrel=eps
            )[0]

        mz = Ic * jax.vmap(integrate)(rz)
        return jp.interp(r / rc, rz, mz)


# Sersic profile
# --------------------------------------------------------
n_ = np.linspace(0.25, 10.00, 1000)
b_ = gammaincinv(2.00 * n_, 0.5)

n_ = jp.array(n_)
b_ = jp.array(b_)


class Sersic(Profile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.re = kwargs.get("re", config.Sersic.re)
        self.Ie = kwargs.get("Ie", config.Sersic.Ie)
        self.ns = kwargs.get("ns", config.Sersic.ns)

        self.units.update(dict(re="deg", Ie="image", ns=""))

        self.description.update(
            dict(
                re="Effective radius",
                Ie="Surface brightness at re",
                ns="Sersic index",
            )
        )

    @staticmethod
    @jax.jit
    def profile(r, Ie, re, ns):
        """
        Sersic profile surface brightness distribution.

        The Sersic profile is a generalization of de Vaucouleurs' law and
        describes the light distribution in elliptical galaxies and bulges.

        Parameters
        ----------
        r : ndarray
            Elliptical radius in degrees.
        Ie : float
            Surface brightness at the effective radius (same units as image).
        re : float
            Effective (half-light) radius in degrees.
        ns : float
            Sersic index (concentration parameter). Typical values:
            - ns = 0.5-1: disk-like profiles
            - ns = 4: de Vaucouleurs profile (elliptical galaxies)
            - ns > 4: highly concentrated profiles

        Returns
        -------
        ndarray
            Surface brightness at radius r.

        Notes
        -----
        The Sersic profile is defined as:
        I(r) = Ie * exp(-bn * [(r/re)^(1/ns) - 1])

        where bn is chosen such that re encloses half the total light.
        The parameter bn is approximated numerically and interpolated.

        Common special cases:
        - ns = 1: Exponential profile
        - ns = 4: de Vaucouleurs profile (elliptical galaxies)

        The valid range for ns is approximately 0.25 to 10.

        References
        ----------
        Sersic, J. L. 1968, Atlas de Galaxias Australes
        Ciotti, L., & Bertin, G. 1999, A&A, 352, 447

        Examples
        --------
        >>> import jax.numpy as jp
        >>> from socca.models import Sersic
        >>> r = jp.linspace(0, 0.1, 100)
        >>> # de Vaucouleurs profile for elliptical galaxy
        >>> I = Sersic.profile(r, Ie=50.0, re=0.02, ns=4.0)
        """
        bn = jp.interp(ns, n_, b_)
        se = jp.power(r / re, 1.00 / ns) - 1.00
        return Ie * jp.exp(-bn * se)


# Exponential profile
# --------------------------------------------------------
class Exponential(Profile):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rs = kwargs.get("rs", config.Exponential.rs)
        self.Is = kwargs.get("Is", config.Exponential.Is)

        self.units.update(dict(rs="deg", Is="image"))
        self.description.update(
            dict(rs="Scale radius", Is="Central surface brightness")
        )

    @staticmethod
    @jax.jit
    def profile(r, Is, rs):
        """
        Exponential disk profile surface brightness distribution.

        The exponential profile (Sersic index n=1) describes the light
        distribution in disk galaxies and spiral arms.

        Parameters
        ----------
        r : ndarray
            Elliptical radius in degrees.
        Is : float
            Central surface brightness (same units as image).
        rs : float
            Scale radius (scale length) in degrees.

        Returns
        -------
        ndarray
            Surface brightness at radius r.

        Notes
        -----
        The exponential profile is defined as:
        I(r) = Is * exp(-r / rs)

        This is a special case of the Sersic profile with n=1 and is the
        canonical profile for modeling disk galaxies. The scale radius rs
        contains about 63% of the total light within that radius.

        The exponential profile has no finite effective radius; the half-light
        radius is approximately 1.678 * rs.

        References
        ----------
        Freeman, K. C. 1970, ApJ, 160, 811
        van der Kruit, P. C., & Searle, L. 1981, A&A, 95, 105

        Examples
        --------
        >>> import jax.numpy as jp
        >>> from socca.models import Exponential
        >>> r = jp.linspace(0, 0.1, 100)
        >>> I = Exponential.profile(r, Is=100.0, rs=0.02)
        """
        return Is * jp.exp(-r / rs)


# PolyExponential
# Mancera Piña et al., A&A, 689, A344 (2024)
# --------------------------------------------------------
class PolyExponential(Exponential):
    def __init__(self, **kwargs):
        nk = 4
        super().__init__(**kwargs)
        for k in range(nk):
            setattr(
                self,
                f"c{k + 1}",
                kwargs.get(f"c{k + 1}", config.PolyExponential.ck),
            )
        self.rc = kwargs.get("rc", config.PolyExponential.rc)

        self.units.update(dict(rc="deg"))
        self.units.update({f"c{k + 1}": "" for k in range(nk)})

        self.description.update(
            dict(
                c1="Polynomial coefficient 1",
                c2="Polynomial coefficient 2",
                c3="Polynomial coefficient 3",
                c4="Polynomial coefficient 4",
                rc="Reference radius for polynomial terms",
            )
        )

    @staticmethod
    @jax.jit
    def profile(r, Is, rs, c1, c2, c3, c4, rc):
        """
        Polynomial-exponential profile surface brightness distribution.

        An exponential profile modulated by a 4th-order polynomial, providing
        flexibility to model deviations from pure exponential profiles in
        disk galaxies.

        Parameters
        ----------
        r : ndarray
            Elliptical radius in degrees.
        Is : float
            Central surface brightness (same units as image).
        rs : float
            Exponential scale radius in degrees.
        c1, c2, c3, c4 : float
            Polynomial coefficients for 1st through 4th order terms.
        rc : float
            Reference radius for polynomial terms in degrees.

        Returns
        -------
        ndarray
            Surface brightness at radius r.

        Notes
        -----
        The profile is defined as:

        I(r) = Is * exp(-r/rs) * [1 + c1*(r/rc) + c2*(r/rc)^2
                                    + c3*(r/rc)^3 + c4*(r/rc)^4]

        This profile allows modeling of:

        - Truncations or drops in outer regions
        - Central enhancements or deficits
        - Breaks in disk profiles
        - Type I, II, and III disk breaks

        The polynomial modulation is normalized to the reference radius rc,
        which should typically be comparable to the scale radius rs.

        References
        ----------
        Mancera Piña et al., A&A, 689, A344 (2024)

        Examples
        --------
        >>> import jax.numpy as jp
        >>> from socca.models import PolyExponential
        >>> r = jp.linspace(0, 0.1, 100)
        >>> # Profile with slight central enhancement
        >>> I = PolyExponential.profile(r, Is=100.0, rs=0.02, c1=0.1,
        ...                             c2=-0.05, c3=0.0, c4=0.0, rc=0.02)
        """
        factor = (
            1.00
            + c1 * (r / rc)
            + c2 * ((r / rc) ** 2)
            + c3 * ((r / rc) ** 3)
            + c4 * ((r / rc) ** 4)
        )
        return factor * Is * jp.exp(-r / rs)


# PolyExponential
# Mancera Piña et al., A&A, 689, A344 (2024)
# --------------------------------------------------------
class PolyExpoRefact(Exponential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.I1 = kwargs.get("I1", config.PolyExpoRefact.I1)
        self.I2 = kwargs.get("I2", config.PolyExpoRefact.I2)
        self.I3 = kwargs.get("I3", config.PolyExpoRefact.I3)
        self.I4 = kwargs.get("I4", config.PolyExpoRefact.I4)
        self.rc = kwargs.get("rc", config.PolyExpoRefact.rc)

        self.units.update(dict(rc="deg"))
        self.units.update({f"I{ci}": "image" for ci in range(1, 5)})

        self.description.update(
            dict(
                I1="Polynomial intensity coefficient 1",
                I2="Polynomial intensity coefficient 2",
                I3="Polynomial intensity coefficient 3",
                I4="Polynomial intensity coefficient 4",
                rc="Reference radius for polynomial terms",
            )
        )

    @staticmethod
    @jax.jit
    def profile(r, Is, rs, I1, I2, I3, I4, rc):
        """
        Refactored polynomial-exponential profile using intensity coefficients.

        Alternative parameterization of the polynomial-exponential profile using
        intensity coefficients rather than dimensionless polynomial coefficients.

        Parameters
        ----------
        r : ndarray
            Elliptical radius in degrees.
        Is : float
            Central surface brightness (same units as image).
        rs : float
            Exponential scale radius in degrees.
        I1, I2, I3, I4 : float
            Intensity coefficients for polynomial terms (same units as Is).
        rc : float
            Reference radius for polynomial terms in degrees.

        Returns
        -------
        ndarray
            Surface brightness at radius r.

        Notes
        -----
        The profile is defined as:

        I(r) = exp(-r/rs) * [Is + I1*(r/rc) + I2*(r/rc)^2
                                + I3*(r/rc)^3 + I4*(r/rc)^4]

        This is mathematically equivalent to PolyExponential but uses intensity
        coefficients Ii instead of dimensionless coefficients ci. The relation is:

        Ii = ci * Is

        This parameterization may be more intuitive when the polynomial terms
        represent physical contributions with intensity units.

        References
        ----------
        Mancera Piña et al., A&A, 689, A344 (2024)

        See Also
        --------
        PolyExponential : Alternative parameterization with dimensionless coefficients.
        """
        factor = Is * jp.exp(-r / rs)
        for ci in range(1, 5):
            factor += eval(f"I{ci}") * jp.exp(-r / rs) * ((r / rc) ** ci)
        return factor

    def refactor(self):
        """
        Convert to equivalent PolyExponential parameterization.

        Transforms the intensity-based parameterization (I1, I2, I3, I4) to the
        dimensionless coefficient parameterization (c1, c2, c3, c4) of PolyExponential.

        Returns
        -------
        PolyExponential
            Equivalent profile with dimensionless coefficients ci = Ii / Is.

        Notes
        -----
        This conversion preserves the exact same surface brightness profile
        but expresses it using normalized polynomial coefficients.

        Examples
        --------
        >>> from socca.models import PolyExpoRefact
        >>> prof = PolyExpoRefact(Is=100, I1=10, I2=5, I3=0, I4=0)
        >>> prof_equiv = prof.refactor()
        >>> # prof_equiv is PolyExponential with c1=0.1, c2=0.05, c3=0, c4=0
        """
        kwargs = {
            key: getattr(self, key)
            for key in ["xc", "yc", "theta", "e", "Is", "rs", "rc"]
        }
        for ci in range(1, 5):
            kwargs.update({f"c{ci}": eval(f"I{ci}") / kwargs["Is"]})

        return PolyExponential(**kwargs)


# Modified Exponential
# Mancera Piña et al., A&A, 689, A344 (2024)
# --------------------------------------------------------
class ModExponential(Exponential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rm = kwargs.get("rm", config.ModExponential.rm)
        self.alpha = kwargs.get("alpha", config.ModExponential.alpha)

        self.units.update(dict(rm="deg", alpha=""))

        self.description.update(
            dict(rm="Modification radius", alpha="Modification exponent")
        )

    @staticmethod
    @jax.jit
    def profile(r, Is, rs, rm, alpha):
        """
        Modified exponential profile with power-law modulation.

        An exponential profile modulated by a power law, providing an additional
        degree of freedom to model disk profiles with deviations from pure
        exponential behavior.

        Parameters
        ----------
        r : ndarray
            Elliptical radius in degrees.
        Is : float
            Central surface brightness (same units as image).
        rs : float
            Exponential scale radius in degrees.
        rm : float
            Modification radius where power law becomes important (deg).
        alpha : float
            Power-law exponent (positive for enhancement, negative for suppression).

        Returns
        -------
        ndarray
            Surface brightness at radius r.

        Notes
        -----
        The profile is defined as:
        I(r) = Is * exp(-r/rs) * (1 + r/rm)^alpha

        This profile can model:
        - Truncations (alpha < 0)
        - Enhancements in outer regions (alpha > 0)
        - Smooth transitions between different slopes

        The modification becomes significant at r ~ rm. For r << rm, the
        profile is approximately exponential. For r >> rm, the behavior
        depends on the sign of alpha.

        References
        ----------
        Mancera Piña et al., A&A, 689, A344 (2024)

        Examples
        --------
        >>> import jax.numpy as jp
        >>> from socca.models import ModExponential
        >>> r = jp.linspace(0, 0.1, 100)
        >>> # Profile with outer truncation
        >>> I = ModExponential.profile(r, Is=100.0, rs=0.02, rm=0.05, alpha=-2.0)
        """
        return Is * jp.exp(-r / rs) * (1.00 + r / rm) ** alpha


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
    xc, yc : float
        Source position in celestial coordinates (deg).
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

        uphase, vphase = img.fft.shift(self.xc, self.yc)
        mgrid = self.Ic * img.fft.pulse * jp.exp(-(uphase + vphase))

        if convolve:
            if img.psf is None:
                warnings.warn(
                    "No PSF defined, so no convolution will be performed."
                )
            else:
                mgrid = mgrid * img.psf_fft
        mgrid = jp.fft.ifftshift(jp.fft.irfft2(mgrid, s=img.data.shape)).real
        return mgrid


# Bakcground
# ---------------------------------------------------
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

    B(x,y) = a0 + a1x*x' + a1y*y'
             + a2xx*x'^2 + a2xy*x'*y' + a2yy*y'^2
             + a3xxx*x'^3 + a3xxy*x'^2*y' + a3xyy*x'*y'^2 + a3yyy*y'^3

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
                for key in self.__dict__.keys()
                if key not in self.okeys and key[0] == "a"
            }
        )

        self.description.update(
            dict(
                rs="Reference radius for polynomial terms",
                a0="Polynomial coefficient 0",
                a1x="Polynomial coefficient 1 in x",
                a1y="Polynomial coefficient 1 in y",
                a2xx="Polynomial coefficient 2 in x*x",
                a2xy="Polynomial coefficient 2 in x*y",
                a2yy="Polynomial coefficient 2 in y*y",
                a3xxx="Polynomial coefficient 3 in x*x*x",
                a3xxy="Polynomial coefficient 3 in x*x*y",
                a3xyy="Polynomial coefficient 3 in x*y*y",
                a3yyy="Polynomial coefficient 3 in y*y*y",
            )
        )

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

        B = a0 + a1x*x' + a1y*y' + a2xx*x'^2 + a2xy*x'*y' + a2yy*y'^2
            + a3xxx*x'^3 + a3xxy*x'^2*y' + a3xyy*x'*y'^2 + a3yyy*y'^3

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

        xc, yc = self.img.wcs.crval
        xgrid, ygrid = self.getgrid(img.grid, xc, yc)
        klist = [
            key
            for key in self.__dict__.keys()
            if key not in self.okeys and key[0] == "a"
        ] + ["rc"]
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

        return self.profile(xgrid, ygrid, **kwarg)


# Vertical profile
# --------------------------------------------------------
class Height(Component):
    """
    Base class for vertical density profiles in 3D disk models.

    Defines the vertical (z-direction) structure of a disk galaxy. Used in
    combination with radial profiles to create 3D disk models via the Disk class.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments including:

        inc : float, optional
            Inclination angle in radians (0 = face-on, pi/2 = edge-on).
        losdepth : float, optional
            Half-extent of line-of-sight integration in degrees.
        losbins : int, optional
            Number of integration points along the line of sight.
        positive : bool, optional
            Whether to enforce positivity constraint.

    Attributes
    ----------
    inc : float
        Inclination angle (rad).
    losdepth : float
        Half line-of-sight depth for numerical integration (deg).
    losbins : int
        Number of points for line-of-sight integration (hyperparameter).

    Notes
    -----
    - Subclasses must implement the abstract profile(z) method
    - The profile should return the density as a function of height z
    - losdepth and losbins control the numerical integration accuracy
    - Larger losdepth needed for extended vertical distributions
    - More losbins increase accuracy but decrease speed
    """

    def __init__(self, **kwargs):
        """
        Initialize a vertical profile component.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments including inc, losdepth, losbins, positive.
        """
        super().__init__(**kwargs)

        self.inc = kwargs.get("inc", config.Height.inc)

        self.losdepth = kwargs.get("losdepth", config.Height.losdepth)
        self.losbins = kwargs.get("losbins", config.Height.losbins)
        self.units = dict(losdepth="deg", losbins="", inc="rad")

        self.hyper = ["losdepth", "losbins"]

        self.description = dict(
            losdepth="Half line-of-sigt extent for integration",
            losbins="Number of points for line-of-sight integration",
            inc="Inclination angle (0=face-on)",
        )

    @abstractmethod
    def profile(z):
        """
        Evaluate vertical density profile at height z.

        Parameters
        ----------
        z : ndarray
            Height above/below disk midplane in degrees.

        Returns
        -------
        ndarray
            Density at height z (normalized).

        Notes
        -----
        This is an abstract method that must be implemented by subclasses.
        """
        pass


# Hyperbolic cosine
# --------------------------------------------------------
class HyperSecantHeight(Height):
    """
    Hyperbolic secant vertical density profile.

    Models the vertical structure of disk galaxies using a hyperbolic secant
    (sech) function raised to a power. Commonly used for thick disks.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments including:

        zs : float, optional
            Scale height in degrees.
        alpha : float, optional
            Exponent applied to sech function (typically 1 or 2).
        inc : float, optional
            Inclination angle (rad).
        losdepth, losbins : float, int, optional
            Integration parameters.

    Attributes
    ----------
    zs : float
        Scale height (deg).
    alpha : float
        Exponent parameter.

    Notes
    -----
    The profile is defined as:

    rho(z) = sech(\|z\|/zs)^alpha

    Common cases:
    - alpha = 1: Simple sech profile
    - alpha = 2: sech^2 profile (isothermal disk)

    The sech profile is smoother than exponential near the midplane and
    has more extended wings, making it suitable for modeling thick disks
    and stellar halos.
    """

    def __init__(self, **kwargs):
        """
        Initialize a hyperbolic secant height profile.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments including zs, alpha, inc, losdepth, losbins.
        """
        super().__init__(**kwargs)
        self.zs = kwargs.get("zs", config.HyperSecantHeight.zs)
        self.alpha = kwargs.get("alpha", config.HyperSecantHeight.alpha)

        self.units.update(dict(zs="deg", alpha=""))
        self.description.update(
            dict(zs="Scale height", alpha="Exponent to the hyperbolic secant")
        )

    @staticmethod
    @jax.jit
    def profile(z, zs, alpha):
        """
        Evaluate hyperbolic secant profile at height z.

        Parameters
        ----------
        z : ndarray
            Height above/below disk midplane (deg).
        zs : float
            Scale height (deg).
        alpha : float
            Exponent parameter.

        Returns
        -------
        ndarray
            Density at height z: sech(\|z\|/zs)^alpha.

        Notes
        -----
        Uses absolute value of z to ensure symmetry about the midplane.
        """
        factor = jp.cosh(jp.abs(z) / zs)
        return 1.00 / factor**alpha


# Exponential height
# --------------------------------------------------------
class ExponentialHeight(Height):
    """
    Exponential vertical density profile.

    Models the vertical structure of thin disk galaxies using an exponential
    function. This is the simplest and most commonly used vertical profile.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments including:

        zs : float, optional
            Scale height in degrees.
        inc : float, optional
            Inclination angle (rad).
        losdepth, losbins : float, int, optional
            Integration parameters.

    Attributes
    ----------
    zs : float
        Scale height (deg).

    Notes
    -----
    The profile is defined as:

    rho(z) = exp(-\|z\|/zs)

    This simple exponential profile is appropriate for thin stellar disks
    and is the vertical analog of the exponential radial profile. The scale
    height zs is typically much smaller than the radial scale length.

    The exponential profile has a sharp peak at the midplane and falls off
    more rapidly than sech profiles, making it suitable for thin disks.
    """

    def __init__(self, **kwargs):
        """
        Initialize an exponential height profile.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments including zs, inc, losdepth, losbins.
        """
        super().__init__(**kwargs)
        self.zs = kwargs.get("zs", config.ExponentialHeight.zs)
        self.units.update(dict(zs="deg"))
        self.description.update(dict(zs="Scale height"))

    @staticmethod
    @jax.jit
    def profile(z, zs):
        """
        Evaluate exponential profile at height z.

        Parameters
        ----------
        z : ndarray
            Height above/below disk midplane (deg).
        zs : float
            Scale height (deg).

        Returns
        -------
        ndarray
            Density at height z: exp(-\|z\|/zs).

        Notes
        -----
        Uses absolute value of z to ensure symmetry about the midplane.
        """
        return jp.exp(-jp.abs(z) / zs)


# Disk model with finite thickness
# --------------------------------------------------------
class Disk(Component):
    """
    3D disk model combining radial and vertical profiles with inclination.

    Creates a realistic disk galaxy model by combining a radial surface brightness
    profile with a vertical density distribution. The model accounts for disk
    inclination via line-of-sight integration.

    Parameters
    ----------
    radial : Profile, optional
        Radial profile defining in-plane surface brightness distribution.
        Default is Sersic(). Can be any 2D profile (Beta, Exponential, etc.).
    vertical : Height, optional
        Vertical profile defining scale height and vertical structure.
        Default is HyperSecantHeight(). Can be HyperSecantHeight or ExponentialHeight.
    **kwargs : dict
        Additional keyword arguments passed to Component.

    Attributes
    ----------
    radial : Profile
        Radial profile component.
    vertical : Height
        Vertical profile component.
    profile : callable
        Combined 3D density profile rho(r, z) = radial(r) * vertical(z).

    Notes
    -----
    The 3D disk model:
    1. Defines density as rho(r,z) = radial_profile(r) * vertical_profile(z)
    2. Integrates along the line of sight to obtain projected surface brightness
    3. Accounts for inclination by rotating the disk before integration
    4. Uses numerical integration (trapezoidal rule) along z-direction

    The inclination angle is stored in vertical.inc:
    - inc = 0: face-on (looking down on the disk)
    - inc = pi/2: edge-on (viewing disk from the side)

    Line-of-sight integration parameters (vertical.losdepth, vertical.losbins)
    control accuracy and computation time. Larger losdepth and more losbins
    improve accuracy for highly inclined disks.

    Examples
    --------
    >>> from socca.models import Disk, Exponential, ExponentialHeight
    >>> # Create edge-on exponential disk
    >>> radial = Exponential(xc=180.5, yc=45.2, rs=0.02, Is=100)
    >>> vertical = ExponentialHeight(zs=0.002, inc=np.pi/2)
    >>> disk = Disk(radial=radial, vertical=vertical)
    """

    def __init__(
        self, radial=Sersic(), vertical=HyperSecantHeight(), **kwargs
    ):
        """
        Initialize a 3D disk model from radial and vertical profiles.

        Parameters
        ----------
        radial : Profile, optional
            Radial profile component. Default is Sersic().
        vertical : Height, optional
            Vertical profile component. Default is HyperSecantHeight().
        **kwargs : dict
            Additional keyword arguments.

        Notes
        -----
        The component ID is synchronized between the Disk, radial, and vertical
        components to ensure consistent parameter naming in composite models.
        """
        super().__init__(**kwargs)
        for key in ["radial", "vertical", "profile"]:
            self.okeys.append(key)

        self.radial = radial
        self.vertical = vertical

        if self.radial.id != self.id:
            type(self).idcls -= 1
            idmin = np.minimum(
                int(self.radial.id.replace("comp_", "")),
                int(self.id.replace("comp_", "")),
            )
            self.id = f"comp_{idmin:02d}"
            self.radial.id = self.id

        rkw = [
            f"r_{key}"
            for key in list(
                inspect.signature(self.radial.profile).parameters.keys()
            )
            if key not in ["r", "z"]
        ]
        zkw = [
            f"z_{key}"
            for key in list(
                inspect.signature(self.vertical.profile).parameters.keys()
            )
            if key not in ["r", "z"]
        ]

        profile_ = [
            "rfoo.profile(r,{0})".format(",".join(rkw)),
            "zfoo.profile(z,{0})".format(",".join(zkw)),
        ]
        profile_ = "*".join(profile_)
        profile_ = "lambda rfoo,zfoo,r,z,{0},{1}: {2}".format(
            ",".join(rkw), ",".join(zkw), profile_
        )

        self.profile = jax.jit(
            partial(eval(profile_), self.radial, self.vertical)
        )

        for key in self.radial.hyper:
            self.hyper.append(f"radial.{key}")

        for key in self.vertical.hyper:
            self.hyper.append(f"vertical.{key}")

        self.units.update(
            {
                f"radial.{key}": self.radial.units[key]
                for key in self.radial.units.keys()
            }
        )
        self.units.update(
            {
                f"vertical.{key}": self.vertical.units[key]
                for key in self.vertical.units.keys()
            }
        )

        self.description.update(
            {
                f"radial.{key}": self.radial.description[key]
                for key in self.radial.description.keys()
            }
        )
        self.description.update(
            {
                f"vertical.{key}": self.vertical.description[key]
                for key in self.vertical.description.keys()
            }
        )

    def getmap(self, img, convolve=False):
        """
        Generate disk image via 3D line-of-sight integration.

        Computes the projected surface brightness by integrating the 3D density
        distribution along the line of sight, accounting for inclination.

        Parameters
        ----------
        img : Image
            Image object containing grid, PSF, and WCS information.
        convolve : bool, optional
            If True, convolve the model with the PSF. Default is False.

        Returns
        -------
        ndarray
            Projected disk image on the image grid.

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
        The algorithm:
        1. Creates 3D coordinate grids (r, z) accounting for inclination
        2. Evaluates 3D density rho(r, z) = radial(r) * vertical(z)
        3. Integrates along line of sight using trapezoidal rule
        4. Averages over sub-pixel sampling
        5. Optionally convolves with PSF

        Integration accuracy controlled by:
        - vertical.losdepth: extent of integration
        - vertical.losbins: number of integration points

        For edge-on disks, use larger losdepth (≥ 5*zs) and more losbins (≥ 200).

        Examples
        --------
        >>> from socca.models import Disk, Exponential, ExponentialHeight
        >>> import numpy as np
        >>> from socca.data import Image
        >>> radial = Exponential(xc=180.5, yc=45.2, rs=0.02, Is=100)
        >>> vertical = ExponentialHeight(zs=0.002, inc=np.pi/4)  # 45 deg
        >>> disk = Disk(radial=radial, vertical=vertical)
        >>> img = Image('observation.fits')
        >>> disk_map = disk.getmap(img, convolve=True)
        """
        kwarg = {}
        for key in list(inspect.signature(self.profile).parameters.keys()):
            if key not in ["r", "z"]:
                if key.startswith("r_"):
                    kwarg[key] = getattr(self.radial, key.replace("r_", ""))
                elif key.startswith("z_"):
                    kwarg[key] = getattr(self.vertical, key.replace("z_", ""))

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

        rcube, zcube = self.getgrid(
            img.grid,
            self.radial.xc,
            self.radial.yc,
            self.vertical.losdepth,
            self.vertical.losbins,
            self.radial.theta,
            self.vertical.inc,
        )

        dx = 2.00 * self.vertical.losdepth / (self.vertical.losbins - 1)

        mgrid = self.profile(rcube, zcube, **kwarg)
        mgrid = jp.trapezoid(mgrid, dx=dx, axis=1)
        mgrid = jp.mean(mgrid, axis=0)

        if convolve:
            if img.psf is None:
                warnings.warn(
                    "No PSF defined, so no convolution will be performed."
                )
            else:
                mgrid = (
                    jp.fft.rfft2(jp.fft.fftshift(mgrid), s=img.data.shape)
                    * img.psf_fft
                )
                mgrid = jp.fft.ifftshift(
                    jp.fft.irfft2(mgrid, s=img.data.shape)
                ).real
        return mgrid

    @staticmethod
    @partial(jax.jit, static_argnames=["grid", "losbins"])
    def getgrid(grid, xc, yc, losdepth, losbins=200, theta=0.00, inc=0.00):
        """
        Compute 3D disk coordinates with rotation and inclination.

        Generates 3D coordinate grids (r, z) for disk model evaluation, accounting
        for position angle and inclination transformations needed for line-of-sight
        integration through an inclined disk.

        Parameters
        ----------
        grid : Grid
            Grid object with .x and .y celestial coordinate arrays (deg).
        xc : float
            Right ascension of disk center (deg).
        yc : float
            Declination of disk center (deg).
        losdepth : float
            Half-extent of line-of-sight integration (deg).
        losbins : int, optional
            Number of integration points along line of sight. Default is 200.
        theta : float, optional
            Position angle of disk major axis, east from north (rad). Default is 0.
        inc : float, optional
            Inclination angle (0 = face-on, pi/2 = edge-on) (rad). Default is 0.

        Returns
        -------
        rcube : ndarray
            4D array of radial distances from disk center (deg).
            Shape: (ssize, losbins, ysize, xsize).
        zcube : ndarray
            4D array of heights above/below disk midplane (deg).
            Shape: (ssize, losbins, ysize, xsize).

        Notes
        -----
        The transformation sequence:
        1. Center coordinates on (xc, yc)
        2. Apply spherical geometry correction (cos(dec))
        3. Rotate by position angle theta
        4. Create line-of-sight grid from -losdepth to +losdepth
        5. Apply inclination rotation
        6. Compute cylindrical radius r and height z

        The inclination is measured from face-on:
        - inc = 0: face-on, z-axis points toward observer
        - inc = pi/2: edge-on, disk in plane of sky

        The 4D arrays allow vectorized evaluation of the 3D density function
        across the entire image with sub-pixel sampling and line-of-sight
        integration.

        This function is JIT-compiled with static losbins for performance.
        """
        ssize, ysize, xsize = grid.x.shape

        zt = jp.linspace(-losdepth, losdepth, losbins)

        sint = jp.sin(theta)
        cost = jp.cos(theta)

        xt = (grid.x - xc) * jp.cos(jp.deg2rad(yc))
        yt = grid.y - yc

        xt, yt = -xt * sint - yt * cost, xt * cost - yt * sint

        xt = jp.broadcast_to(
            xt[:, None, :, :], (ssize, losbins, ysize, xsize)
        ).copy()
        yt = jp.broadcast_to(
            yt[:, None, :, :], (ssize, losbins, ysize, xsize)
        ).copy()
        zt = jp.broadcast_to(
            zt[None, :, None, None], (ssize, losbins, ysize, xsize)
        ).copy()

        sini = jp.sin(inc - 0.50 * jp.pi)
        cosi = jp.cos(inc - 0.50 * jp.pi)

        zt, yt = yt * cosi - zt * sini, yt * sini + zt * cosi

        return jp.sqrt(xt**2 + yt**2), zt

    def parameters(self):
        """
        Print formatted table of disk parameters from radial and vertical components.

        Displays parameters from both the radial and vertical profile components,
        organized as 'radial.parameter' and 'vertical.parameter'. Separates regular
        parameters from hyperparameters (integration settings).

        Notes
        -----
        Output format:

        Model parameters
        ================
        radial.xc        [deg]   : value | Right ascension of centroid
        radial.yc        [deg]   : value | Declination of centroid
        radial.rs        [deg]   : value | Scale radius
        vertical.zs      [deg]   : value | Scale height
        vertical.inc     [rad]   : value | Inclination angle (0=face-on)

        Hyperparameters
        ===============
        vertical.losdepth [deg]  : value | Half line-of-sight extent
        vertical.losbins  []     : value | Number of integration points

        Hyperparameters control the numerical integration accuracy but are not
        fitted parameters.

        Examples
        --------
        >>> from socca.models import Disk, Exponential, ExponentialHeight
        >>> import numpy as np
        >>> radial = Exponential(xc=180.5, yc=45.2, rs=0.02, Is=100)
        >>> vertical = ExponentialHeight(zs=0.002, inc=np.pi/4)
        >>> disk = Disk(radial=radial, vertical=vertical)
        >>> disk.parameters()
        Model parameters
        ================
        radial.xc         [deg]    : 1.8050E+02 | Right ascension of centroid
        radial.yc         [deg]    : 4.5200E+01 | Declination of centroid
        ...
        """
        keyout = []
        for key in self.radial.__dict__.keys():
            if (
                key not in self.okeys
                and f"radial.{key}" not in self.hyper
                and key != "okeys"
            ):
                keyout.append(f"radial.{key}")
        for key in self.vertical.__dict__.keys():
            if (
                key not in self.okeys
                and f"vertical.{key}" not in self.hyper
                and key != "okeys"
            ):
                keyout.append(f"vertical.{key}")

        if len(keyout) > 0:
            maxlen = np.max(
                np.array(
                    [
                        len(f"{key} [{self.units[key]}]")
                        for key in keyout + self.hyper
                    ]
                )
            )

            print("\nModel parameters")
            print("=" * 16)
            for key in keyout:
                keylen = maxlen - len(f" [{self.units[key]}]")
                kvalue = (
                    self.radial.__dict__[key.replace("radial.", "")]
                    if "radial" in key
                    else self.vertical.__dict__[key.replace("vertical.", "")]
                )
                kvalue = None if kvalue is None else f"{kvalue:.4E}"
                print(
                    f"{key:<{keylen}} [{self.units[key]}] : "
                    + f"{kvalue}".ljust(10)
                    + f" | {self.description[key]}"
                )

            if len(self.hyper) > 0:
                print("\nHyperparameters")
                print("=" * 15)
                for key in self.hyper:
                    keylen = maxlen - len(f" [{self.units[key]}]")
                    kvalue = (
                        self.radial.__dict__[key.replace("radial.", "")]
                        if "radial" in key
                        else self.vertical.__dict__[
                            key.replace("vertical.", "")
                        ]
                    )
                    kvalue = None if kvalue is None else f"{kvalue:.4E}"
                    print(
                        f"{key:<{keylen}} [{self.units[key]}] : "
                        + f"{kvalue}".ljust(10)
                        + f" | {self.description[key]}"
                    )
        else:
            print("No parameters defined.")

    def parlist(self):
        """
        Return list of parameter names from radial and vertical components.

        Collects all parameters from both the radial and vertical profile
        components, with names prefixed as 'radial.' and 'vertical.'.

        Returns
        -------
        list of str
            Combined list of parameter names from radial and vertical components.
            Names are prefixed with component type (e.g., 'radial.xc', 'vertical.zs').

        Notes
        -----
        This method is used internally by Model.addcomponent() when adding
        a Disk component to a composite model, ensuring all parameters from
        both sub-components are registered.

        Examples
        --------
        >>> from socca.models import Disk, Exponential, ExponentialHeight
        >>> radial = Exponential(xc=180.5, yc=45.2, rs=0.02, Is=100)
        >>> vertical = ExponentialHeight(zs=0.002, inc=0.5)
        >>> disk = Disk(radial=radial, vertical=vertical)
        >>> disk.parlist()
        ['radial.xc', 'radial.yc', 'radial.theta', 'radial.e', 'radial.cbox',
         'radial.rs', 'radial.Is', 'vertical.losdepth', 'vertical.losbins',
         'vertical.inc', 'vertical.zs']
        """
        pars_ = [
            f"radial.{key}"
            for key in self.radial.__dict__.keys()
            if key not in self.okeys and key != "okeys"
        ]
        pars_ += [
            f"vertical.{key}"
            for key in self.vertical.__dict__.keys()
            if key not in self.okeys and key != "okeys"
        ]
        return pars_
