import matplotlib.pyplot as plt
import inspect
import re
import types
import warnings

import jax
import jax.numpy as jp
import numpy as np

import numpyro.distributions

from socca.units import conversion_factor as _cfactor

from ..priors import _BoundTo
from . import config
from .base import Component
from .misc import Point, Background
from .disk import Disk
from .ellipsoid import Ellipsoid
from .bridge import SimpleBridge, MesaBridge
from .bridge.base import Bridge
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

__all__ = [
    "config",
    "Component",
    "Model",
    "zoo",
    # radial profiles
    "Profile",
    "CustomProfile",
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
    # misc
    "Point",
    "Background",
    # disk
    "Disk",
    # bridge
    "SimpleBridge",
    "MesaBridge",
]


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
    - Ellipsoid: 3D triaxial-ellipsoid model evaluated via line-of-sight
      integration (e.g. a galactic bar)

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
    Ellipsoid
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

    print("\nEllipsoid-like model")
    print("====================")
    print("Ellipsoid")

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
        tied : list of bool
            Indicates which parameters are tied to other parameters.
        type : list of str
            Class names of each component in the model.
        units : dict
            Dictionary mapping parameter names to their physical units.
        periodic : dict
            Dictionary mapping parameter names to their period, for
            parameters that are periodic (e.g. an elliptical profile's
            ``theta``). Keys absent from this dict are not periodic.

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
        self.components = []
        self.tied = []
        self.type = []
        self.units = {}
        self.periodic = {}
        self.conversions = {}

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
        if prof.id in self._index:
            warnings.warn(
                f"Component id {prof.id} is already in the model; tie "
                f"arguments naming {prof.id} from other components refer "
                "to the first one.",
                stacklevel=2,
            )

        self.type.append(prof.__class__.__name__)
        self.positive.append(prof.positive if positive is None else positive)
        for p in prof.parlist():
            par = eval(f"prof.{p}")

            if par is None:
                raise ValueError(
                    f"Parameter {p} in component {self.ncomp:02d} is set "
                    f"to None. Please provide a valid value or prior."
                )

            if isinstance(par, (types.LambdaType, types.FunctionType)):
                _glb = {
                    name: par.__globals__[name]
                    for name in par.__code__.co_names
                    if name in par.__globals__
                }
                _glb["jp"] = jp
                _glb.setdefault("__builtins__", __builtins__)
                par = types.FunctionType(
                    par.__code__,
                    _glb,
                    par.__name__,
                    par.__defaults__,
                    par.__closure__,
                )

            self.params.append(f"comp_{self.ncomp:02d}_{p}")
            self.priors.update({f"comp_{self.ncomp:02d}_{p}": par})
            if isinstance(par, numpyro.distributions.Distribution):
                self.paridx.append(len(self.params) - 1)

            if isinstance(
                par, (types.LambdaType, types.FunctionType, _BoundTo)
            ):
                self.tied.append(True)
            else:
                self.tied.append(False)

            key = f"comp_{self.ncomp:02d}_{p}"
            native_unit = prof.units[p]
            input_unit = prof._input_units.get(p, native_unit)
            if input_unit != native_unit:
                self.conversions[key] = _cfactor(input_unit, native_unit)
            self.units[key] = input_unit
            if p in prof.periodic:
                self.periodic[key] = prof.periodic[p]

        self.components.append(prof)
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
                elif isinstance(
                    par, (types.LambdaType, types.FunctionType, _BoundTo)
                ):
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
        for key in self.params:
            if isinstance(
                self.priors[key], numpyro.distributions.Distribution
            ):
                raise ValueError(
                    f"Parameter {key} is a distribution. "
                    f"Use getmodel() with sampled values instead."
                )

        mraw, msmo, mbkg, _ = self.getmodel(img, [], doresp=False, doexp=False)

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
        elif hasattr(component, "id"):
            component = [self._comp_index(component)]
        elif isinstance(component, (list, tuple)):
            component_ = []
            for c in component:
                if isinstance(c, int):
                    component_.append(c)
                elif isinstance(c, str):
                    c_ = c.replace("comp", "").replace("_", "")
                    component_.append(int(c_))
                elif hasattr(c, "id"):
                    component_.append(self._comp_index(c))
            component = component_
            del component_
        return component

    @property
    def _index(self):
        """Model index of each component id (the first one, if repeated)."""
        index = {}
        for nc, comp in enumerate(self.components):
            index.setdefault(comp.id, nc)
        return index

    def _comp_index(self, comp):
        """Model index of a component object."""
        for nc, c in enumerate(self.components):
            if c is comp:
                return nc
        if comp.id in self._index:
            return self._index[comp.id]
        raise ValueError(f"Component {comp.id} is not in the model.")

    def _tiewarn(self, message):
        """Warn once per model, however often ties are resolved."""
        warned = self.__dict__.setdefault("_tiewarned", set())
        if message not in warned:
            warned.add(message)
            warnings.warn(message)

    def _tiekey(self, name, owner=None):
        """
        Model key for a parameter name used in a tie.

        Ties name parameters by component id (``Component.id``), which can
        differ from the model index used in the model keys, e.g. when
        components are added in a different order than they were created.
        The id of `owner`, the model index of the component holding the
        tie, always refers to that component.
        """
        match = re.match(r"(comp_\d+)_(.+)$", name)
        if match is None:
            return name
        index = self._index
        if owner is not None and match.group(1) == self.components[owner].id:
            index = {**index, match.group(1): owner}
        modelkeys = {p.replace(".", "_") for p in self.params}
        cid = match.group(1)
        if cid not in index:
            if name in modelkeys:
                self._tiewarn(
                    f"Tie arguments naming {cid} name no component id in "
                    f"the model, so they are read as model keys ({cid}_*). "
                    "Ties name components by id (see Component.id)."
                )
            return name
        comp = f"comp_{index[match.group(1)]:02d}"
        key = f"{comp}_{match.group(2)}"
        if key != name and name in modelkeys:
            self._tiewarn(
                f"Tie arguments naming {cid} refer to the component with id "
                f"{cid}, which is {comp} in the model, not to the model "
                f"keys {cid}_*. Ties name components by id (see "
                "Component.id)."
            )
        return key

    def _boundkey(self, prior, key=None):
        """Model key of the parameter a boundto value refers to."""
        key = prior.key if key is None else key
        cid = prior._comp.id
        for nc, comp in enumerate(self.components):
            if comp is prior._comp:
                return f"comp_{nc:02d}_{prior._attr}"
        if cid not in self._index:
            raise ValueError(
                f"{key} is bound to {prior.key}, but component "
                f"{cid} is not in the model."
            )
        self._tiewarn(
            f"{key} is bound to a component that is not in the model; "
            f"using comp_{self._index[cid]:02d}, which has the same id {cid}."
        )
        return f"comp_{self._index[cid]:02d}_{prior._attr}"

    def _tieargs(self, key):
        """Argument names of a tie and the model keys they refer to."""
        prior = self.priors[key]
        if isinstance(prior, _BoundTo):
            return {None: self._boundkey(prior, key)}
        owner = int(re.match(r"comp_(\d+)_", key).group(1))
        alias = {p.replace(".", "_"): p for p in self.params}
        args = {}
        for name in inspect.signature(prior).parameters:
            arg = self._tiekey(name, owner)
            if arg not in self.priors and arg not in alias:
                raise KeyError(
                    f"Tie argument {name} of {key} matches no parameter "
                    "of the model. Ties name components by id (see "
                    "Component.id)."
                )
            args[name] = arg if arg in self.priors else alias[arg]
        return args

    def _resolve_ties(self, pars):
        """Evaluate tied parameters in `pars`, after the ones they use."""
        pending = {
            key: self._tieargs(key)
            for ki, key in enumerate(self.params)
            if self.tied[ki]
        }
        while pending:
            ready = [
                key
                for key, args in pending.items()
                if not any(arg in pending for arg in args.values())
            ]
            if not ready:
                raise ValueError(f"Circular ties among {', '.join(pending)}.")
            for key in ready:
                args = pending.pop(key)
                if None in args:
                    pars[key] = self.priors[key](pars[args[None]])
                else:
                    pars[key] = self.priors[key](
                        **{name: pars[arg] for name, arg in args.items()}
                    )

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
        pp : array_like
            Values of the parameters with prior distributions, in the order
            they appear in paridx, in the variable each prior samples: for
            LogPowerLaw priors this is log(x). Fit outputs are in x; use
            `fitter.getmodel` to evaluate them.
        doresp : bool, optional
            If True, multiply by the instrumental response map (img.response).
            Default is False.
        doexp : bool, optional
            If True, multiply by the exposure map (img.exposure). Default is False.
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
                if hasattr(self.priors[key], "to_model"):
                    pars[key] = self.priors[key].to_model(pars[key])

        for key, factor in self.conversions.items():
            if key in pars:
                pars[key] = pars[key] * factor

        self._resolve_ties(pars)

        mbkg = jp.zeros(img.data.shape)
        mraw = jp.zeros(img.data.shape)
        mpts = jp.zeros(img.fft.pulse.shape) + 0j

        mneg = jp.zeros(img.data.shape)

        comp_ = self._comp_filter(component)

        for nc in comp_:
            comp = self.components[nc]
            if isinstance(comp, Bridge):
                kwarg = comp._build_kwargs(
                    pars, f"comp_{nc:02d}", self._tiekey, self._boundkey
                )
            else:
                kwarg = comp._build_kwargs(pars, f"comp_{nc:02d}")
            mone = comp._evaluate(img, **kwarg)

            if self.positive[nc] and self.type[nc] != "Point":
                mneg = jp.where(mone < 0.00, 1.00, mneg)

            if self.type[nc] == "Point":
                if doresp:
                    xpts, ypts = img.fft.world2pix(kwarg["xc"], kwarg["yc"])
                    mone *= jax.scipy.ndimage.map_coordinates(
                        img.response,
                        [jp.array([ypts]), jp.array([xpts])],
                        order=1,
                        mode="nearest",
                    )[0]

                mpts += mone.copy()

            elif self.type[nc] == "Background":
                mbkg += mone.copy()

            else:  # Profiles, Disk, etc.
                mraw += mone.copy()

            del mone

        haspts = any(self.type[nc] == "Point" for nc in comp_)

        msmo = mraw.copy()

        if doresp:
            msmo *= img.response

        if haspts:
            mraw = mraw + img.fft.ifft(mpts)

        if img.psf is not None:
            mfft = img.fft.fft(msmo)
            if haspts:
                mfft = mfft + mpts
            msmo = img.convolve.crop(mfft * img.convolve.psf_fft)
        elif haspts:
            msmo = msmo + img.fft.ifft(mpts)

        if doexp:
            msmo *= img.exposure
            mbkg *= img.exposure

        msmo = msmo + mbkg

        return mraw, msmo, mbkg, mneg


Component.idcls = 0
