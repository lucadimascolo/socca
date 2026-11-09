import matplotlib.pyplot as plt
import inspect
import types
import warnings

import jax
import jax.numpy as jp
import numpy as np

import numpyro.distributions

from socca.units import conversion_factor as _cfactor

from ..priors import _TIED, _Function
from . import config
from .base import Component, _check_name, _format_rows, _format_value
from .misc import Point, Background
from .disk import Disk
from .ellipsoid import Ellipsoid
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
        self._prefix = []
        self._owner = []

        if prof is not None:
            self._addcomponent(prof, positive, stacklevel=3)

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
            value or prior distribution, if the component is already in the
            model, or if its id is already used.

        Notes
        -----
        - Parameters are stored as '<id>_<parameter>' (e.g. 'comp_00_xc'),
          where the id is the component's name, or 'comp_NN' from its
          position in the model. The component's ``id`` is set to it.
        - Parameters can be fixed values, prior distributions, or tied
          (socca.priors.function or boundto).
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
        self._addcomponent(prof, positive, stacklevel=3)

    def _addcomponent(self, prof, positive, stacklevel):
        """Add a component; see addcomponent()."""
        if any(c is prof for c in self.components):
            raise ValueError(
                f"This {type(prof).__name__} is already in the model."
            )
        prefix = prof.name
        if prefix is None:
            n = self.ncomp
            while f"comp_{n:02d}" in self._prefix:
                n += 1
            prefix = f"comp_{n:02d}"
        if prefix in self._prefix:
            raise ValueError(
                f"The id '{prefix}' is already used in this model: give "
                f"this {type(prof).__name__} another name."
            )

        self.type.append(prof.__class__.__name__)
        self.positive.append(prof.positive if positive is None else positive)
        for p in prof.parlist():
            par = eval(f"prof.{p}")

            if par is None:
                raise ValueError(
                    f"Parameter {p} in component {prefix} is set "
                    f"to None. Please provide a valid value or prior."
                )

            if isinstance(par, types.FunctionType):
                warnings.warn(
                    f"Tying {prefix}_{p} with a lambda is deprecated and "
                    "will be removed: use socca.priors.function() instead.",
                    FutureWarning,
                    stacklevel=stacklevel,
                )
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

            self.params.append(f"{prefix}_{p}")
            self._owner.append(self.ncomp)
            self.priors.update({f"{prefix}_{p}": par})
            if isinstance(par, numpyro.distributions.Distribution):
                self.paridx.append(len(self.params) - 1)

            if isinstance(par, _TIED):
                self.tied.append(True)
            else:
                self.tied.append(False)

            key = f"{prefix}_{p}"
            native_unit = prof.units[p]
            input_unit = prof._input_units.get(p, native_unit)
            if input_unit != native_unit:
                self.conversions[key] = _cfactor(input_unit, native_unit)
            self.units[key] = input_unit
            if p in prof.periodic:
                self.periodic[key] = prof.periodic[p]

        self.components.append(prof)
        self._prefix.append(prefix)
        prof.id = prefix
        self.ncomp += 1

    def parameters(self, freeonly=False):
        """
        Print the model parameters, grouped by component.

        Each component is listed under a ``Type (id)`` line. The model key
        of a parameter, used in ``fit.labels``, is the id followed by
        ``_`` and the parameter name (e.g. ``comp_00_xc``,
        ``core_radial.xc``).

        Parameters
        ----------
        freeonly : bool, optional
            If True, only list parameters with prior distributions.
            Default is False.

        Examples
        --------
        >>> model = Model(Beta(xc=priors.uniform(180, 181), yc=45.2))
        >>> model.parameters()
        <BLANKLINE>
        Beta (comp_00)
        ==============
        xc [deg] : Uniform [180, 181] | Right ascension of centroid
        yc [deg] : 4.5200E+01         | Declination of centroid
        ...
        """
        blocks = []
        for nc, prefix in enumerate(self._prefix):
            comp = self.components[nc]
            rows = []
            for key, owner in zip(self.params, self._owner):
                par = self.priors[key]
                if owner != nc or (
                    freeonly
                    and not isinstance(par, numpyro.distributions.Distribution)
                ):
                    continue
                name = key[len(prefix) + 1 :]
                rows.append(
                    (
                        name,
                        self.units[key],
                        _format_value(par, self._source_label),
                        comp.description.get(name, ""),
                    )
                )
            if rows:
                blocks.append((f"{self.type[nc]} ({prefix})", rows))

        if not blocks:
            print("No free parameters." if freeonly else "No parameters.")
            return
        lines = _format_rows([row for _, rows in blocks for row in rows])
        for title, rows in blocks:
            print(f"\n{title}\n" + "=" * len(title))
            print("\n".join(lines[: len(rows)]))
            lines = lines[len(rows) :]

    def __repr__(self):
        """List the model components with their ids."""
        lines = [
            f"{ctype} ({prefix})"
            for ctype, prefix in zip(self.type, self._prefix)
        ]
        return "\n".join(["Model"] + ["  " + line for line in lines])

    def _source_label(self, comp, name):
        """Label of a tie source: the component id and parameter name."""
        try:
            key = self._keyof(comp, name)
        except ValueError:
            return f"{type(comp).__name__}.{name} (not in model)"
        prefix = self._prefix[self._owner[self.params.index(key)]]
        return f"{prefix}.{key[len(prefix) + 1 :]}"

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
            if isinstance(
                self.priors[key], numpyro.distributions.Distribution
            ):
                raise ValueError(
                    f"Parameter {key} is a distribution. "
                    f"Use getmap() with sampled values instead."
                )
            elif not self.tied[ki]:
                pars[key] = self.priors[key]

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

    def _keyof(self, comp, attr):
        """Return the model key of a component object's parameter."""
        for c, prefix in zip(self.components, self._prefix):
            if c is comp:
                return f"{prefix}_{attr}"
            for name, sub in getattr(c, "_namespaces", {}).items():
                if sub is comp:
                    return f"{prefix}_{name}.{attr}"
        raise ValueError(
            f"A tie refers to a {type(comp).__name__} that is not in this "
            "model."
        )

    def _tieargs(self, key):
        """Return the model keys a tied parameter depends on."""
        tie = self.priors[key]
        if isinstance(tie, _Function):
            return [self._keyof(comp, name) for comp, name in tie.sources]
        args = []
        for name in inspect.signature(tie).parameters:
            match = [
                k for k in self.params if name in (k, k.replace(".", "_"))
            ]
            if not match:
                raise KeyError(
                    f"Tie argument '{name}' of {key} matches no parameter "
                    "of this model."
                )
            args.append(match[0])
        return args

    def _resolve_ties(self, pars):
        """Evaluate tied parameters in dependency order, in place."""
        deps = {
            key: self._tieargs(key)
            for key, tied in zip(self.params, self.tied)
            if tied
        }
        done = set()

        def resolve(key, chain):
            if key in done:
                return
            if key in chain:
                raise ValueError(
                    "Circular tie: " + " -> ".join(chain + [key]) + "."
                )
            for dep in deps[key]:
                if dep in deps:
                    resolve(dep, chain + [key])
            pars[key] = self.priors[key](*[pars[dep] for dep in deps[key]])
            done.add(key)

        for key in deps:
            resolve(key, [])

    def _comp_filter(self, component=None):
        """
        Parse component selections into a list of model positions.

        Parameters
        ----------
        component : None, int, str, Component, or list of these, optional
            None selects every component. An int is a position in the
            model, a str an id (e.g. 'comp_01' or a name), and a
            component object is found by identity.

        Returns
        -------
        list of int
            Positions of the selected components.

        Raises
        ------
        ValueError
            If a selection matches no component of the model.

        Examples
        --------
        >>> model._comp_filter(None)
        [0, 1, 2]
        >>> model._comp_filter([0, "comp_02"])
        [0, 2]
        """
        if component is None:
            return list(range(self.ncomp))
        if not isinstance(component, (list, tuple)):
            component = [component]
        positions = []
        for c in component:
            if isinstance(c, (int, np.integer)) and not isinstance(c, bool):
                if not 0 <= c < self.ncomp:
                    raise ValueError(f"No component at position {c}.")
                positions.append(int(c))
            elif isinstance(c, str):
                cid = _check_name(c)
                if cid not in self._prefix:
                    raise ValueError(f"No component with id '{c}'.")
                positions.append(self._prefix.index(cid))
            else:
                match = [i for i, m in enumerate(self.components) if m is c]
                if not match:
                    raise ValueError(
                        f"This {type(c).__name__} is not in the model."
                    )
                positions.append(match[0])
        return positions

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
            if isinstance(
                self.priors[key], numpyro.distributions.Distribution
            ):
                pars[key], pp = pp[0], pp[1:]
            elif not self.tied[ki]:
                pars[key] = self.priors[key]

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
            kwarg = self.components[nc]._build_kwargs(pars, self._prefix[nc])
            mone = self.components[nc]._evaluate(img, **kwarg)

            if self.positive[nc] and self.type[nc] != "Point":
                mneg = jp.where(mone < 0.00, 1.00, mneg)

            if self.type[nc] == "Point":
                if doresp:
                    dw = jp.array(
                        [
                            (kwarg["xc"] - img.hdu.header["CRVAL1"])
                            * jp.cos(jp.deg2rad(img.hdu.header["CRVAL2"])),
                            kwarg["yc"] - img.hdu.header["CRVAL2"],
                        ]
                    )
                    dp = img.fft.cd_inv @ dw
                    xpts = img.hdu.header["CRPIX1"] - 1 + dp[0]
                    ypts = img.hdu.header["CRPIX2"] - 1 + dp[1]
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

        msmo = mraw.copy()

        if doresp:
            msmo *= img.response

        mraw = mraw + img.fft.ifft(mpts * img.fft.center).real

        if img.psf is not None:
            msmo = img.convolve(msmo)
            mpts = img.convolve.crop(mpts * img.convolve.psf_fft).real
        else:
            mpts = img.fft.ifft(mpts).real
        msmo = msmo + mpts

        if doexp:
            msmo *= img.exposure
            mbkg *= img.exposure

        msmo = msmo + mbkg

        return mraw, msmo, mbkg, mneg
