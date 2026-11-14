import copy
import inspect
import types
import warnings

import jax
import jax.numpy as jp
import numpy as np

import numpyro.distributions


from ..priors import _TIED, _BoundTo, _Function
from . import config
from .base import (
    _a,
    _too_old,
    Component,
    _check_name,
    _format_rows,
    _format_value,
    _getdotted,
    _outside,
    _warn_span,
    _whole,
)
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
    # ellipsoid
    "Ellipsoid",
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
    - SimpleBridge, MesaBridge: Bridge/filament models
    - Disk: 3D disk model with finite thickness
    - Ellipsoid: 3D triaxial-ellipsoid model evaluated via line-of-sight
      integration (e.g. a galactic bar)

    Examples
    --------
    >>> import socca.models as models
    >>> models.zoo()
    <BLANKLINE>
    Radial models
    =============
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
    <BLANKLINE>
    Filaments
    =========
    SimpleBridge
    MesaBridge
    <BLANKLINE>
    Disk-like model
    ===============
    Disk
    <BLANKLINE>
    Ellipsoid-like model
    ====================
    Ellipsoid
    <BLANKLINE>
    Other models
    ============
    Point
    Background
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
    print("=========")
    models = [
        "SimpleBridge",
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
def _period(prof, name):
    """Return the native period of a (possibly dotted) parameter, or None."""
    if name in prof.periodic:
        return prof.periodic[name]
    sub, local = prof._sub(name)
    if sub is not None:
        return sub.periodic.get(local)
    return None


def _warn_lambda(key):
    """Warn that a lambda tie is deprecated."""
    warnings.warn(
        f"Tying {key} with a lambda is deprecated and will be removed: "
        "use socca.priors.function() instead.",
        FutureWarning,
        stacklevel=_outside(),
    )


def _snapshot(value):
    """Copy a fixed array value, so later in-place edits are seen."""
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.array(value)
    return value


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
            Dictionary mapping parameter names to their period, in their
            declared units (see Component.set_units), for parameters that
            are periodic (e.g. an elliptical profile's ``theta``). Keys
            absent from this dict are not periodic.

        Examples
        --------
        >>> from socca.models import Model, Beta
        >>> # Create empty model and add component later
        >>> model = Model()
        >>> model.addcomponent(Beta(xc=180.5, yc=45.2, rc=0.01, Ic=1.0))
        >>> # Or initialize with a profile directly
        >>> model = Model(Beta(xc=180.5, yc=45.2, rc=0.01, Ic=1.0))
        >>> model
        Model
          Beta (comp_00)
        """
        self.ncomp = 0
        self.priors = {}
        self.params = []
        self.paridx = []
        self.positive = []
        self.components = ()
        self.tied = []
        self.type = []
        self.units = {}
        self.periodic = {}
        self.conversions = {}
        self._prefix = ()
        self._owner = []
        self._positive = ()
        self._lambdas = {}
        self._warned = set()

        if prof is not None:
            self._addcomponent(prof, positive)

    def __setstate__(self, state):
        """Restore a pickled model, refusing those saved by socca 1.4.0."""
        if "_prefix" not in state:
            raise _too_old("model")
        self.__dict__.update(state)
        for key in ("components", "_prefix", "_positive"):
            self.__dict__[key] = tuple(self.__dict__[key])
        # rekey on the ids of the unpickled lambdas
        self._lambdas = {id(p): (p, f) for p, f in self._lambdas.values()}

    def __copy__(self):
        """Copy the model's tables; the components themselves are shared."""
        new = Model.__new__(Model)
        new.__dict__.update(
            {
                k: copy.copy(v) if isinstance(v, (list, dict, set)) else v
                for k, v in self.__dict__.items()
            }
        )
        return new

    # Add a new component to the model
    # --------------------------------------------------------
    def addcomponent(self, prof, positive=None):
        """
        Add a profile component to the composite model.

        This method incorporates a new profile component into the model, handling
        parameter registration, prior assignments, and constraint management.
        Parameters are prefixed with the component id: its ``name=`` if
        given, else ``comp_NN`` with the lowest free number from the
        model's current count.

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
            model or is a sub-profile wrapped by a Disk, Ellipsoid or
            Bridge, if one of its sub-profiles is already in the model, if
            its id is already used, if two parameters share a model key, or
            if a hyperparameter is not fixed.
        KeyError
            If a lambda tie argument matches no parameter of the model, or
            several (when the model is used: getmap, getmodel, fitter).

        Warns
        -----
        UserWarning
            If a prior on a periodic parameter spans a whole number of
            periods (sampled as periodic by nautilus, dynesty and pocomc)
            or more than one period.
        FutureWarning
            If a parameter is tied with a lambda (use priors.function).

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
        >>> model.addcomponent(Beta(xc=180.5, yc=45.2, rc=0.01, Ic=1.0))
        >>> # Add a point source
        >>> model.addcomponent(Point(xc=180.6, yc=45.3, Ic=1.0), positive=True)
        >>> print(model.ncomp)
        2
        >>> model
        Model
          Beta (comp_00)
          Point (comp_01)
        """
        self._addcomponent(prof, positive)

    def _addcomponent(self, prof, positive):
        """Add a component; see addcomponent()."""
        if any(c is prof for c in self.components):
            raise ValueError(
                f"This {type(prof).__name__} is already in the model."
            )
        if prof._wrapper is not None:
            raise ValueError(
                f"This {type(prof).__name__} is part of "
                f"{_a(type(prof._wrapper).__name__)}: add the "
                f"{type(prof._wrapper).__name__} instead."
            )
        _whole(prof)
        subs = getattr(prof, "_namespaces", {}).values()
        if any(sub is c for sub in subs for c in self.components):
            raise ValueError(
                f"A sub-component of this {type(prof).__name__} is already "
                "in the model."
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

        old = (prof.id, self.components, self._prefix, self._positive)
        self.components += (prof,)
        self._prefix += (prefix,)
        self._positive += (positive,)
        prof.id = prefix
        try:
            self._sync()
        except Exception:
            prof.id, self.components, self._prefix, self._positive = old
            raise

    def _legacy(self, par):
        """Return a lambda tie with jax.numpy available as jp."""
        cached = self._lambdas.get(id(par))
        if cached is not None and cached[0] is par:
            return cached[1]
        _glb = {
            name: par.__globals__[name]
            for name in par.__code__.co_names
            if name in par.__globals__
        }
        _glb["jp"] = jp
        _glb.setdefault("__builtins__", __builtins__)
        new = types.FunctionType(
            par.__code__,
            _glb,
            par.__name__,
            par.__defaults__,
            par.__closure__,
        )
        new.__dict__.update(par.__dict__)
        self._lambdas[id(par)] = (par, new)
        return new

    def _sync(self, strict=True):
        """
        Rebuild the parameter tables from the components.

        Called when a component is added and whenever the model is used
        (getmap, parameters, fitter), so changes made to the components
        after adding them are followed: values, priors, ties, units and
        truncations. With strict=False (listings), unset values are
        allowed.
        """
        tables = dict(
            type=[],
            positive=[],
            params=[],
            _owner=[],
            priors={},
            paridx=[],
            tied=[],
            units={},
            conversions={},
            periodic={},
        )
        notes = []
        for nc, (prof, prefix) in enumerate(
            zip(self.components, self._prefix)
        ):
            if strict and prof._wrapper is not None:
                raise ValueError(
                    f"The {type(prof).__name__} '{prefix}' is now part of a "
                    f"{type(prof._wrapper).__name__}: a component can "
                    "belong to one model entry only."
                )
            positive = self._positive[nc]
            tables["type"].append(prof.__class__.__name__)
            tables["positive"].append(
                prof.positive if positive is None else positive
            )
            for p in prof.parlist():
                par = _getdotted(prof, p)
                key = f"{prefix}_{p}"
                if key in tables["priors"]:
                    raise ValueError(
                        f"Two parameters share the model key {key}: rename "
                        "a component or the parameter."
                    )

                if par is None and strict:
                    raise ValueError(
                        f"Parameter {p} in component {prefix} is set "
                        f"to None. Please provide a valid value or prior."
                    )
                if (
                    strict
                    and p in prof.hyper
                    and isinstance(
                        par, (numpyro.distributions.Distribution, *_TIED)
                    )
                ):
                    raise ValueError(
                        f"{key} is a hyperparameter, not fitted: set it to a "
                        "fixed value."
                    )

                unit = prof._input_unit(p)
                factor = prof._factor(p)
                period = _period(prof, p)
                if period is not None:
                    period = period / factor
                    notes.append((_warn_span, par, key, (period, unit)))

                if isinstance(par, types.FunctionType):
                    notes.append((_warn_lambda, par, key, ()))
                    par = self._legacy(par)

                tables["params"].append(key)
                tables["_owner"].append(nc)
                tables["priors"][key] = par
                if isinstance(par, numpyro.distributions.Distribution):
                    tables["paridx"].append(len(tables["params"]) - 1)
                tables["tied"].append(isinstance(par, _TIED))
                tables["units"][key] = unit
                if factor != 1.0:
                    tables["conversions"][key] = factor
                if period is not None:
                    tables["periodic"][key] = period

        for warn, par, key, args in notes:
            tag = (warn.__name__, key, id(par), args)
            if tag not in self._warned:
                if warn is _warn_span:
                    warn(par, key, *args)
                else:
                    warn(key)
                self._warned.add(tag)

        self.__dict__.update(tables)
        self.ncomp = len(self.components)

    def _state(self):
        """Return a snapshot of the parameter tables, to detect changes."""
        return (
            tuple(self.params),
            tuple(self.positive),
            tuple(_snapshot(self.priors[key]) for key in self.params),
            tuple(sorted(self.conversions.items())),
            tuple(
                (
                    type(getattr(prof, "truncation", None)),
                    getattr(prof, "_numerics_key", tuple)(),
                )
                for comp in self.components
                for prof in [comp, *getattr(comp, "_namespaces", {}).values()]
            ),
        )

    @staticmethod
    def _samestate(old, new):
        """Whether two snapshots match: values by value, priors by identity."""
        if old[:2] != new[:2] or old[3:] != new[3:]:
            return False

        def same(a, b):
            if isinstance(a, (int, float, np.number, np.ndarray)):
                return isinstance(
                    b, (int, float, np.number, np.ndarray)
                ) and np.array_equal(a, b)
            return a is b

        return all(same(a, b) for a, b in zip(old[2], new[2]))

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
        >>> from socca import priors
        >>> from socca.models import Model, Beta
        >>> beta = Beta(xc=priors.uniform(180, 181), yc=45.2, rc=0.01, Ic=1.0)
        >>> model = Model(beta)
        >>> model.parameters()
        <BLANKLINE>
        Beta (comp_00)
        ==============
        xc    [deg] : Uniform [180, 181] | Right ascension of centroid
        yc    [deg] : 4.5200E+01         | Declination of centroid
        theta [rad] : 0.0000E+00         | Position angle (east from north)
        e        [] : 0.0000E+00         | Projected ellipticity (1 - axis ratio)
        cbox     [] : 0.0000E+00         | Projected boxiness
        rc    [deg] : 1.0000E-02         | Core radius
        alpha    [] : 2.0000E+00         | Radial exponent
        beta     [] : 5.5000E-01         | Slope parameter
        Ic  [image] : 1.0000E+00         | Central surface brightness
        """
        self._sync(strict=False)
        blocks = []
        for nc, prefix in enumerate(self._prefix):
            comp = self.components[nc]
            rows, hyper = [], []
            for key, owner in zip(self.params, self._owner):
                par = self.priors[key]
                if owner != nc or (
                    freeonly
                    and not isinstance(par, numpyro.distributions.Distribution)
                ):
                    continue
                name = key[len(prefix) + 1 :]
                (hyper if name in comp.hyper else rows).append(
                    (
                        name,
                        self.units[key],
                        _format_value(par, self._source_label),
                        comp.description.get(name, ""),
                    )
                )
            if rows or hyper:
                blocks.append((f"{self.type[nc]} ({prefix})", rows, hyper))

        if not blocks:
            print("No free parameters." if freeonly else "No parameters.")
            return
        lines = _format_rows(
            [row for _, rows, hyper in blocks for row in rows + hyper]
        )
        for title, rows, hyper in blocks:
            print(f"\n{title}\n" + "=" * len(title))
            if rows:
                print("\n".join(lines[: len(rows)]))
            lines = lines[len(rows) :]
            if hyper:
                print("\nHyperparameters\n" + "-" * 15)
                print("\n".join(lines[: len(hyper)]))
            lines = lines[len(hyper) :]

    def __repr__(self):
        """List the model components with their ids."""
        self._sync(strict=False)
        lines = [
            f"{ctype} ({prefix})"
            for ctype, prefix in zip(self.type, self._prefix)
        ]
        return "\n".join(["Model"] + ["  " + line for line in lines])

    def _source_label(self, comp, name):
        """Label of a tie source: the component id and parameter name."""
        try:
            key = self._keyof(comp, name)
            prefix = self._prefix[self._owner[self.params.index(key)]]
        except ValueError:
            return f"{type(comp).__name__}.{name} (not in model)"
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
        - All model parameters must be fixed values (any numeric type).
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
        self._sync()
        for key in self.params:
            if isinstance(
                self.priors[key], numpyro.distributions.Distribution
            ):
                raise ValueError(
                    f"Parameter {key} is a distribution. "
                    f"Use getmodel() with sampled values instead."
                )

        mraw, msmo, mbkg, _ = self._getmodel(
            img, [], doresp=False, doexp=False
        )

        if not addbackground:
            msmo = msmo - mbkg

        if convolve:
            return msmo
        else:
            if addbackground:
                warnings.warn(
                    "The background component is added only to the "
                    "convolved map. Returning the raw map without background.",
                    stacklevel=_outside(),
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
            if len(match) > 1:
                raise KeyError(
                    f"Tie argument '{name}' of {key} matches several "
                    f"parameters ({', '.join(match)}): use "
                    "socca.priors.function() instead."
                )
            args.append(match[0])
        return args

    def _resolve_ties(self, pars):
        """
        Evaluate tied parameters in dependency order, in place.

        pars holds native values. Functions take and return declared
        units (see Component.set_units); boundto copies the value.
        """
        scale = self.conversions
        deps, order = self._tieorder()
        for key in order:
            tie = self.priors[key]
            if isinstance(tie, _BoundTo):
                pars[key] = pars[deps[key][0]]
            elif getattr(tie, "native", False):
                pars[key] = tie(*[pars[dep] for dep in deps[key]])
            else:
                args = [pars[dep] / scale.get(dep, 1.0) for dep in deps[key]]
                pars[key] = tie(*args) * scale.get(key, 1.0)

    def _tieorder(self):
        """
        Return the tie dependencies and an evaluation order.

        Raises if a tie refers to a component outside the model, to an
        unknown parameter, or if ties form a cycle.
        """
        deps = {
            key: self._tieargs(key)
            for key, tied in zip(self.params, self.tied)
            if tied
        }
        for key, args in deps.items():
            for dep in args:
                if dep not in self.priors:
                    raise ValueError(
                        f"{key} is tied to {dep}, which is not a parameter "
                        "of this model."
                    )
        order = []

        def visit(key, chain):
            if key in order:
                return
            if key in chain:
                raise ValueError(
                    "Circular tie: " + " -> ".join(chain + [key]) + "."
                )
            for dep in deps[key]:
                if dep in deps:
                    visit(dep, chain + [key])
            order.append(key)

        for key in deps:
            visit(key, [])
        return deps, order

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
            If a selection matches no component of the model, or names
            a component more than once.

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
        if len(set(positions)) < len(positions):
            raise ValueError(
                f"The selection {component!r} names a component more "
                "than once."
            )
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
        component : None, str, int, Component, or list of these, optional
            Component(s) to include in the model computation:

            - None: all components (default)
            - str: a component id, its ``name=`` or ``comp_NN``
            - int: a position in the model (e.g., 0 for the first)
            - Component: a component of the model, matched by identity
            - list: several of the above

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
        >>> import numpy as np
        >>> from astropy.io import fits
        >>> from astropy.wcs import WCS
        >>> from socca import priors
        >>> from socca.data import Image
        >>> from socca.models import Model, Beta
        >>> wcs = WCS(naxis=2)
        >>> wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        >>> wcs.wcs.crval = [180.5, 45.2]
        >>> wcs.wcs.crpix = [16.5, 16.5]
        >>> wcs.wcs.cdelt = [-1e-3, 1e-3]
        >>> hdu = fits.PrimaryHDU(np.zeros((32, 32)), header=wcs.to_header())
        >>> img = Image(hdu)
        >>> beta = Beta(xc=priors.uniform(180, 181), yc=45.2, rc=0.01, Ic=1.0)
        >>> model = Model(beta)
        >>> # During fitting, pp contains sampled parameter values
        >>> pp = [180.5]  # value for xc
        >>> mraw, msmo, mbkg, mneg = model.getmodel(img, pp)
        >>> mraw.shape
        (32, 32)
        """
        self._sync()
        return self._getmodel(img, pp, doresp, doexp, component)

    def _getmodel(self, img, pp, doresp=False, doexp=False, component=None):
        """Compute the model images; the tables must already be synced."""
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
