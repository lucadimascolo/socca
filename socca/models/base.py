"""Base component class for model profiles."""

import inspect
import threading
import types
import warnings

import numpy as np
import numpyro.distributions

from . import config
from ..priors import _BoundTo
from socca.units import _NON_CONVERTIBLE, conversion_factor as _cfactor


_RESOLVING = threading.local()


def _support_bounds(prior):
    """
    Return a prior's support bounds in model space.

    Bounds are infinite where the support is unbounded.
    """
    lo = getattr(prior.support, "lower_bound", -float("inf"))
    hi = getattr(prior.support, "upper_bound", float("inf"))
    if hasattr(prior, "model_bounds"):
        lo, hi = (float(b) for b in prior.model_bounds())
    elif hasattr(prior, "to_model"):
        lo, hi = (float(prior.to_model(b)) for b in (lo, hi))
    return lo, hi


def _spans(extent, period, strict=False):
    """Whether `extent` reaches (or, if strict, exceeds) `period`."""
    if strict:
        return extent > period * (1.0 + 1e-9)
    return extent >= period * (1.0 - 1e-9)


def _warn_range(
    value,
    name,
    lower=None,
    upper=None,
    lower_strict=False,
    upper_strict=False,
    note="",
):
    """
    Warn if a fixed value, or a prior's support, falls outside [lower, upper].

    lower_strict/upper_strict make the corresponding bound itself invalid
    too (e.g. lower=0, lower_strict=True rejects exactly 0, for parameters
    like scale radii that must be strictly positive; lower=0 alone allows
    0, for parameters like ellipticities that are valid down to 0).

    Tied values (lambdas, functions, or _BoundTo) and None (an unset
    parameter, e.g. before a fixed value or prior has been assigned) are
    skipped.
    """
    if value is None:
        return
    if isinstance(value, (types.LambdaType, types.FunctionType, _BoundTo)):
        return
    if isinstance(value, numpyro.distributions.Distribution):
        kind = f"The {name} prior support includes values"
        vlo, vhi = _support_bounds(value)
    else:
        kind = f"The {name} parameter is"
        vlo = vhi = value
    if upper is not None:
        bad = vhi >= upper if upper_strict else vhi > upper
        if bad:
            suffix = "or equal to " if upper_strict else ""
            warnings.warn(
                f"{kind} greater than {suffix}{upper}. {note}",
                UserWarning,
                stacklevel=3,
            )
    if lower is not None:
        bad = vlo <= lower if lower_strict else vlo < lower
        if bad:
            suffix = "or equal to " if lower_strict else ""
            warnings.warn(
                f"{kind} less than {suffix}{lower}. {note}",
                UserWarning,
                stacklevel=3,
            )


def _warn_span(value, name, span, unit="rad", note=""):
    """
    Warn about a prior distribution whose support is span or wider.

    Two independent conditions are checked. A prior wider than `span`
    triggers `note` -- typically a caution that this needlessly
    duplicates the same physical solution at two points in the prior
    (e.g. a position angle prior wider than its symmetry period). A
    prior that is at least `span` wide -- including exactly `span`,
    the normal way to mark a periodic parameter's full range -- also
    triggers a second, separate warning explaining that it will
    automatically be sampled as periodic (see `Component.periodic`
    and `fitter.run`'s `periodic` argument). Priors with unbounded
    support are not checked. `span` is in `unit`, the unit of the prior.
    """
    if unit == "rad":
        period = f"{np.degrees(span):.0f} deg"
    else:
        period = f"{span:.6g} {unit}"
    if isinstance(value, numpyro.distributions.Distribution):
        lo, hi = _support_bounds(value)
        extent = hi - lo
        if not np.isfinite(extent):
            return
        if _spans(extent, span, strict=True):
            warnings.warn(
                f"The {name} prior support spans more than {period}. {note}",
                UserWarning,
                stacklevel=4,
            )
        if _spans(extent, span):
            warnings.warn(
                f"The {name} prior spans a full period ({period}), so it will "
                "automatically be treated as periodic if sampled with "
                "the nautilus, dynesty, or pocomc backends. Pass "
                "periodic=[...] to run() with this parameter's name to "
                "choose a different set, or periodic=[] to disable "
                "this entirely.",
                UserWarning,
                stacklevel=4,
            )


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
    periodic : dict
        Mapping from parameter names to their period in internal units,
        for parameters that are periodic (e.g. an elliptical profile's
        ``theta``, symmetric under ``theta -> theta + pi``). Absent keys
        are not periodic.
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
        self.id = f"comp_{Component.idcls:02d}"
        Component.idcls += 1

        self.positive = kwargs.get("positive", config.Component.positive)
        self.hyper = []
        self.units = {}
        self._input_units = {}
        self.description = {}
        self.periodic = {}
        self._initialized = False

    def __setattr__(self, name, value):
        """Prevent adding undefined attributes after initialization."""
        if getattr(self, "_initialized", False) and not hasattr(self, name):
            raise AttributeError(
                f"Cannot add new attribute '{name}' to "
                f"{self.__class__.__name__}. "
                f"Use addparameter() to add new parameters."
            )
        super().__setattr__(name, value)

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
        keyout = [key for key in self.units.keys() if key not in self.hyper]

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
                keypar = getattr(self, key)

                if keypar is None:
                    keyval = None
                elif isinstance(keypar, numpyro.distributions.Distribution):
                    keyval = f"Distribution: {keypar.__class__.__name__}"
                elif isinstance(
                    keypar, (types.LambdaType, types.FunctionType, _BoundTo)
                ):
                    keyval = "Tied keyparameter"
                else:
                    keyval = f"{keypar:.4E}"

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
                    kvalue = getattr(self, key)
                    kvalue = None if kvalue is None else f"{kvalue:.4E}"
                    print(
                        f"{key:<{keylen}} [{self.units[key]}] : "
                        + f"{kvalue}".ljust(10)
                        + f" | {self.description[key]}"
                    )

        else:
            print("No parameters defined.")

    def _getpar(self, name):
        """
        Return a parameter by tied-lambda argument name.

        The component prefix is stripped, names such as ``radial_rs``
        resolve to the namespaced parameter ``radial.rs``, and tied values
        are evaluated.
        """
        name = name.removeprefix(f"{self.id}_")
        for ns in ("radial", "vertical", "parallel"):
            sub = getattr(self, ns, None)
            if isinstance(sub, Component) and name.startswith(f"{ns}_"):
                value = getattr(sub, name[len(ns) + 1 :])
                break
        else:
            value = getattr(self, name)
        if isinstance(value, numpyro.distributions.Distribution):
            raise ValueError("Priors must be fixed values, not distributions.")
        resolving = _RESOLVING.__dict__.setdefault("keys", set())
        key = (id(self), name)
        if key in resolving:
            raise ValueError(
                f"Circular ties among the parameters of {self.id}."
            )
        resolving.add(key)
        try:
            return self._resolve(value)
        finally:
            resolving.discard(key)

    def _input_unit(self, name):
        """
        Input unit declared for a parameter, or None.

        For a namespaced parameter (e.g. ``radial.re``), a unit declared on
        this component takes precedence over one set on the sub-component.
        """
        if name in self._input_units:
            return self._input_units[name]
        ns, _, sub = name.partition(".")
        comp = getattr(self, ns, None) if sub else None
        if isinstance(comp, Component):
            return comp._input_units.get(sub)
        return None

    def _resolve(self, value):
        """Evaluate a tied value outside a Model, from this component."""
        if isinstance(value, _BoundTo):
            return value.resolve()
        if isinstance(value, (types.LambdaType, types.FunctionType)):
            names = inspect.signature(value).parameters
            return value(**{p: self._getpar(p) for p in names})
        return value

    def parlist(self):
        """
        Return a list of parameter names for this component.

        Returns
        -------
        list of str
            Names of all model parameters, as defined in the units dict.

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
        return list(self.units.keys())

    def set_units(self, **kwargs):
        """Declare the input units for one or more parameters.

        This tells the model that your parameter values (or prior bounds) are
        expressed in the given units.  The model will transparently convert them
        to its internal units before evaluation.

        Parameters
        ----------
        **kwargs : str
            Keyword arguments mapping parameter names to unit strings
            (any string recognised by :mod:`astropy.units`).

        Warns
        -----
        UserWarning
            For surface brightness parameters (internal unit ``"image"``) or
            dimensionless parameters (internal unit ``""``): the declared unit
            will be ignored and conversion skipped.

        Raises
        ------
        ValueError
            If a parameter name is unknown, the unit string is not parseable
            by astropy, or the declared unit cannot be converted to the
            internal unit.

        Examples
        --------
        >>> comp = Gaussian(xc=priors.uniform(100, 200), rs=priors.uniform(1, 10))
        >>> comp.set_units(xc="arcsec", rs="arcmin")
        """
        for name, unit_str in kwargs.items():
            if name not in self.units:
                raise ValueError(
                    f"Unknown parameter '{name}' on {self.__class__.__name__}."
                )
            native = self.units[name]
            if native in _NON_CONVERTIBLE:
                kind = (
                    "surface brightness"
                    if native == "image"
                    else "dimensionless"
                )
                warnings.warn(
                    f"Parameter '{name}' is {kind} (internal unit: "
                    f"'{native}'). Unit '{unit_str}' will be neglected.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            _cfactor(unit_str, native)
            self._input_units[name] = unit_str

    def addparameter(self, name, value=None, units="", description=""):
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
        >>> comp.addparameter('amplitude', value=1.0, units='Jy', description='Source flux')
        >>> comp.amplitude
        1.0
        >>> comp.units['amplitude']
        'Jy'
        """
        self._initialized = False
        setattr(self, name, value)
        self._initialized = True
        self.units.update({name: units})
        self.description.update({name: description})

    def removeparameter(self, name):
        """
        Remove a parameter from the component.

        Parameters
        ----------
        name : str
            Name of the parameter to remove.

        Raises
        ------
        AttributeError
            If the parameter does not exist on this component.

        Examples
        --------
        >>> from socca.models import Beta
        >>> beta = Beta(xc=180.5, yc=45.2)
        >>> beta.addparameter('custom', value=1.0, units='Jy')
        >>> beta.removeparameter('custom')
        """
        if not hasattr(self, name):
            raise AttributeError(
                f"Parameter '{name}' does not exist on "
                f"{self.__class__.__name__}."
            )
        self._initialized = False
        delattr(self, name)
        self._initialized = True
        self.units.pop(name, None)
        self._input_units.pop(name, None)
        self.description.pop(name, None)
        self.hyper = [h for h in self.hyper if h != name]
