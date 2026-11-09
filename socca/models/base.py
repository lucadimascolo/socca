"""Base component class for model profiles."""

import inspect
import keyword
import re
import warnings

import numpy as np
import numpyro.distributions

from . import config
from ..priors import _TIED, _Function
from socca.units import _NON_CONVERTIBLE, conversion_factor as _cfactor


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

    Tied values (lambdas, function() or boundto()) and None (an unset
    parameter, e.g. before a fixed value or prior has been assigned) are
    skipped.
    """
    if value is None:
        return
    if isinstance(value, _TIED):
        return
    if isinstance(value, numpyro.distributions.Distribution):
        kind = f"The {name} prior support includes values"
        vlo, vhi = value.support.lower_bound, value.support.upper_bound
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


def _warn_span(value, name, span, note=""):
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
    and `fitter.run`'s `periodic` argument).
    """
    if isinstance(value, numpyro.distributions.Distribution):
        extent = value.support.upper_bound - value.support.lower_bound
        if extent > span:
            warnings.warn(
                f"The {name} prior support spans more than "
                f"{np.degrees(span):.0f} deg. {note}",
                UserWarning,
                stacklevel=3,
            )
        if extent >= span:
            warnings.warn(
                f"The {name} prior spans a full "
                f"{np.degrees(span):.0f}-degree period, so it will "
                "automatically be treated as periodic if sampled with "
                "the nautilus, dynesty, or pocomc backends. Pass "
                "periodic=[...] to run() with this parameter's name to "
                "choose a different set, or periodic=[] to disable "
                "this entirely.",
                UserWarning,
                stacklevel=3,
            )


def _check_name(name):
    """Normalise a component name, checking it can be used in keys."""
    if name is None:
        return None
    if isinstance(name, (int, np.integer)) and not isinstance(name, bool):
        number = int(name)
    elif isinstance(name, str) and re.fullmatch(r"comp_[0-9]+", name):
        number = int(name[5:])
    elif (
        isinstance(name, str)
        and name.isidentifier()
        and not keyword.iskeyword(name)
    ):
        return name
    else:
        raise ValueError(
            f"Invalid component name {name!r}: use a Python identifier "
            "(e.g. 'core'), a non-negative integer or 'comp_NN'."
        )
    if number < 0:
        raise ValueError(f"Invalid component name {name!r}: negative.")
    return f"comp_{number:02d}"


class Component:
    """
    Base class for all model components.

    This abstract base class provides the infrastructure for model components,
    including parameter management, unique identification, and formatted output.
    All specific profile types (Beta, Sersic, Point, etc.) inherit from this class.

    Attributes
    ----------
    id : str or None
        Identifier in the model the component was last added to: its name
        if given, else ``comp_NN`` from its position. None until added.
    name : str or None
        Name given at creation (``name=``), used as the id.
    positive : bool
        Whether to enforce positivity constraint on this component.
    hyper : list of str
        Names of hyperparameters (not fitted, control behavior).
    units : dict
        Mapping from parameter names to their physical units.
    description : dict
        Mapping from parameter names to their descriptions.
    periodic : dict
        Mapping from parameter names to their period, for parameters that
        are periodic (e.g. an elliptical profile's ``theta``, symmetric
        under ``theta -> theta + pi``). Absent keys are not periodic.
    """

    id = None
    name = None

    def __init__(self, **kwargs):
        """
        Initialize a component with parameter tracking structures.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for component initialization.

            positive : bool, optional
                Override default positivity constraint.
            name : str or int, optional
                Identifier used for the component's parameters in a model
                (e.g. ``"core"`` gives ``core_xc``). An integer ``n`` or
                ``"comp_n"`` gives ``comp_NN``, as for the component at
                position ``n``. Default is the position in the model.
        """
        self.id = None
        self.name = _check_name(kwargs.get("name"))

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
                elif isinstance(keypar, _TIED):
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

    def _resolve(self, attr, _seen=frozenset()):
        """
        Return the value of a (possibly dotted) parameter, resolving ties.

        Used by standalone getmap() calls, where there is no model
        parameter dict: function() and boundto() follow their source
        objects, and a lambda's arguments must name parameters of this
        component.
        """
        obj = self
        for part in attr.split(".")[:-1]:
            obj = getattr(obj, part)
        value = getattr(obj, attr.split(".")[-1])
        if not isinstance(value, _TIED):
            return value

        tag = (id(self), attr)
        if tag in _seen:
            raise ValueError(f"Circular tie involving {attr}.")
        _seen = _seen | {tag}

        if isinstance(value, _Function):
            return value(
                *[comp._resolve(name, _seen) for comp, name in value.sources]
            )
        return value(
            *[
                self._resolve(self._tiearg(name), _seen)
                for name in inspect.signature(value).parameters
            ]
        )

    def _tiearg(self, name):
        """Map a lambda argument name to one of this component's parameters."""
        if self.id is None:
            raise ValueError(
                f"Cannot resolve the lambda argument '{name}': this "
                f"{type(self).__name__} is not in a model, so it has no id. "
                "Use socca.priors.function() instead."
            )
        local = re.sub(rf"^{re.escape(self.id)}_", "", name)
        for key in self.units:
            if local in (key, key.replace(".", "_")) and local != name:
                return key
        raise ValueError(
            f"Tie argument '{name}' is not a parameter of this "
            f"{type(self).__name__} ({self.id}). Standalone getmap() can "
            "only resolve ties to the component's own parameters; add the "
            "components to a Model and use Model.getmap() instead."
        )

    def _getpars(self, keys):
        """
        Resolve parameters and check they are all fixed.

        keys is a list of parameter names, or a dict mapping output
        keyword names to parameter names.
        """
        if not isinstance(keys, dict):
            keys = {key: key for key in keys}
        kwarg = {out: self._resolve(key) for out, key in keys.items()}
        for key, value in kwarg.items():
            if isinstance(value, numpyro.distributions.Distribution):
                raise ValueError(
                    "Priors must be fixed values, not distributions."
                )
            if value is None:
                raise ValueError(
                    f"keyword {key} is set to None. "
                    f"Please provide a valid value."
                )
        return kwarg

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
