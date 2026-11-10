"""Base component class for model profiles."""

import copy
import functools
import inspect
import keyword
import re
import warnings

import jax
import numpy as np
import numpyro.distributions

from . import config
from ..priors import _TIED, _BoundTo, _Function
from socca.units import _NON_CONVERTIBLE, conversion_factor as _cfactor


def _warn_range(
    value,
    name,
    lower=None,
    upper=None,
    lower_strict=False,
    upper_strict=False,
    note="",
    error=False,
):
    """
    Warn if a fixed value, or a prior's support, falls outside [lower, upper].

    With error=True, raise ValueError instead.

    lower_strict/upper_strict make the corresponding bound itself invalid
    too (e.g. lower=0, lower_strict=True rejects exactly 0, for parameters
    like scale radii that must be strictly positive; lower=0 alone allows
    0, for parameters like ellipticities that are valid down to 0).
    They apply to fixed values only: a prior's endpoints have zero
    probability.

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
        vlo, vhi = _support_bounds(value)
        lower_strict = upper_strict = False
    else:
        kind = f"The {name} parameter is"
        vlo = vhi = value
        if error and np.isnan(np.asarray(value, dtype=float)).any():
            raise ValueError(f"{kind} NaN. {note}".strip())
    messages = []
    if upper is not None and (vhi >= upper if upper_strict else vhi > upper):
        suffix = "or equal to " if upper_strict else ""
        messages.append(f"{kind} greater than {suffix}{upper}. {note}")
    if lower is not None and (vlo <= lower if lower_strict else vlo < lower):
        suffix = "or equal to " if lower_strict else ""
        messages.append(f"{kind} less than {suffix}{lower}. {note}")
    for message in messages:
        if error:
            raise ValueError(message.strip())
        warnings.warn(message.strip(), UserWarning, stacklevel=3)


def _support_bounds(prior):
    """Return a prior's support as (low, high), infinite where unbounded."""
    support = prior.support
    return (
        float(getattr(support, "lower_bound", -np.inf)),
        float(getattr(support, "upper_bound", np.inf)),
    )


def _whole_periods(low, high, period):
    """Whether [low, high] spans a whole number of periods (at least one)."""
    extent = high - low
    turns = np.round(extent / period)
    return bool(
        np.isfinite(extent)
        and turns >= 1
        and abs(extent - turns * period) <= 1e-9 * extent
    )


def _warn_span(value, name, period, unit, stacklevel):
    """
    Warn about a prior on a periodic parameter spanning its period.

    A prior wider than the period repeats the same solutions. A prior
    spanning a whole number of periods is sampled as periodic by
    nautilus, dynesty and pocomc (see fitter.run's periodic argument).
    """
    if not isinstance(value, numpyro.distributions.Distribution):
        return
    low, high = _support_bounds(value)
    extent = high - low
    if not np.isfinite(extent):
        return
    text = f"{period:.6g} {unit}".strip()
    if extent > period * (1 + 1e-9):
        warnings.warn(
            f"The {name} prior spans more than its period ({text}), so "
            "it repeats the same solutions, which can make the posterior "
            "multimodal and harder to sample.",
            UserWarning,
            stacklevel=stacklevel,
        )
    if _whole_periods(low, high, period):
        warnings.warn(
            f"The {name} prior spans whole periods ({text}), so it will "
            "be treated as periodic by the nautilus, dynesty and pocomc "
            "backends. Pass periodic=[...] to run() to choose a different "
            "set, or periodic=[] to disable this.",
            UserWarning,
            stacklevel=stacklevel,
        )


def _source_label(comp, name):
    """Label of a tie source outside a model: id (or type) and name."""
    return f"{comp.id or type(comp).__name__}.{name}"


def _format_bounds(low, high):
    """Short texts for prior bounds, with enough digits to tell them apart."""
    for digits in range(4, 18):
        texts = [
            "inf"
            if v == np.inf
            else "-inf"
            if v == -np.inf
            else f"{v:.{digits}g}"
            for v in (low, high)
        ]
        if texts[0] != texts[1] or low == high:
            return texts
    return texts


def _format_value(value, label=_source_label):
    """
    Short text for a parameter value, prior or tie.

    label(comp, name) names the source parameters of function() and
    boundto() ties.
    """
    if value is None:
        return "None"
    if isinstance(value, numpyro.distributions.Distribution):
        text = type(value).__name__
        try:
            low, high = _support_bounds(value)
        except (NotImplementedError, TypeError):
            return text
        if np.isinf(low) and np.isinf(high):
            return text
        left = "(" if np.isinf(low) else "["
        right = ")" if np.isinf(high) else "]"
        low, high = _format_bounds(low, high)
        return f"{text} {left}{low}, {high}{right}"
    if isinstance(value, _BoundTo):
        return "= " + label(*value.sources[0])
    if isinstance(value, _Function):
        return "function of " + ", ".join(label(*s) for s in value.sources)
    if isinstance(value, _TIED):
        return "lambda of " + ", ".join(inspect.signature(value).parameters)
    try:
        return f"{float(value):.4E}"
    except (TypeError, ValueError):
        return str(value)


def _format_rows(rows, indent=""):
    """Lines for (name, unit, value, description) rows in aligned columns."""
    width = max(len(f"{name} [{unit}]") for name, unit, _, _ in rows)
    vwidth = max(len(value) for _, _, value, _ in rows)
    return [
        f"{indent}{name:<{width - len(unit) - 3}} [{unit}] : "
        f"{value:<{vwidth}} | {desc}".rstrip(" |")
        for name, unit, value, desc in rows
    ]


def _repoint(value, old, new):
    """Return value, with function()/boundto() sources on old moved to new."""
    if not isinstance(value, _Function):
        return value
    value = copy.copy(value)
    value.sources = [
        (new if comp is old else comp, name) for comp, name in value.sources
    ]
    return value


def _own_tables(state):
    """Return state with its dicts, lists and sets copied."""
    return {
        key: copy.copy(value)
        if isinstance(value, (dict, list, set))
        else value
        for key, value in state.items()
    }


def _fixed(value):
    """Return whether a parameter value is fixed (not a prior or tie)."""
    return not isinstance(value, (numpyro.distributions.Distribution, _TIED))


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


def _whole(prof, free=()):
    """Reject a profile missing parameters stripped by a wrapper, or a copy."""
    lost = sorted(set(getattr(prof, "_stripped", ())) - set(free))
    if lost:
        raise ValueError(
            f"This {type(prof).__name__} lost {', '.join(lost)} to a wrapper "
            "(Disk, Ellipsoid or Bridge), possibly as a copy of a wrapped "
            "profile: create a new one."
        )


_REFIT = (
    "re-run the fit with this version, or load it with socca 1.4.0 "
    "(pip install socca==1.4.0)."
)


def _check_losdepth(value):
    """Reject a non-positive line-of-sight depth (inf is allowed)."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"losdepth must be a number, not {value!r}.")
    _warn_range(
        value,
        "losdepth",
        lower=0,
        lower_strict=True,
        note="The line-of-sight depth must be strictly positive.",
        error=True,
    )


def _check_losbins(value):
    """Reject a number of line-of-sight nodes that is not an integer >= 2."""
    if isinstance(value, (numpyro.distributions.Distribution, *_TIED)):
        return  # not a fixed value: the model raises for hyperparameters
    if not (
        isinstance(value, (int, np.integer))
        and not isinstance(value, bool)
        and value >= 2
    ):
        raise ValueError(
            f"losbins must be an integer of at least 2, not {value!r}."
        )


_LOS_CHECKS = {"losdepth": _check_losdepth, "losbins": _check_losbins}


def _too_old(kind):
    """Error for a pickle saved by socca 1.4.0 or earlier."""
    return ValueError(
        f"This {kind} was saved by socca 1.4.0 or earlier, which this "
        f"version cannot load: {_REFIT}"
    )


def _a(name):
    """Name with its indefinite article."""
    return f"{'an' if name[0] in 'AEIOU' else 'a'} {name}"


def _claim(*subs, free=()):
    """
    Check that sub-profiles are free to be wrapped by a new component.

    free names the parameters the wrapper strips, so a sub-profile may
    already lack them.
    """
    for sub in subs:
        if sub.id is not None:
            raise ValueError(
                f"This {type(sub).__name__} is in a model ({sub.id}), which "
                "wrapping it would break: use a separate profile, e.g. "
                "copy.copy() of it."
            )
        if sub._wrapper is not None:
            raise ValueError(
                f"This {type(sub).__name__} is already part of "
                f"{_a(type(sub._wrapper).__name__)}: use a separate profile "
                "for each component, e.g. a copy made before wrapping."
            )
        _whole(sub, free)
    if len({id(sub) for sub in subs}) < len(subs):
        raise ValueError("The same profile cannot be used twice.")


def _strip(sub, keys):
    """Remove parameters that a wrapper owns or ignores from a sub-profile."""
    sub.__dict__["_stripped"] = set(getattr(sub, "_stripped", ())) | set(keys)
    for key in keys:
        attr = f"_{key}" if f"_{key}" in sub.__dict__ else key
        if attr in sub.__dict__:
            delattr(sub, attr)
        for table in (
            sub.units,
            sub._input_units,
            sub.description,
            sub.periodic,
        ):
            table.pop(key, None)


def _subkeys(prof, prefix=None):
    """Keyword names of a profile, truncation included, optionally prefixed."""
    keys = [
        key
        for key in inspect.signature(prof.profile).parameters
        if key not in ["r", "z"]
    ]
    if getattr(prof, "truncation", None) is not None:
        keys += ["rt", "wt"]
    if prefix is None:
        return keys
    return [f"{prefix}_{key}" for key in keys]


def _subprofile(prof, x, kw, prefix=None):
    """Evaluate a profile on its keywords, optionally prefixed."""
    if prefix is None:
        pars = dict(kw)
    else:
        pars = {
            key[len(prefix) + 1 :]: val
            for key, val in kw.items()
            if key.startswith(f"{prefix}_")
        }
    rt, wt = pars.pop("rt", None), pars.pop("wt", None)
    out = prof.profile(x, **pars)
    if rt is not None:
        out = out * prof.truncation.profile(x, rt, wt)
    return out


def _getdotted(obj, name):
    """Attribute of obj named by a dotted path, e.g. radial.xc."""
    return functools.reduce(getattr, name.split("."), obj)


def _subkeymap(prof, prefix, namespace):
    """Map a sub-profile's prefixed keywords to its parameter names."""
    n = len(prefix) + 1
    return {key: f"{namespace}.{key[n:]}" for key in _subkeys(prof, prefix)}


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
    _wrapper = None
    _format = 1  # of pickles; those of socca 1.4.0 have none
    _periods = {}

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
        self.name = kwargs.get("name")

        self.positive = kwargs.get("positive", config.Component.positive)
        self.hyper = []
        self.units = {}
        self._input_units = {}
        self.description = {}
        self.periodic = dict(self._periods)
        self._initialized = False

    def __getstate__(self):
        """Return the state for pickling and copying, without compiled code."""
        state = self.__dict__.copy()
        state.pop("_jit", None)
        state.pop("_wrapper", None)  # set again by the wrapper's __setstate__
        state["_format"] = self._format
        return state

    def __setstate__(self, state):
        """Restore a pickled component, refusing those of socca 1.4.0."""
        state = dict(state)
        if state.pop("_format", 0) < 1:
            raise _too_old(type(self).__name__)
        self.__dict__.update(state)
        for sub in state.get("_namespaces", {}).values():
            if sub._wrapper is None:
                sub._wrapper = self

    def __copy__(self):
        """
        Return a shallow copy with its own parameter tables.

        Ties to the component point to the copy, which has no id until it
        is added to a model. A wrapper's sub-profiles are copied with it,
        so they belong to the copy alone. copy.deepcopy keeps the id, as
        whole models need: wrap a copy.copy of a model member instead.
        """
        new = type(self).__new__(type(self))
        new.__setstate__(_own_tables(self.__getstate__()))
        new.__dict__["id"] = None
        subs = self.__dict__.get("_namespaces", {})
        twins = {id(sub): copy.copy(sub) for sub in subs.values()}
        swap = {id(self): new, **twins}
        for obj in (new, *twins.values()):
            for key, value in obj.__dict__.items():
                if isinstance(value, _Function):
                    value = copy.copy(value)
                    value.sources = [
                        (swap.get(id(comp), comp), name)
                        for comp, name in value.sources
                    ]
                elif isinstance(value, Component):
                    value = swap.get(id(value), value)
                obj.__dict__[key] = value
        new.__dict__.pop("_jit", None)
        if subs:
            new.__dict__["_namespaces"] = {
                name: twins[id(sub)] for name, sub in subs.items()
            }
            new._wrap()
        return new

    def _compiled(self):
        """
        Return the compiled profile of a wrapper.

        It is cached until a sub-profile, its profile function or its
        truncation is replaced.
        """
        key = [
            obj
            for sub in self._namespaces.values()
            for obj in (
                sub,
                sub.__dict__.get("profile"),
                getattr(sub, "truncation", None),
            )
        ]
        cached = self.__dict__.get("_jit")
        if (
            cached is None
            or len(cached[0]) != len(key)
            or any(a is not b for a, b in zip(cached[0], key))
        ):
            cached = (key, jax.jit(self.profile))
            self.__dict__["_jit"] = cached
        return cached[1]

    def _wrap(self):
        """Mark the sub-profiles as wrapped by this component."""
        for sub in self._namespaces.values():
            sub._wrapper = self

    def _sub(self, key):
        """Sub-profile holding a namespaced key and the key within it."""
        space, _, local = key.partition(".")
        sub = getattr(self, "_namespaces", {}).get(space)
        return (sub, local) if local and sub is not None else (None, key)

    def _adopt(self, hidden=()):
        """
        Add the sub-profiles' hyperparameters, units and descriptions.

        Keys are prefixed with their namespace; hidden names the sub-profile
        attributes holding keys to leave out.
        """
        for name, sub in self._namespaces.items():
            skip = [getattr(sub, attr) for attr in hidden]
            self.hyper += [f"{name}.{key}" for key in sub.hyper]
            for table in ("units", "description"):
                getattr(self, table).update(
                    {
                        f"{name}.{key}": value
                        for key, value in getattr(sub, table).items()
                        if key not in skip
                    }
                )

    def _retruncate(self, sub):
        """Follow a truncation added to or removed from a sub-profile."""
        for name, prof in self._namespaces.items():
            if prof is not sub:
                continue
            for key in ["rt", "wt"]:
                if key in sub.units:
                    self.units[f"{name}.{key}"] = sub.units[key]
                    self.description[f"{name}.{key}"] = sub.description[key]
                else:
                    self.units.pop(f"{name}.{key}", None)
                    self.description.pop(f"{name}.{key}", None)

    def __setattr__(self, name, value):
        """Prevent adding undefined attributes after initialization."""
        if getattr(self, "_initialized", False) and name in self.__dict__.get(
            "_namespaces", ()
        ):
            raise AttributeError(
                f"The {name} profile of a {type(self).__name__} cannot be "
                f"replaced: create a new {type(self).__name__} instead."
            )
        if getattr(self, "_initialized", False) and not hasattr(self, name):
            raise AttributeError(
                f"Cannot add new attribute '{name}' to "
                f"{self.__class__.__name__}. "
                f"Use addparameter() to add new parameters."
            )
        if name in _LOS_CHECKS:
            _LOS_CHECKS[name](value)
        if (
            callable(value)
            and not isinstance(
                value, (_TIED, numpyro.distributions.Distribution)
            )
            and not name.startswith("_")
            and name != "profile"
        ):
            raise TypeError(
                f"{name} cannot be set to {type(value).__name__} "
                f"{value!r}: tie it with socca.priors.function(func, "
                "(component, 'name'), ...) instead."
            )
        if name == "name":
            value = _check_name(value)
            if self.id is not None and value != self.name:
                raise ValueError(
                    f"This {type(self).__name__} is in a model as "
                    f"'{self.id}': rename a copy.copy of it instead."
                )
        super().__setattr__(name, value)

    #   Print model parameters
    #   --------------------------------------------------------
    def parameters(self):
        """
        Print the component's parameters and hyperparameters.

        Each line shows the parameter name, its units, its value (a fixed
        value, a prior with its range, or what a tie depends on) and its
        description. The title gives the component type and, once it is
        in a model, its id.

        Examples
        --------
        >>> from socca.models import Beta
        >>> beta = Beta(xc=180.5, yc=45.2, rc=0.01, Ic=100, beta=0.5)
        >>> beta.parameters()
        <BLANKLINE>
        Beta
        ====
        xc    [deg] : 1.8050E+02 | Right ascension of centroid
        ...
        """
        keyout = [key for key in self.units if key not in self.hyper]
        if not keyout:
            print("No parameters defined.")
            return
        lines = _format_rows(
            [
                (
                    key,
                    self._input_unit(key),
                    _format_value(_getdotted(self, key)),
                    self.description.get(key, ""),
                )
                for key in keyout + self.hyper
            ]
        )
        title = type(self).__name__
        if self.id is not None:
            title += f" ({self.id})"
        print(f"\n{title}\n" + "=" * len(title))
        print("\n".join(lines[: len(keyout)]))
        if self.hyper:
            print("\nHyperparameters\n" + "=" * 15)
            print("\n".join(lines[len(keyout) :]))

    def _copy(self):
        """
        Return a copy that keeps every parameter, unit and setting.

        Ties to the component itself point to the copy, and the copy has
        no id until it is added to a model.
        """
        new = copy.copy(self)
        state = {
            key: copy.copy(value) if isinstance(value, (dict, list)) else value
            for key, value in self.__dict__.items()
        }
        state = {
            key: _repoint(value, self, new) for key, value in state.items()
        }
        state["id"] = None
        state.pop("_wrapper", None)
        state.pop("_truncation", None)  # the copy has its own
        new.__dict__.update(state)
        return new

    def _resolve(self, attr, _seen=frozenset()):
        """
        Return the value of a (possibly dotted) parameter, resolving ties.

        Used by standalone getmap() calls, where there is no model
        parameter dict: function() and boundto() follow their source
        objects, and a lambda's arguments must name parameters of this
        component. Values are in declared units (see set_units):
        functions take and return declared units, and boundto copies the
        physical value.
        """
        value = _getdotted(self, attr)
        if not isinstance(value, _TIED):
            return value

        tag = (id(self), attr)
        if tag in _seen:
            raise ValueError(f"Circular tie involving {attr}.")
        _seen = _seen | {tag}

        sources = getattr(value, "sources", [])
        args = [comp._resolve(name, _seen) for comp, name in sources]
        for arg, (comp, name) in zip(args, sources):
            if isinstance(arg, numpyro.distributions.Distribution):
                raise ValueError(
                    f"{attr} is tied to {_source_label(comp, name)}, which "
                    "has a prior: standalone getmap() needs fixed values. "
                    "Use Model.getmodel() with sampled values instead."
                )
        if isinstance(value, _BoundTo):
            comp, name = sources[0]
            return args[0] * comp._factor(name) / self._factor(attr)
        if isinstance(value, _Function):
            if value.native:
                args = [
                    arg * comp._factor(name)
                    for arg, (comp, name) in zip(args, value.sources)
                ]
                return value(*args) / self._factor(attr)
            return value(*args)
        keys = [
            self._tiearg(name) for name in inspect.signature(value).parameters
        ]
        args = [self._resolve(key, _seen) for key in keys]
        return value(*args)

    def _tiearg(self, name):
        """Map a lambda argument name to one of this component's parameters."""
        cid = self.id or self.name
        if cid is None:
            raise ValueError(
                f"Cannot resolve the lambda argument '{name}': this "
                f"{type(self).__name__} is not in a model, so it has no id. "
                "Use socca.priors.function() instead."
            )
        local = re.sub(rf"^{re.escape(cid)}_", "", name)
        for key in self.units:
            if local in (key, key.replace(".", "_")) and local != name:
                return key
        raise ValueError(
            f"Tie argument '{name}' is not a parameter of this "
            f"{type(self).__name__} ({cid}). Standalone getmap() can "
            "only resolve ties to the component's own parameters; add the "
            "components to a Model and use Model.getmap() instead."
        )

    def _keymap(self):
        """Map the keywords of _evaluate to parameter names."""
        raise NotImplementedError

    def _build_kwargs(self, pars, comp_prefix):
        """Keywords for _evaluate from the model's parameter values."""
        return {
            out: pars[f"{comp_prefix}_{key}"]
            for out, key in self._keymap().items()
        }

    @staticmethod
    def _convolve(img, mgrid, convolve):
        """Convolve with the image PSF if asked and if there is one."""
        if not convolve:
            return mgrid
        if img.psf is None:
            warnings.warn(
                "No PSF defined, so no convolution will be performed.",
                stacklevel=3,
            )
            return mgrid
        return img.convolve(mgrid)

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
        for out, key in keys.items():
            factor = self._factor(key)
            if factor != 1.0:
                kwarg[out] = kwarg[out] * factor
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

    def _input_unit(self, key):
        """Return the declared unit of a (dotted) parameter, else native."""
        if key in self._input_units:
            return self._input_units[key]
        sub, local = self._sub(key)
        if sub is not None:
            return sub._input_unit(local)
        return self.units.get(key, "")

    def _factor(self, key):
        """Return the factor from a parameter's declared to native unit."""
        native = self.units.get(key)
        unit = self._input_unit(key)
        if native is None or unit == native:
            return 1.0
        return _cfactor(unit, native)

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
            sub, local = self._sub(name)
            if sub is not None:
                sub.set_units(**{local: unit_str})
                continue
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
        if not name.isidentifier():
            raise ValueError(
                f"Invalid parameter name {name!r}: use a Python identifier "
                "(dotted names are reserved for sub-profiles)."
            )
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
        if name in ("rt", "wt") and getattr(self, "truncation", None):
            raise ValueError(
                f"'{name}' belongs to the truncation: set truncation = None "
                "to remove it."
            )
        self._initialized = False
        delattr(self, name)
        self._initialized = True
        self.units.pop(name, None)
        self._input_units.pop(name, None)
        self.description.pop(name, None)
        self.hyper = [h for h in self.hyper if h != name]
