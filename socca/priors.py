"""Prior distributions for Bayesian inference."""

import inspect
import types

import jax
import jax.numpy as jp
import numpy as np

import numpyro.distributions


# Uniform
# --------------------------------------------------------
def uniform(low, high):
    """
    Create a uniform prior distribution with validation.

    Constructs a uniform distribution over the interval [low, high] with
    automatic validation that the lower bound is less than the upper bound.

    Parameters
    ----------
    low : float
        Lower bound of the uniform distribution.
    high : float
        Upper bound of the uniform distribution.

    Returns
    -------
    numpyro.distributions.Uniform
        Uniform distribution object.

    Raises
    ------
    ValueError
        If a limit is not finite, or low >= high.
    """
    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError(
            f"The limits must be finite, not low={low} high={high}."
        )
    if low >= high:
        message = "The lower limit must be smaller than the upper limit."
        message = message + f"\nInput values > low={low} high={high}"
        raise ValueError(message)
    return numpyro.distributions.Uniform(low, high)


# Log-uniform
# --------------------------------------------------------
class LogPowerLaw:
    """Removed prior, kept so that its pickles raise a clear error."""

    def __setstate__(self, state):
        """Refuse the pickle, saved by socca 1.4.0 or earlier."""
        from .models.base import _too_old

        raise _too_old("LogPowerLaw prior")


def loguniform(low, high):
    """
    Create a log-uniform prior distribution with validation.

    Constructs a log-uniform distribution over the interval [low, high],
    where log(x) is uniformly distributed. Useful for parameters that span
    many orders of magnitude.

    Parameters
    ----------
    low : float
        Lower bound of the log-uniform distribution (must be positive).
    high : float
        Upper bound of the log-uniform distribution.

    Returns
    -------
    numpyro.distributions.LogUniform
        Log-uniform distribution object.

    Raises
    ------
    ValueError
        If low >= high.
    """
    if low >= high:
        message = "The lower limit must be smaller than the upper limit."
        message = message + f"\nInput values > low={low} high={high}"
        raise ValueError(message)
    return numpyro.distributions.LogUniform(low, high)


# Normal
# --------------------------------------------------------
def normal(loc, scale):
    """
    Create a normal (Gaussian) prior distribution.

    Constructs a normal distribution with specified mean and standard
    deviation. Commonly used for parameters with expected values and
    uncertainties.

    Parameters
    ----------
    loc : float
        Mean (location parameter) of the normal distribution.
    scale : float
        Standard deviation (scale parameter) of the normal distribution.

    Returns
    -------
    numpyro.distributions.Normal
        Normal distribution object.
    """
    return numpyro.distributions.Normal(loc, scale)


# Split-normal
# --------------------------------------------------------
class SplitNormal(numpyro.distributions.Distribution):
    """
    Split Normal (two-piece normal) distribution centered at 0.

    Parameters
    ----------
    losig : scale for x < 0
    hisig : scale for x >= 0
    """

    arg_constraints = {
        "losig": numpyro.distributions.constraints.positive,
        "hisig": numpyro.distributions.constraints.positive,
    }
    support = numpyro.distributions.constraints.real
    reparametrized_params = ["losig", "hisig"]

    def __init__(self, losig, hisig, validate_args=None):
        self.losig, self.hisig = numpyro.distributions.util.promote_shapes(
            losig, hisig
        )
        batch_shape = jp.broadcast_shapes(
            jp.shape(self.losig),
            jp.shape(self.hisig),
        )
        super().__init__(
            batch_shape=batch_shape,
            event_shape=(),
            validate_args=validate_args,
        )

    def log_prob(self, x):
        """
        Compute log probability density for split normal distribution.

        Evaluates the log probability using different scales for positive
        and negative values, with proper normalization to ensure the
        distribution integrates to 1.

        Parameters
        ----------
        x : float or array_like
            Value(s) at which to evaluate the log probability.

        Returns
        -------
        float or array_like
            Log probability density at x.
        """
        losig = self.losig
        hisig = self.hisig

        sig_prob = jp.where(x >= 0.0, hisig, losig)
        log_norm = jp.log(jp.sqrt(2.0 / jp.pi)) - jp.log(losig + hisig)
        return log_norm - 0.5 * (x / sig_prob) ** 2


def splitnormal(loc, losig, hisig):
    """
    Create a split normal distribution centered at a specified location.

    Constructs an asymmetric normal distribution with different standard
    deviations for values below and above the center. Useful for modeling
    parameters with asymmetric uncertainties.

    Parameters
    ----------
    loc : float
        Center (location) of the split normal distribution.
    losig : float
        Standard deviation for values below the center (x < loc).
    hisig : float
        Standard deviation for values above the center (x >= loc).

    Returns
    -------
    numpyro.distributions.TransformedDistribution
        Split normal distribution centered at `loc`.

    Notes
    -----
    The distribution is constructed by applying an affine transformation
    to shift the base SplitNormal (centered at 0) to the desired location.
    """
    return numpyro.distributions.TransformedDistribution(
        base_distribution=SplitNormal(losig, hisig),
        transforms=[
            numpyro.distributions.transforms.AffineTransform(loc, 1.0)
        ],
    )


class PowerLaw(numpyro.distributions.Distribution):
    """
    Power-law prior, p(x) ∝ x^alpha for x in [low, high].

    Computed in log space, so steep slopes and narrow ranges do not
    overflow or lose precision; alpha = -1 is log-uniform.

    Parameters
    ----------
    alpha : float
        Power-law index (0: uniform, -1: log-uniform, 1: linear).
    low : float
        Lower bound, positive.
    high : float
        Upper bound, greater than low.
    """

    arg_constraints = {
        "alpha": numpyro.distributions.constraints.real,
        "low": numpyro.distributions.constraints.positive,
        "high": numpyro.distributions.constraints.positive,
    }
    reparametrized_params = ["alpha", "low", "high"]

    def __init__(self, alpha, low, high, validate_args=None):
        self.alpha, self.low, self.high = (
            numpyro.distributions.util.promote_shapes(alpha, low, high)
        )
        batch_shape = jp.broadcast_shapes(
            jp.shape(self.alpha), jp.shape(self.low), jp.shape(self.high)
        )
        super().__init__(
            batch_shape=batch_shape,
            event_shape=(),
            validate_args=validate_args,
        )

    @numpyro.distributions.constraints.dependent_property(
        is_discrete=False, event_dim=0
    )
    def support(self):  # noqa: D102
        # built on demand: unflattened copies (jax pytrees) skip __init__
        return numpyro.distributions.constraints.interval(self.low, self.high)

    def log_prob(self, x):
        """
        Log probability density.

        Parameters
        ----------
        x : float or array_like
            Values at which to evaluate the density.

        Returns
        -------
        float or array_like
            Log density, -inf outside [low, high].
        """
        beta, _, _, loglow, span = _powerlaw_terms(
            self.alpha, self.low, self.high
        )
        small = jp.abs(beta * span) < 1.0e-03
        b = jp.where(small, 1.0, beta)
        exact = (
            jp.maximum(b * loglow, b * (loglow + span))
            + jp.log(-jp.expm1(-jp.abs(b) * span))
            - jp.log(jp.abs(b))
        )
        series = (
            jp.log(span) + beta * loglow + jp.log(_expm1_ratio(beta * span))
        )
        lognorm = jp.where(small, series, exact)
        inside = (x >= self.low) & (x <= self.high)
        xsafe = jp.where(inside, x, self.low)
        return jp.where(inside, self.alpha * jp.log(xsafe) - lognorm, -jp.inf)

    def cdf(self, x):
        """
        Cumulative distribution function.

        Parameters
        ----------
        x : float or array_like
            Values at which to evaluate the CDF.

        Returns
        -------
        float or array_like
            Probability of a value not larger than x.
        """
        x = jp.clip(x, self.low, self.high)
        return _powerlaw_cdf(x, self.alpha, self.low, self.high, upper=False)

    def icdf(self, u):
        """
        Inverse of the cumulative distribution function.

        Parameters
        ----------
        u : float or array_like
            Probabilities in [0, 1].

        Returns
        -------
        float or array_like
            Values in [low, high].
        """
        return _powerlaw_icdf(u, self.alpha, self.low, self.high)

    def sample(self, key, sample_shape=()):
        """
        Draw random samples.

        Parameters
        ----------
        key : jax.random.PRNGKey
            JAX random key.
        sample_shape : tuple, optional
            Shape of the sample batch.

        Returns
        -------
        array_like
            Samples in [low, high].
        """
        u = jax.random.uniform(key, sample_shape + self.batch_shape)
        return self.icdf(u)


def _expm1_ratio(z):
    """expm1(z) / z for small z."""
    return 1.0 + z / 2.0 + z**2 / 6.0 + z**3 / 24.0


def _powerlaw_terms(alpha, low, high):
    """Return beta = alpha + 1, beta != 0, |beta| (1 where 0), log low, log range."""
    beta = alpha + 1.0
    nonzero = beta != 0
    safe = jp.where(nonzero, jp.abs(beta), 1.0)
    loglow = jp.log(low)
    return beta, nonzero, safe, loglow, jp.log(high) - loglow


@jax.custom_jvp
def _powerlaw_icdf(u, alpha, low, high):
    """PowerLaw icdf, with exact derivatives also at u = 0 and 1."""
    beta, nonzero, safe, loglow, span = _powerlaw_terms(alpha, low, high)
    decay = -safe * span
    small = decay > -1.0
    rising = (
        loglow
        + span
        + jp.where(
            small,
            jp.log1p((1.0 - u) * jp.expm1(decay)),
            jp.logaddexp(jp.log(u), jp.log1p(-u) + decay),
        )
        / safe
    )
    falling = (
        loglow
        - jp.where(
            small,
            jp.log1p(u * jp.expm1(decay)),
            jp.logaddexp(jp.log1p(-u), jp.log(u) + decay),
        )
        / safe
    )
    logx = jp.where(
        nonzero, jp.where(beta > 0, rising, falling), loglow + u * span
    )
    return jp.clip(jp.exp(logx), low, high)


def _powerlaw_cdf(x, alpha, low, high, upper):
    """
    PowerLaw cdf without clipping, also used for the icdf derivatives.

    Smooth in alpha across alpha = -1; with upper, 1 - (1 - cdf) is used,
    which keeps the derivatives accurate near x = high.
    """
    beta = alpha + 1.0
    t = jp.log(x) - jp.log(low)
    span = jp.log(high) - jp.log(low)
    small = jp.abs(beta * span) < 1.0e-03
    b = jp.where(small, 1.0, beta)
    s = jp.abs(b)  # exponents kept <= 0, so neither branch overflows
    norm = -jp.expm1(-s * span)
    near = -jp.expm1(-s * t)
    far = -jp.expm1(-s * (span - t))
    lower = jp.where(b > 0, jp.exp(-s * (span - t)) * near, near)
    above = jp.where(b > 0, far, jp.exp(-s * t) * far)
    exact = jp.where(upper, 1.0 - above / norm, lower / norm)
    series = t / span * _expm1_ratio(beta * t) / _expm1_ratio(beta * span)
    return jp.where(small, series, exact)


@_powerlaw_icdf.defjvp
def _powerlaw_icdf_jvp(primals, tangents):
    # implicit derivative of cdf(x) = u: dx = (du - dcdf/dpars) / pdf(x)
    u, alpha, low, high = primals
    du, dalpha, dlow, dhigh = tangents
    x = _powerlaw_icdf(u, alpha, low, high)
    _, dcdf = jax.jvp(
        lambda a, lo, hi: _powerlaw_cdf(x, a, lo, hi, u >= 0.5),
        (alpha, low, high),
        (dalpha, dlow, dhigh),
    )
    pdf = jp.exp(PowerLaw(alpha, low, high).log_prob(x))
    return x, (du - dcdf) / pdf


def powerlaw(alpha, low, high):
    """
    Create a power-law prior, p(x) ∝ x^alpha on [low, high].

    Parameters
    ----------
    alpha : float
        Power-law index (0: uniform, -1: log-uniform, 1: linear).
    low : float
        Lower bound, positive.
    high : float
        Upper bound, finite and greater than low.

    Returns
    -------
    PowerLaw
        The power-law distribution.

    Raises
    ------
    ValueError
        If alpha is not finite, or the bounds do not satisfy
        0 < low < high < inf.
    """
    if not np.all(np.isfinite(alpha)):
        raise ValueError("The power-law index must be finite.")
    low_, high_ = np.asarray(low, float), np.asarray(high, float)
    if not (
        np.all(low_ > 0)
        and np.all(low_ < high_)
        and np.all(np.isfinite(high_))
    ):
        raise ValueError("Power-law bounds must satisfy 0 < low < high < inf.")
    return PowerLaw(alpha, low=low, high=high)


# Tied parameters
# --------------------------------------------------------
def _identity(value):
    return value


def _checked(comp, var):
    """Reject parameters a wrapper computes internally (Bridge radii)."""
    obj = comp
    for part in var.split(".")[:-1]:
        obj = getattr(obj, part)
    value = getattr(obj, var.split(".")[-1])
    if isinstance(value, _Function) and value.native:
        owner = getattr(obj, "_wrapper", None) or comp
        raise ValueError(
            f"{type(obj).__name__}.{var.split('.')[-1]} is computed by its "
            f"{type(owner).__name__} from its own parameters (rs, e): tie "
            f"to those of the {type(owner).__name__} instead."
        )
    return comp, var


def _source(comp, var):
    """Return (component, parameter name) for a tie source, checking it."""
    units = getattr(comp, "units", None)
    if not isinstance(units, dict):
        raise TypeError(
            f"Expected a model component, got {type(comp).__name__}."
        )
    if var in units:
        return _checked(comp, var)
    found = [
        f"{name}.{var}"
        for name in getattr(comp, "_namespaces", {})
        if f"{name}.{var}" in units
    ]
    if len(found) > 1:
        raise ValueError(
            f"'{var}' is ambiguous for {type(comp).__name__}: use one of "
            f"{', '.join(repr(f) for f in found)}."
        )
    if not found:
        raise ValueError(f"{type(comp).__name__} has no parameter '{var}'.")
    return _checked(comp, found[0])


class _Function:
    """
    A parameter computed from other parameters, built by function().

    sources holds (component, parameter) pairs, passed to func in order.
    Arguments and result are in declared units, unless native is set.
    """

    native = False

    def __init__(self, func, sources):
        self.func = func
        self.sources = sources

    def __call__(self, *args):
        return self.func(*args)


class _BoundTo(_Function):
    """A parameter equal to another parameter, built by boundto()."""

    def __init__(self, comp, var):
        super().__init__(_identity, [_source(comp, var)])


_TIED = (types.FunctionType, _Function)


def function(func, *sources):
    """
    Compute a parameter from other parameters.

    Parameters
    ----------
    func : callable
        Function of the source parameters, taking them positionally in
        the order given. Arguments and result are in declared units (see
        Component.set_units).
    *sources : tuple of (Component, str)
        The parameters passed to func, as (component, name) pairs. The
        component can be the one being set, any other component, or a
        sub-profile of a Disk, Ellipsoid or Bridge (e.g.
        ``(disk.radial, "re")``). On Disk, Ellipsoid and Bridge a bare
        name is looked up in the sub-profiles (``(disk, "re")`` is
        ``disk.radial.re``).

    Returns
    -------
    callable
        An object marking the parameter as tied.

    Raises
    ------
    ValueError
        If a source names no parameter, or a bare name matches more than
        one sub-profile.
    TypeError
        If func cannot take as many arguments as there are sources.

    Examples
    --------
    >>> from socca.models import Sersic
    >>> from socca.priors import function
    >>> comp1, comp2 = Sersic(), Sersic()
    >>> comp2.addparameter("re_scale", 2.0)
    >>> comp2.re = function(
    ...     lambda s, r: s * r, (comp2, "re_scale"), (comp1, "re")
    ... )
    """
    if not callable(func):
        raise TypeError("func must be callable.")
    for source in sources:
        if not (isinstance(source, tuple) and len(source) == 2):
            raise TypeError(
                "Each source must be a (component, parameter name) pair."
            )
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        try:
            signature.bind(*sources)
        except TypeError:
            raise TypeError(
                f"func cannot take {len(sources)} positional argument(s), "
                "one per source."
            ) from None
    return _Function(func, [_source(comp, var) for comp, var in sources])


def boundto(comp, var):
    """
    Set a parameter equal to another parameter.

    The physical value is copied: unit conversions between the declared
    units of the two parameters are applied (see Component.set_units).

    Parameters
    ----------
    comp : Component
        Component holding the parameter: another component or the one
        being set. On Disk, Ellipsoid and Bridge a bare name is looked up
        in the sub-profiles, and a sub-profile (e.g. ``disk.radial``) can
        be passed directly.
    var : str
        Name of the parameter.

    Returns
    -------
    callable
        An object marking the parameter as tied.

    Raises
    ------
    ValueError
        If var names no parameter, or matches more than one sub-profile.

    See Also
    --------
    function : compute a parameter from other parameters.

    Examples
    --------
    >>> import socca
    >>> from socca.priors import boundto
    >>> comp1, comp2 = socca.models.Sersic(), socca.models.Sersic()
    >>> comp2.xc = boundto(comp1, "xc")

    Tie a bar's orientation to its host disk:

    >>> disk = socca.models.Disk()
    >>> bar = socca.models.Ellipsoid(radial=socca.models.Sersic())
    >>> bar.theta = socca.priors.boundto(disk, "theta")
    >>> bar.inc = socca.priors.boundto(disk, "inc")
    """
    return _BoundTo(comp, var)


# pocoMC prior refactor for numpyro
# --------------------------------------------------------
class pocomcPrior:
    """
    Prior distribution adapter for pocoMC sampler.

    Wraps numpyro distributions to provide the interface required by the
    pocoMC sampler, including log probability density, sampling, and bounds.
    """

    def __init__(self, dists, seed=0):
        self.dists = dists
        self.key = jax.random.PRNGKey(seed)

    def logpdf(self, x):
        """
        Compute log probability density for the joint prior distribution.

        Evaluates the sum of log probabilities across all parameter dimensions,
        assuming independence between parameters.

        Parameters
        ----------
        x : array_like, shape (n_samples, n_dims)
            Sample points at which to evaluate the log probability density.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Log probability density for each sample.
        """
        logp = np.zeros(len(x))
        for i, dist in enumerate(self.dists):
            logp += dist.log_prob(x[:, i])
        return logp

    def rvs(self, size=1):
        """
        Generate random samples from the prior distribution.

        Draws independent samples from each parameter's prior distribution
        and stacks them into a 2D array.

        Parameters
        ----------
        size : int, optional
            Number of samples to generate. Default is 1.

        Returns
        -------
        jax.numpy.ndarray, shape (size, n_dims)
            Array of random samples, where each column corresponds to
            a parameter dimension.

        Notes
        -----
        Uses JAX's PRNG system with automatic key splitting for
        reproducible random number generation.
        """
        keys = jax.random.split(self.key, len(self.dists) + 1)
        self.key, keys = keys[0], keys[1:]

        samples = [d.sample(k, (size,)) for d, k in zip(self.dists, keys)]
        return jp.column_stack(samples)

    @property
    def bounds(self):
        """
        Get parameter bounds from prior distributions.

        Returns
        -------
        numpy.ndarray, shape (n_dims, 2)
            Array of [lower, upper] bounds for each parameter.
        """
        bounds = []
        for dist in self.dists:
            bounds.append([float(dist.icdf(0.00)), float(dist.icdf(1.00))])
        return np.array(bounds)

    @property
    def dim(self):
        """
        Get number of parameter dimensions.

        Returns
        -------
        int
            Number of parameters in the prior distribution.
        """
        return len(self.dists)
