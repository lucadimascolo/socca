"""Prior distributions for Bayesian inference."""

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
        If low >= high.
    """
    if low >= high:
        message = "The lower limit must be smaller than the upper limit."
        message = message + f"\nInput values > low={low} high={high}"
        raise ValueError(message)
    return numpyro.distributions.Uniform(low, high)


# Log-uniform
# --------------------------------------------------------
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

    def cdf(self, x):
        """
        Compute the cumulative distribution function.

        Parameters
        ----------
        x : float or array_like
            Value(s) at which to evaluate the CDF.

        Returns
        -------
        float or array_like
            Probability that a draw is <= x.
        """
        norm = self.losig + self.hisig
        lo = 2.0 * self.losig / norm * jax.scipy.special.ndtr(x / self.losig)
        hi = self.losig / norm + 2.0 * self.hisig / norm * (
            jax.scipy.special.ndtr(x / self.hisig) - 0.5
        )
        return jp.where(x < 0.0, lo, hi)

    def icdf(self, u):
        """
        Compute the inverse CDF (quantile function).

        Parameters
        ----------
        u : float or array_like
            Uniform random variable(s) in [0, 1].

        Returns
        -------
        float or array_like
            Corresponding sample(s).
        """
        norm = self.losig + self.hisig
        low = u < self.losig / norm
        qlo = jp.where(low, 0.5 * u * norm / self.losig, 0.25)
        qhi = jp.where(low, 0.75, 1.0 - 0.5 * (1.0 - u) * norm / self.hisig)
        return jp.where(
            low,
            self.losig * jax.scipy.special.ndtri(qlo),
            self.hisig * jax.scipy.special.ndtri(qhi),
        )

    def sample(self, key, sample_shape=()):
        """
        Draw random samples from the distribution.

        Parameters
        ----------
        key : jax.random.PRNGKey
            JAX random key.
        sample_shape : tuple, optional
            Shape of the sample batch.

        Returns
        -------
        array_like
            Random samples.
        """
        u = jax.random.uniform(key, sample_shape + self.batch_shape)
        return self.icdf(u)


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


def _check_limits(low, high):
    lo, hi = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    if not np.all((0 < lo) & (lo < hi) & np.isfinite(hi)):
        raise ValueError(
            "The limits must satisfy 0 < low < high < inf."
            f"\nInput values > low={low} high={high}"
        )


class _PowerLawBase(numpyro.distributions.Distribution):
    """
    Power law p(x) ∝ x^alpha between two bounds.

    The normalisation, cdf and inverse cdf are computed in log space and
    anchored at the bound that keeps them finite for any alpha.
    """

    reparametrized_params = ["alpha", "low", "high"]

    def __init__(self, alpha, low, high, validate_args=None):
        self.alpha, self.low, self.high = (
            numpyro.distributions.util.promote_shapes(alpha, low, high)
        )
        batch_shape = jp.broadcast_shapes(
            jp.shape(self.alpha),
            jp.shape(self.low),
            jp.shape(self.high),
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
        return numpyro.distributions.constraints.interval(self.low, self.high)

    def _terms(self):
        la, lb = self._log_bounds()
        beta = jp.asarray(self.alpha + 1.0)
        flat = beta == 0.0
        beta = jp.where(flat, 1.0, beta)
        c = jp.abs(beta) * (lb - la)
        anchor = jp.where(beta > 0, lb, la)
        return la, lb, flat, beta, c, -jp.expm1(-c), anchor

    def _log_norm(self):
        la, lb, flat, beta, c, q, _ = self._terms()
        return jp.where(
            flat,
            jp.log(lb - la),
            jp.maximum(beta * la, beta * lb)
            + jp.log(q)
            - jp.log(jp.abs(beta)),
        )

    def _cdf(self, lx):
        la, lb, flat, beta, c, q, anchor = self._terms()
        lx = jp.clip(lx, la, lb)
        d = beta * (lx - anchor)
        t = -jp.exp(d) * jp.expm1(-(c + d)) / q
        return jp.where(
            flat, (lx - la) / (lb - la), jp.where(beta > 0, t, 1.0 - t)
        )

    def _log_icdf(self, u):
        la, lb, flat, beta, c, q, anchor = self._terms()
        w = jp.where(beta > 0, u, 1.0 - u)
        cs = jp.minimum(c, 1.0)
        lw = jp.where(
            c < 1.0,
            jp.log1p(w * jp.expm1(cs)) - cs,
            jp.log(jp.exp(-c) + w * q),
        )
        return jp.where(flat, la + u * (lb - la), anchor + lw / beta)

    def sample(self, key, sample_shape=()):
        """
        Draw random samples from the distribution.

        Parameters
        ----------
        key : jax.random.PRNGKey
            JAX random key.
        sample_shape : tuple, optional
            Shape of the sample batch.

        Returns
        -------
        array_like
            Random samples.
        """
        u = jax.random.uniform(key, sample_shape + self.batch_shape)
        return self.icdf(u)


class LogPowerLaw(_PowerLawBase):
    """
    Power-law prior for a parameter sampled in log space.

    The sampled variable is theta = log(x), with density
    p(theta) ∝ exp((alpha + 1) * theta) for theta in [low, high], where
    (alpha + 1) accounts for the Jacobian of the log transform, so that
    the implied prior on x is p(x) ∝ x^alpha. Samplers work in theta;
    the model receives x = exp(theta) (see `to_model`), and fit results
    are reported in x. Quantities defined on the sampled variable, such as
    the prior-deboosted evidence from `getzprior` or the optimizer's MAP
    estimate, refer to theta.

    Parameters
    ----------
    alpha : float
        Power-law index in linear space.
        alpha = -1 : log-uniform (flat in log space)
        alpha =  0 : flat/uniform in linear space
        alpha = +1 : linear prior (favours large R_e)
    low : float
        Lower bound in log space
    high : float
        Upper bound in log space
    """

    arg_constraints = {
        "alpha": numpyro.distributions.constraints.real,
        "low": numpyro.distributions.constraints.real,
        "high": numpyro.distributions.constraints.real,
    }

    def _log_bounds(self):
        return self.low, self.high

    @staticmethod
    def to_model(theta):
        """Map a sampled value theta = log(x) to the model value x."""
        return jp.exp(theta)

    @staticmethod
    def from_model(x):
        """Map a model value x to the sampled value theta = log(x)."""
        return jp.log(x)

    def model_bounds(self):
        """Support bounds in model space, as given to `logpowerlaw`."""
        return getattr(self, "_xbounds", (jp.exp(self.low), jp.exp(self.high)))

    def log_prob(self, theta):
        """
        Compute log probability density in log space.

        Parameters
        ----------
        theta : float or array_like
            Value(s) in log space at which to evaluate the log probability.

        Returns
        -------
        float or array_like
            Log probability density at theta.
        """
        return jp.where(
            (theta >= self.low) & (theta <= self.high),
            (self.alpha + 1.0) * theta - self._log_norm(),
            -jp.inf,
        )

    def cdf(self, theta):
        """
        Compute the cumulative distribution function in log space.

        Parameters
        ----------
        theta : float or array_like
            Value(s) in log space at which to evaluate the CDF.

        Returns
        -------
        float or array_like
            Probability that a draw is <= theta.
        """
        return self._cdf(theta)

    def icdf(self, u):
        """
        Compute the inverse CDF (quantile function) in log space.

        Parameters
        ----------
        u : float or array_like
            Uniform random variable(s) in [0, 1].

        Returns
        -------
        float or array_like
            Corresponding sample(s) in log space.
        """
        return jp.clip(self._log_icdf(u), self.low, self.high)


def logpowerlaw(alpha, low, high):
    """
    Create a power-law prior for a parameter sampled in log space.

    Parameters
    ----------
    alpha : float
        Power-law index in linear space.
    low : float
        Lower bound in linear space (log-transformed internally).
    high : float
        Upper bound in linear space (log-transformed internally).

    Returns
    -------
    LogPowerLaw
        Power-law distribution sampled in log space; fit results are
        reported in linear space.

    Raises
    ------
    ValueError
        If the limits do not satisfy 0 < low < high < inf.
    """
    _check_limits(low, high)
    prior = LogPowerLaw(alpha, low=jp.log(low), high=jp.log(high))
    prior._xbounds = (low, high)
    return prior


class PowerLaw(_PowerLawBase):
    """
    Power-law prior in linear space.

    p(x) ∝ x^alpha for x in [low, high]

    Parameters
    ----------
    alpha : float
        Power-law index.
        alpha =  0 : flat/uniform
        alpha = +1 : linear (favours large values)
        alpha = -1 : log-uniform
    low : float
        Lower bound in linear space
    high : float
        Upper bound in linear space
    """

    arg_constraints = {
        "alpha": numpyro.distributions.constraints.real,
        "low": numpyro.distributions.constraints.positive,
        "high": numpyro.distributions.constraints.positive,
    }

    def _log_bounds(self):
        return jp.log(self.low), jp.log(self.high)

    def log_prob(self, x):
        """
        Compute log probability density in linear space.

        Parameters
        ----------
        x : float or array_like
            Value(s) at which to evaluate the log probability.

        Returns
        -------
        float or array_like
            Log probability density at x.
        """
        return jp.where(
            (x >= self.low) & (x <= self.high),
            self.alpha * jp.log(x) - self._log_norm(),
            -jp.inf,
        )

    def cdf(self, x):
        """
        Compute the cumulative distribution function.

        Parameters
        ----------
        x : float or array_like
            Value(s) at which to evaluate the CDF.

        Returns
        -------
        float or array_like
            Probability that a draw is <= x.
        """
        lx = jp.log(jp.where(x > 0, x, 1.0))
        return self._cdf(jp.where(x > 0, lx, -jp.inf))

    def icdf(self, u):
        """
        Compute the inverse CDF (quantile function).

        Parameters
        ----------
        u : float or array_like
            Uniform random variable(s) in [0, 1].

        Returns
        -------
        float or array_like
            Corresponding sample(s) in linear space.
        """
        return jp.clip(jp.exp(self._log_icdf(u)), self.low, self.high)


def powerlaw(alpha, low, high):
    """
    Create a power-law prior distribution in linear space.

    Parameters
    ----------
    alpha : float
        Power-law index.
    low : float
        Lower bound in linear space.
    high : float
        Upper bound in linear space.

    Returns
    -------
    PowerLaw
        Power-law distribution in linear space.

    Raises
    ------
    ValueError
        If the limits do not satisfy 0 < low < high < inf.
    """
    _check_limits(low, high)
    return PowerLaw(alpha, low=low, high=high)


# Parameter bound to another component's parameter
# --------------------------------------------------------
class _BoundTo:
    """
    Internal marker tying one component's parameter to another's.

    Not part of the public API -- construct these via boundto(), never
    directly. Namespaced parameter keys (e.g. "comp_00_radial.xc") contain a
    literal dot and can't be used as a Python identifier, so a lambda can't
    represent the binding; this class carries the resolved key instead.
    """

    def __init__(self, key, comp, attr):
        self.key = key
        self._comp = comp
        self._attr = attr

    def __call__(self, value):
        """Return the bound value unchanged, matching the lambda it replaces."""
        return value

    def resolve(self):
        """
        Return the current value of the referenced parameter directly.

        Model.getmap()/getmodel() resolve ties by looking `key` up in
        their flattened parameter dict. A single component's own
        getmap(), called outside of a Model, has no such dict -- this
        walks the referenced component/attribute directly instead.
        Resolves recursively if the referenced value is itself tied.
        """
        value = self._comp
        for part in self._attr.split("."):
            value = getattr(value, part)
        if isinstance(value, _BoundTo):
            value = value.resolve()
        return value


def boundto(comp, var):
    """
    Bind a parameter to another component's parameter.

    Creates a functional relationship where one component's parameter is
    constrained to equal another component's parameter. Useful for tying
    parameters together across multiple model components.

    Parameters
    ----------
    comp : Component or str
        Component object or string representation of component whose
        parameter should be referenced. If string, it is evaluated.
    var : str
        Name of the parameter variable to bind to.

    Returns
    -------
    callable
        An opaque, callable object that takes the bound parameter and
        returns its value unchanged.

    Examples
    --------
    Bind the x-coordinate of component2 to match component1's x-coordinate:
    >>> comp2.xc = boundto(comp1, 'xc')

    Bind the inclination and position angle of an Ellipsoid (e.g. a
    galactic bar) to that of its host Disk:
    >>> disk_radial = socca.models.Sersic()
    >>> disk_radial.theta = np.pi/4
    >>> disk_vertical = socca.models.disk.vertical.HyperSecantHeight()
    >>> disk_vertical.inc = np.radians(80)
    >>>
    >>> disk = socca.models.Disk(radial=disk_radial, vertical=disk_vertical)
    >>>
    >>> bar = socca.models.Ellipsoid(radial=socca.models.Sersic())
    >>> bar.theta = socca.priors.boundto(disk, "theta")
    >>> bar.inc   = socca.priors.boundto(disk, "inc")

    Notes
    -----
    Always pass the top-level component, not its sub-components. For components
    with namespaced sub-components (Disk, Ellipsoid), the correct sub-component
    is found automatically by searching the component's own exposed parameter
    names (its units dict) -- not the sub-component's, since a sub-component
    may have attributes the parent doesn't actually expose as a parameter
    (e.g. Ellipsoid.radial is a full Sersic profile, but Ellipsoid keeps
    xc/yc/theta/e on itself rather than delegating to radial).

    Tied parameters resolve correctly both through Model.getmap()/
    getmodel() and when calling a single component's own getmap()
    directly (outside of a Model) -- the latter reads the referenced
    component/attribute directly, since it has no flattened parameter
    dict to resolve `key` against.
    """
    if isinstance(comp, str):
        comp = eval(comp)

    namespaces = getattr(comp, "_namespaces", None)
    if namespaces:
        for namespace in namespaces:
            if f"{namespace}.{var}" in comp.units:
                attr = f"{namespace}.{var}"
                return _BoundTo(f"{comp.id}_{attr}", comp, attr)

    return _BoundTo(f"{comp.id}_{var}", comp, var)


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
