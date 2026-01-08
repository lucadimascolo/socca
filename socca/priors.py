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
    if low > high:
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
    if low > high:
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


# Parameter bound to another component's parameter
# --------------------------------------------------------
def boundto(comp, var):
    """
    Create a lambda function to bind a parameter to another component's parameter.

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
    lambda function
        A lambda function that takes the bound parameter and returns its value.

    Examples
    --------
    Bind the x-coordinate of component2 to match component1's x-coordinate:
    >>> comp2.xc = boundto(comp1, 'xc')

    Notes
    -----
    The binding is implemented by creating a lambda that takes the
    referenced parameter as input and returns it unchanged.
    """
    if isinstance(comp, str):
        comp = eval(comp)

    x = f"{comp.id}_{var}"
    return eval(f"lambda {x}: {x}")


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
