"""Tests for socca.priors module."""

import jax
import jax.numpy as jp
import numpy as np
import pytest

import socca.models as models
import socca.priors as priors


class TestUniform:
    """Tests for uniform prior distribution."""

    def test_valid_bounds(self):
        """Test creating uniform prior with valid bounds."""
        dist = priors.uniform(low=0.0, high=1.0)
        assert dist is not None

    def test_invalid_bounds_raises_error(self):
        """Test that low >= high raises ValueError."""
        with pytest.raises(ValueError, match="lower limit must be smaller"):
            priors.uniform(low=1.0, high=0.0)

    def test_equal_bounds_raises_error(self):
        """Test that low == high raises ValueError."""
        with pytest.raises(ValueError, match="lower limit must be smaller"):
            priors.uniform(low=0.5, high=0.5)

    @pytest.mark.parametrize("low, high", [(0.0, np.inf), (-np.inf, 1.0)])
    def test_infinite_bounds_raise(self, low, high):
        """Infinite limits, whose samples would be inf or nan, raise."""
        with pytest.raises(ValueError, match="must be finite"):
            priors.uniform(low=low, high=high)

    def test_icdf_at_boundaries(self):
        """Test inverse CDF at 0 and 1 returns bounds."""
        low, high = 2.0, 5.0
        dist = priors.uniform(low=low, high=high)
        assert float(dist.icdf(0.0)) == pytest.approx(low)
        assert float(dist.icdf(1.0)) == pytest.approx(high)

    def test_icdf_at_midpoint(self):
        """Test inverse CDF at 0.5 returns midpoint."""
        low, high = 0.0, 10.0
        dist = priors.uniform(low=low, high=high)
        assert float(dist.icdf(0.5)) == pytest.approx((low + high) / 2)

    def test_log_prob_in_range(self):
        """Test log probability is constant within range."""
        low, high = 0.0, 1.0
        dist = priors.uniform(low=low, high=high)
        expected_log_prob = -np.log(high - low)
        assert float(dist.log_prob(0.5)) == pytest.approx(expected_log_prob)

    def test_log_prob_outside_range(self):
        """Test log probability outside range is not positive."""
        dist = priors.uniform(low=0.0, high=1.0)
        log_prob_outside = float(dist.log_prob(-0.5))
        log_prob_inside = float(dist.log_prob(0.5))
        assert log_prob_outside <= log_prob_inside


class TestLoguniform:
    """Tests for log-uniform prior distribution."""

    def test_valid_bounds(self):
        """Test creating log-uniform prior with valid bounds."""
        dist = priors.loguniform(low=0.01, high=10.0)
        assert dist is not None

    def test_invalid_bounds_raises_error(self):
        """Test that low >= high raises ValueError."""
        with pytest.raises(ValueError, match="lower limit must be smaller"):
            priors.loguniform(low=10.0, high=1.0)

    def test_equal_bounds_raises_error(self):
        """Test that low == high raises ValueError."""
        with pytest.raises(ValueError, match="lower limit must be smaller"):
            priors.loguniform(low=1.0, high=1.0)

    def test_icdf_at_boundaries(self):
        """Test inverse CDF at 0 and 1 returns bounds."""
        low, high = 0.1, 100.0
        dist = priors.loguniform(low=low, high=high)
        assert float(dist.icdf(0.0)) == pytest.approx(low)
        assert float(dist.icdf(1.0)) == pytest.approx(high)

    def test_icdf_at_midpoint(self):
        """Test inverse CDF at 0.5 returns geometric mean."""
        low, high = 1.0, 100.0
        dist = priors.loguniform(low=low, high=high)
        expected = np.sqrt(low * high)
        assert float(dist.icdf(0.5)) == pytest.approx(expected)

    def test_log_prob_positive_values(self):
        """Test log probability for positive values."""
        low, high = 1.0, 10.0
        dist = priors.loguniform(low=low, high=high)
        log_prob = float(dist.log_prob(5.0))
        assert np.isfinite(log_prob)


class TestNormal:
    """Tests for normal prior distribution."""

    def test_create_distribution(self):
        """Test creating normal distribution."""
        dist = priors.normal(loc=0.0, scale=1.0)
        assert dist is not None

    def test_log_prob_at_mean(self):
        """Test log probability is maximum at mean."""
        dist = priors.normal(loc=5.0, scale=1.0)
        log_prob_at_mean = float(dist.log_prob(5.0))
        log_prob_off_mean = float(dist.log_prob(6.0))
        assert log_prob_at_mean > log_prob_off_mean

    def test_icdf_at_midpoint(self):
        """Test inverse CDF at 0.5 returns mean."""
        loc = 3.0
        dist = priors.normal(loc=loc, scale=2.0)
        assert float(dist.icdf(0.5)) == pytest.approx(loc)


class TestSplitNormal:
    """Tests for split normal distribution."""

    def test_create_distribution(self):
        """Test creating split normal distribution."""
        dist = priors.splitnormal(loc=0.0, losig=1.0, hisig=2.0)
        assert dist is not None

    def test_log_prob_asymmetry(self):
        """Test that distribution is asymmetric."""
        dist = priors.splitnormal(loc=0.0, losig=1.0, hisig=2.0)
        log_prob_neg = float(dist.log_prob(-1.0))
        log_prob_pos = float(dist.log_prob(1.0))
        assert log_prob_neg != log_prob_pos

    def test_log_prob_at_center(self):
        """Test log probability at center location."""
        loc = 5.0
        dist = priors.splitnormal(loc=loc, losig=1.0, hisig=1.0)
        log_prob = float(dist.log_prob(loc))
        assert np.isfinite(log_prob)

    def test_symmetric_case(self):
        """Test symmetric case with equal sigmas."""
        loc = 0.0
        sigma = 1.5
        dist = priors.splitnormal(loc=loc, losig=sigma, hisig=sigma)
        log_prob_neg = float(dist.log_prob(-0.5))
        log_prob_pos = float(dist.log_prob(0.5))
        assert log_prob_neg == pytest.approx(log_prob_pos, rel=1e-5)


class TestSplitNormalClass:
    """Tests for the SplitNormal distribution class."""

    def test_log_prob_normalization(self):
        """Test that log_prob is properly normalized."""
        losig, hisig = 1.0, 2.0
        dist = priors.SplitNormal(losig=losig, hisig=hisig)
        expected_log_norm = np.log(np.sqrt(2.0 / np.pi)) - np.log(
            losig + hisig
        )
        log_prob_at_zero = float(dist.log_prob(0.0))
        assert log_prob_at_zero == pytest.approx(expected_log_norm)

    def test_negative_values_use_losig(self):
        """Test that negative values use losig."""
        losig, hisig = 1.0, 10.0
        dist = priors.SplitNormal(losig=losig, hisig=hisig)
        log_prob = float(dist.log_prob(-losig))
        log_norm = np.log(np.sqrt(2.0 / np.pi)) - np.log(losig + hisig)
        expected = log_norm - 0.5
        assert log_prob == pytest.approx(expected)


class TestBoundto:
    """Tests for boundto."""

    @staticmethod
    def beta():
        """Create a fixed Beta profile."""
        return models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0)

    def test_returns_identity(self):
        """The binding returns its input unchanged."""
        bound = priors.boundto(self.beta(), "xc")
        assert bound(42.0) == 42.0

    def test_reads_current_value(self):
        """The bound value is read when resolved, not when bound."""
        comp = self.beta()
        other = self.beta()
        other.xc = priors.boundto(comp, "xc")
        comp.xc = 200.0
        assert other._resolve("xc") == 200.0

    def test_namespaced_lookup(self):
        """A bare name on a Disk is found in its sub-profiles."""
        disk = models.Disk()
        bound = priors.boundto(disk, "inc")
        assert bound.sources == [(disk, "vertical.inc")]

    def test_sub_profile(self):
        """A sub-profile can be passed directly."""
        disk = models.Disk()
        bound = priors.boundto(disk.radial, "xc")
        assert bound.sources == [(disk.radial, "xc")]

    def test_bridge_lookup(self):
        """A bare name on a Bridge is found in its sub-profiles."""
        bridge = models.SimpleBridge()
        bound = priors.boundto(bridge, "alpha")
        assert bound.sources == [(bridge, "radial.alpha")]

    def test_chained_tie(self):
        """Ties to tied parameters resolve recursively."""
        disk = models.Disk()
        disk.radial.theta = 0.7
        ellipsoid = models.Ellipsoid()
        ellipsoid.theta = priors.boundto(disk, "theta")
        other = models.Ellipsoid()
        other.theta = priors.boundto(ellipsoid, "theta")
        assert other._resolve("theta") == 0.7

    def test_ambiguous_name(self):
        """A name found in two sub-profiles raises."""
        disk = models.Disk(
            radial=models.Beta(),
            vertical=models.disk.vertical.HyperSecantHeight(),
        )
        with pytest.raises(ValueError, match="ambiguous"):
            priors.boundto(disk, "alpha")

    def test_unknown_name(self):
        """An unknown parameter raises."""
        with pytest.raises(ValueError, match="no parameter 'foo'"):
            priors.boundto(self.beta(), "foo")

    @pytest.mark.parametrize("comp", ["comp_00", None, models.Beta])
    def test_not_a_component(self, comp):
        """Anything but a component instance raises."""
        with pytest.raises(TypeError, match="model component"):
            priors.boundto(comp, "xc")


class TestFunction:
    """Tests for function."""

    @staticmethod
    def beta():
        """Create a fixed Beta profile."""
        return models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0)

    def test_sources_in_order(self):
        """Sources are passed to func positionally, in order."""
        a, b = self.beta(), self.beta()
        b.rc = 0.03
        a.Ic = priors.function(lambda x, y: x - y, (b, "rc"), (a, "rc"))
        assert a._resolve("Ic") == pytest.approx(0.02)

    def test_any_callable(self):
        """Named functions and partials are accepted."""
        import functools

        a = self.beta()

        def double(x):
            return 2 * x

        a.Ic = priors.function(double, (a, "rc"))
        assert a._resolve("Ic") == pytest.approx(0.02)
        a.Ic = priors.function(functools.partial(pow, 2.0), (a, "rc"))
        assert a._resolve("Ic") == pytest.approx(2.0**0.01)

    def test_wrong_argument_count(self):
        """A func that cannot take one argument per source raises."""
        a = self.beta()
        with pytest.raises(TypeError, match="2 positional"):
            priors.function(lambda x: x, (a, "rc"), (a, "xc"))

    def test_bad_source(self):
        """A source that is not a (component, name) pair raises."""
        a = self.beta()
        with pytest.raises(TypeError, match="pair"):
            priors.function(lambda x: x, a)

    def test_not_callable(self):
        """A non-callable func raises."""
        with pytest.raises(TypeError, match="callable"):
            priors.function(1.0, (self.beta(), "rc"))


class TestPocomcPrior:
    """Tests for pocomcPrior adapter class."""

    def test_initialization(self):
        """Test initialization with distributions."""
        dists = [
            priors.uniform(0.0, 1.0),
            priors.uniform(2.0, 4.0),
        ]
        prior = priors.pocomcPrior(dists)
        assert prior.dim == 2

    def test_dim_property(self):
        """Test dim property returns correct number."""
        dists = [
            priors.uniform(0.0, 1.0),
            priors.loguniform(0.1, 10.0),
            priors.normal(0.0, 1.0),
        ]
        prior = priors.pocomcPrior(dists)
        assert prior.dim == 3

    def test_bounds_property(self):
        """Test bounds property returns correct bounds."""
        dists = [
            priors.uniform(0.0, 1.0),
            priors.uniform(2.0, 5.0),
        ]
        prior = priors.pocomcPrior(dists)
        bounds = prior.bounds
        assert bounds.shape == (2, 2)
        assert bounds[0, 0] == pytest.approx(0.0)
        assert bounds[0, 1] == pytest.approx(1.0)
        assert bounds[1, 0] == pytest.approx(2.0)
        assert bounds[1, 1] == pytest.approx(5.0)

    def test_logpdf_shape(self):
        """Test logpdf returns correct shape."""
        dists = [
            priors.uniform(0.0, 1.0),
            priors.uniform(0.0, 1.0),
        ]
        prior = priors.pocomcPrior(dists)
        x = np.array([[0.5, 0.5], [0.2, 0.8], [0.9, 0.1]])
        logp = prior.logpdf(x)
        assert logp.shape == (3,)

    def test_logpdf_values_in_range(self):
        """Test logpdf returns finite values for valid samples."""
        dists = [
            priors.uniform(0.0, 1.0),
            priors.uniform(0.0, 1.0),
        ]
        prior = priors.pocomcPrior(dists)
        x = np.array([[0.5, 0.5]])
        logp = prior.logpdf(x)
        assert np.isfinite(logp[0])

    def test_rvs_shape(self):
        """Test rvs returns correct shape."""
        dists = [
            priors.uniform(0.0, 1.0),
            priors.uniform(0.0, 1.0),
        ]
        prior = priors.pocomcPrior(dists)
        samples = prior.rvs(size=10)
        assert samples.shape == (10, 2)

    def test_rvs_in_bounds(self):
        """Test rvs samples are within bounds."""
        low, high = 2.0, 5.0
        dists = [priors.uniform(low, high)]
        prior = priors.pocomcPrior(dists)
        samples = prior.rvs(size=100)
        assert np.all(samples >= low)
        assert np.all(samples <= high)


class TestPowerLaw:
    """Tests for the power-law prior."""

    CASES = [
        (-1.0, 1e-4, 1e-2),
        (-1.0 - 1e-9, 1e-4, 1e-2),
        (0.0, 1.0, 2.0),
        (2.0, 1.0, 1.0001),
        (-2.5, 1e-6, 1e3),
        (50.0, 1.0, 10.0),
        (-300.0, 1.0, 10.0),
    ]

    @staticmethod
    def reference_icdf(alpha, low, high, u):
        """Textbook inverse CDF, fine away from extreme slopes."""
        beta = alpha + 1.0
        if beta == 0:
            return low * (high / low) ** u
        return (low**beta + u * (high**beta - low**beta)) ** (1.0 / beta)

    @pytest.mark.parametrize("alpha, low, high", CASES)
    def test_endpoints_and_roundtrip(self, alpha, low, high):
        """The inverse CDF spans [low, high] and inverts the CDF."""
        dist = priors.powerlaw(alpha, low, high)
        u = jp.array([0.0, 1e-12, 0.1, 0.5, 0.9, 1 - 1e-12, 1.0])
        x = dist.icdf(u)
        assert bool(jp.all(jp.isfinite(x)))
        assert float(x[0]) == pytest.approx(low, rel=1e-12)
        assert float(x[-1]) == pytest.approx(high, rel=1e-12)
        np.testing.assert_allclose(dist.cdf(x[2:5]), u[2:5], atol=1e-10)

    @pytest.mark.parametrize("alpha", [-2.5, -1.0, 0.0, 1.0, 3.0])
    def test_matches_textbook(self, alpha):
        """The inverse CDF matches the textbook formula."""
        dist = priors.powerlaw(alpha, 1e-3, 1e-1)
        u = np.linspace(0.0, 1.0, 11)
        np.testing.assert_allclose(
            dist.icdf(jp.asarray(u)),
            self.reference_icdf(alpha, 1e-3, 1e-1, u),
            rtol=1e-12,
        )

    @pytest.mark.parametrize("alpha, low, high", CASES)
    def test_normalised(self, alpha, low, high):
        """The density integrates to one."""
        dist = priors.powerlaw(alpha, low, high)
        logx = np.linspace(np.log(low), np.log(high), 200001)
        x = np.clip(np.exp(logx), low, high)
        density = np.exp(np.asarray(dist.log_prob(jp.asarray(x)))) * x
        total = float(jp.trapezoid(density, logx))
        assert total == pytest.approx(1.0, rel=1e-5)

    def test_support(self):
        """The support is [low, high] and log_prob is -inf outside."""
        dist = priors.powerlaw(-1.0, 1e-4, 1e-2)
        assert float(dist.support.lower_bound) == 1e-4
        assert float(dist.support.upper_bound) == 1e-2
        assert float(dist.log_prob(2e-2)) == -np.inf
        assert float(dist.log_prob(5e-5)) == -np.inf

    def test_sampling(self):
        """Samples follow the distribution."""
        dist = priors.powerlaw(-2.5, 1e-6, 1e3)
        x = dist.sample(jax.random.PRNGKey(0), (200000,))
        assert float(jp.median(x)) == pytest.approx(
            float(dist.icdf(0.5)), rel=0.02
        )

    @pytest.mark.parametrize(
        "alpha, low, high",
        [(0.0, 0.0, 1.0), (0.0, 2.0, 1.0), (0.0, 1.0, np.inf), (np.nan, 1, 2)],
    )
    def test_invalid(self, alpha, low, high):
        """Invalid indices or bounds raise."""
        with pytest.raises(ValueError):
            priors.powerlaw(alpha, low, high)

    def test_logpowerlaw_removed(self):
        """The log-sampled power law is gone."""
        assert not hasattr(priors, "logpowerlaw")
        assert not hasattr(priors, "LogPowerLaw")


class TestPowerLawPytree:
    """PowerLaw priors through jax transformations."""

    def test_support_after_unflattening(self):
        """An unflattened copy, which skips __init__, keeps its support."""
        prior = priors.powerlaw(-2.0, 1.0, 10.0)
        copy = jax.tree_util.tree_map(lambda x: x, prior)
        assert float(copy.support.lower_bound) == 1.0
        assert float(copy.support.upper_bound) == 10.0
        assert np.isfinite(float(copy.log_prob(3.0)))


class TestPowerLawGradient:
    """Exact icdf derivatives, including at the edges of the unit interval."""

    @pytest.mark.parametrize(
        "alpha, low, high",
        [
            (-1.0, 1e-3, 1e3),
            (2.0, 1.0, 5.0),
            (-2.5, 0.1, 10.0),
            (0.0, 1.0, 2.0),
            (50.0, 1.0, 10.0),
            (-60.0, 1.0, 10.0),
        ],
    )
    def test_du_is_inverse_density(self, alpha, low, high):
        """The icdf derivative is 1 / pdf, also at u = 0 and 1."""
        dist = priors.powerlaw(alpha, low, high)
        u = jp.array([0.0, 1e-9, 0.3, 1.0 - 1e-9, 1.0])
        grad = jax.vmap(jax.grad(dist.icdf))(u)
        pdf = jp.exp(dist.log_prob(dist.icdf(u)))
        np.testing.assert_allclose(grad * pdf, 1.0, rtol=1e-9)

    @pytest.mark.parametrize("u", [0.0, 0.3, 1.0])
    @pytest.mark.parametrize("alpha", [-40.0, -2.5, -1.0, 0.0, 2.0, 40.0])
    def test_parameter_derivatives(self, alpha, u):
        """Derivatives with respect to alpha, low and high match differences."""

        def icdf(a, lo, hi):
            return priors.PowerLaw(a, lo, hi).icdf(u)

        pars = (alpha, 0.5, 4.0)
        grad = jax.grad(icdf, argnums=(0, 1, 2))(*pars)
        for i, g in enumerate(grad):
            h = 1e-6
            up, dn = list(pars), list(pars)
            up[i] += h
            dn[i] -= h
            fd = (icdf(*up) - icdf(*dn)) / (2 * h)
            np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-8)

    @pytest.mark.parametrize(
        "alpha, low, high, x",
        [
            (30.0, 1e-10, 1.0, 0.9),
            (200.0, 1e-3, 1e3, 500.0),
            (-200.0, 1e-3, 1e3, 1.1e-3),
        ],
    )
    def test_steep_cdf_derivatives(self, alpha, low, high, x):
        """The cdf derivatives stay finite where exp(beta * span) overflows."""
        dist = priors.PowerLaw(alpha, low, high)
        np.testing.assert_allclose(
            jax.grad(dist.cdf)(x), jp.exp(dist.log_prob(x)), rtol=1e-7
        )

        def cdf(a):
            return priors.PowerLaw(a, low, high).cdf(x)

        h = 1e-6 * abs(alpha)
        fd = (cdf(alpha + h) - cdf(alpha - h)) / (2 * h)
        np.testing.assert_allclose(
            jax.grad(cdf)(alpha), fd, rtol=1e-5, atol=1e-13
        )

    @pytest.mark.parametrize("alpha", [-1.0, -1.0 + 1e-7, -1.002])
    def test_log_prob_alpha_derivative(self, alpha):
        """The alpha derivative of log_prob is right, also at alpha = -1."""

        def logp(a):
            return priors.PowerLaw(a, 1.0, 10.0).log_prob(2.0)

        h = 1e-5
        fd = (logp(alpha + h) - logp(alpha - h)) / (2 * h)
        np.testing.assert_allclose(jax.grad(logp)(alpha), fd, rtol=1e-7)
