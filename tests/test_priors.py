"""Tests for socca.priors module."""

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


class TestPowerLaws:
    """Tests for the PowerLaw and LogPowerLaw distributions."""

    ALPHAS = [-1.0, 0.0, 1.5, -2.5]

    @staticmethod
    def _grid(dist, n):
        lo, hi = (
            float(b)
            for b in (dist.support.lower_bound, dist.support.upper_bound)
        )
        if isinstance(dist, priors.PowerLaw):
            return np.geomspace(lo, hi, n)
        return np.linspace(lo, hi, n)

    @pytest.mark.parametrize("factory", [priors.powerlaw, priors.logpowerlaw])
    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_density_is_normalised(self, factory, alpha):
        """The density integrates to 1 over its own variable."""
        dist = factory(alpha, 0.1, 10.0)
        x = self._grid(dist, 200001)
        pdf = np.exp(np.asarray(dist.log_prob(jp.asarray(x))))
        assert np.trapezoid(pdf, x) == pytest.approx(1.0, rel=1e-6)

    @pytest.mark.parametrize("factory", [priors.powerlaw, priors.logpowerlaw])
    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_cdf_and_icdf(self, factory, alpha):
        """The cdf integrates log_prob, and icdf inverts the cdf."""
        dist = factory(alpha, 0.1, 10.0)
        x = self._grid(dist, 200001)
        pdf = np.exp(np.asarray(dist.log_prob(jp.asarray(x))))
        cdf = np.concatenate(
            [[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(x))]
        )
        np.testing.assert_allclose(dist.cdf(jp.asarray(x)), cdf, atol=1e-8)
        u = jp.asarray([0.0, 0.05, 0.3, 0.5, 0.8, 0.99, 1.0])
        np.testing.assert_allclose(dist.cdf(dist.icdf(u)), u, atol=1e-12)

    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_same_implied_prior(self, alpha):
        """LogPowerLaw samples log(x), with the PowerLaw prior on x."""
        lin = priors.powerlaw(alpha, 0.1, 10.0)
        log = priors.logpowerlaw(alpha, 0.1, 10.0)
        u = jp.asarray([0.0, 0.2, 0.7, 1.0])
        np.testing.assert_allclose(
            log.to_model(log.icdf(u)), lin.icdf(u), rtol=1e-12
        )
        x = jp.asarray([0.2, 0.5, 3.0, 9.0])
        np.testing.assert_allclose(
            log.log_prob(log.from_model(x)) - jp.log(x),
            lin.log_prob(x),
            rtol=1e-12,
        )

    def test_support(self):
        """Supports are the bounds in each distribution's own variable."""
        lin = priors.powerlaw(1.0, 0.1, 10.0)
        log = priors.logpowerlaw(1.0, 0.1, 10.0)
        assert float(lin.support.lower_bound) == 0.1
        assert float(lin.support.upper_bound) == 10.0
        assert float(log.support.lower_bound) == pytest.approx(np.log(0.1))
        assert float(log.support.upper_bound) == pytest.approx(np.log(10.0))
        assert bool(lin.support.check(0.1)) and bool(lin.support.check(10.0))
        assert bool(log.support.check(log.from_model(0.1)))

    @pytest.mark.parametrize("factory", [priors.powerlaw, priors.logpowerlaw])
    @pytest.mark.parametrize("alpha", [-1.0, 1.5])
    def test_icdf_gradient_is_finite(self, factory, alpha):
        """The icdf has finite gradients, including the log-uniform case."""
        import jax

        dist = factory(alpha, 0.1, 10.0)
        grad = jax.vmap(jax.grad(dist.icdf))(jp.asarray([0.1, 0.5, 0.9]))
        assert np.all(np.isfinite(grad)) and np.all(np.asarray(grad) > 0)

    @pytest.mark.parametrize("factory", [priors.powerlaw, priors.logpowerlaw])
    @pytest.mark.parametrize(
        "low, high", [(10.0, 1.0), (1.0, 1.0), (0.0, 1.0)]
    )
    def test_invalid_limits(self, factory, low, high):
        """Limits must satisfy 0 < low < high."""
        with pytest.raises(ValueError, match="0 < low < high"):
            factory(1.0, low, high)

    @pytest.mark.parametrize("alpha", [-6.0, -3.0, 5.0])
    def test_steep_power_laws_reach_their_bounds(self, alpha):
        """icdf(0) and icdf(1) are the bounds even for steep power laws."""
        dist = priors.powerlaw(alpha, 1e-3, 1e3)
        np.testing.assert_allclose(
            dist.icdf(jp.asarray([0.0, 1.0])), [1e-3, 1e3], rtol=1e-12
        )

    def test_logpowerlaw_model_bounds_exact(self):
        """The model-space bounds are the limits given to logpowerlaw."""
        from socca.models.base import _support_bounds

        assert _support_bounds(priors.logpowerlaw(-1.0, 0.25, 10.0)) == (
            0.25,
            10.0,
        )
        models.Sersic(ns=priors.logpowerlaw(-1.0, 0.25, 10.0))

    def test_infinite_limit_raises(self):
        """An infinite upper limit is rejected."""
        with pytest.raises(ValueError, match="0 < low < high < inf"):
            priors.powerlaw(-1.0, 1.0, np.inf)

    def test_array_limits(self):
        """Batched limits are validated element-wise."""
        dist = priors.powerlaw(0.0, np.array([1.0, 2.0]), np.array([2.0, 3.0]))
        np.testing.assert_allclose(dist.icdf(0.5), [1.5, 2.5])
        with pytest.raises(ValueError):
            priors.powerlaw(0.0, np.array([1.0, 2.0]), np.array([2.0, 1.0]))

    def test_cdf_narrow_range(self):
        """The cdf inverts the icdf on very narrow ranges near alpha = -1."""
        dist = priors.powerlaw(-0.999998, 1.0, 1.0 + 1e-8)
        u = jp.linspace(0.0, 1.0, 11)
        np.testing.assert_allclose(dist.cdf(dist.icdf(u)), u, atol=1e-6)

    @pytest.mark.parametrize("beta", [1e-7, -1e-7, 1e-12])
    def test_near_log_uniform(self, beta):
        """Indices just off alpha = -1 use the exact power law."""
        la, lb = np.log(1e-10), np.log(1e10)
        dist = priors.logpowerlaw(beta - 1.0, 1e-10, 1e10)
        c = beta * (lb - la)
        expected = la + np.log1p(0.5 * np.expm1(c)) / beta
        assert float(dist.icdf(0.5)) == pytest.approx(expected, abs=1e-12)


class TestSplitNormalSampling:
    """Tests for the SplitNormal cdf, icdf and sampling."""

    def test_cdf_matches_density(self):
        """The cdf is the integral of the density."""
        dist = priors.splitnormal(1.0, 0.5, 2.0)
        x = np.linspace(-6.0, 20.0, 400001)
        pdf = np.exp(np.asarray(dist.log_prob(jp.asarray(x))))
        cdf = np.concatenate(
            [[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(x))]
        )
        np.testing.assert_allclose(dist.cdf(jp.asarray(x)), cdf, atol=1e-7)

    def test_icdf_inverts_cdf(self):
        """icdf(cdf(x)) returns x on both sides of the mode."""
        dist = priors.splitnormal(1.0, 0.5, 2.0)
        x = jp.asarray([-1.0, 0.0, 0.9, 1.0, 1.1, 3.0, 8.0])
        np.testing.assert_allclose(dist.icdf(dist.cdf(x)), x, atol=1e-12)
        assert float(dist.icdf(0.2)) == pytest.approx(1.0)

    def test_samples_follow_cdf(self):
        """Samples reproduce the quantiles."""
        import jax

        dist = priors.splitnormal(1.0, 0.5, 2.0)
        s = np.asarray(dist.sample(jax.random.PRNGKey(0), (200000,)))
        q = np.array([0.05, 0.2, 0.5, 0.9])
        np.testing.assert_allclose(
            np.quantile(s, q), dist.icdf(jp.asarray(q)), atol=0.02
        )

    @pytest.mark.parametrize(
        "losig, hisig", [(0.004, 0.002), (0.002, 0.005), (0.5, 2.0)]
    )
    def test_icdf_gradient_at_mode(self, losig, hisig):
        """The icdf slope at the mode is 1 / pdf(mode), on both sides."""
        import jax

        dist = priors.splitnormal(0.0, losig, hisig)
        umode = dist.cdf(0.0)
        expected = (losig + hisig) / np.sqrt(2.0 / np.pi)
        for u in (umode, jp.nextafter(umode, 0.0), jp.nextafter(umode, 1.0)):
            assert float(jax.grad(dist.icdf)(u)) == pytest.approx(expected)


class TestBoundto:
    """Tests for boundto function."""

    def test_creates_callable(self):
        """Test that boundto returns a callable."""

        class MockComponent:
            id = "mock"

        comp = MockComponent()
        result = priors.boundto(comp, "xc")
        assert callable(result)

    def test_lambda_returns_input(self):
        """Test that the lambda returns its input unchanged."""

        class MockComponent:
            id = "comp1"

        comp = MockComponent()
        bound_fn = priors.boundto(comp, "xc")
        test_value = 42.0
        result = bound_fn(test_value)
        assert result == test_value

    def test_resolve_flat_attribute(self):
        """Test that resolve() reads a top-level component's current value."""
        comp = models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0)
        bound = priors.boundto(comp, "xc")
        assert bound.resolve() == 180.0

    def test_resolve_reflects_current_value(self):
        """Test that resolve() re-reads the value rather than caching it."""
        comp = models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0)
        bound = priors.boundto(comp, "xc")
        comp.xc = 200.0
        assert bound.resolve() == 200.0

    def test_resolve_namespaced_attribute(self):
        """Test that resolve() works for namespaced sub-component parameters."""
        models.Component.reset()
        disk = models.Disk()
        disk.vertical.inc = 0.9
        bound = priors.boundto(disk, "inc")
        assert bound.key.endswith("vertical.inc")
        assert bound.resolve() == 0.9

    def test_resolve_chained_tie(self):
        """Test that resolve() follows a chain of ties recursively."""
        models.Component.reset()
        disk = models.Disk()
        disk.radial.theta = 0.7
        ellipsoid = models.Ellipsoid()
        ellipsoid.theta = priors.boundto(disk, "theta")
        chained = priors.boundto(ellipsoid, "theta")
        assert chained.resolve() == 0.7

    def test_key_unchanged_for_model_level_resolution(self):
        """Test that .key (used by Model.getmap/getmodel) is unaffected."""
        comp = models.Beta(xc=180.0, yc=45.0, rc=0.01, Ic=1.0)
        bound = priors.boundto(comp, "xc")
        assert bound.key == f"{comp.id}_xc"


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
        samples[0] = samples[1]
        logp = prior.logpdf(samples)
        logp[0] = logp[1]

    def test_rvs_in_bounds(self):
        """Test rvs samples are within bounds."""
        low, high = 2.0, 5.0
        dists = [priors.uniform(low, high)]
        prior = priors.pocomcPrior(dists)
        samples = prior.rvs(size=100)
        assert np.all(samples >= low)
        assert np.all(samples <= high)
