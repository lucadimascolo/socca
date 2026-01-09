"""Tests for socca.priors module."""

import jax.numpy as jp
import numpy as np
import pytest

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

    def test_equal_bounds_creates_distribution(self):
        """Test that low == high creates a degenerate distribution."""
        dist = priors.uniform(low=0.5, high=0.5)
        assert dist is not None

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
