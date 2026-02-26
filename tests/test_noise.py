"""Tests for socca.noise module."""

import jax.numpy as jp
import numpy as np
import pytest
from astropy.io import fits

import socca.noise as noise


class TestNormal:
    """Tests for Normal (uncorrelated) noise model."""

    def test_initialization_default(self):
        """Test default initialization without parameters."""
        n = noise.Normal()
        assert n.select is None
        assert n.kwargs == {"idx": 0}

    def test_initialization_with_sigma(self):
        """Test initialization with sigma parameter."""
        n = noise.Normal(sigma=0.1)
        assert n.select == "sigma"
        assert n.kwargs["sigma"] == 0.1

    def test_initialization_with_variance(self):
        """Test initialization with variance parameter."""
        n = noise.Normal(var=0.01)
        assert n.select == "var"
        assert n.kwargs["var"] == 0.01

    def test_initialization_with_weight(self):
        """Test initialization with weight parameter."""
        n = noise.Normal(wht=100.0)
        assert n.select == "wht"
        assert n.kwargs["wht"] == 100.0

    def test_initialization_with_alias(self):
        """Test initialization with alias parameter names."""
        for alias in ["sig", "std", "rms", "stddev"]:
            n = noise.Normal(**{alias: 0.1})
            assert n.select == alias

    def test_multiple_identifiers_raises_error(self):
        """Test that multiple noise identifiers raise ValueError."""
        with pytest.raises(ValueError, match="Multiple noise identifiers"):
            noise.Normal(sigma=0.1, var=0.01)

    def test_getsigma_from_float(self, simple_hdu, uniform_mask):
        """Test getsigma with float sigma value."""
        n = noise.Normal(sigma=0.5)
        n.data = jp.array(simple_hdu.data)
        n.mask = jp.array(uniform_mask)
        sigma = n.getsigma()
        assert sigma.shape == simple_hdu.data.shape
        np.testing.assert_allclose(sigma, 0.5)

    def test_getsigma_from_variance(self, simple_hdu, uniform_mask):
        """Test getsigma with variance value (converts to sigma)."""
        n = noise.Normal(var=0.25)
        n.data = jp.array(simple_hdu.data)
        n.mask = jp.array(uniform_mask)
        sigma = n.getsigma()
        np.testing.assert_allclose(sigma, 0.5)

    def test_getsigma_from_weight(self, simple_hdu, uniform_mask):
        """Test getsigma with weight value (converts to sigma)."""
        n = noise.Normal(wht=4.0)
        n.data = jp.array(simple_hdu.data)
        n.mask = jp.array(uniform_mask)
        sigma = n.getsigma()
        np.testing.assert_allclose(sigma, 0.5)

    def test_getsigma_mad_estimation(self, uniform_mask, capsys):
        """Test getsigma with MAD estimation (default)."""
        np.random.seed(42)
        data = np.random.normal(0, 1.0, (32, 32))
        n = noise.Normal()
        n.data = jp.array(data)
        n.mask = jp.array(np.ones_like(data, dtype=int))
        sigma = n.getsigma()
        captured = capsys.readouterr()
        assert "MAD" in captured.out
        assert sigma.shape == data.shape
        assert float(sigma[0, 0]) == pytest.approx(1.0, rel=0.2)

    def test_call_sets_up_model(self, simple_hdu, uniform_mask):
        """Test __call__ sets up the noise model correctly."""
        n = noise.Normal(sigma=0.1)
        data = jp.array(simple_hdu.data)
        mask = jp.array(uniform_mask)
        n(data, mask)
        assert n.data is not None
        assert n.sigma is not None
        assert n.logpdf is not None

    def test_logpdf_returns_float(self, simple_hdu, uniform_mask):
        """Test logpdf returns a scalar float."""
        n = noise.Normal(sigma=0.1)
        data = jp.array(simple_hdu.data)
        mask = jp.array(uniform_mask)
        n(data, mask)
        model_values = n.data + jp.zeros_like(n.data)
        logp = n.logpdf(model_values)
        assert np.isscalar(logp) or logp.shape == ()

    def test_logpdf_perfect_fit(self, simple_hdu, uniform_mask):
        """Test logpdf is maximized when model equals data."""
        n = noise.Normal(sigma=0.1)
        data = jp.array(simple_hdu.data)
        mask = jp.array(uniform_mask)
        n(data, mask)
        logp_perfect = n.logpdf(n.data)
        logp_offset = n.logpdf(n.data + 1.0)
        assert float(logp_perfect) > float(logp_offset)

    def test_static_logpdf(self):
        """Test _logpdf static method directly."""
        x = jp.array([1.0, 2.0, 3.0])
        data = jp.array([1.1, 1.9, 3.1])
        sigma = jp.array([0.1, 0.1, 0.1])
        logp = noise.Normal._logpdf(x, data, sigma)
        assert np.isfinite(float(logp))

    def test_handles_inf_sigma(self, simple_hdu, uniform_mask, tmp_path):
        """Test that infinite sigma values are handled (set to 0)."""
        sigma_map = np.full(simple_hdu.data.shape, 0.1)
        sigma_map[0, 0] = np.inf
        sigma_file = tmp_path / "sigma_inf.fits"
        fits.PrimaryHDU(data=sigma_map, header=simple_hdu.header).writeto(
            sigma_file, overwrite=True
        )
        n = noise.Normal(sigma=str(sigma_file))
        n.data = jp.array(simple_hdu.data)
        n.mask = jp.array(uniform_mask)
        sigma = n.getsigma()
        assert sigma[0, 0] == 0.0

    def test_handles_nan_sigma(self, simple_hdu, uniform_mask, tmp_path):
        """Test that NaN sigma values are handled (set to 0)."""
        sigma_map = np.full(simple_hdu.data.shape, 0.1)
        sigma_map[0, 0] = np.nan
        sigma_file = tmp_path / "sigma_nan.fits"
        fits.PrimaryHDU(data=sigma_map, header=simple_hdu.header).writeto(
            sigma_file, overwrite=True
        )
        n = noise.Normal(sigma=str(sigma_file))
        n.data = jp.array(simple_hdu.data)
        n.mask = jp.array(uniform_mask)
        sigma = n.getsigma()
        assert sigma[0, 0] == 0.0

    def test_zero_weight_updates_mask(
        self, simple_hdu, uniform_mask, tmp_path
    ):
        """Test that zero weight pixels update the mask."""
        weight_map = np.full(simple_hdu.data.shape, 100.0)
        weight_map[5:10, 5:10] = 0.0
        weight_file = tmp_path / "weight.fits"
        fits.PrimaryHDU(data=weight_map, header=simple_hdu.header).writeto(
            weight_file, overwrite=True
        )
        n = noise.Normal(wht=str(weight_file))
        n.data = jp.array(simple_hdu.data)
        n.mask = jp.array(uniform_mask)
        n.getsigma()


class TestNormalCorrelated:
    """Tests for NormalCorrelated noise model."""

    def test_initialization_with_icov(self):
        """Test initialization with inverse covariance matrix."""
        icov = np.eye(16)
        n = noise.NormalCorrelated(icov=icov)
        assert n.icov is not None
        assert n.cov is None

    def test_initialization_with_cov(self):
        """Test initialization with covariance matrix."""
        cov = np.eye(16)
        n = noise.NormalCorrelated(cov=cov)
        assert n.cov is not None
        assert n.icov is not None

    def test_initialization_with_cube(self):
        """Test initialization with noise cube."""
        np.random.seed(42)
        cube = np.random.normal(0, 1, (100, 4, 4))
        n = noise.NormalCorrelated(cube=cube)
        assert n.cov is not None
        assert n.icov is not None

    def test_no_input_raises_error(self):
        """Test that missing inputs raise ValueError."""
        with pytest.raises(ValueError, match="Either covariance matrix"):
            noise.NormalCorrelated()

    def test_icov_takes_precedence(self):
        """Test that icov is used when both cov and icov provided."""
        cov = np.eye(4) * 2
        icov = np.eye(4)
        with pytest.warns(UserWarning):
            n = noise.NormalCorrelated(cov=cov, icov=icov)
        np.testing.assert_array_equal(n.icov, icov)

    def test_call_sets_up_model(self):
        """Test __call__ sets up the noise model correctly."""
        icov = np.eye(16)
        n = noise.NormalCorrelated(icov=icov)
        data = jp.array(np.random.rand(4, 4))
        mask = jp.array(np.ones((4, 4), dtype=int))
        n(data, mask)
        assert n.logpdf is not None

    def test_static_logpdf(self):
        """Test _logpdf static method directly."""
        x = jp.array([1.0, 2.0])
        data = jp.array([1.1, 1.9])
        icov = jp.eye(2) * 100
        logp = noise.NormalCorrelated._logpdf(x, data, icov)
        assert np.isfinite(float(logp))

    def test_logpdf_perfect_fit(self):
        """Test logpdf is maximized when model equals data."""
        icov = np.eye(4)
        n = noise.NormalCorrelated(icov=icov)
        data = jp.array(np.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2))
        mask = jp.array(np.ones((2, 2), dtype=int))
        n(data, mask)
        logp_perfect = n.logpdf(n.data)
        logp_offset = n.logpdf(n.data + 1.0)
        assert float(logp_perfect) > float(logp_offset)


class TestNormalFourier:
    """Tests for NormalFourier noise model."""

    def test_initialization_with_icov(self):
        """Test initialization with inverse covariance."""
        icov = np.ones((8, 5))
        n = noise.NormalFourier(icov=icov, ftype="real")
        assert n.icov is not None

    def test_initialization_with_cube(self):
        """Test initialization with noise cube."""
        np.random.seed(42)
        cube = np.random.normal(0, 1, (100, 8, 8))
        n = noise.NormalFourier(cube=cube)
        assert n.cov is not None
        assert n.icov is not None

    def test_invalid_ftype_raises_error(self):
        """Test that invalid ftype raises ValueError."""
        with pytest.raises(ValueError, match="ftype must be"):
            noise.NormalFourier(icov=np.ones((4, 3)), ftype="invalid")

    def test_ftype_options(self):
        """Test valid ftype options."""
        icov = np.ones((4, 3))
        for ftype in ["real", "rfft", "full", "fft"]:
            n = noise.NormalFourier(icov=icov, ftype=ftype)
            assert n.ftype == ftype

    def test_no_input_raises_error(self):
        """Test that missing inputs raise ValueError."""
        with pytest.raises(ValueError, match="Either covariance matrix"):
            noise.NormalFourier()

    def test_masked_pixels_raise_error(self):
        """Test that masked pixels raise ValueError."""
        icov = np.ones((4, 3))
        n = noise.NormalFourier(icov=icov, ftype="real")
        data = jp.array(np.random.rand(4, 4))
        mask = jp.array(np.ones((4, 4), dtype=int))
        mask = mask.at[0, 0].set(0)
        with pytest.raises(ValueError, match="requires full image"):
            n(data, mask)


class TestNormalRI:
    """Tests for NormalRI noise model."""

    def test_initialization_default(self):
        """Test default initialization without parameters."""
        n = noise.NormalRI()
        assert n.select is None

    def test_initialization_with_sigma(self):
        """Test initialization with sigma parameter."""
        n = noise.NormalRI(sigma=0.1)
        assert n.select == "sigma"
        assert n.kwargs["sigma"] == 0.1

    def test_initialization_with_variance(self):
        """Test initialization with variance parameter."""
        n = noise.NormalRI(var=0.01)
        assert n.select == "var"
        assert n.kwargs["var"] == 0.01

    def test_initialization_with_weight(self):
        """Test initialization with weight parameter."""
        n = noise.NormalRI(wht=100.0)
        assert n.select == "wht"
        assert n.kwargs["wht"] == 100.0

    def test_initialization_with_alias(self):
        """Test initialization with alias parameter names."""
        for alias in ["sig", "std", "rms", "stddev"]:
            n = noise.NormalRI(**{alias: 0.1})
            assert n.select == alias

    def test_multiple_identifiers_raises_error(self):
        """Test that multiple noise identifiers raise ValueError."""
        with pytest.raises(ValueError, match="Multiple noise identifiers"):
            noise.NormalRI(sigma=0.1, var=0.01)

    def test_getsigma_from_float(self):
        """Test getsigma with float sigma value."""
        n = noise.NormalRI(sigma=0.5)
        n.data = jp.array(np.random.rand(8, 8))
        n.mask = jp.array(np.ones((8, 8), dtype=int))
        sigma = n.getsigma()
        assert sigma == pytest.approx(0.5)

    def test_getsigma_from_variance(self):
        """Test getsigma with variance value (converts to sigma)."""
        n = noise.NormalRI(var=0.25)
        n.data = jp.array(np.random.rand(8, 8))
        n.mask = jp.array(np.ones((8, 8), dtype=int))
        sigma = n.getsigma()
        assert sigma == pytest.approx(0.5)

    def test_getsigma_from_weight(self):
        """Test getsigma with weight value (converts to sigma)."""
        n = noise.NormalRI(wht=4.0)
        n.data = jp.array(np.random.rand(8, 8))
        n.mask = jp.array(np.ones((8, 8), dtype=int))
        sigma = n.getsigma()
        assert sigma == pytest.approx(0.5)

    def test_getsigma_mad_estimation(self, capsys):
        """Test getsigma with MAD estimation (default)."""
        np.random.seed(42)
        data = np.random.normal(0, 1.0, (32, 32))
        n = noise.NormalRI()
        n.data = jp.array(data)
        n.mask = jp.array(np.ones_like(data, dtype=int))
        sigma = n.getsigma()
        captured = capsys.readouterr()
        assert "MAD" in captured.out
        assert sigma == pytest.approx(1.0, rel=0.2)

    def test_call_sets_up_model(self):
        """Test __call__ sets up the noise model correctly."""
        n = noise.NormalRI(sigma=0.1)
        data = jp.array(np.random.rand(8, 8))
        mask = jp.array(np.ones((8, 8), dtype=int))
        n(data, mask)
        assert n.data is not None
        assert n.sigma is not None
        assert n.logpdf is not None

    def test_logpdf_returns_scalar(self):
        """Test logpdf returns a scalar value."""
        n = noise.NormalRI(sigma=0.1)
        data = jp.array(np.random.rand(8, 8))
        mask = jp.array(np.ones((8, 8), dtype=int))
        n(data, mask)
        xr = jp.ones(64)
        xs = jp.ones(64)
        logp = n.logpdf(xr, xs)
        assert np.isscalar(logp) or logp.shape == ()

    def test_static_logpdf(self):
        """Test _logpdf static method directly."""
        data = jp.array(np.random.rand(16))
        xr = jp.ones(16)
        xs = jp.ones(16)
        logp = noise.NormalRI._logpdf(xr, xs, data, 0.1)
        assert np.isfinite(float(logp))
