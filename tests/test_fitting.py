"""Tests for socca.fitting module."""

import jax.numpy as jp
import numpy as np
import pytest
from astropy.io import fits

import socca.data as data
import socca.fitting as fitting
from socca.fitting.methods.utils import get_imp_weights
import socca.models as models
import socca.noise as noise
import socca.priors as priors


class TestGetImpWeights:
    """Tests for get_imp_weights function."""

    def test_basic_computation(self):
        """Test basic importance weight computation."""
        logw = np.array([0.0, -1.0, -2.0])
        weights = get_imp_weights(logw)
        assert weights.shape == logw.shape
        assert np.isclose(weights.sum(), 1.0)

    def test_with_logz(self):
        """Test with explicit log-evidence."""
        logw = np.array([0.0, -1.0, -2.0])
        logz = 0.0
        weights = get_imp_weights(logw, logz=logz)
        assert weights.shape == logw.shape
        assert np.isclose(weights.sum(), 1.0)

    def test_with_logz_list(self):
        """Test with log-evidence as list."""
        logw = np.array([0.0, -1.0, -2.0])
        logz = [0.0, 0.1]
        weights = get_imp_weights(logw, logz=logz)
        assert weights.shape == logw.shape
        assert np.isclose(weights.sum(), 1.0)

    def test_uniform_weights(self):
        """Test that equal log-weights give uniform weights."""
        logw = np.array([0.0, 0.0, 0.0])
        weights = get_imp_weights(logw)
        np.testing.assert_allclose(weights, 1 / 3, rtol=1e-5)

    def test_weights_sum_to_one(self):
        """Test that weights sum to 1.0."""
        np.random.seed(42)
        logw = np.random.randn(100)
        weights = get_imp_weights(logw)
        assert np.isclose(weights.sum(), 1.0)


class TestFitter:
    """Tests for fitter class."""

    @pytest.fixture
    def simple_fitter(self, simple_hdu, gaussian_psf):
        """Create a simple fitter for testing."""
        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))
        img.addpsf(gaussian_psf)

        xc = simple_hdu.header["CRVAL1"]
        yc = simple_hdu.header["CRVAL2"]

        gaussian = models.Gaussian(
            xc=priors.uniform(xc - 0.01, xc + 0.01),
            yc=priors.uniform(yc - 0.01, yc + 0.01),
            rs=0.005,
            Is=10.0,
        )
        mod = models.Model(gaussian)
        return fitting.fitter(img, mod)

    def test_initialization(self, simple_fitter):
        """Test fitter initialization."""
        assert simple_fitter.img is not None
        assert simple_fitter.mod is not None
        assert simple_fitter.mask is not None
        assert simple_fitter.pdfnoise is not None

    def test_labels_extracted(self, simple_fitter):
        """Test that parameter labels are extracted."""
        assert len(simple_fitter.labels) > 0
        assert all(isinstance(label, str) for label in simple_fitter.labels)

    def test_units_extracted(self, simple_fitter):
        """Test that parameter units are extracted."""
        assert len(simple_fitter.units) > 0
        assert len(simple_fitter.units) == len(simple_fitter.labels)

    def test_plotter_created(self, simple_fitter):
        """Test that Plotter instance is created."""
        from socca.plotting import Plotter

        assert isinstance(simple_fitter.plot, Plotter)

    def test_getmodel(self, simple_fitter):
        """Test _getmodel method."""
        pp = [
            simple_fitter.img.hdu.header["CRVAL1"],
            simple_fitter.img.hdu.header["CRVAL2"],
        ]
        result = simple_fitter._getmodel(pp)
        assert len(result) == 4
        mraw, msmo, mbkg, mneg = result
        assert mraw.shape == simple_fitter.img.data.shape
        assert msmo.shape == simple_fitter.img.data.shape

    def test_log_likelihood_returns_scalar(self, simple_fitter):
        """Test that _log_likelihood returns a scalar."""
        pp = jp.array(
            [
                simple_fitter.img.hdu.header["CRVAL1"],
                simple_fitter.img.hdu.header["CRVAL2"],
            ]
        )
        logL = simple_fitter._log_likelihood(pp)
        assert jp.isscalar(logL) or logL.shape == ()

    def test_log_likelihood_finite(self, simple_fitter):
        """Test that log-likelihood is finite for valid parameters."""
        pp = jp.array(
            [
                simple_fitter.img.hdu.header["CRVAL1"],
                simple_fitter.img.hdu.header["CRVAL2"],
            ]
        )
        logL = simple_fitter._log_likelihood(pp)
        assert jp.isfinite(logL)

    def test_prior_transform(self, simple_fitter):
        """Test _prior_transform method."""
        pp = jp.array([0.5])
        transformed = simple_fitter._prior_transform(pp)
        assert transformed.shape == pp.shape

    def test_prior_transform_bounds(self, simple_fitter):
        """Test that prior transform respects bounds."""
        pp_low = jp.array([0.0])
        pp_high = jp.array([1.0])
        transformed_low = simple_fitter._prior_transform(pp_low)
        transformed_high = simple_fitter._prior_transform(pp_high)
        assert float(transformed_low[0]) < float(transformed_high[0])

    def test_log_prior(self, simple_fitter):
        """Test _log_prior method."""
        theta = jp.array(
            [
                simple_fitter.img.hdu.header["CRVAL1"],
                simple_fitter.img.hdu.header["CRVAL2"],
            ]
        )
        log_prob = simple_fitter._log_prior(theta)
        assert jp.isfinite(log_prob)


class TestFitterWithMultipleParameters:
    """Tests for fitter with multiple free parameters."""

    @pytest.fixture
    def multi_param_fitter(self, simple_hdu, gaussian_psf):
        """Create a fitter with multiple free parameters."""
        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))
        img.addpsf(gaussian_psf)

        xc = simple_hdu.header["CRVAL1"]
        yc = simple_hdu.header["CRVAL2"]

        beta = models.Beta(
            xc=priors.uniform(xc - 0.01, xc + 0.01),
            yc=priors.uniform(yc - 0.01, yc + 0.01),
            rc=priors.loguniform(0.001, 0.01),
            Ic=1.0,
            beta=0.5,
        )
        mod = models.Model(beta)
        return fitting.fitter(img, mod)

    def test_multiple_labels(self, multi_param_fitter):
        """Test that multiple labels are extracted."""
        assert len(multi_param_fitter.labels) == 3

    def test_prior_transform_multiple(self, multi_param_fitter):
        """Test prior transform with multiple parameters."""
        pp = jp.array([0.5, 0.5, 0.5])
        transformed = multi_param_fitter._prior_transform(pp)
        assert transformed.shape == pp.shape

    def test_log_likelihood_multiple(self, multi_param_fitter):
        """Test log-likelihood with multiple parameters."""
        xc = multi_param_fitter.img.hdu.header["CRVAL1"]
        yc = multi_param_fitter.img.hdu.header["CRVAL2"]
        rc = 0.005
        pp = jp.array([xc, yc, rc])
        logL = multi_param_fitter._log_likelihood(pp)
        assert jp.isfinite(logL)


class TestFitterWithBackground:
    """Tests for fitter with background component."""

    @pytest.fixture
    def fitter_with_background(self, simple_hdu, gaussian_psf):
        """Create a fitter with background component."""
        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))
        img.addpsf(gaussian_psf)

        xc = simple_hdu.header["CRVAL1"]
        yc = simple_hdu.header["CRVAL2"]

        beta = models.Beta(
            xc=xc,
            yc=yc,
            rc=0.005,
            Ic=priors.loguniform(0.1, 10.0),
            beta=0.5,
        )
        bkg = models.Background(a0=priors.uniform(-1.0, 1.0))

        mod = models.Model(beta)
        mod.addcomponent(bkg)
        return fitting.fitter(img, mod)

    def test_background_parameter_in_labels(self, fitter_with_background):
        """Test that background parameter is in labels."""
        assert any("a0" in label for label in fitter_with_background.labels)

    def test_log_likelihood_with_background(self, fitter_with_background):
        """Test log-likelihood with background component."""
        pp = jp.array([1.0, 0.0])
        logL = fitter_with_background._log_likelihood(pp)
        assert jp.isfinite(logL)


class TestFitterRunMethod:
    """Tests for fitter.run method (without actually running samplers)."""

    @pytest.fixture
    def simple_fitter(self, simple_hdu, gaussian_psf):
        """Create a simple fitter for testing."""
        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))
        img.addpsf(gaussian_psf)

        xc = simple_hdu.header["CRVAL1"]
        yc = simple_hdu.header["CRVAL2"]

        beta = models.Beta(
            xc=priors.uniform(xc - 0.01, xc + 0.01),
            yc=yc,
            rc=0.005,
            Ic=1.0,
            beta=0.5,
        )
        mod = models.Model(beta)
        return fitting.fitter(img, mod)

    def test_invalid_method_raises_error(self, simple_fitter):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown sampling method"):
            simple_fitter.run(method="invalid_method")


class TestFitterIntegration:
    """Integration tests for the full fitting workflow (light tests)."""

    @pytest.fixture
    def mock_data_and_model(self, simple_hdu, gaussian_psf):
        """Create mock data and model for integration testing."""
        np.random.seed(42)
        shape = simple_hdu.data.shape

        y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
        cx, cy = shape[1] / 2, shape[0] / 2
        sigma = 5.0
        mock_source = 10.0 * np.exp(
            -((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2)
        )
        mock_noise = np.random.normal(0, 0.5, shape)
        mock_data = mock_source + mock_noise

        hdu = fits.PrimaryHDU(
            data=mock_data.astype(np.float64), header=simple_hdu.header
        )
        img = data.Image(hdu, noise=noise.Normal(sigma=0.5))
        img.addpsf(gaussian_psf)

        xc = hdu.header["CRVAL1"]
        yc = hdu.header["CRVAL2"]

        beta = models.Beta(
            xc=xc,
            yc=yc,
            rc=0.005,
            Ic=priors.loguniform(1.0, 100.0),
            beta=0.5,
        )
        mod = models.Model(beta)
        return fitting.fitter(img, mod)

    def test_can_create_fitter(self, mock_data_and_model):
        """Test that fitter can be created with mock data."""
        assert mock_data_and_model is not None

    def test_likelihood_evaluates(self, mock_data_and_model):
        """Test that likelihood can be evaluated on mock data."""
        pp = jp.array([10.0])
        logL = mock_data_and_model._log_likelihood(pp)
        assert jp.isfinite(logL)

    def test_prior_transform_evaluates(self, mock_data_and_model):
        """Test that prior transform can be evaluated."""
        pp = jp.array([0.5])
        transformed = mock_data_and_model._prior_transform(pp)
        assert jp.isfinite(transformed[0])


class TestFitterResponseAndExposure:
    """Tests for fitter with response and exposure maps."""

    @pytest.fixture
    def fitter_with_resp_exp(self, simple_hdu, gaussian_psf):
        """Create a fitter with non-uniform response and exposure."""
        resp_map = np.ones(simple_hdu.data.shape) * 0.9
        exp_map = np.ones(simple_hdu.data.shape) * 100.0

        header = simple_hdu.header.copy()
        resp_hdu = fits.PrimaryHDU(data=resp_map, header=header)
        exp_hdu = fits.PrimaryHDU(data=exp_map, header=header)

        img = data.Image(
            simple_hdu,
            response=resp_hdu,
            exposure=exp_hdu,
            noise=noise.Normal(sigma=0.1),
        )
        img.addpsf(gaussian_psf)

        xc = simple_hdu.header["CRVAL1"]
        yc = simple_hdu.header["CRVAL2"]

        beta = models.Beta(
            xc=priors.uniform(xc - 0.01, xc + 0.01),
            yc=yc,
            rc=0.005,
            Ic=1.0,
            beta=0.5,
        )
        mod = models.Model(beta)
        return fitting.fitter(img, mod)

    def test_response_applied(self, fitter_with_resp_exp):
        """Test that response is considered in model evaluation."""
        assert not np.all(np.array(fitter_with_resp_exp.img.response) == 1.0)

    def test_exposure_applied(self, fitter_with_resp_exp):
        """Test that exposure is considered in model evaluation."""
        assert not np.all(np.array(fitter_with_resp_exp.img.exposure) == 1.0)

    def test_getmodel_with_resp_exp(self, fitter_with_resp_exp):
        """Test _getmodel with response and exposure."""
        pp = [fitter_with_resp_exp.img.hdu.header["CRVAL1"]]
        result = fitter_with_resp_exp._getmodel(pp)
        assert len(result) == 4


@pytest.mark.slow
class TestSamplerIntegration:
    """Integration tests that run actual samplers with minimal iterations.

    These tests are marked as slow and can be skipped with:
        pytest -m "not slow"

    To run only these tests:
        pytest -m slow
    """

    @pytest.fixture
    def multi_param_fitter(self, simple_hdu, gaussian_psf):
        """Create a fitter with 2+ parameters for sampler testing.

        Note: Nautilus requires at least 2 parameters.
        Uses Gaussian model to match the Gaussian data in simple_hdu fixture.
        """
        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))

        xc = simple_hdu.header["CRVAL1"]
        yc = simple_hdu.header["CRVAL2"]

        gaussian = models.Gaussian(
            xc=priors.uniform(xc - 0.01, xc + 0.01),
            yc=priors.uniform(yc - 0.01, yc + 0.01),
            rs=0.005,
            Is=10.0,
        )
        mod = models.Model(gaussian)
        return fitting.fitter(img=img, mod=mod)

    def test_nautilus_sampler(self, multi_param_fitter, tmp_path):
        """Test that nautilus sampler runs and produces valid output."""
        checkpoint = tmp_path / "nautilus_test.hdf5"
        multi_param_fitter.run(
            method="nautilus",
            n_live=1000,
            n_like_max=5000,
            checkpoint=str(checkpoint),
            verbose=False,
            resume=False,
        )

        assert hasattr(multi_param_fitter, "samples")
        assert hasattr(multi_param_fitter, "weights")
        assert hasattr(multi_param_fitter, "logz")
        assert multi_param_fitter.samples.shape[0] > 0
        assert multi_param_fitter.samples.shape[1] == 2
        assert (
            len(multi_param_fitter.weights)
            == multi_param_fitter.samples.shape[0]
        )
        assert np.isfinite(multi_param_fitter.logz)

    def test_dynesty_sampler(self, multi_param_fitter):
        """Test that dynesty sampler runs and produces valid output."""
        multi_param_fitter.run(
            method="dynesty",
            nlive=1000,
            maxiter=5000,
            resume=False,
        )

        assert hasattr(multi_param_fitter, "samples")
        assert hasattr(multi_param_fitter, "weights")
        assert hasattr(multi_param_fitter, "logz")
        assert multi_param_fitter.samples.shape[0] > 0
        assert multi_param_fitter.samples.shape[1] == 2
        assert (
            len(multi_param_fitter.weights)
            == multi_param_fitter.samples.shape[0]
        )
        assert np.isfinite(multi_param_fitter.logz)

    @pytest.mark.xfail(
        reason="pocomc's Student-t fitting can fail in pytest due to "
        "environment-specific random state issues. Works in standalone scripts.",
        raises=ValueError,
    )
    def test_pocomc_sampler(self, multi_param_fitter):
        """Test that pocomc sampler runs and produces valid output."""
        multi_param_fitter.run(
            method="pocomc", n_effective=1000, n_active=500, resume=False
        )

        assert hasattr(multi_param_fitter, "samples")
        assert hasattr(multi_param_fitter, "weights")
        assert hasattr(multi_param_fitter, "logz")
        assert multi_param_fitter.samples.shape[0] > 0
        assert multi_param_fitter.samples.shape[1] == 2
        assert (
            len(multi_param_fitter.weights)
            == multi_param_fitter.samples.shape[0]
        )
        assert np.isfinite(multi_param_fitter.logz)

    def test_emcee_sampler(self, multi_param_fitter, tmp_path):
        """Test that emcee sampler runs and produces valid output."""
        checkpoint = tmp_path / "emcee_test.hdf5"
        multi_param_fitter.run(
            method="emcee",
            nwalkers=20,
            nsteps=500,
            discard=100,
            thin=1,
            checkpoint=str(checkpoint),
            progress=False,
            resume=False,
        )

        assert hasattr(multi_param_fitter, "samples")
        assert hasattr(multi_param_fitter, "weights")
        assert hasattr(multi_param_fitter, "sampler")
        assert multi_param_fitter.samples.shape[0] > 0
        assert multi_param_fitter.samples.shape[1] == 2
        assert (
            len(multi_param_fitter.weights)
            == multi_param_fitter.samples.shape[0]
        )
        assert np.allclose(multi_param_fitter.weights, 1.0)

    def test_emcee_with_convergence(self, multi_param_fitter, tmp_path):
        """Test emcee with convergence checking enabled."""
        checkpoint = tmp_path / "emcee_converge_test.hdf5"
        multi_param_fitter.run(
            method="emcee",
            nwalkers=20,
            nsteps=1000,
            converge=True,
            check_every=100,
            tau_factor=10,
            checkpoint=str(checkpoint),
            progress=False,
            resume=False,
        )

        assert hasattr(multi_param_fitter, "samples")
        assert hasattr(multi_param_fitter, "tau_history")
        assert multi_param_fitter.samples.shape[0] > 0

    def test_emcee_resume(self, multi_param_fitter, tmp_path):
        """Test emcee checkpoint and resume functionality."""
        checkpoint = tmp_path / "emcee_resume_test.hdf5"

        multi_param_fitter.run(
            method="emcee",
            nwalkers=20,
            nsteps=200,
            discard=0,
            checkpoint=str(checkpoint),
            progress=False,
            resume=False,
        )
        initial_iterations = multi_param_fitter.sampler.iteration

        multi_param_fitter.run(
            method="emcee",
            nwalkers=20,
            nsteps=400,
            discard=0,
            checkpoint=str(checkpoint),
            progress=False,
            resume=True,
        )
        final_iterations = multi_param_fitter.sampler.iteration

        assert final_iterations > initial_iterations
        assert final_iterations == 400

    def test_emcee_nwalkers_minimum(self, multi_param_fitter, tmp_path):
        """Test that emcee enforces minimum nwalkers = 2 * ndim."""
        checkpoint = tmp_path / "emcee_nwalkers_test.hdf5"
        ndim = 2

        multi_param_fitter.run(
            method="emcee",
            nwalkers=1,
            nsteps=100,
            discard=0,
            checkpoint=str(checkpoint),
            progress=False,
            resume=False,
        )

        assert multi_param_fitter.sampler.nwalkers >= 2 * ndim

    def test_sampler_results_reasonable(
        self, simple_hdu, gaussian_psf, tmp_path
    ):
        """Test that sampler finds approximately correct parameter values."""
        xc_true = simple_hdu.header["CRVAL1"]
        yc_true = simple_hdu.header["CRVAL2"]

        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))
        img.addpsf(gaussian_psf)

        beta = models.Beta(
            xc=priors.uniform(xc_true - 0.01, xc_true + 0.01),
            yc=priors.uniform(yc_true - 0.01, yc_true + 0.01),
            rc=0.005,
            Ic=1.0,
            beta=0.5,
        )
        mod = models.Model(beta)
        fit = fitting.fitter(img, mod)

        checkpoint = tmp_path / "test_reasonable.hdf5"
        fit.run(
            method="nautilus",
            n_live=2000,
            n_like_max=5000,
            checkpoint=str(checkpoint),
            verbose=False,
            resume=False,
        )

        xc_mean = np.average(fit.samples[:, 0], weights=fit.weights)
        yc_mean = np.average(fit.samples[:, 1], weights=fit.weights)
        assert abs(xc_mean - xc_true) < 0.01
        assert abs(yc_mean - yc_true) < 0.01


@pytest.mark.slow
class TestEmceeSpecific:
    """Specific tests for emcee sampler functionality."""

    @pytest.fixture
    def multi_param_fitter(self, simple_hdu, gaussian_psf):
        """Create a fitter with 2+ parameters for sampler testing."""
        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))

        xc = simple_hdu.header["CRVAL1"]
        yc = simple_hdu.header["CRVAL2"]

        gaussian = models.Gaussian(
            xc=priors.uniform(xc - 0.01, xc + 0.01),
            yc=priors.uniform(yc - 0.01, yc + 0.01),
            rs=0.005,
            Is=10.0,
        )
        mod = models.Model(gaussian)
        return fitting.fitter(img=img, mod=mod)

    def test_emcee_seed_reproducibility(self, multi_param_fitter, tmp_path):
        """Test that emcee produces reproducible results with same seed."""
        checkpoint1 = tmp_path / "emcee_seed1.hdf5"
        checkpoint2 = tmp_path / "emcee_seed2.hdf5"

        multi_param_fitter.run(
            method="emcee",
            nwalkers=10,
            nsteps=100,
            discard=0,
            seed=42,
            checkpoint=str(checkpoint1),
            progress=False,
            resume=False,
        )
        samples1 = multi_param_fitter.samples.copy()

        multi_param_fitter.run(
            method="emcee",
            nwalkers=10,
            nsteps=100,
            discard=0,
            seed=42,
            checkpoint=str(checkpoint2),
            progress=False,
            resume=False,
        )
        samples2 = multi_param_fitter.samples.copy()

        np.testing.assert_allclose(samples1, samples2, rtol=1e-10)

    def test_emcee_discard_thin_params(self, multi_param_fitter, tmp_path):
        """Test emcee discard and thin parameters."""
        checkpoint = tmp_path / "emcee_discard_thin.hdf5"

        nwalkers = 20
        nsteps = 500
        discard = 100
        thin = 5

        multi_param_fitter.run(
            method="emcee",
            nwalkers=nwalkers,
            nsteps=nsteps,
            discard=discard,
            thin=thin,
            checkpoint=str(checkpoint),
            progress=False,
            resume=False,
        )

        expected_samples = nwalkers * ((nsteps - discard) // thin)
        assert multi_param_fitter.samples.shape[0] == expected_samples

    def test_emcee_auto_thin_discard_with_converge(
        self, multi_param_fitter, tmp_path
    ):
        """Test emcee auto-sets thin and discard with converge=True."""
        checkpoint = tmp_path / "emcee_auto_thin.hdf5"

        multi_param_fitter.run(
            method="emcee",
            nwalkers=20,
            nsteps=2000,
            converge=True,
            check_every=200,
            tau_factor=10,
            checkpoint=str(checkpoint),
            progress=False,
            resume=False,
        )

        assert hasattr(multi_param_fitter, "tau")
        assert hasattr(multi_param_fitter, "tau_history")

    def test_emcee_alternative_param_names(self, multi_param_fitter, tmp_path):
        """Test emcee accepts alternative parameter names."""
        checkpoint = tmp_path / "emcee_alt_params.hdf5"

        multi_param_fitter.run(
            method="emcee",
            nwalkers=10,
            n_steps=200,
            n_burn=50,
            checkpoint=str(checkpoint),
            progress=False,
            resume=False,
        )

        assert multi_param_fitter.samples.shape[0] > 0

    def test_emcee_uniform_weights(self, multi_param_fitter, tmp_path):
        """Test that emcee produces uniform weights (all ones)."""
        checkpoint = tmp_path / "emcee_weights.hdf5"

        multi_param_fitter.run(
            method="emcee",
            nwalkers=10,
            nsteps=200,
            discard=50,
            checkpoint=str(checkpoint),
            progress=False,
            resume=False,
        )

        assert np.all(multi_param_fitter.weights == 1.0)
        assert len(multi_param_fitter.weights) == len(
            multi_param_fitter.samples
        )

    def test_emcee_sampler_attribute(self, multi_param_fitter, tmp_path):
        """Test that emcee sampler object is stored."""
        import emcee

        checkpoint = tmp_path / "emcee_sampler_attr.hdf5"

        multi_param_fitter.run(
            method="emcee",
            nwalkers=10,
            nsteps=100,
            discard=0,
            checkpoint=str(checkpoint),
            progress=False,
            resume=False,
        )

        assert isinstance(multi_param_fitter.sampler, emcee.EnsembleSampler)
        assert multi_param_fitter.sampler.iteration == 100
        assert multi_param_fitter.sampler.nwalkers == 10
