"""Tests for socca.fitting module."""

from unittest.mock import patch
import warnings

import jax.numpy as jp
import numpy as np
import pytest
from astropy.io import fits

import socca.data as data
import socca.fitting as fitting
import socca.fitting.methods.pocomc as pocomc_backend
from socca.fitting.methods.pocomc import _Sampler, _latest_state
from socca.fitting.methods.utils import (
    check_resume,
    circular_quantile,
    circular_recenter,
    get_imp_weights,
    weighted_quantile,
)
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


class TestWeightedQuantile:
    """Tests for weighted_quantile (NumPy<2.0-compatible weighted quantile)."""

    def test_uniform_weights_matches_unweighted_quantile(self):
        """Uniform weights must reproduce the plain inverted_cdf quantile."""
        rng = np.random.default_rng(0)
        a = rng.normal(size=500)
        weights = np.ones(500)
        result = weighted_quantile(a, [0.16, 0.50, 0.84], weights)
        expected = np.quantile(a, [0.16, 0.50, 0.84], method="inverted_cdf")
        np.testing.assert_allclose(result, expected)

    def test_nonuniform_weights_shift_the_quantile(self):
        """Heavily up-weighting the low tail should pull the median down."""
        a = np.arange(10.0)
        weights = np.ones(10)
        weights[0] = 100.0
        result = weighted_quantile(a, 0.50, weights)
        assert result < 2.0

    def test_scalar_quantile_returns_scalar_shape(self):
        """A scalar `quantiles` input should drop the leading axis, like np.quantile."""
        a = np.arange(10.0)
        weights = np.ones(10)
        result = weighted_quantile(a, 0.50, weights)
        assert np.asarray(result).shape == ()

    def test_axis0_on_2d_array(self):
        """axis=0 on a 2D array reduces over samples per column independently."""
        rng = np.random.default_rng(1)
        a = rng.normal(size=(300, 4))
        weights = rng.uniform(0.1, 1.0, size=300)
        result = weighted_quantile(a, 0.50, weights, axis=0)
        assert result.shape == (4,)
        for k in range(4):
            expected = weighted_quantile(a[:, k], 0.50, weights)
            assert result[k] == pytest.approx(float(expected))


class TestCircularQuantile:
    """Tests for circular_recenter/circular_quantile (periodic statistics)."""

    def test_wrapped_boundary_quantile(self):
        """A posterior clustered at the wrap point must not be reported as a spurious ~half-period-wide spread by a naive quantile."""
        rng = np.random.default_rng(0)
        n = 20000
        theta = rng.normal(0.0, 0.15, n) % np.pi
        weights = np.ones(n) / n

        naive = weighted_quantile(theta, [0.16, 0.50, 0.84], weights)
        circ = circular_quantile(theta, weights, np.pi, [0.16, 0.50, 0.84])

        # Naive quantile spuriously spans most of the period.
        assert naive[2] - naive[0] > 1.0
        # Circular quantile correctly recovers a tight interval, wrapping
        # through the 0/pi boundary.
        upper = (circ[2] - circ[1] + np.pi / 2.00) % np.pi - np.pi / 2.00
        lower = (circ[1] - circ[0] + np.pi / 2.00) % np.pi - np.pi / 2.00
        assert upper == pytest.approx(0.15, abs=0.02)
        assert lower == pytest.approx(0.15, abs=0.02)

    def test_non_wrapping_matches_naive_quantile(self):
        """Away from the wrap boundary, circular and naive quantiles must agree (circular handling shouldn't distort the ordinary case)."""
        rng = np.random.default_rng(0)
        n = 20000
        theta = rng.normal(np.pi / 2.00, 0.15, n) % np.pi
        weights = np.ones(n) / n

        naive = weighted_quantile(theta, [0.16, 0.50, 0.84], weights)
        circ = circular_quantile(theta, weights, np.pi, [0.16, 0.50, 0.84])

        np.testing.assert_allclose(circ, naive, atol=1e-6)

    def test_recenter_returns_mean_in_range(self):
        """circular_recenter's returned mean must lie in [0, period)."""
        rng = np.random.default_rng(1)
        n = 5000
        theta = rng.normal(0.0, 0.10, n) % np.pi
        weights = np.ones(n) / n

        recentered, mean = circular_recenter(theta, weights, np.pi)
        assert 0.00 <= mean < np.pi
        # Recentered samples no longer straddle the wrap boundary.
        assert recentered.max() - recentered.min() < 1.0


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

    def test_parameters_grouped_by_component(self, simple_fitter, capsys):
        """Best-fit values are listed under 'Type (id)'."""
        simple_fitter.method = "optimizer"
        simple_fitter.results = type(
            "R", (), {"parameters": np.array([1.0, -2.0])}
        )()
        simple_fitter.parameters()
        lines = capsys.readouterr().out.split("\n")
        assert lines[1:4] == [
            "Best-fit parameters",
            "=" * 19,
            "Gaussian (comp_00)",
        ]
        assert lines[4] == "  xc :  1.0000E+00"
        assert lines[5] == "  yc : -2.0000E+00"

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

    def test_corner_selects_component_by_id(self, simple_hdu):
        """corner(component=...) plots the parameters of that component."""
        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))
        models.Sersic()
        models.Sersic()
        first = models.Gaussian(
            xc=priors.uniform(179.99, 180.01), yc=45.0, rs=0.005, Is=1.0
        )
        second = models.Gaussian(
            xc=180.0, yc=priors.uniform(44.99, 45.01), rs=0.005, Is=1.0
        )
        mod = models.Model(second)
        mod.addcomponent(first)
        fit = fitting.fitter(img, mod)
        fit.samples = np.column_stack([np.full(50, 45.0), np.full(50, 180.0)])
        fit.weights = np.ones(50) / 50
        fit.method = "nautilus"
        with patch("socca.plotting.corner.corner") as plot:
            for component in (first, first.id, 1):
                fit.plot.corner(component=component)
                assert np.all(plot.call_args.kwargs["data"] == 180.0)


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


class TestPeriodicAutoDetection:
    """Tests for automatic periodic-prior detection in fitter.run()."""

    @pytest.fixture
    def periodic_fitter(self, simple_hdu, gaussian_psf):
        """Fitter with a free theta prior spanning a full (pi) period."""
        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))
        xc = simple_hdu.header["CRVAL1"]
        yc = simple_hdu.header["CRVAL2"]

        gaussian = models.Gaussian(
            xc=priors.uniform(xc - 0.01, xc + 0.01),
            yc=priors.uniform(yc - 0.01, yc + 0.01),
            rs=0.005,
            Is=10.0,
            theta=priors.uniform(0.00, np.pi),
        )
        mod = models.Model(gaussian)
        return fitting.fitter(img=img, mod=mod)

    def test_fit_periodic_populated(self, periodic_fitter):
        """fit.periodic is parallel to fit.labels: pi at theta's position, None everywhere else."""
        assert len(periodic_fitter.periodic) == len(periodic_fitter.labels)
        theta_idx = periodic_fitter.labels.index("comp_00_theta")
        assert periodic_fitter.periodic[theta_idx] == pytest.approx(np.pi)
        assert all(
            p is None
            for i, p in enumerate(periodic_fitter.periodic)
            if i != theta_idx
        )

    def test_getquantiles_uses_circular_for_periodic_columns(
        self, periodic_fitter
    ):
        """fitter.getquantiles dispatches per column: circular_quantile for periodic parameters, plain weighted quantile otherwise."""
        rng = np.random.default_rng(0)
        n = 2000
        theta_idx = periodic_fitter.labels.index("comp_00_theta")
        n_free = len(periodic_fitter.labels)

        samples = rng.normal(0.5, 0.05, (n, n_free))
        samples[:, theta_idx] = rng.normal(0.0, 0.10, n) % np.pi
        weights = np.ones(n) / n

        periodic_fitter.samples = samples
        periodic_fitter.weights = weights

        qvals = periodic_fitter.getquantiles([0.16, 0.50, 0.84])
        assert qvals.shape == (n_free, 3)

        expected_theta = circular_quantile(
            samples[:, theta_idx], weights, np.pi, [0.16, 0.50, 0.84]
        )
        np.testing.assert_allclose(qvals[theta_idx], expected_theta)

        other_idx = 0 if theta_idx != 0 else 1
        expected_other = weighted_quantile(
            samples[:, other_idx], [0.16, 0.50, 0.84], weights
        )
        np.testing.assert_allclose(qvals[other_idx], expected_other)

    def test_nautilus_auto_injects_periodic(self, periodic_fitter):
        """A full-period theta prior gets forwarded as periodic=[...] to nautilus.Sampler, without warning again at run() time (the warning already fired when the component was added to the model)."""
        import socca.fitting.methods.nautilus as nautilus_mod

        theta_idx = periodic_fitter.labels.index("comp_00_theta")
        n_free = len(periodic_fitter.labels)

        with patch.object(
            nautilus_mod.nautilus, "Sampler", autospec=True
        ) as mock_cls:
            inst = mock_cls.return_value
            inst.posterior.return_value = (
                np.zeros((5, n_free)),
                np.zeros(5),
                None,
            )
            inst.log_z = -1.0
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                periodic_fitter.run(method="nautilus", n_live=10)
            msgs = [str(wi.message) for wi in w]

        assert not any("periodic" in m.lower() for m in msgs)
        assert mock_cls.call_args.kwargs["periodic"] == [theta_idx]

    def test_unbounded_prior_not_periodic(self, simple_hdu):
        """A theta prior with unbounded support is not made periodic."""
        import socca.fitting.methods.nautilus as nautilus_mod

        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))
        gaussian = models.Gaussian(
            xc=simple_hdu.header["CRVAL1"],
            yc=simple_hdu.header["CRVAL2"],
            rs=0.005,
            Is=priors.uniform(5.0, 15.0),
            theta=priors.normal(0.5, 0.1),
        )
        fit = fitting.fitter(img=img, mod=models.Model(gaussian))
        with patch.object(
            nautilus_mod.nautilus, "Sampler", autospec=True
        ) as mock_cls:
            inst = mock_cls.return_value
            inst.posterior.return_value = (np.zeros((5, 2)), np.zeros(5), None)
            inst.log_z = -1.0
            fit.run(method="nautilus", n_live=10)
        assert "periodic" not in mock_cls.call_args.kwargs

    def test_explicit_periodic_overrides_auto_detection(self, periodic_fitter):
        """Passing periodic= explicitly suppresses auto-detection and is forwarded as-is."""
        import socca.fitting.methods.nautilus as nautilus_mod

        n_free = len(periodic_fitter.labels)

        with patch.object(
            nautilus_mod.nautilus, "Sampler", autospec=True
        ) as mock_cls:
            inst = mock_cls.return_value
            inst.posterior.return_value = (
                np.zeros((5, n_free)),
                np.zeros(5),
                None,
            )
            inst.log_z = -1.0
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                periodic_fitter.run(method="nautilus", n_live=10, periodic=[])
            msgs = [str(wi.message) for wi in w]

        assert not any("periodic" in m.lower() for m in msgs)
        assert mock_cls.call_args.kwargs["periodic"] == []

    def test_explicit_periodic_accepts_parameter_names(self, periodic_fitter):
        """periodic=[...] accepts fit.labels parameter names, resolved to indices before being forwarded to the sampler."""
        import socca.fitting.methods.nautilus as nautilus_mod

        theta_idx = periodic_fitter.labels.index("comp_00_theta")
        n_free = len(periodic_fitter.labels)

        with patch.object(
            nautilus_mod.nautilus, "Sampler", autospec=True
        ) as mock_cls:
            inst = mock_cls.return_value
            inst.posterior.return_value = (
                np.zeros((5, n_free)),
                np.zeros(5),
                None,
            )
            inst.log_z = -1.0
            periodic_fitter.run(
                method="nautilus",
                n_live=10,
                periodic=["comp_00_theta"],
            )

        assert mock_cls.call_args.kwargs["periodic"] == [theta_idx]

    def test_explicit_periodic_rejects_unknown_name(self, periodic_fitter):
        """An unknown parameter name in periodic=[...] raises a clear error instead of silently mis-indexing."""
        with pytest.raises(ValueError, match="not a free parameter"):
            periodic_fitter.run(
                method="nautilus", n_live=10, periodic=["not_a_real_param"]
            )

    def test_numpyro_never_receives_periodic(self, periodic_fitter):
        """Backend gating: numpyro must never get an auto-injected periodic kwarg -- run_numpyro forwards leftover kwargs straight into mcmc.run(), which would TypeError on an unrecognized one."""
        import socca.fitting.methods.numpyro as numpyro_mod

        with patch.object(
            numpyro_mod.numpyro.infer, "MCMC", autospec=True
        ) as mock_mcmc_cls:
            inst = mock_mcmc_cls.return_value
            inst.get_samples.return_value = {
                label: np.zeros(5) for label in periodic_fitter.labels
            }
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                periodic_fitter.run(
                    method="numpyro", num_warmup=5, num_samples=5
                )
            msgs = [str(wi.message) for wi in w]

        assert not any("periodic" in m.lower() for m in msgs)
        assert "periodic" not in inst.run.call_args.kwargs


class TestLogSampledPriors:
    """Tests for priors sampled in another variable (LogPowerLaw)."""

    @pytest.fixture
    def setup(self, simple_hdu):
        """Build a Gaussian fit whose scale radius is sampled in log space."""
        models.Component.reset()
        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))
        truth = models.Model(
            models.Gaussian(xc=180.0, yc=45.0, rs=0.005, Is=10.0)
        )
        img.data = truth.getmap(img)
        models.Component.reset()
        gaussian = models.Gaussian(
            xc=180.0,
            yc=45.0,
            rs=priors.logpowerlaw(-1.0, 1e-4, 1e-1),
            Is=priors.uniform(5.0, 15.0),
        )
        return img, fitting.fitter(img=img, mod=models.Model(gaussian))

    def test_model_receives_linear_value(self, setup):
        """The likelihood evaluates the model at exp(theta)."""
        img, fit = setup
        best = fit._log_likelihood(jp.array([np.log(0.005), 10.0]))
        off = fit._log_likelihood(jp.array([0.005, 10.0]))
        assert float(best) > float(off)
        assert float(fit._log_prior(jp.array([np.log(0.005), 10.0]))) > -np.inf

    def test_optimizer_reports_linear_values(self, setup):
        """Optimizer results and pinits are in linear space."""
        img, fit = setup
        fit.run(method="optimizer", pinits=[0.004, 9.0])
        np.testing.assert_allclose(
            fit.results.parameters, [0.005, 10.0], rtol=1e-4
        )
        _, msmo, _ = fit.getmodel()
        np.testing.assert_allclose(msmo, img.data, atol=1e-6)

    def test_periodic_rejected(self, setup):
        """A log-sampled parameter cannot be passed as periodic."""
        img, fit = setup
        with pytest.raises(ValueError, match="cannot be periodic"):
            fit.run(method="nautilus", periodic=["comp_00_rs"])

    def test_invalid_pinits(self, setup):
        """Invalid initial values raise a clear error."""
        img, fit = setup
        with pytest.raises(ValueError, match="Unknown pinits"):
            fit.run(method="optimizer", pinits=None)
        with pytest.raises(ValueError, match="one value per free parameter"):
            fit.run(method="optimizer", pinits=[0.004, 9.0, 1.0])

    def test_pinits_outside_log_support(self, setup):
        """A non-positive start for a log-sampled parameter raises."""
        img, fit = setup
        with pytest.raises(ValueError, match="pinits must be finite"):
            fit.run(method="optimizer", pinits=[-0.001, 9.0])

    def test_pinits_outside_support(self, setup):
        """A start outside a prior's support raises."""
        img, fit = setup
        with pytest.raises(ValueError, match="comp_00_Is=20.0"):
            fit.run(method="optimizer", pinits=[0.004, 20.0])

    def test_optimizer_leaves_no_samples(self, setup):
        """After an optimizer run there are no posterior samples."""
        img, fit = setup
        fit.run(method="optimizer")
        assert not hasattr(fit, "samples")

    def test_optimizer_starts_from_linear_pinits(self, setup):
        """Initial values are given in linear space."""
        import socca.fitting.methods.optimizer as optimizer_mod

        img, fit = setup
        minimize = optimizer_mod.scipy.optimize.minimize
        with patch.object(
            optimizer_mod.scipy.optimize, "minimize", wraps=minimize
        ) as spy:
            fit.run(method="optimizer", pinits=[0.004, 9.0])
        start = fit._prior_transform(jp.asarray(spy.call_args.kwargs["x0"]))
        np.testing.assert_allclose(
            fit._to_model(start), [0.004, 9.0], rtol=1e-10
        )

    def test_samples_are_reported_in_linear_space(self, setup):
        """Sampler output in theta is converted to x once."""
        import socca.fitting.methods.nautilus as nautilus_mod

        img, fit = setup
        theta = np.array([[np.log(0.005), 10.0]] * 5)
        with patch.object(
            nautilus_mod.nautilus, "Sampler", autospec=True
        ) as mock_cls:
            inst = mock_cls.return_value
            inst.posterior.return_value = (theta, np.zeros(5), None)
            inst.log_z = -1.0
            fit.run(method="nautilus", n_live=10)
        np.testing.assert_allclose(fit.samples[:, 0], 0.005)
        np.testing.assert_allclose(fit.samples[:, 1], 10.0)
        _, msmo, _ = fit.getmodel()
        np.testing.assert_allclose(msmo, img.data, atol=1e-6)


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


class TestResume:
    """Resuming checks the checkpoint against the current model."""

    @staticmethod
    def _fitter(hdu, scale=1.0, width=0.01, positive=False):
        hdu = fits.PrimaryHDU(data=hdu.data * scale, header=hdu.header)
        img = data.Image(hdu, noise=noise.Normal(sigma=0.1))
        xc, yc = hdu.header["CRVAL1"], hdu.header["CRVAL2"]
        gaussian = models.Gaussian(
            xc=priors.uniform(xc - width, xc + width),
            yc=priors.uniform(yc - width, yc + width),
            rs=0.005,
            Is=priors.uniform(-10.0, 20.0) if positive else 10.0,
        )
        return fitting.fitter(
            img=img, mod=models.Model(gaussian, positive=positive)
        )

    @staticmethod
    def _dynesty(tmp_path, **kwargs):
        checkpoint = str(tmp_path / "d")
        return dict(kwargs, method="dynesty", nlive=50, checkpoint=checkpoint)

    def test_check_resume(self):
        """Matching saved values pass, others raise, nothing saved passes."""
        points = np.arange(12.0).reshape(6, 2)
        saved = points.sum(1)
        check_resume("ckpt", np.sum, points, saved)
        check_resume("ckpt", np.sum, points[:0], saved[:0])
        with pytest.raises(ValueError, match="ckpt was made with"):
            check_resume("ckpt", np.sum, points, saved + 1e-3)
        with pytest.raises(ValueError, match="ckpt was made with"):
            check_resume("ckpt", np.sum, points * 1e-12, saved * 2e-12)
        with pytest.raises(ValueError, match="ckpt was made with"):
            check_resume("ckpt", np.sum, points, (saved + 1) * (1 + 1e-9))
        check_resume("ckpt", lambda p: -np.inf, points, np.full(6, -1e300))
        check_resume("ckpt", lambda p: np.nan, points, np.full(6, np.nan))
        with pytest.raises(ValueError, match="ckpt was made with"):
            check_resume("ckpt", lambda p: p[3], points, saved)

    @pytest.mark.parametrize(
        "change, ncores", [({"scale": 2.0}, None), ({"width": 0.02}, 2)]
    )
    def test_dynesty(self, simple_hdu, tmp_path, change, ncores):
        """A dynesty run resumes the same model and refuses a changed one."""
        kwargs = self._dynesty(tmp_path, ncores=ncores)
        first = self._fitter(simple_hdu)
        first.run(maxiter=100, add_live=False, resume=False, **kwargs)
        with pytest.raises(ValueError, match="different model"):
            self._fitter(simple_hdu, **change).run(**kwargs)
        fit = self._fitter(simple_hdu)
        fit.run(**kwargs)
        assert len(fit.samples) > len(first.samples)
        assert np.isfinite(fit.logz)

    def test_dynesty_finished(self, simple_hdu, tmp_path):
        """A finished dynesty run continues to a stricter dlogz."""
        kwargs = self._dynesty(tmp_path)
        first = self._fitter(simple_hdu)
        first.run(dlogz=50.0, resume=False, **kwargs)
        same = self._fitter(simple_hdu)
        same.run(dlogz=50.0, **kwargs)
        assert len(same.samples) == len(first.samples)
        fit = self._fitter(simple_hdu)
        fit.run(**kwargs)
        assert len(fit.samples) > len(first.samples)
        assert np.isfinite(fit.logz)

    def test_dynesty_infinite_likelihood(self, simple_hdu, tmp_path):
        """Live points at -inf likelihood do not block resuming."""
        kwargs = self._dynesty(tmp_path)
        self._fitter(simple_hdu, positive=True).run(
            maxiter=5, add_live=False, resume=False, **kwargs
        )
        fit = self._fitter(simple_hdu, positive=True)
        fit.run(maxiter=50, **kwargs)
        assert np.isfinite(fit.logz)

    def test_dynesty_uses_current_fitter(self, simple_hdu, tmp_path):
        """A resumed dynesty run evaluates the current fitter, not a copy."""
        import dynesty

        restored = []
        restore = dynesty.NestedSampler.restore

        def capture(*args, **kwargs):
            restored.append(restore(*args, **kwargs))
            return restored[-1]

        kwargs = self._dynesty(tmp_path)
        self._fitter(simple_hdu).run(maxiter=100, resume=False, **kwargs)
        fit = self._fitter(simple_hdu)
        with patch.object(dynesty.NestedSampler, "restore", capture):
            fit.run(maxiter=200, **kwargs)
        sampler = restored[0]
        for func in [
            sampler.loglikelihood.loglikelihood.func,
            sampler.prior_transform.func,
        ]:
            assert func.__closure__[0].cell_contents is fit

    @pytest.mark.parametrize("first, then", [(2, None), (None, 2)])
    def test_dynesty_pool_switch(self, simple_hdu, tmp_path, first, then):
        """A dynesty run resumes after switching between serial and a pool."""
        kwargs = self._dynesty(tmp_path)
        self._fitter(simple_hdu).run(
            maxiter=100, resume=False, ncores=first, **kwargs
        )
        fit = self._fitter(simple_hdu)
        fit.run(maxiter=200, ncores=then, **kwargs)
        assert np.isfinite(fit.logz)

    @pytest.mark.parametrize("change", [{"scale": 2.0}, {"width": 0.02}])
    def test_emcee(self, simple_hdu, tmp_path, change):
        """An emcee run resumes the same model and refuses a changed one."""
        kwargs = dict(
            method="emcee",
            nwalkers=8,
            checkpoint=str(tmp_path / "emcee.hdf5"),
            progress=False,
        )
        self._fitter(simple_hdu).run(nsteps=20, resume=False, **kwargs)
        fit = self._fitter(simple_hdu)
        fit.run(nsteps=40, **kwargs)
        assert fit.sampler.iteration == 40
        with pytest.raises(ValueError, match="different model"):
            self._fitter(simple_hdu, **change).run(nsteps=60, **kwargs)

    @pytest.mark.parametrize("change", [{"scale": 2.0}, {"width": 0.02}])
    def test_nautilus(self, simple_hdu, tmp_path, change):
        """A nautilus run resumes the same model and refuses a changed one."""
        kwargs = dict(
            method="nautilus",
            n_live=100,
            checkpoint=str(tmp_path / "nautilus.hdf5"),
            verbose=False,
        )
        self._fitter(simple_hdu).run(n_like_max=500, resume=False, **kwargs)
        self._fitter(simple_hdu).run(n_like_max=1000, **kwargs)
        with pytest.raises(ValueError, match="different model"):
            self._fitter(simple_hdu, **change).run(n_like_max=1500, **kwargs)

    def test_no_default_checkpoint(self, simple_hdu, tmp_path, monkeypatch):
        """Without a checkpoint, emcee writes and resumes nothing."""
        monkeypatch.chdir(tmp_path)
        self._fitter(simple_hdu).run(
            method="emcee", nwalkers=8, nsteps=5, progress=False
        )
        assert not list(tmp_path.iterdir())

    def test_pocomc_latest_state(self, tmp_path):
        """Resuming pocomc takes the latest iteration, then the final state."""
        for tag in ["10", "90", "110"]:
            (tmp_path / f"pmc_{tag}.state").touch()
        assert _latest_state(str(tmp_path)).endswith("pmc_110.state")
        (tmp_path / "pmc_final.state").touch()
        assert _latest_state(str(tmp_path)).endswith("pmc_final.state")
        assert _latest_state(str(tmp_path / "missing")) is None

    @pytest.mark.parametrize("particles", [None, "jax"])
    def test_pocomc_load_state(self, tmp_path, particles):
        """A loaded pocomc state keeps the current functions and arrays."""
        prior = priors.pocomcPrior(
            [priors.uniform(0.0, 1.0), priors.uniform(0.0, 1.0)]
        )
        old = _Sampler(likelihood=lambda x: 0.0, prior=prior, n_active=10)
        old.t = 7
        old.pbar = None
        if particles:
            old.prior_samples = jp.zeros((10, 2))
            old.current_particles = {"x": jp.zeros((10, 2)), "iter": 3}
        old.save_state(tmp_path / "pmc_7.state")
        new = _Sampler(likelihood=lambda x: 1.0, prior=prior, n_active=10)
        new.load_state(tmp_path / "pmc_7.state")
        assert new.t == 7
        assert new.log_likelihood(np.zeros(2)) == 1.0
        if particles:
            new.prior_samples[0] = 1.0
            new.current_particles["x"][0] = 1.0

    def test_pocomc_infinite_likelihood(self, simple_hdu):
        """Prior draws with -inf likelihood do not stop pocomc."""
        fit = self._fitter(simple_hdu, positive=True)
        fit.run(method="pocomc", nlive=100, progress=False)
        assert np.isfinite(fit.logz)

    def test_pocomc_log_weights(self, simple_hdu, monkeypatch):
        """Results of pocomc use its log-weights, exponentiated once."""
        logw = np.log(np.array([0.1, 0.2, 0.7]))

        class Fake:
            def __init__(self, **kwargs):
                pass

            def run(self, **kwargs):
                pass

            def posterior(self, return_logw=False):
                weights = logw if return_logw else np.exp(logw)
                return np.zeros((3, 2)), weights, None, None

            def evidence(self):
                return 0.0, 0.0

        monkeypatch.setattr(pocomc_backend, "_Sampler", Fake)
        fit = self._fitter(simple_hdu)
        fit.run(method="pocomc", progress=False)
        np.testing.assert_allclose(fit.weights, [0.1, 0.2, 0.7])

    def test_unknown_keyword_warns(self, simple_hdu):
        """Keyword arguments a backend does not use are reported."""
        with pytest.warns(UserWarning, match="method='optimizer': bogus"):
            self._fitter(simple_hdu).run(method="optimizer", bogus=1)

    @pytest.mark.parametrize("verbose", [True, False])
    def test_nautilus_verbose(self, simple_hdu, capsys, verbose):
        """Progress of nautilus is printed only when verbose."""
        self._fitter(simple_hdu).run(
            method="nautilus", n_live=100, n_like_max=500, verbose=verbose
        )
        assert ("Status" in capsys.readouterr().out) == verbose


class TestPocoMCVectorize:
    """Tests for pocomc vectorize argument validation."""

    @pytest.fixture
    def simple_fitter(self, simple_hdu, gaussian_psf):
        """Create a simple fitter for validation testing."""
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
        return fitting.fitter(img=img, mod=mod)

    def test_vectorize_with_pool_raises(self, simple_fitter):
        """Test that vectorize=True with pool raises ValueError."""
        with pytest.raises(ValueError, match="pocomc does not support"):
            simple_fitter.run(
                method="pocomc", vectorize=True, pool=2, resume=False
            )

    def test_vectorize_with_ncores_raises(self, simple_fitter):
        """Test that vectorize=True with ncores raises ValueError."""
        with pytest.raises(ValueError, match="pocomc does not support"):
            simple_fitter.run(
                method="pocomc", vectorize=True, ncores=2, resume=False
            )


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


class TestLikelihoodWithPSF:
    """End-to-end checks of the likelihood with a PSF."""

    @staticmethod
    def _hdu(shape, data):
        header = fits.Header()
        header["CRPIX1"] = shape[1] // 2 + 1
        header["CRPIX2"] = shape[0] // 2 + 1
        header["CRVAL1"] = 150.0
        header["CRVAL2"] = 2.0
        header["CDELT1"] = -1.0 / 3600
        header["CDELT2"] = 1.0 / 3600
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        return fits.PrimaryHDU(data, header=header)

    @staticmethod
    def _model(img, point):
        models.Component.reset()
        ra, dec = img.wcs.all_pix2world(img.data.shape[1] // 2 + 1.3, 9.8, 0)
        mod = models.Model(
            models.Sersic(
                xc=float(ra),
                yc=float(dec),
                theta=0.3,
                e=0.3,
                re=3.0 / 3600,
                Ie=1.0,
                ns=1.0,
            )
        )
        ra, dec = img.wcs.all_pix2world(point[1], point[0], 0)
        mod.addcomponent(
            models.Point(
                xc=float(ra), yc=float(dec), Ic=priors.uniform(0.0, 10.0)
            )
        )
        return mod

    @pytest.mark.parametrize("shape", [(24, 31), (32, 32)])
    def test_normal_matches_direct_convolution(self, shape):
        """Masked, responded and exposed likelihood matches a reference."""
        from scipy.signal import fftconvolve

        rng = np.random.default_rng(3)
        sky = rng.normal(size=shape)
        resp, expo = 0.5 + rng.random(shape), 1.0 + rng.random(shape)
        hdu = self._hdu(shape, sky)
        img = data.Image(
            hdu,
            response=fits.PrimaryHDU(resp, header=hdu.header),
            exposure=fits.PrimaryHDU(expo, header=hdu.header),
            noise=noise.Normal(sigma=0.7),
        )
        k = np.arange(-3.0, 4.0)[:, None], np.arange(-4.0, 5.0)[None, :]
        psf = np.exp(-0.5 * ((k[1] / 1.4) ** 2 + (k[0] / 1.1) ** 2))
        psf /= psf.sum()
        img.addpsf(psf, normalize=False)
        mask = np.ones(shape, int)
        mask[3:7, 5:12] = 0
        img.addmask([mask])

        point = (shape[0] // 2 + 3, shape[1] // 2 - 4)
        mod = self._model(img, point)
        fit = fitting.fitter(img, mod)
        pp = jp.array([3.0])

        raw = np.asarray(mod.getmodel(img, pp, component=0)[0]) * resp
        raw[point] += 3.0 * resp[point]
        ref = fftconvolve(raw, psf, mode="same") * expo
        sel = np.asarray(fit.mask)
        expected = float(img.noise.logpdf(jp.asarray(ref[sel])))
        assert np.isclose(float(fit._log_likelihood(pp)), expected, atol=1e-4)

    def test_normalri_matches_direct_convolution(self):
        """The NormalRI likelihood matches a reference with a dirty beam."""
        from scipy.signal import fftconvolve

        shape = (32, 32)
        sky = np.random.default_rng(4).normal(size=shape)
        img = data.Image(
            self._hdu(shape, sky), noise=noise.NormalRI(sigma=1.0)
        )
        y, x = np.indices(shape) - 16.0
        r = np.hypot(x, y)
        beam = np.exp(-0.5 * r**2 / 2.0**2)
        beam -= 0.1 * np.exp(-0.5 * ((r - 8.0) / 2.0) ** 2)
        img.addpsf(beam, normalize=False)
        assert img.convolve.center == (16, 16)

        point = (18, 13)
        mod = self._model(img, point)
        fit = fitting.fitter(img, mod)
        pp = jp.array([2.0])

        raw = np.array(mod.getmodel(img, pp, component=0)[0])
        raw[point] += 2.0
        smo = fftconvolve(raw, beam[1:, 1:], mode="same")
        sel = np.asarray(fit.mask).astype(bool)
        expected = -0.5 * np.sum((smo - 2.0 * sky)[sel] * raw[sel])
        assert np.isclose(float(fit._log_likelihood(pp)), expected, atol=1e-4)

    def test_gradients_through_psf(self):
        """Gradients in the source position and flux match finite differences."""
        import jax

        shape = (30, 37)
        sky = np.random.default_rng(5).normal(size=shape)
        img = data.Image(self._hdu(shape, sky), noise=noise.Normal(sigma=1.0))
        k = np.arange(-5.0, 6.0)
        img.addpsf(np.exp(-0.5 * (k[:, None] ** 2 / 3.0 + k**2 / 2.0)))
        ra, dec = img.wcs.all_pix2world(17.3, 14.6, 0)
        models.Component.reset()
        point = models.Point(
            xc=priors.uniform(ra - 1e-3, ra + 1e-3),
            yc=priors.uniform(dec - 1e-3, dec + 1e-3),
            Ic=priors.uniform(0.0, 10.0),
        )
        fit = fitting.fitter(img, models.Model(point))

        pp = np.array(
            [
                {"comp_00_xc": ra, "comp_00_yc": dec, "comp_00_Ic": 5.0}[p]
                for p in fit.labels
            ],
            dtype=float,
        )
        grad = np.asarray(jax.grad(fit._log_likelihood)(jp.asarray(pp)))
        step = {"comp_00_xc": 1e-6, "comp_00_yc": 1e-6, "comp_00_Ic": 1e-4}
        for i, p in enumerate(fit.labels):
            dp = np.zeros_like(pp)
            dp[i] = step[p]
            fd = (
                float(fit._log_likelihood(jp.asarray(pp + dp)))
                - float(fit._log_likelihood(jp.asarray(pp - dp)))
            ) / (2.0 * step[p])
            assert np.isclose(grad[i], fd, rtol=1e-4)

        jitted = jax.jit(fit._log_likelihood)(jp.asarray(pp))
        assert np.isclose(
            jitted, fit._log_likelihood(jp.asarray(pp)), rtol=1e-12
        )


class TestLoadWithPSF:
    """Tests for saving and loading fits with a PSF."""

    @pytest.fixture
    def psf_fitter(self, simple_hdu):
        """Create a fitter with a PSF and a point source."""
        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))
        k = np.arange(-6.0, 7.0)
        img.addpsf(np.exp(-0.5 * ((k[:, None] - 0.4) ** 2 + k**2) / 2.0))
        models.Component.reset()
        point = models.Point(
            xc=simple_hdu.header["CRVAL1"] + 0.003,
            yc=simple_hdu.header["CRVAL2"] - 0.002,
            Ic=priors.uniform(0.0, 10.0),
        )
        return fitting.fitter(img, models.Model(point))

    def test_roundtrip(self, psf_fitter, tmp_path):
        """A dumped fit gives the same likelihood when loaded."""
        pp = jp.array([4.0])
        before = float(psf_fitter._log_likelihood(pp))
        psf_fitter.dump(tmp_path / "fit.pickle")
        fit = fitting.load(tmp_path / "fit.pickle")
        assert float(fit._log_likelihood(pp)) == before

    def test_new_components_skip_loaded_ids(self, psf_fitter, tmp_path):
        """In a new session, components added to a loaded fit get new ids."""
        psf_fitter.dump(tmp_path / "fit.pickle")
        models.Component.reset()
        fit = fitting.load(tmp_path / "fit.pickle")
        point = models.Point(xc=0.0, yc=0.0, Ic=1.0)
        assert point.id not in fit.mod._prefixes
        fit.mod.addcomponent(point)
        assert fit.mod._prefixes[-1] == point.id

    def test_legacy_operators_rebuilt(self, psf_fitter, tmp_path):
        """Fits saved with the old FFT conventions are rebuilt on load."""
        pp = jp.array([4.0])
        before = float(psf_fitter._log_likelihood(pp))
        psf_fitter.img.fft.center = jp.ones(psf_fitter.img.fft.pulse.shape)
        psf_fitter.img.convolve.psf_fft = jp.zeros_like(
            psf_fitter.img.convolve.psf_fft
        )
        del psf_fitter.img.grid.tx
        del psf_fitter.img.fft._header
        periodic = dict(psf_fitter.mod.periodic)
        del psf_fitter.mod.periodic
        for comp in psf_fitter.mod.components:
            del comp.periodic
        psf_fitter.dump(tmp_path / "fit.pickle")
        with pytest.warns(UserWarning, match="socca <= 1.4"):
            fit = fitting.load(tmp_path / "fit.pickle")
        assert fit.mod.periodic == periodic
        assert [c.periodic for c in fit.mod.components] == [
            c._periods for c in fit.mod.components
        ]
        assert fit.plot.fit is fit
        assert not hasattr(fit.img.fft, "center")
        assert hasattr(fit.img.grid, "tx")
        assert hasattr(fit.img.fft, "world2pix")
        assert np.isclose(float(fit._log_likelihood(pp)), before, rtol=1e-12)
