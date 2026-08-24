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

    @pytest.mark.parametrize("nrow,ncol", [(8, 8), (8, 9), (9, 8), (9, 9)])
    def test_real_full_ftype_equivalence(self, nrow, ncol):
        """ftype='real' must give the same logpdf as ftype='full'.

        rfft2 drops conjugate-redundant Fourier modes; the mode-multiplicity
        weighting in NormalFourier must exactly compensate for this,
        regardless of row/column parity (i.e. whether a Nyquist column
        exists).
        """
        rng = np.random.default_rng(0)
        cube = rng.normal(size=(50, nrow, ncol))
        data = rng.normal(size=(nrow, ncol))
        model = jp.array(rng.normal(size=(nrow, ncol))).flatten()
        mask = np.ones((nrow, ncol), dtype=int)

        n_real = noise.NormalFourier(cube=cube, ftype="real")
        n_full = noise.NormalFourier(cube=cube, ftype="full")

        n_real(data, mask)
        n_full(data, mask)

        logp_real = float(n_real.logpdf(model))
        logp_full = float(n_full.logpdf(model))

        assert logp_real == pytest.approx(logp_full, rel=1e-8)

    def test_clean_cube_excludes_no_modes(self):
        """A well-behaved noise cube should not trip the auto tol cutoff."""
        rng = np.random.default_rng(0)
        cube = rng.normal(size=(200, 16, 16))
        data = rng.normal(size=(16, 16))
        mask = np.ones((16, 16), dtype=int)

        n = noise.NormalFourier(cube=cube, ftype="real")
        n(data, mask)

        assert bool(jp.all(n.cmask))

    def test_auto_tol_excludes_crushed_mode(self):
        """A single artificially crushed Fourier mode should be excluded.

        Needs smoothing (the default) to have an effect: the exclusion
        compares each mode's raw estimate to its own local (smoothed)
        neighborhood, so with no smoothing there's nothing to compare
        against.
        """
        rng = np.random.default_rng(0)
        cube = rng.normal(size=(200, 16, 16))
        fft = np.fft.fft2(cube, axes=(-2, -1))
        fft[:, 3, 5] *= 1.00e-08
        cube = np.fft.ifft2(fft, axes=(-2, -1)).real
        data = rng.normal(size=(16, 16))
        mask = np.ones((16, 16), dtype=int)

        n = noise.NormalFourier(cube=cube, ftype="real")
        n(data, mask)

        assert not bool(n.cmask[3, 5])

    def test_tol_zero_disables_exclusion(self):
        """tol=0.0 should keep every mode, even a crushed one."""
        rng = np.random.default_rng(0)
        cube = rng.normal(size=(200, 16, 16))
        fft = np.fft.fft2(cube, axes=(-2, -1))
        fft[:, 3, 5] *= 1.00e-08
        cube = np.fft.ifft2(fft, axes=(-2, -1)).real
        data = rng.normal(size=(16, 16))
        mask = np.ones((16, 16), dtype=int)

        n = noise.NormalFourier(cube=cube, ftype="real", tol=0.00)
        n(data, mask)

        assert bool(n.cmask[3, 5])

    def test_red_noise_spectrum_does_not_over_exclude(self):
        """A genuinely red/pink (steeply k-dependent) noise spectrum should not trip the exclusion just for being legitimately low-power at high k.

        Regression test: an earlier version compared each mode's power to
        the *global* median, which flagged a large, contiguous band of
        legitimately low-power (but perfectly well-estimated) high-k modes
        as "outliers" whenever the spectrum had broad smooth dynamic range
        -- 26% of modes excluded on real data. Comparing against each
        mode's own local (smoothed) neighborhood instead should keep the
        excluded fraction small regardless of the spectrum's overall shape.
        """
        rng = np.random.default_rng(0)
        nside = 32
        kx = np.fft.fftfreq(nside)[:, None]
        ky = np.fft.fftfreq(nside)[None, :]
        kk = np.sqrt(kx**2 + ky**2)
        kk[0, 0] = kk[kk > 0].min()
        filt = 1.00 / kk

        white = rng.normal(size=(300, nside, nside))
        cube = np.fft.ifft2(
            np.fft.fft2(white, axes=(-2, -1)) * filt, axes=(-2, -1)
        ).real
        data = rng.normal(size=(nside, nside))
        mask = np.ones((nside, nside), dtype=int)

        n = noise.NormalFourier(cube=cube, ftype="real")
        n(data, mask)

        excluded_fraction = float((~n.cmask).mean())
        assert excluded_fraction < 0.05

    def test_covmodel_options(self):
        """Both covmodel values are accepted and stored."""
        icov = np.ones((4, 3))
        assert noise.NormalFourier(icov=icov).covmodel == "diagonal"
        n = noise.NormalFourier(icov=icov, covmodel="diagonal")
        assert n.covmodel == "diagonal"

    def test_invalid_covmodel_raises_error(self):
        """An unknown covmodel raises ValueError."""
        with pytest.raises(ValueError, match="covmodel must be"):
            noise.NormalFourier(icov=np.ones((4, 3)), covmodel="invalid")

    def test_banded_covmodel_requires_cube(self):
        """covmodel='banded' needs cube, and rejects cov/icov."""
        with pytest.raises(ValueError, match="requires a noise"):
            noise.NormalFourier(icov=np.ones((4, 3)), covmodel="banded")

        rng = np.random.default_rng(0)
        cube = rng.normal(size=(50, 8, 8))
        with pytest.raises(ValueError, match="does not accept cov/icov"):
            noise.NormalFourier(
                cov=np.ones((8, 5)), cube=cube, covmodel="banded"
            )

    def test_banded_covmodel_sets_expected_attributes(self):
        """Setup produces finite, correctly-shaped banded-mode attributes."""
        rng = np.random.default_rng(0)
        cube = rng.normal(size=(50, 8, 8))
        n = noise.NormalFourier(cube=cube, covmodel="banded", radius=3)

        assert n.chol_banded.shape[1] == 64
        assert n.chol_rows.shape == (64, n.bandwidth)
        assert n.chol_diag.shape == (64,)
        assert bool(jp.all(jp.isfinite(n.chol_diag)))
        assert bool(jp.isfinite(n.norm_full))

    @pytest.mark.parametrize("nrow,ncol", [(8, 8), (8, 9), (9, 8), (9, 9)])
    def test_banded_reduces_to_diagonal_when_unwindowed(self, nrow, ncol):
        """covmodel='banded' with apod=1 must match covmodel='diagonal'.

        With no apodization there is no mode coupling to approximate, so
        the two covariance treatments should agree on logpdf to tight
        tolerance, independent of grid parity.
        """
        rng = np.random.default_rng(0)
        cube = rng.normal(size=(300, nrow, ncol))
        data = rng.normal(size=(nrow, ncol))
        model = jp.array(rng.normal(size=(nrow, ncol))).flatten()
        mask = np.ones((nrow, ncol), dtype=int)
        apod = jp.ones((nrow, ncol))

        n_diag = noise.NormalFourier(cube=cube, ftype="real", smooth=0)
        n_banded = noise.NormalFourier(
            cube=cube,
            covmodel="banded",
            apod=apod,
            radius=max(nrow, ncol),
            smooth=0,
        )

        n_diag(data, mask)
        n_banded(data, mask)

        logp_diag = float(n_diag.logpdf(model))
        logp_banded = float(n_banded.logpdf(model))

        assert logp_diag == pytest.approx(logp_banded, rel=1e-8)

    def test_banded_matches_dense_reference_away_from_seam(self):
        """The banded covariance must match a from-scratch dense reference.

        Builds the same real covariance two independent ways: densely, by
        applying the real linear map (apodize -> FFT -> extract the
        reduced real degrees of freedom) to a basis of the image and
        propagating a circulant pixel-space covariance through it; and via
        the production banded-kernel construction. They should agree
        exactly for mode pairs that don't need to wrap around the
        ky=0/ny degrees-of-freedom seam (a documented limitation of the
        banded storage, checked separately).
        """
        rng = np.random.default_rng(1)
        ny, nx = 12, 12
        radius = 5

        raw = rng.uniform(0.5, 3.0, size=(ny, nx))
        iy, ix = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        my, mx = (-iy) % ny, (-ix) % nx
        spectrum = 0.50 * (raw + raw[my, mx])

        yy, xx = np.mgrid[0:ny, 0:nx]
        apod = (
            0.20
            + 0.80
            * np.cos(np.pi * (yy - ny / 2) / ny) ** 2
            * np.cos(np.pi * (xx - nx / 2) / nx) ** 2
        )

        dof = noise.NormalFourier._dof_index_maps(ny, nx)
        n = len(dof)
        npix = ny * nx

        c = np.real(np.fft.ifft2(spectrum)) / npix
        mm, nn = np.meshgrid(np.arange(npix), np.arange(npix), indexing="ij")
        my_, mx_ = np.divmod(mm, nx)
        ny_, nx_ = np.divmod(nn, nx)
        sigma_pixel = c[(my_ - ny_) % ny, (mx_ - nx_) % nx]

        a = np.zeros((n, npix))
        for j in range(npix):
            e = np.zeros((ny, nx))
            e.flat[j] = 1.00
            r = np.fft.fft2(apod * e)
            for i, (part, ky, kx) in enumerate(dof):
                a[i, j] = r[ky, kx].real if part == 0 else r[ky, kx].imag

        sigma_dense = a @ sigma_pixel @ a.T

        ab = noise.NormalFourier._build_banded_covariance(
            apod, spectrum, dof, radius, 0.00
        )
        bandwidth = ab.shape[0] - 1

        away = np.array([radius < ky < ny - 1 - radius for (_, ky, _) in dof])

        max_rel = 0.00
        for j in range(n):
            if not away[j]:
                continue
            for k in range(bandwidth + 1):
                i = j + k
                if i >= n or not away[i]:
                    continue
                diff = abs(ab[k, j] - sigma_dense[i, j])
                max_rel = max(max_rel, diff / np.max(np.abs(sigma_dense)))

        assert max_rel < 1.00e-10

    def test_banded_logpdf_finite_and_differentiable(self):
        """Logpdf under covmodel='banded' is finite and has finite grad."""
        import jax

        rng = np.random.default_rng(0)
        cube = rng.normal(size=(50, 8, 8))
        data = rng.normal(size=(8, 8))
        model = jp.array(rng.normal(size=(8, 8))).flatten()
        mask = np.ones((8, 8), dtype=int)

        n = noise.NormalFourier(cube=cube, covmodel="banded", radius=3)
        n(data, mask)

        assert bool(jp.isfinite(n.logpdf(model)))

        grad = jax.grad(n.logpdf)(model)
        assert bool(jp.all(jp.isfinite(grad)))

    def test_banded_logpdf_vmap_compatible(self):
        """Logpdf under covmodel='banded' composes with jax.vmap."""
        import jax

        rng = np.random.default_rng(0)
        cube = rng.normal(size=(50, 8, 8))
        data = rng.normal(size=(8, 8))
        mask = np.ones((8, 8), dtype=int)
        batch = jp.array(rng.normal(size=(4, 64)))

        n = noise.NormalFourier(cube=cube, covmodel="banded", radius=3)
        n(data, mask)

        out = jax.vmap(n.logpdf)(batch)
        assert out.shape == (4,)
        assert bool(jp.all(jp.isfinite(out)))

    def test_banded_noise_model_is_picklable(self):
        """A covmodel='banded' noise object survives a dill round trip."""
        import dill

        rng = np.random.default_rng(0)
        cube = rng.normal(size=(50, 8, 8))
        data = rng.normal(size=(8, 8))
        model = jp.array(rng.normal(size=(8, 8))).flatten()
        mask = np.ones((8, 8), dtype=int)

        n = noise.NormalFourier(cube=cube, covmodel="banded", radius=3)
        n(data, mask)

        reloaded = dill.loads(dill.dumps(n))

        assert float(reloaded.logpdf(model)) == pytest.approx(
            float(n.logpdf(model)), rel=1e-10
        )


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

    def test_getsigma_invalid_type_raises_error(self):
        """Test getsigma raises ValueError for non-scalar sigma."""
        n = noise.NormalRI(sigma=np.ones((8, 8)))
        n.data = jp.array(np.random.rand(8, 8))
        n.mask = jp.array(np.ones((8, 8), dtype=int))
        with pytest.raises(
            ValueError, match="Invalid type for noise parameter"
        ):
            n.getsigma()

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
