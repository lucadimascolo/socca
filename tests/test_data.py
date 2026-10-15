"""Tests for socca.data module."""

import jax.numpy as jp
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

import socca.data as data
import socca.noise as noise


class TestWCSgrid:
    """Tests for WCSgrid class."""

    def test_initialization(self, simple_hdu):
        """Test basic WCSgrid initialization."""
        grid = data.WCSgrid(simple_hdu)
        assert grid.x is not None
        assert grid.y is not None

    def test_grid_shape(self, simple_hdu):
        """Test that grid has correct shape."""
        grid = data.WCSgrid(simple_hdu)
        assert grid.x.shape[1:] == simple_hdu.data.shape
        assert grid.y.shape[1:] == simple_hdu.data.shape

    def test_subgrid_factor(self, simple_hdu):
        """Test subgrid sampling factor."""
        subgrid = 2
        grid = data.WCSgrid(simple_hdu, subgrid=subgrid)
        assert grid.x.shape[0] == subgrid * subgrid
        assert grid.y.shape[0] == subgrid * subgrid

    def test_subgrid_1_single_layer(self, simple_hdu):
        """Test that subgrid=1 produces a single layer."""
        grid = data.WCSgrid(simple_hdu, subgrid=1)
        assert grid.x.shape[0] == 1
        assert grid.y.shape[0] == 1


class TestWCSgridGetmesh:
    """Tests for WCSgrid.getmesh static method."""

    def test_getmesh_from_hdu(self, simple_hdu):
        """Test getmesh with HDU input."""
        gridwx, gridwy = data.WCSgrid.getmesh(hdu=simple_hdu)
        assert gridwx.shape == simple_hdu.data.shape
        assert gridwy.shape == simple_hdu.data.shape

    def test_getmesh_from_header(self, simple_wcs_header):
        """Test getmesh with header input."""
        gridwx, gridwy = data.WCSgrid.getmesh(header=simple_wcs_header)
        assert gridwx.shape == (
            simple_wcs_header["NAXIS2"],
            simple_wcs_header["NAXIS1"],
        )
        assert gridwy.shape == (
            simple_wcs_header["NAXIS2"],
            simple_wcs_header["NAXIS1"],
        )

    def test_getmesh_both_raises_error(self, simple_hdu, simple_wcs_header):
        """Test that providing both hdu and header raises error."""
        with pytest.raises(ValueError, match="Either header or hdu"):
            data.WCSgrid.getmesh(hdu=simple_hdu, header=simple_wcs_header)

    def test_getmesh_neither_raises_error(self):
        """Test that providing neither hdu nor header raises error."""
        with pytest.raises(ValueError, match="Either header or hdu"):
            data.WCSgrid.getmesh()

    def test_getmesh_with_wcs(self, simple_hdu):
        """Test getmesh with explicit WCS object."""
        wcs = WCS(simple_hdu.header)
        gridwx, gridwy = data.WCSgrid.getmesh(hdu=simple_hdu, wcs=wcs)
        assert gridwx.shape == simple_hdu.data.shape


class TestFFTspec:
    """Tests for FFTspec class."""

    def test_initialization(self, simple_hdu):
        """Test FFTspec initialization."""
        fft_spec = data.FFTspec(simple_hdu)
        assert fft_spec.pulse is not None
        assert fft_spec.image_shape == simple_hdu.data.shape
        assert fft_spec.padded_shape == data.pad_size(simple_hdu.data.shape)
        assert len(fft_spec.freq) == 2
        assert fft_spec.header is not None

    def test_pulse_shape(self, simple_hdu):
        """Test that pulse has correct shape (rfft2 on padded grid)."""
        fft_spec = data.FFTspec(simple_hdu)
        padded = data.pad_size(simple_hdu.data.shape)
        expected_shape = (padded[0], padded[1] // 2 + 1)
        assert fft_spec.pulse.shape == expected_shape

    def test_freq_shape(self, simple_hdu):
        """Test that frequency grids have correct shape."""
        fft_spec = data.FFTspec(simple_hdu)
        for freq in fft_spec.freq:
            assert freq.shape == fft_spec.pulse.shape

    def test_header_contains_required_keys(self, simple_hdu):
        """Test that header dictionary has required keys."""
        fft_spec = data.FFTspec(simple_hdu)
        required_keys = [
            "CRPIX1",
            "CRPIX2",
            "CRVAL1",
            "CRVAL2",
            "CDELT1",
            "CDELT2",
            "NAXIS1",
            "NAXIS2",
        ]
        for key in required_keys:
            assert key in fft_spec.header

    def test_shift_returns_phase(self, simple_hdu):
        """Test that shift method returns a single complex phase array."""
        fft_spec = data.FFTspec(simple_hdu)
        xc = simple_hdu.header["CRVAL1"]
        yc = simple_hdu.header["CRVAL2"]
        phase = fft_spec.shift(xc, yc)
        assert phase.shape == fft_spec.pulse.shape
        assert jp.iscomplexobj(phase)


class TestImage:
    """Tests for Image class."""

    def test_initialization_from_hdu(self, simple_hdu):
        """Test Image initialization from HDU."""
        img = data.Image(simple_hdu)
        assert img.data is not None
        assert img.mask is not None
        assert img.grid is not None

    def test_initialization_from_file(self, temp_fits_file):
        """Test Image initialization from file path."""
        img = data.Image(temp_fits_file)
        assert img.data is not None
        assert img.hdu is not None

    def test_data_is_jax_array(self, simple_hdu):
        """Test that data is converted to JAX array."""
        img = data.Image(simple_hdu)
        assert isinstance(img.data, jp.ndarray)

    def test_mask_initialized_to_ones(self, simple_hdu):
        """Test that mask is initialized to all ones (valid)."""
        img = data.Image(simple_hdu)
        assert jp.all(img.mask == 1) or jp.all(img.noise.mask)

    def test_nan_pixels_masked(self, simple_wcs_header):
        """Test that NaN pixels are masked."""
        nan_data = np.ones((64, 64))
        nan_data[10:20, 10:20] = np.nan
        hdu = fits.PrimaryHDU(data=nan_data, header=simple_wcs_header)
        img = data.Image(hdu, noise=noise.Normal(sigma=0.1))
        assert img.mask[15, 15] == 0

    def test_grid_created(self, simple_hdu):
        """Test that WCSgrid is created."""
        img = data.Image(simple_hdu)
        assert isinstance(img.grid, data.WCSgrid)

    def test_fft_created(self, simple_hdu):
        """Test that FFTspec is created."""
        img = data.Image(simple_hdu)
        assert isinstance(img.fft, data.FFTspec)

    def test_noise_model_default(self, simple_hdu):
        """Test that default noise model is Normal."""
        img = data.Image(simple_hdu)
        assert isinstance(img.noise, noise.Normal)

    def test_noise_model_custom(self, simple_hdu):
        """Test custom noise model."""
        custom_noise = noise.Normal(sigma=0.5)
        img = data.Image(simple_hdu, noise=custom_noise)
        assert img.noise is custom_noise

    def test_exposure_default(self, simple_hdu):
        """Test default exposure is ones."""
        img = data.Image(simple_hdu)
        assert jp.all(img.exposure == 1.0)

    def test_response_default(self, simple_hdu):
        """Test default response is ones."""
        img = data.Image(simple_hdu)
        assert jp.all(img.response == 1.0)

    def test_subgrid_parameter(self, simple_hdu):
        """Test subgrid parameter is passed to grid."""
        img = data.Image(simple_hdu, subgrid=2)
        assert img.subgrid == 2
        assert img.grid.x.shape[0] == 4

    def test_psf_none_by_default(self, simple_hdu):
        """Test that PSF is None by default."""
        img = data.Image(simple_hdu)
        assert img.psf is None


class TestImageAddpsf:
    """Tests for Image.addpsf method."""

    def test_addpsf_from_array(self, simple_hdu, gaussian_psf):
        """Test adding PSF from numpy array."""
        img = data.Image(simple_hdu)
        img.addpsf(gaussian_psf)
        assert img.psf is not None
        assert img.convolve is not None
        assert img.convolve.psf_fft is not None

    def test_addpsf_normalizes_by_default(self, simple_hdu, gaussian_psf):
        """Test that PSF is normalized by default."""
        img = data.Image(simple_hdu)
        psf_unnorm = gaussian_psf * 10.0
        img.addpsf(psf_unnorm, normalize=True)
        assert np.isclose(img.psf.sum(), 1.0)

    def test_addpsf_no_normalize(self, simple_hdu, gaussian_psf):
        """Test PSF without normalization."""
        img = data.Image(simple_hdu)
        psf_scaled = gaussian_psf * 2.0
        img.addpsf(psf_scaled, normalize=False)
        assert np.isclose(img.psf.sum(), 2.0)

    def test_addpsf_from_file(self, simple_hdu, temp_psf_file):
        """Test adding PSF from file."""
        img = data.Image(simple_hdu)
        img.addpsf(temp_psf_file)
        assert img.psf is not None

    def test_addpsf_from_hdu(
        self, simple_hdu, gaussian_psf, simple_wcs_header
    ):
        """Test adding PSF from HDU."""
        img = data.Image(simple_hdu)
        psf_header = simple_wcs_header.copy()
        psf_header["NAXIS1"] = gaussian_psf.shape[1]
        psf_header["NAXIS2"] = gaussian_psf.shape[0]
        psf_hdu = fits.PrimaryHDU(data=gaussian_psf, header=psf_header)
        img.addpsf(psf_hdu)
        assert img.psf is not None

    def test_addpsf_fft_computed(self, simple_hdu, gaussian_psf):
        """Test that PSF FFT is computed on the padded grid."""
        img = data.Image(simple_hdu)
        img.addpsf(gaussian_psf)
        padded = data.pad_size(simple_hdu.data.shape)
        expected_shape = (padded[0], padded[1] // 2 + 1)
        assert img.convolve.psf_fft.shape == expected_shape

    def test_addpsf_larger_than_image_kept(self, simple_hdu):
        """Test that a PSF larger than the image is stored unchanged."""
        img = data.Image(simple_hdu)
        large_psf = np.ones((100, 100))
        img.addpsf(large_psf)
        assert img.psf.shape == large_psf.shape
        padded = data.pad_size(simple_hdu.data.shape)
        assert img.convolve.psf_fft.shape == (padded[0], padded[1] // 2 + 1)


class TestConvolveCentering:
    """Tests for the automatic recentring of the PSF by Convolve."""

    @staticmethod
    def _image(shape):
        header = fits.Header()
        header["CRPIX1"] = shape[1] // 2 + 1
        header["CRPIX2"] = shape[0] // 2 + 1
        header["CRVAL1"] = 180.0
        header["CRVAL2"] = 0.0
        header["CDELT1"] = -0.001
        header["CDELT2"] = 0.001
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        return data.Image(
            fits.PrimaryHDU(np.zeros(shape), header=header),
            noise=noise.Normal(sigma=1.0),
        )

    @staticmethod
    def _stamp(shape, seed=0):
        """Random stamp, point-symmetric about pixel n // 2."""
        m = [2 * ((k - 1) // 2) + 1 for k in shape]
        core = np.random.default_rng(seed).random(m)
        psf = np.zeros(shape)
        psf[shape[0] - m[0] :, shape[1] - m[1] :] = core + core[::-1, ::-1]
        return psf / psf.sum()

    @staticmethod
    def _gauss(shape, center, fwhm=4.0):
        y, x = np.indices(shape).astype(float)
        s = fwhm / 2.355
        r2 = (y - center[0]) ** 2 + (x - center[1]) ** 2
        return np.exp(-0.5 * r2 / s**2)

    @staticmethod
    def _placed(psf, center, shape, pos):
        out = np.zeros(shape)
        y0, x0 = pos[0] - center[0], pos[1] - center[1]
        for i, j in np.ndindex(psf.shape):
            if 0 <= y0 + i < shape[0] and 0 <= x0 + j < shape[1]:
                out[y0 + i, x0 + j] = psf[i, j]
        return out

    @staticmethod
    def _pulse(img, pos):
        sky = np.zeros(img.data.shape)
        sky[pos] = 1.0
        return np.asarray(img.convolve(sky))

    @pytest.mark.parametrize("shape", [(32, 32), (33, 33), (32, 35)])
    @pytest.mark.parametrize(
        "kshape", [(9, 9), (8, 8), (8, 11), (40, 40), (41, 70)]
    )
    def test_reproduces_centred_stamp(self, shape, kshape):
        """A stamp centred on a pixel is used with its values untouched."""
        img = self._image(shape)
        psf = self._stamp(kshape)
        img.addpsf(psf)
        center = (kshape[0] // 2, kshape[1] // 2)
        assert img.convolve.center == center
        for pos in [(shape[0] // 2, shape[1] // 2), (0, 0), (shape[0] - 1, 3)]:
            np.testing.assert_allclose(
                self._pulse(img, pos),
                self._placed(psf, center, shape, pos),
                rtol=0,
                atol=1e-14,
            )

    @pytest.mark.parametrize("shape", [(32, 32), (33, 33), (32, 35)])
    def test_matches_direct_convolution(self, shape):
        """Odd centred stamps reproduce a direct convolution."""
        from scipy.signal import fftconvolve

        img = self._image(shape)
        psf = self._stamp((9, 11))
        img.addpsf(psf)

        sky = np.random.default_rng(1).normal(size=shape)
        expected = fftconvolve(sky, psf, mode="same")
        np.testing.assert_allclose(img.convolve(sky), expected, atol=1e-12)

    @pytest.mark.parametrize(
        "center", [(10.0, 12.0), (12.5, 9.5), (9.3, 13.8), (14.71, 8.06)]
    )
    def test_recentres_offset_psf(self, center):
        """An offset PSF is moved onto the source, and only moved."""
        img = self._image((32, 32))
        psf = self._gauss((23, 24), center)
        img.addpsf(psf)
        np.testing.assert_allclose(img.convolve.center, center, atol=1e-4)

        pos = (15, 17)
        expected = self._gauss((32, 32), pos) / psf.sum()
        np.testing.assert_allclose(
            self._pulse(img, pos), expected, atol=1e-4 * expected.max()
        )

    def test_position_in_stamp_does_not_matter(self):
        """An asymmetric PSF gives the same model wherever it sits."""
        core = np.random.default_rng(3).random((9, 11))
        a, b = np.zeros((25, 25)), np.zeros((25, 25))
        a[8:17, 7:18] = core
        b[3:12, 11:22] = core
        img = self._image((32, 32))
        img.addpsf(a)
        out_a = self._pulse(img, (16, 15))
        img.addpsf(b)
        out_b = self._pulse(img, (16, 15))
        np.testing.assert_allclose(out_a, out_b, atol=1e-12)

    def test_psf_values_untouched(self):
        """The stored kernel keeps the input values."""
        img = self._image((32, 32))
        psf = self._gauss((15, 16), (6.3, 8.7))
        img.addpsf(psf, normalize=False)
        np.testing.assert_array_equal(img.psf, psf)

    def test_readd_is_identical(self):
        """Re-adding the stored PSF leaves the operator unchanged."""
        img = self._image((32, 32))
        img.addpsf(self._gauss((15, 16), (6.3, 8.7)))
        before = np.asarray(img.convolve.psf_fft)
        img.addpsf(img.psf, normalize=False)
        np.testing.assert_array_equal(img.convolve.psf_fft, before)

    @pytest.mark.parametrize("csize", [15, 16])
    def test_cutout_keeps_psf(self, csize):
        """A cutout keeps the PSF stamp and its centre."""
        from astropy.coordinates import SkyCoord

        img = self._image((32, 32))
        psf = self._gauss((15, 16), (6.3, 8.7))
        img.addpsf(psf)
        center = img.convolve.center
        before = self._pulse(img, (16, 16))
        ra, dec = img.wcs.all_pix2world(16, 16, 0)
        img.cutout(SkyCoord(float(ra), float(dec), unit="deg"), csize)
        np.testing.assert_allclose(img.psf, psf / psf.sum(), rtol=1e-14)
        assert img.convolve.center == center
        h = csize // 2
        np.testing.assert_allclose(
            self._pulse(img, (h, h)),
            before[16 - h : 16 - h + csize, 16 - h : 16 - h + csize],
            atol=1e-14,
        )

    def test_no_recentring(self):
        """center=None uses pixel n // 2 with the values untouched."""
        img = self._image((32, 32))
        psf = self._gauss((15, 16), (6.3, 8.7))
        img.addpsf(psf, normalize=False, center=None)
        assert img.convolve.center == (7, 8)
        np.testing.assert_allclose(
            self._pulse(img, (16, 16)),
            self._placed(psf, (7, 8), (32, 32), (16, 16)),
            rtol=0,
            atol=1e-14,
        )

    def test_explicit_center(self):
        """An explicit centre overrides the estimate."""
        img = self._image((32, 32))
        psf = self._gauss((23, 24), (9.3, 13.8))
        img.addpsf(psf, center=(9.3, 13.8))
        assert img.convolve.center == (9.3, 13.8)
        expected = self._gauss((32, 32), (15, 17)) / psf.sum()
        np.testing.assert_allclose(
            self._pulse(img, (15, 17)), expected, atol=1e-4 * expected.max()
        )
        img.addpsf(psf, center=(5, 11))
        np.testing.assert_allclose(
            self._pulse(img, (15, 17)),
            self._placed(psf / psf.sum(), (5, 11), (32, 32), (15, 17)),
            rtol=0,
            atol=1e-14,
        )

    def test_center_from_constructor(self, simple_hdu):
        """The constructor forwards the centre option."""
        psf = self._gauss((15, 16), (6.3, 8.7))
        img = data.Image(simple_hdu, addpsf=dict(img=psf, center=None))
        assert img.convolve.center == (7, 8)

    def test_invalid_center_option(self):
        """Unknown centre strings are rejected."""
        img = self._image((32, 32))
        with pytest.raises(ValueError):
            img.addpsf(self._stamp((8, 8)), center="peak")

    @pytest.mark.parametrize("center", [(-1.0, 4.0), (4.0, 8.0), (1.0,)])
    def test_invalid_center(self, center):
        """Centres outside the kernel are rejected."""
        with pytest.raises(ValueError):
            data.Convolve(self._stamp((8, 8)), (32, 32), center=center)


class TestPSFCenter:
    """Tests for the estimate of the PSF centre."""

    @staticmethod
    def _gauss(shape, center, sigma=1.5):
        y, x = np.indices(shape).astype(float)
        r2 = (y - center[0]) ** 2 + (x - center[1]) ** 2
        return np.exp(-0.5 * r2 / sigma**2)

    @staticmethod
    def _beam(shape, center):
        """Interferometer-like beam with negative sidelobes and zero sum."""
        y, x = np.indices(shape).astype(float)
        r = np.hypot(y - center[0], x - center[1])
        beam = np.exp(-0.5 * r**2 / 2.0**2)
        beam -= 0.1 * np.exp(-0.5 * ((r - 10.0) / 3.0) ** 2)
        return beam - beam.mean()

    @pytest.mark.parametrize(
        "shape, center",
        [((15, 15), (7, 7)), ((16, 16), (8, 8)), ((16, 16), (7.5, 7.5))],
    )
    def test_symmetric_stamp_is_exact(self, shape, center):
        """Whole- and half-pixel centres of symmetric stamps are exact."""
        psf = self._gauss(shape, center)
        np.testing.assert_allclose(data._psf_center(psf), center, atol=1e-12)

    @pytest.mark.parametrize(
        "center, atol",
        [((12.3, 9.8), 1e-6), ((4.6, 17.25), 5e-3), ((20.1, 3.4), 5e-3)],
    )
    def test_subpixel_centre(self, center, atol):
        """Sub-pixel centres are recovered, also with the wings cut off."""
        psf = self._gauss((25, 23), center)
        np.testing.assert_allclose(data._psf_center(psf), center, atol=atol)

    def test_zero_sum_beam(self):
        """A beam summing to zero is centred on its reference pixel."""
        assert data._psf_center(self._beam((64, 64), (32, 32))) == (32, 32)

    def test_follows_integer_shifts(self):
        """Moving a PSF in its stamp moves the estimate by the same amount."""
        core = np.random.default_rng(0).random((9, 11))
        a, b = np.zeros((25, 25)), np.zeros((25, 25))
        a[8:17, 7:18] = core
        b[3:12, 11:22] = core
        shift = np.subtract(data._psf_center(b), data._psf_center(a))
        np.testing.assert_allclose(shift, (-5, 4), atol=1e-12)

    def test_ignores_nan(self):
        """NaN pixels are ignored."""
        psf = self._gauss((15, 15), (7, 7))
        psf[0, 0] = np.nan
        assert data._psf_center(psf) == (7, 7)

    def test_empty_psf(self):
        """An empty PSF is rejected."""
        with pytest.raises(ValueError, match="empty"):
            data._psf_center(np.zeros((9, 9)))

    @pytest.mark.parametrize("seed", [272, 1739, 2069])
    def test_noise_stays_in_stamp(self, seed):
        """Noise-like stamps give a centre within the stamp."""
        rng = np.random.default_rng(seed)
        psf = rng.normal(size=tuple(rng.integers(1, 12, 2)))
        c = np.asarray(data._psf_center(psf))
        assert np.all(c >= 0) and np.all(c <= np.array(psf.shape) - 1)

    @pytest.mark.parametrize("center", [(8.2, 20.7), (18.3, 12.6)])
    def test_truncated_sidelobes(self, center):
        """Sidelobes cut by the stamp edge do not bias the centre."""
        y, x = np.indices((33, 33)).astype(float)
        r = np.hypot(y - center[0], x - center[1])
        beam = np.exp(-0.25 * r**2)
        beam -= 0.1 * np.exp(-0.5 * ((r - 10.0) / 3.0) ** 2)
        np.testing.assert_allclose(data._psf_center(beam), center, atol=2e-3)


class TestConvolveEdgeCases:
    """Tests for small or unusual images and PSFs."""

    @staticmethod
    def _image(shape):
        header = fits.Header()
        header["CRPIX1"] = shape[1] // 2 + 1
        header["CRPIX2"] = shape[0] // 2 + 1
        header["CRVAL1"] = 180.0
        header["CRVAL2"] = 0.0
        header["CDELT1"] = -0.001
        header["CDELT2"] = 0.001
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        return data.Image(
            fits.PrimaryHDU(np.zeros(shape), header=header),
            noise=noise.Normal(sigma=1.0),
        )

    @pytest.mark.parametrize("shape", [(2, 2), (3, 3), (4, 7)])
    @pytest.mark.parametrize(
        "kshape", [(1, 1), (1, 3), (3, 1), (2, 2), (1, 9)]
    )
    def test_tiny_images_and_stamps(self, shape, kshape):
        """Tiny images and thin stamps give finite models."""
        img = self._image(shape)
        img.addpsf(np.random.default_rng(0).random(kshape) + 0.1)
        sky = np.zeros(shape)
        sky[shape[0] // 2, shape[1] // 2] = 1.0
        assert np.all(np.isfinite(img.convolve(sky)))

    def test_single_pixel_psf_is_identity(self):
        """A 1x1 PSF leaves the image unchanged."""
        img = self._image((16, 17))
        img.addpsf(np.ones((1, 1)))
        sky = np.random.default_rng(1).normal(size=(16, 17))
        np.testing.assert_allclose(img.convolve(sky), sky, atol=1e-14)

    @pytest.mark.parametrize("kind", ["int", "float32", "jax"])
    def test_input_types(self, kind):
        """Integer, single-precision and JAX PSFs match float64 ones."""
        psf = np.random.default_rng(2).integers(1, 9, size=(7, 7))
        cast = {
            "int": psf,
            "float32": psf.astype(np.float32),
            "jax": jp.asarray(psf),
        }[kind]
        ref, img = self._image((16, 16)), self._image((16, 16))
        ref.addpsf(psf.astype(float))
        img.addpsf(cast)
        assert img.convolve.center == ref.convolve.center
        np.testing.assert_allclose(
            img.convolve.psf_fft, ref.convolve.psf_fft, atol=1e-12
        )

    def test_empty_psf(self):
        """An empty PSF is rejected."""
        img = self._image((16, 16))
        with pytest.raises(ValueError):
            img.addpsf(np.zeros((5, 5)), normalize=False)


class TestImageAddmask:
    """Tests for Image.addmask method."""

    def test_addmask_from_array(self, simple_hdu):
        """Test adding mask from numpy array."""
        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))
        mask_array = np.ones(simple_hdu.data.shape)
        mask_array[10:20, 10:20] = 0
        img.addmask(regions=[mask_array], combine=False)
        assert img.mask[15, 15] == 0
        assert img.mask[0, 0] == 1

    def test_addmask_combine_true(self, simple_hdu):
        """Test combining masks with existing mask."""
        img = data.Image(simple_hdu, noise=noise.Normal(sigma=0.1))
        mask1 = np.ones(simple_hdu.data.shape)
        mask1[0:10, :] = 0
        mask2 = np.ones(simple_hdu.data.shape)
        mask2[:, 0:10] = 0
        img.addmask(regions=[mask1], combine=False)
        img.addmask(regions=[mask2], combine=True)
        assert img.mask[5, 5] == 0
        assert img.mask[5, 50] == 0
        assert img.mask[50, 5] == 0


class TestHduMask:
    """Tests for _hdu_mask helper function."""

    def test_hdu_mask_reprojection(self, simple_hdu):
        """Test that mask HDU is reprojected to target WCS."""
        mask_data = np.ones(simple_hdu.data.shape)
        mask_data[20:40, 20:40] = 0
        mask_hdu = fits.PrimaryHDU(
            data=mask_data, header=simple_hdu.header.copy()
        )
        result = data._hdu_mask(mask_hdu, simple_hdu)
        assert result.shape == simple_hdu.data.shape
        assert np.all((result == 0) | (result == 1))
