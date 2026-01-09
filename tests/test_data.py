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
        assert len(fft_spec.freq) == 2
        assert fft_spec.head is not None

    def test_pulse_shape(self, simple_hdu):
        """Test that pulse has correct shape (rfft2 output)."""
        fft_spec = data.FFTspec(simple_hdu)
        expected_shape = (
            simple_hdu.data.shape[0],
            simple_hdu.data.shape[1] // 2 + 1,
        )
        assert fft_spec.pulse.shape == expected_shape

    def test_freq_shape(self, simple_hdu):
        """Test that frequency grids have correct shape."""
        fft_spec = data.FFTspec(simple_hdu)
        for freq in fft_spec.freq:
            assert freq.shape == fft_spec.pulse.shape

    def test_head_contains_required_keys(self, simple_hdu):
        """Test that head dictionary has required keys."""
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
            assert key in fft_spec.head

    def test_shift_returns_phases(self, simple_hdu):
        """Test that shift method returns phase arrays."""
        fft_spec = data.FFTspec(simple_hdu)
        xc = simple_hdu.header["CRVAL1"]
        yc = simple_hdu.header["CRVAL2"]
        uphase, vphase = fft_spec.shift(xc, yc)
        assert uphase.shape == fft_spec.pulse.shape
        assert vphase.shape == fft_spec.pulse.shape


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
        assert not img.noise.mask[15, 15]

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
        assert jp.all(img.exp == 1.0)

    def test_response_default(self, simple_hdu):
        """Test default response is ones."""
        img = data.Image(simple_hdu)
        assert jp.all(img.resp == 1.0)

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
        assert img.psf_fft is not None

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
        """Test that PSF FFT is computed."""
        img = data.Image(simple_hdu)
        img.addpsf(gaussian_psf)
        expected_shape = (
            simple_hdu.data.shape[0],
            simple_hdu.data.shape[1] // 2 + 1,
        )
        assert img.psf_fft.shape == expected_shape

    def test_addpsf_larger_than_image_cropped(self, simple_hdu):
        """Test that PSF larger than image is cropped."""
        img = data.Image(simple_hdu)
        large_psf = np.ones((100, 100))
        img.addpsf(large_psf)
        assert img.psf.shape[0] <= img.data.shape[0]
        assert img.psf.shape[1] <= img.data.shape[1]


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
