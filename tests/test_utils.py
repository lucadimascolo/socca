"""Tests for socca.utils module."""

import numpy as np
import pytest
from astropy.io import fits

from socca.utils import _img_loader, _reduce_axes, apodization


class TestImgLoader:
    """Tests for _img_loader function."""

    def test_load_primary_hdu(self, simple_hdu):
        """Test loading from a PrimaryHDU object."""
        result = _img_loader(simple_hdu)
        assert isinstance(result, fits.PrimaryHDU)
        assert result is simple_hdu

    def test_load_image_hdu(self, simple_hdu):
        """Test loading from an ImageHDU object."""
        image_hdu = fits.ImageHDU(
            data=simple_hdu.data.copy(), header=simple_hdu.header.copy()
        )
        result = _img_loader(image_hdu)
        assert isinstance(result, fits.ImageHDU)
        assert result is image_hdu

    def test_load_hdu_list(self, simple_hdu):
        """Test loading from an HDUList."""
        hdu_list = fits.HDUList([simple_hdu])
        result = _img_loader(hdu_list, idx=0)
        assert result is simple_hdu

    def test_load_hdu_list_with_index(self, simple_hdu):
        """Test loading a specific index from HDUList."""
        hdu0 = simple_hdu
        hdu1 = fits.ImageHDU(
            data=simple_hdu.data * 2, header=simple_hdu.header
        )
        hdu_list = fits.HDUList([hdu0, hdu1])
        result = _img_loader(hdu_list, idx=1)
        assert result is hdu1

    def test_load_from_file(self, temp_fits_file):
        """Test loading from a FITS file path."""
        result = _img_loader(temp_fits_file, idx=0)
        assert isinstance(result, (fits.PrimaryHDU, fits.ImageHDU))
        assert result.data is not None

    def test_invalid_input_raises_error(self):
        """Test that invalid input raises ValueError."""
        with pytest.raises(ValueError, match="img must be"):
            _img_loader([1, 2, 3])


class TestReduceAxes:
    """Tests for _reduce_axes function."""

    def test_reduce_3d_to_2d(self, simple_wcs_header):
        """Test reducing a 3D array to 2D."""
        data_3d = np.random.rand(1, 64, 64)
        header = simple_wcs_header.copy()
        header["NAXIS"] = 3
        header["NAXIS3"] = 1
        header["CRPIX3"] = 1
        header["CRVAL3"] = 1.0
        header["CDELT3"] = 1.0
        header["CTYPE3"] = "FREQ"

        hdu = fits.PrimaryHDU(data=data_3d, header=header)
        result = _reduce_axes(hdu)

        assert result.data.ndim == 2
        assert result.data.shape == (64, 64)
        assert result.header["NAXIS"] == 2
        assert "NAXIS3" not in result.header

    def test_reduce_4d_to_2d(self, simple_wcs_header):
        """Test reducing a 4D array to 2D."""
        data_4d = np.random.rand(1, 1, 64, 64)
        header = simple_wcs_header.copy()
        header["NAXIS"] = 4
        header["NAXIS3"] = 1
        header["NAXIS4"] = 1
        header["CRPIX3"] = 1
        header["CRVAL3"] = 1.0
        header["CDELT3"] = 1.0
        header["CTYPE3"] = "FREQ"
        header["CRPIX4"] = 1
        header["CRVAL4"] = 1.0
        header["CDELT4"] = 1.0
        header["CTYPE4"] = "STOKES"

        hdu = fits.PrimaryHDU(data=data_4d, header=header)
        result = _reduce_axes(hdu)

        assert result.data.ndim == 2
        assert result.data.shape == (64, 64)
        assert result.header["NAXIS"] == 2
        assert "NAXIS3" not in result.header
        assert "NAXIS4" not in result.header

    def test_2d_unchanged(self, simple_hdu):
        """Test that a 2D array is returned unchanged (except header cleanup)."""
        result = _reduce_axes(simple_hdu)
        assert result.data.shape == simple_hdu.data.shape
        np.testing.assert_array_equal(result.data, simple_hdu.data)

    def test_header_keywords_removed(self, simple_wcs_header):
        """Test that header keywords for axes 3 and 4 are removed."""
        data_3d = np.random.rand(1, 64, 64)
        header = simple_wcs_header.copy()
        header["NAXIS"] = 3
        header["NAXIS3"] = 1
        header["CRPIX3"] = 1
        header["CRVAL3"] = 1.0
        header["CDELT3"] = 1.0
        header["CTYPE3"] = "FREQ"
        header["CUNIT3"] = "Hz"
        header["CD3_3"] = 1.0
        header["PC3_3"] = 1.0

        hdu = fits.PrimaryHDU(data=data_3d, header=header)
        result = _reduce_axes(hdu)

        for key in ["CRPIX3", "CRVAL3", "CDELT3", "CTYPE3", "CUNIT3"]:
            assert key not in result.header

    def test_returns_primary_hdu(self, simple_hdu):
        """Test that result is always a PrimaryHDU."""
        result = _reduce_axes(simple_hdu)
        assert isinstance(result, fits.PrimaryHDU)


class TestApodize:
    """Tests for apodization function."""

    def test_output_shape_rectangular(self):
        """Test that a rectangular shape is respected."""
        result = apodization((20, 30), 5)
        assert result.shape == (20, 30)
        assert result.dtype == np.float64

    def test_nonpositive_width_returns_ones(self):
        """Test that width <= 0 returns an all-ones array."""
        result = apodization((10, 10), 0)
        np.testing.assert_array_equal(result, np.ones((10, 10)))

        result = apodization((10, 10), -3)
        np.testing.assert_array_equal(result, np.ones((10, 10)))

    def test_square_symmetric_under_rotation(self):
        """Test that a square window is unchanged under 180-degree rotation."""
        result = apodization((20, 20), 6)
        np.testing.assert_allclose(result, result[::-1, ::-1])

    def test_edges_taper_toward_zero(self):
        """Test that the very edge values are close to zero."""
        result = apodization((30, 30), 8)
        assert result[0, 15] < 0.05
        assert result[15, 0] < 0.05
        assert result[-1, 15] < 0.05
        assert result[15, -1] < 0.05

    def test_interior_is_unity(self):
        """Test that pixels beyond the tapered border are untouched."""
        width = 5
        result = apodization((30, 30), width)
        interior = result[2 * width : -2 * width, 2 * width : -2 * width]
        np.testing.assert_array_equal(interior, np.ones_like(interior))

    def test_linear_profile(self):
        """Test the linear taper profile matches a direct ramp."""
        width = 4
        result = apodization((20, 20), width, profile="lin")
        expected = np.arange(1, width + 1) / (width + 1)
        np.testing.assert_allclose(result[10, :width], expected)

    def test_invalid_profile_raises_error(self):
        """Test that an unknown profile raises ValueError."""
        with pytest.raises(ValueError, match="Unknown apodization profile"):
            apodization((10, 10), 3, profile="bogus")

    def test_cos_profile_matches_reference_values(self):
        """Test the cosine taper against hardcoded reference values."""
        width = 3
        result = apodization((10, 10), width)
        x = np.arange(1, width + 1) / (width + 1)
        expected = 0.50 * (1.00 - np.cos(np.pi * x))
        np.testing.assert_allclose(result[5, :width], expected)

    def test_alpha_default_matches_unmodified_taper(self):
        """Test that alpha=1.00 reproduces the default (unmodified) taper."""
        width = 4
        result_default = apodization((20, 20), width)
        result_alpha1 = apodization((20, 20), width, alpha=1.00)
        np.testing.assert_array_equal(result_default, result_alpha1)

    def test_alpha_reshapes_taper(self):
        """Test that alpha != 1 raises the taper to that power."""
        width = 4
        alpha = 2.50
        result = apodization((20, 20), width, alpha=alpha)
        x = np.arange(1, width + 1) / (width + 1)
        expected = (0.50 * (1.00 - np.cos(np.pi * x))) ** alpha
        np.testing.assert_allclose(result[10, :width], expected)

        result_lin = apodization((20, 20), width, profile="lin", alpha=alpha)
        expected_lin = x**alpha
        np.testing.assert_allclose(result_lin[10, :width], expected_lin)

    def test_fractional_width_matches_equivalent_pixel_count(self):
        """Test that a fraction resolves to round(fraction * axis length)."""
        result_frac = apodization((20, 20), 0.20)
        result_pix = apodization((20, 20), 4)
        np.testing.assert_array_equal(result_frac, result_pix)

    def test_fractional_width_scales_per_axis(self):
        """Test that a fraction is resolved independently for each axis."""
        result = apodization((10, 20), 0.20)
        result_ref = apodization((10, 20), 2)
        np.testing.assert_array_equal(result[:, 10], result_ref[:, 10])

        result_x = apodization((10, 20), 4)
        np.testing.assert_array_equal(result[5, :], result_x[5, :])
